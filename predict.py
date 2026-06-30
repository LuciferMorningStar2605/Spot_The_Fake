import os
import sys
import pickle
from PIL import Image
import numpy as np
import cv2
import pillow_heif

# Register HEIF opener
pillow_heif.register_heif_opener()

# Image dimensions
IMG_SIZE = 512

# Precompute distances and masks
y_coords, x_coords = np.indices((IMG_SIZE, IMG_SIZE))
center_y, center_x = IMG_SIZE // 2, IMG_SIZE // 2
distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)

# Define radial bins (concentric rings from radius 20 to 250)
num_bins = 10
bin_edges = np.linspace(20, 250, num_bins + 1)
radial_masks = []
for i in range(num_bins):
    mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
    radial_masks.append(mask)

# High-frequency non-axis mask for Cr/Cb/Sat
valid_mask_chroma = (distances > 40) & (np.abs(x_coords - center_x) > 6) & (np.abs(y_coords - center_y) > 6)

def extract_features(path: str):
    """
    Extracts the identical 80-feature vector as train.py.
    """
    try:
        img = Image.open(path)
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_np = np.array(img.convert("RGB"))
        
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        
        sat = hsv[:,:,1]
        cr = ycrcb[:,:,1]
        cb = ycrcb[:,:,2]
        
        features = []
        
        # 1. Grayscale Sharpness & Edge Features
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(lap_var)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        features.extend([np.mean(sobel_mag), np.std(sobel_mag)])
        
        # 2. Glare Features
        glare_pixels = np.sum(gray > 240) / (IMG_SIZE * IMG_SIZE)
        max_brightness = np.max(gray)
        bright_mask = gray > 200
        bright_std = np.std(gray[bright_mask]) if np.sum(bright_mask) > 0 else 0.0
        features.extend([glare_pixels, float(max_brightness), bright_std])
        
        # 3. Texture Uniformity
        cells = [gray[i*64:(i+1)*64, j*64:(j+1)*64] for i in range(8) for j in range(8)]
        cell_stds = [np.std(cell) for cell in cells]
        uniformity = np.std(cell_stds)
        features.append(uniformity)
        
        # 4. Grayscale FFT Features
        f_gray = np.fft.fft2(gray)
        fshift_gray = np.fft.fftshift(f_gray)
        mag_gray = 20 * np.log(np.abs(fshift_gray) + 1e-8)
        
        mag_blur_gray = cv2.GaussianBlur(mag_gray, (21, 21), 0)
        mag_diff_gray = mag_gray - mag_blur_gray
        
        local_max = cv2.dilate(mag_diff_gray, np.ones((9, 9)))
        peaks_mask = (mag_diff_gray == local_max) & (mag_diff_gray > 15.0) & (distances > 80)
        peaks_count = np.sum(peaks_mask)
        features.append(float(peaks_count))
        
        features.extend([np.mean(mag_diff_gray), np.std(mag_diff_gray), np.max(mag_diff_gray)])
        high_freq_mag_gray = mag_gray[distances > 80]
        features.extend([np.mean(high_freq_mag_gray), np.std(high_freq_mag_gray), np.max(high_freq_mag_gray)])
        
        for mask in radial_masks:
            mag_bin = mag_gray[mask]
            mag_diff_bin = mag_diff_gray[mask]
            if len(mag_bin) > 0:
                features.extend([np.mean(mag_bin), np.std(mag_bin), np.max(mag_bin)])
                features.extend([np.mean(mag_diff_bin), np.std(mag_diff_bin), np.max(mag_diff_bin)])
            else:
                features.extend([0, 0, 0, 0, 0, 0])
                
        # 5. Chrominance / Saturation FFT peaks
        for ch_name, channel in [("sat", sat), ("cr", cr), ("cb", cb)]:
            f_ch = np.fft.fft2(channel)
            fshift_ch = np.fft.fftshift(f_ch)
            abs_f_ch = np.abs(fshift_ch)
            
            dc = abs_f_ch[center_y, center_x]
            norm_mag = abs_f_ch / (dc + 1e-8)
            
            norm_blur = cv2.GaussianBlur(norm_mag, (21, 21), 0)
            norm_diff = norm_mag - norm_blur
            
            diff_vals = norm_diff[valid_mask_chroma]
            if len(diff_vals) > 0:
                std_diff = np.std(diff_vals)
                max_diff = np.max(diff_vals)
                pmr = max_diff / (std_diff + 1e-8)
                features.extend([max_diff, pmr])
            else:
                features.extend([0.0, 0.0])
                
        return features
    except Exception as e:
        sys.stderr.write(f"Error processing image {path}: {e}\n")
        return None

def predict(image_path: str) -> float:
    # 1. Resolve path to model.pkl (look in script folder, then parent folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths_to_check = [
        os.path.join(script_dir, "model.pkl"),
        os.path.join(script_dir, "..", "model.pkl")
    ]
    
    loaded_data = None
    for model_path in model_paths_to_check:
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    loaded_data = pickle.load(f)
                break
            except Exception as e:
                sys.stderr.write(f"Error loading model from {model_path}: {e}\n")
                
    if loaded_data is None:
        raise FileNotFoundError("Could not find or load model.pkl. Make sure train.py has been run.")
        
    scaler = loaded_data["scaler"]
    model = loaded_data["model"]
    
    # 2. Extract features
    features = extract_features(image_path)
    if features is None:
        raise ValueError(f"Failed to extract features from image: {image_path}")
        
    # 3. Normalize & Predict
    features_scaled = scaler.transform([features])
    
    # Probability of class 1 (photo of a screen)
    prob = model.predict_proba(features_scaled)[0][1]
    return float(prob)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    try:
        score = predict(sys.argv[1])
        print(f"{score:.4f}")
    except Exception as e:
        sys.stderr.write(f"Prediction failed: {e}\n")
        sys.exit(1)
