import os
import sys
import pickle
from PIL import Image
import numpy as np
import cv2
import pillow_heif

# Register HEIF opener so PIL can open HEIC files
pillow_heif.register_heif_opener()

# Image dimensions
IMG_SIZE = 512

# Precompute radial coordinates
y_coords, x_coords = np.indices((IMG_SIZE, IMG_SIZE))
center_y, center_x = IMG_SIZE // 2, IMG_SIZE // 2
distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)

# Define radial bins (concentric rings from radius 20 to 250)
num_bins = 12
bin_edges = np.linspace(20, 250, num_bins + 1)
radial_masks = []
for i in range(num_bins):
    mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
    radial_masks.append(mask)

def extract_features(path: str):
    """
    Extracts the identical 123-feature vector as train.py.
    """
    try:
        img = Image.open(path)
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_np = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        features = []
        
        # 1. Sharpness & Edge Features
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(lap_var)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        features.extend([np.mean(sobel_mag), np.std(sobel_mag)])
        
        # 2. Color Statistics
        for c in range(3):
            ch = img_np[:, :, c]
            features.extend([np.mean(ch), np.std(ch), np.percentile(ch, 10), np.percentile(ch, 90)])
            
        for c in range(3):
            ch = hsv[:, :, c]
            features.extend([np.mean(ch), np.std(ch)])
            
        # Color histograms (8 bins per channel in RGB)
        for c in range(3):
            hist, _ = np.histogram(img_np[:, :, c], bins=8, range=(0, 256))
            hist = hist / (IMG_SIZE * IMG_SIZE)
            features.extend(hist)
            
        # 3. FFT Frequency Radial Features
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-8)
        
        # High-pass filter via Gaussian blur subtraction
        mag_blur = cv2.GaussianBlur(mag, (21, 21), 0)
        mag_diff = mag - mag_blur
        
        # Global FFT magnitude stats
        features.extend([np.mean(mag), np.std(mag), np.max(mag)])
        features.extend([np.mean(mag_diff), np.std(mag_diff), np.max(mag_diff)])
        
        # Radial statistics
        for mask in radial_masks:
            mag_bin = mag[mask]
            mag_diff_bin = mag_diff[mask]
            
            if len(mag_bin) > 0:
                features.extend([np.mean(mag_bin), np.std(mag_bin), np.max(mag_bin)])
                features.extend([np.mean(mag_diff_bin), np.std(mag_diff_bin), np.max(mag_diff_bin)])
            else:
                features.extend([0, 0, 0, 0, 0, 0])
                
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
