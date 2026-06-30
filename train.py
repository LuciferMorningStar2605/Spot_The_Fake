import os
import glob
import time
import pickle
from PIL import Image
import numpy as np
import cv2
import pillow_heif
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Register HEIF opener
pillow_heif.register_heif_opener()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
real_dir = os.path.join(BASE_DIR, "real_world")
screen_dir = os.path.join(BASE_DIR, "spoof")

# Precompute radial masks for 512x512 images
IMG_SIZE = 512
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

def extract_features(path):
    """
    Extracts 80 features:
    - 7 structural/sharpness features (Laplacian var, Sobel mean/std, glare %/max/std, uniformity)
    - 67 grayscale FFT features (global mag diff, hf mag, and radial bin profiles)
    - 6 chromatic aliasing features (FFT diff max and PMR on Sat, Cr, Cb channels)
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
        print(f"Error loading {path}: {e}")
        return None

if __name__ == "__main__":
    real_files = sorted(glob.glob(os.path.join(real_dir, "*")))
    screen_files = sorted(glob.glob(os.path.join(screen_dir, "*")))
    
    print(f"Found {len(real_files)} real_world images and {len(screen_files)} spoof images.")
    
    X = []
    y = []
    
    t0 = time.time()
    print("Extracting features from real_world images...")
    for f in real_files:
        feat = extract_features(f)
        if feat is not None:
            X.append(feat)
            y.append(0)
            
    print("Extracting features from spoof images...")
    for f in screen_files:
        feat = extract_features(f)
        if feat is not None:
            X.append(feat)
            y.append(1)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature extraction complete in {time.time() - t0:.2f} seconds.")
    print(f"Dataset features shape: {X.shape}")
    
    # 5-Fold Cross-Validation evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        clf = SVC(kernel="rbf", C=2.0, probability=True, random_state=42)
        clf.fit(X_tr, y[train_idx])
        preds = clf.predict(X_va)
        cv_accs.append(accuracy_score(y[val_idx], preds))
        
    print(f"\n--- Model Cross-Validation (diverse 322 dataset) ---")
    print(f"5-Fold CV Accuracy: {np.mean(cv_accs) * 100:.2f}%")
    
    # Fit model on the full 322-image dataset
    full_scaler = StandardScaler()
    X_scaled = full_scaler.fit_transform(X)
    
    full_clf = SVC(kernel="rbf", C=2.0, probability=True, random_state=42)
    full_clf.fit(X_scaled, y)
    
    # Save model and scaler
    model_path = os.path.join(BASE_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "scaler": full_scaler,
            "model": full_clf
        }, f)
        
    print(f"Successfully saved model to {model_path}")
    
    # Evaluate on the independent test images
    test_images = {
        "Real Group Photo": "/Users/namishrathy/.gemini/antigravity-ide/brain/6015b748-be2d-42a0-b2ed-4ef691050aed/media__1782811734512.jpg",
        "Screen Recapture Group": "/Users/namishrathy/.gemini/antigravity-ide/brain/6015b748-be2d-42a0-b2ed-4ef691050aed/media__1782814179007.jpg",
        "Real Lady Portrait": "/Users/namishrathy/.gemini/antigravity-ide/brain/6015b748-be2d-42a0-b2ed-4ef691050aed/media__1782814705322.jpg",
        "Screen Recapture Lady": "/Users/namishrathy/.gemini/antigravity-ide/brain/6015b748-be2d-42a0-b2ed-4ef691050aed/media__1782814816753.jpg",
        "Screen Laptop Finder": "/Users/namishrathy/.gemini/antigravity-ide/brain/6015b748-be2d-42a0-b2ed-4ef691050aed/media__1782814899503.jpg",
        "Real Office": "/Users/namishrathy/.gemini/antigravity-ide/brain/6015b748-be2d-42a0-b2ed-4ef691050aed/media__1782814953357.jpg",
        "Real Ceremony": "/Users/namishrathy/.gemini/antigravity-ide/brain/6015b748-be2d-42a0-b2ed-4ef691050aed/media__1782816342410.jpg",
        "Screen Video Call": "/Users/namishrathy/.gemini/antigravity-ide/brain/6015b748-be2d-42a0-b2ed-4ef691050aed/media__1782816470956.jpg",
    }
    
    print("\n--- Out-of-Distribution Independent Test Results ---")
    correct = 0
    for name, path in test_images.items():
        feat = extract_features(path)
        if feat is not None:
            feat_sc = full_scaler.transform([feat])
            prob = full_clf.predict_proba(feat_sc)[0][1]
            pred = "Screen" if prob >= 0.5 else "Real"
            expected = "Real" if "Real" in name else "Screen"
            is_correct = pred == expected
            if is_correct:
                correct += 1
            print(f"{name:25s} -> score: {prob:.4f} ({pred}) | {'OK' if is_correct else 'FAIL'}")
    print(f"Total Correct: {correct}/8")
