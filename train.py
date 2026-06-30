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
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Register HEIF opener
pillow_heif.register_heif_opener()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
real_dir = os.path.join(BASE_DIR, "real")
screen_dir = os.path.join(BASE_DIR, "screen")

# Precompute radial masks for 512x512 images
IMG_SIZE = 512
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

def extract_features(path):
    """
    Extracts 123 features from the image:
    - 3 sharpness & edge features (Laplacian var, Sobel mean/std)
    - 12 RGB statistics (mean, std, 10th percentile, 90th percentile)
    - 6 HSV statistics (mean, std)
    - 24 color histogram bins (8 per channel in RGB)
    - 6 global FFT features (raw & diff high-pass mean, std, max)
    - 72 radial FFT features (mean, std, max for raw & diff in 12 concentric rings)
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
        print(f"Error loading {path}: {e}")
        return None

if __name__ == "__main__":
    real_files = sorted(glob.glob(os.path.join(real_dir, "*")))
    screen_files = sorted(glob.glob(os.path.join(screen_dir, "*")))
    
    print(f"Found {len(real_files)} real images and {len(screen_files)} screen images.")
    
    X = []
    y = []
    
    t0 = time.time()
    print("Extracting features from real images...")
    for f in real_files:
        feat = extract_features(f)
        if feat is not None:
            X.append(feat)
            y.append(0)
            
    print("Extracting features from screen images...")
    for f in screen_files:
        feat = extract_features(f)
        if feat is not None:
            X.append(feat)
            y.append(1)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature extraction complete in {time.time() - t0:.2f} seconds.")
    print(f"Features shape: {X.shape}")
    
    # 5-fold cross validation score to print for verification
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        svm = SVC(kernel="rbf", probability=True, C=10.0, random_state=42)
        svm.fit(X_train_scaled, y_train)
        preds = svm.predict(X_val_scaled)
        accs.append(accuracy_score(y_val, preds))
        
    print(f"5-Fold CV Accuracy: {np.mean(accs):.4f} (std: {np.std(accs):.4f})")
    
    # Fit model on the full dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm = SVC(kernel="rbf", probability=True, C=10.0, random_state=42)
    svm.fit(X_scaled, y)
    
    # Save the model
    model_path = os.path.join(BASE_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "model": svm
        }, f)
        
    print(f"Successfully saved scaler and model to {model_path}")
