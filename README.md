# Take-Home Submission: Spot the Fake Photo

This repository contains the solution to identify whether an image is a **real photo** or a **photo of a screen** (screencapture/fraud recapture).

---

## 1. Approach & Design Decisions

### Feature Engineering (80 Features)
Rather than relying on resource-intensive deep learning models (which are prone to environment conflicts and require heavy compute), we build a classic computer vision feature extraction pipeline. We extract **80 hand-crafted features** capturing sharpness, glare, texture uniformity, and periodic textures (moiré patterns):

1. **Sharpness, Edges & Glare (7 features)**:
   - **Laplacian Variance**: Measures the sharpness/focus of the image. Screen photos often have flat surfaces or localized blur.
   - **Sobel Magnitude (Mean & Std)**: Measures edge density and transitions.
   - **Glare Statistics (glare %, max brightness, bright region std)**: Captures screen backlight highlights and dynamic range clipping.
   - **Texture Uniformity**: Standard deviation of standard deviations across 8x8 image cells to separate uniform display noise from natural texture.
2. **Grayscale Moiré FFT concentric rings (67 features)**:
   - We compute the **2D Fast Fourier Transform (FFT)** magnitude spectrum of the 512x512 downsampled image.
   - We apply a high-pass filter by subtracting a Gaussian-blurred version of the FFT to isolate sharp spikes/peaks representing periodic grid frequencies.
   - We divide the 2D frequency domain into **10 concentric ring bins** (radius 20 to 250 pixels).
   - In each ring, we compute the mean, standard deviation, and maximum value for both the raw FFT magnitude and the high-pass filtered FFT magnitude.
   - **Isolated Peaks Count**: Detect local maxima in a 9x9 neighborhood in high frequencies.
3. **Chrominance Color-Aliasing Peaks (6 features)**:
   - **Peak-to-Noise Ratio (PMR) and Maximum Peak value** of the high-pass normalized FFT magnitude in Saturation, Cr, and Cb channels. This directly isolates the periodic chromatic moiré bands, which survive downsampling and JPEG compression.

### Model Architecture
- **Classifier**: Support Vector Machine (SVM) with an **RBF (Radial Basis Function) Kernel** ($C=2.0$) and probability output.
- **Normalization**: `StandardScaler` to normalize the 80 features before feeding them to the SVM.

---

## 2. Accuracy & Performance

We evaluated the pipeline using **Stratified 5-Fold Cross-Validation** on the provided dataset of 100 images (50 real, 50 screen).

| Model | Cross-Validation Accuracy | Standard Deviation |
| :--- | :---: | :---: |
| **SVM (RBF Kernel)** | **98.00%** | **±2.45%** |

Our selected model **SVM (RBF Kernel, C=2.0)** delivers an outstanding **98% accuracy** and is highly robust.

---

## 3. Required Report Metrics

### Latency
Measurements taken on a **Macbook CPU (Apple M3)**:

*   **JPG Image**: **~75 ms** total internal execution time (including image loading, downsampling to 512x512, color conversions, feature extraction, and SVM prediction).
*   **HEIC Image**: **~750 ms** total internal execution time (contains ~670 ms overhead from decoding high-resolution HEIC files using CPU-based `pillow-heif` library).
*   **Cold Startup / Process Overhead**: ~1.5 seconds when run as a standalone script (e.g., `python predict.py image.jpg`) due to one-time Python startup and `scikit-learn` import overhead. In production (e.g. web API or app process), this import occurs once at startup, so the per-image latency is the internal speed of **~75 ms**.

### Cost per Image
*   **On-Device (Client-side / Mobile)**: **$0.00 (Free)**. Since the model and scaler are small (~500 KB total) and use basic numpy/cv2 operations, they can run entirely on the user's phone or client app, completely offloading server costs.
*   **Cloud Server Scale**: **~$0.047 per 1,000 images** (or **~$47.00 per million images**).
    *   *Assumptions*: Hosted as a serverless function on AWS Lambda (allocated 2 GB RAM, 1 vCPU).
    *   AWS Lambda cost rate: $0.000033334 per GB-second.
    *   Average execution time: 150 ms.
    *   Cost per execution = $0.000033334 * 2 * 0.15 = $0.000010 per run ($10 per million executions).
    *   Add API Gateway ($3.50/million requests) and network bandwidth ($33.50/million requests), totaling ~$47 per million images.

---

## 4. Discussion Questions

### How to keep it accurate as cheaters adapt?
1. **Adversarial Augmentation**: If cheaters use high-quality printouts or high-density Retina displays, standard moiré patterns may change. We should continuously collect adversarial recaptures (from new screen types, different print qualities) and augment the training set.
2. **Dynamic Thresholding & Monitoring**: Keep track of the prediction score distribution. If scores begin to cluster close to the decision boundary, it indicates adaptation.
3. **Active Frequency Domain Patches**: Moiré signatures shift dynamically based on distance and angle. We can run multiple local FFTs on cropped patches (e.g., center vs corners) to detect multi-angle recaptures.

### How to make it tiny and fast enough for a phone?
1. **Model Conversion**: Convert the scikit-learn SVM and scaler model into a lightweight format like **ONNX Runtime Mobile** or compile it to native code (using C/C++ or Swift/Kotlin). The file size will shrink to under **500 KB**.
2. **Native OpenCV Implementation**: Implement the resize, color conversion, Sobel/Laplacian filters, and FFT using the native C++ OpenCV library on Android/iOS. This bypasses the Python interpreter overhead.
3. **Optimized Inference**: On native devices, a 512x512 float FFT and linear/RBF SVM evaluation takes **<5 ms** on modern mobile CPUs/NPUs.

### How to choose the cut-off score for flagging fraud?
1. **High Precision vs. Recall Trade-off**: In real-world fraud detection, false positives (flagging a genuine user as fraud) can ruin the user experience.
2. **Multi-tier Action Plan**:
   - **Green Channel (Score < 0.20)**: Automatically approve (low probability of fraud).
   - **Review Channel (Score 0.20 to 0.85)**: Flag for manual human review or trigger a secondary verification step (e.g., ask the user to turn their camera slightly or blink).
   - **Red Channel (Score > 0.85)**: Block immediately (high confidence of screen recapture).
