import streamlit as st
import time
import os
import tempfile
from PIL import Image
import pillow_heif
from predict import predict

# Register HEIF opener
pillow_heif.register_heif_opener()

# Page configuration
st.set_page_config(
    page_title="Spot the Fake Photo - Recapture Detector",
    page_icon="📸",
    layout="centered"
)

# Custom premium styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1e293b;
        text-align: center;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #64748b;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .result-container {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin-top: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        text-align: center;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>📸 Spot the Fake Photo</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Recapture Fraud Detection using Advanced FFT Moiré Analysis & SVM</p>", unsafe_allow_html=True)

st.write("Upload any image (JPG, PNG, or HEIC) to verify if it is a **real photo** of a physical object or a **re-photographed screen** (fraud capture).")

# Image uploader
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "heic"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".heic"]:
        suffix = ".jpg"
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Columns for side-by-side display
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Uploaded Image")
        try:
            image = Image.open(temp_path)
            st.image(image, use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {e}")

    with col2:
        st.markdown("### Analysis Results")
        
        # Add a spinner while predicting
        with st.spinner("Analyzing textures and moiré frequency patterns..."):
            try:
                t0 = time.time()
                # Run prediction
                score = predict(temp_path)
                latency = (time.time() - t0) * 1000  # in ms
                
                # Format predictions
                is_screen = score >= 0.5
                confidence = score * 100 if is_screen else (1.0 - score) * 100
                
                # Show results with premium styling
                if is_screen:
                    st.markdown(
                        f"<div class='result-container' style='background-color: #fef2f2; border: 1px solid #fee2e2; color: #991b1b;'>"
                        f"⚠️ SCREEN CAPTURED IMAGE<br>"
                        f"<span style='font-size: 0.9rem; font-weight: normal; color: #7f1d1d;'>Recapture Fraud Detected</span>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='result-container' style='background-color: #f0fdf4; border: 1px solid #dcfce7; color: #166534;'>"
                        f"✅ REAL PHOTO<br>"
                        f"<span style='font-size: 0.9rem; font-weight: normal; color: #14532d;'>Authentic Physical Image</span>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                
                # Metrics cards
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<span style='color: #64748b; font-size: 0.85rem; font-weight: 500;'>FRAUD CONFIDENCE SCORE</span><br>"
                    f"<span style='font-size: 1.6rem; font-weight: 700; color: #0f172a;'>{confidence:.2f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<span style='color: #64748b; font-size: 0.85rem; font-weight: 500;'>INFERENCE LATENCY</span><br>"
                    f"<span style='font-size: 1.6rem; font-weight: 700; color: #0f172a;'>{latency:.1f} ms</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
else:
    # Information callout when no image is uploaded
    st.info("Please upload an image above to run the recaptured photo detector.")
