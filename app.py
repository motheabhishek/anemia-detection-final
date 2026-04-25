import streamlit as st
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model

# Google Drive direct file ID link
FILE_ID = "1CuLOlei4T2zUjwyKGCbi02Z6H5IglIEi"
MODEL_PATH = "model.h5"

# Download model (only once)
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model with caching to avoid reloading on every interaction
@st.cache_resource
def load_anemia_model():
    return load_model(MODEL_PATH, compile=False, safe_mode=False)

model = load_anemia_model()

# Image size
IMG_SIZE = 224

# Page config
st.set_page_config(page_title="Anemia Detection App", page_icon="🩺")

# UI
st.title("🩺 Anemia Detection App")
st.write("Upload an image to predict Hb level and anemia status.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("❌ Failed to read the image. Please upload a valid image file.")
        st.stop()

    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # Preprocess for model (separate from display image)
    img_input = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_input = img_input / 255.0
    img_input = np.expand_dims(img_input, axis=0)  # Shape: (1, 224, 224, 3)

    # Predict
    with st.spinner("Analyzing image..."):
        output = model.predict(img_input)

    # Handle model output safely
    if isinstance(output, list) and len(output) == 2:
        pred_class, pred_hb = output
    else:
        st.error(f"❌ Unexpected model output format: {type(output)}, length: {len(output) if isinstance(output, list) else 'N/A'}")
        st.stop()

    hb_value = pred_hb[0][0]
    anemia_prob = pred_class[0][0]

    # Display results
    st.subheader("📊 Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Hb Level (g/dL)", value=f"{hb_value:.2f}")

    with col2:
        st.metric(label="Anemia Probability", value=f"{anemia_prob:.2%}")

    st.divider()

    if anemia_prob > 0.5:
        st.error("⚠️ Result: **Anemic** — Please consult a healthcare professional.")
    else:
        st.success("✅ Result: **Normal** — No signs of anemia detected.")

st.warning("⚠️ This app is for educational purposes only and is NOT a substitute for medical diagnosis. Always consult a qualified doctor.")
