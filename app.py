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
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(
        url,
        MODEL_PATH,
        quiet=False,
        fuzzy=True
    )
# Load model
model = load_model(MODEL_PATH, compile=False)

# Image size
IMG_SIZE = 224

# UI
st.title("🩺 Anemia Detection App")
st.write("Upload an image to predict Hb level and anemia")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred_class, pred_hb = model.predict(img)

    st.subheader("Result")
    st.write(f"Hb Level: {pred_hb[0][0]:.2f}")
    st.write(f"Anemia Probability: {pred_class[0][0]:.2f}")

    if pred_class[0][0] > 0.5:
        st.error("⚠️ Anemic")
    else:
        st.success("✅ Normal")

st.warning("This is not a medical diagnosis. Please consult a doctor.")
