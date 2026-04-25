import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import gdown

# Download model from Google Drive
url = "https://drive.google.com/uc?id=1CuLOlei4T2zUjwyKGCbi02Z6H5IglIEi"
gdown.download(url, "model.h5", quiet=False)

# Load model
model = load_model("model.h5", compile=False)

IMG_SIZE = 224

st.title("💅 Anemia Detection App")
st.write("Upload an image to predict Hb level and anemia")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
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