import os
import streamlit as st
import numpy as np
import cv2
import gdown
import keras

FILE_ID = "1NN4mDv3aM-ttrQEZsfzxpQxeBzpg5DRA"
MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_anemia_model():
    try:
        return keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

model = load_anemia_model()
IMG_SIZE = 224

st.set_page_config(page_title="Anemia Detection App", page_icon="🩺")
st.title("🩺 Anemia Detection App")
st.write("Upload an image to predict anemia status.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("❌ Failed to read image.")
        st.stop()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    img_input = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_input = img_input / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    with st.spinner("Analyzing image..."):
        output = model.predict(img_input)

    anemia_prob = float(output[0][0])

    st.subheader("📊 Results")
    st.metric("Anemia Probability", f"{anemia_prob:.2%}")

    st.divider()
    if anemia_prob > 0.5:
        st.error("⚠️ Result: **Anemic** — Please consult a doctor.")
    else:
        st.success("✅ Result: **Normal** — No signs of anemia detected.")

st.warning("⚠️ Not a medical diagnosis. Always consult a doctor.")
