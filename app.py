import streamlit as st
import numpy as np
import cv2
import pickle
import time
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    with open("Facial_emotion.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="Facial Emotion Scanner", layout="centered")
st.title("Facial Emotion Scanner")
st.markdown("Upload your face photo and let the AI guess your emotion!")

uploaded_file = st.file_uploader("📸 Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    st.markdown("🔍 **Scanning face...**")
    progress = st.progress(0)
    for percent in range(100):
        time.sleep(0.01)
        progress.progress(percent + 1)

    # Dummy preprocessing
    face_resized = cv2.resize(img_cv, (48, 48))
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    input_features = face_gray.flatten().reshape(1, -1)

    # Check input shape matches model
    if input_features.shape[1] == model.named_steps['standardscaler'].n_features_in_:
        pred = model.predict(input_features)[0]
        st.success(f"🎯 **Predicted Emotion:** `{pred}`")
    else:
        st.error("Image does not match expected input size. Try a face-only image.")



