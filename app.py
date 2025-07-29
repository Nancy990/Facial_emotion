import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
import pickle
import cv2

# Load the model
@st.cache_resource
def load_model():
    with open("Facial_emotion.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="Live Facial Emotion Detection", layout="wide")
st.title("🧠 Real-Time Facial Emotion Detection")
st.markdown("Webcam-based facial emotion detection using a trained model.")

# Dummy face detection using OpenCV Haar cascades (for simplicity)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))  # assuming 48x48 input size
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_flat = face_gray.flatten().reshape(1, -1)

            if face_flat.shape[1] == model.named_steps['standardscaler'].n_features_in_:
                pred = model.predict(face_flat)[0]
                cv2.putText(img, f"{pred}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Invalid input size", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return img

# Start the webcam stream
webrtc_streamer(key="emotion", video_transformer_factory=EmotionDetector)

