import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("Facial_emotion.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Streamlit UI
st.set_page_config(page_title="Live Facial Emotion Detection", layout="wide")
st.title("🧠 Real-Time Facial Emotion Detection")
st.markdown("Allow webcam access and click 'Start Webcam'.")

# Start webcam stream
run = st.checkbox("Start Webcam")

frame_window = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)

while run and cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.warning("Failed to grab frame.")
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract normalized landmarks as input features
            face_points = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]
            ).flatten().reshape(1, -1)

            # If landmark count doesn't match model input, skip
            if face_points.shape[1] != model.named_steps['standardscaler'].n_features_in_:
                cv2.putText(frame, "Face landmark mismatch", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue

            prediction = model.predict(face_points)[0]
            cv2.putText(frame, f"Emotion: {prediction}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_window.image(frame_bgr)

# Release camera
if cap:
    cap.release()

