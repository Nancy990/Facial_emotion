import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import io

# Try to import MediaPipe, show error if not available
try:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    st.error("""
    ❌ **MediaPipe not available**
    
    This app requires MediaPipe for pose and facial landmark detection. 
    MediaPipe installation failed on this platform.
    
    **Solutions:**
    1. **For local development:** Install MediaPipe manually:
       ```
       pip install mediapipe==0.9.3.0
       ```
    
    2. **For deployment:** Try using a different hosting platform like:
       - Heroku
       - Google Cloud Run  
       - AWS Lambda
       - Local hosting
    
    3. **Alternative approach:** Consider using a simpler emotion detection model 
       that doesn't require MediaPipe (face-only detection with dlib or OpenCV)
    """)
    st.stop()

def load_model():
    """Load the trained emotion detection model"""
    try:
        with open('Facial_emotion.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'Facial_emotion.pkl' not found. Please make sure it's in the same directory.")
        return None

def extract_landmarks(image):
    """Extract pose and face landmarks from image"""
    if not MEDIAPIPE_AVAILABLE:
        return None
        
    # Convert PIL image to OpenCV format
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Initialize holistic model
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1
    ) as holistic:
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = holistic.process(rgb_image)
        
        return results

def process_landmarks(results):
    """Process landmarks and return feature vector"""
    try:
        # Check if we have both pose and face landmarks
        if results.pose_landmarks is None or results.face_landmarks is None:
            return None, "Could not detect both face and pose landmarks. Please try a clearer image with full body visible."
        
        # Extract Pose landmarks
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                                 for landmark in pose]).flatten())
        
        # Extract Face landmarks
        face = results.face_landmarks.landmark
        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                                 for landmark in face]).flatten())
        
        # Concatenate rows
        row = pose_row + face_row
        
        return row, None
        
    except Exception as e:
        return None, f"Error processing landmarks: {str(e)}"

def predict_emotion(model, landmarks):
    """Predict emotion from landmarks"""
    try:
        # Create DataFrame with landmarks
        X = pd.DataFrame([landmarks])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        return prediction, probabilities
        
    except Exception as e:
        return None, f"Error making prediction: {str(e)}"

def draw_landmarks_on_image(image, results):
    """Draw landmarks on the image for visualization"""
    if not MEDIAPIPE_AVAILABLE:
        return image
        
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert RGB to BGR for OpenCV processing
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    # Draw hand landmarks
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    
    # Convert back to RGB for display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def main():
    st.title("🎭 Emotion Detection from Images")
    
    if not MEDIAPIPE_AVAILABLE:
        st.stop()
    
    st.write("Upload an image and let our AI detect the emotion!")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="For best results, use images with clear view of face and body posture"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image
        with st.spinner("Analyzing image..."):
            # Extract landmarks
            results = extract_landmarks(image)
            
            # Process landmarks
            landmarks, error = process_landmarks(results)
            
            if error:
                st.error(error)
                st.info("Tips for better results:")
                st.write("- Use images with clear lighting")
                st.write("- Make sure both face and body are visible")
                st.write("- Avoid heavily cropped images")
                st.write("- Try images where the person is facing the camera")
            else:
                # Make prediction
                prediction, probabilities = predict_emotion(model, landmarks)
                
                if isinstance(probabilities, str):  # Error case
                    st.error(probabilities)
                else:
                    with col2:
                        st.subheader("Analysis Results")
                        
                        # Draw landmarks on image
                        annotated_image = draw_landmarks_on_image(image.copy(), results)
                        st.image(annotated_image, caption="Detected Landmarks", use_column_width=True)
                    
                    # Display results
                    st.success(f"🎯 Detected Emotion: **{prediction}**")
                    
                    # Display confidence
                    max_prob = np.max(probabilities)
                    st.info(f"📊 Confidence: {max_prob:.2%}")
                    
                    # Display all class probabilities
                    st.subheader("📈 Probability Distribution")
                    
                    # Get class names (adjust based on your model)
                    class_names = ['Doubt', 'Happy', 'Normal', 'Stressed']
                    
                    # Create a dataframe for better visualization
                    prob_df = pd.DataFrame({
                        'Emotion': class_names,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    # Display as bar chart
                    st.bar_chart(prob_df.set_index('Emotion'))
                    
                    # Display as table
                    st.dataframe(prob_df, use_container_width=True)
    
    # Add information about the model
    with st.expander("ℹ️ About this Model"):
        st.write("""
        This emotion detection model uses MediaPipe for pose and facial landmark detection,
        combined with machine learning to classify emotions based on body language and facial expressions.
        
        **Supported Emotions:**
        - Normal
        - Happy  
        - Doubt
        - Stressed
        
        **How it works:**
        1. Extracts facial and pose landmarks from the image
        2. Processes the landmarks into feature vectors
        3. Uses a trained Random Forest classifier to predict emotions
        
        **For best results:**
        - Use clear, well-lit images
        - Include both face and upper body in the frame
        - Avoid heavily filtered or distorted images
        """)

if __name__ == "__main__":
    main()
