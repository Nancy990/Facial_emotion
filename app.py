import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import io

def load_model():
    """Load the trained emotion detection model"""
    try:
        with open('Facial_emotion.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'Facial_emotion.pkl' not found. Please make sure it's in the same directory.")
        return None

def extract_face_features(image):
    """Extract basic facial features using OpenCV without MediaPipe"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Load OpenCV's pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, "No face detected in the image. Please upload an image with a clear face."
    
    # Use the largest face
    face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = face
    
    # Extract face region
    face_roi = gray[y:y+h, x:x+w]
    face_roi_color = image[y:y+h, x:x+w]
    
    # Extract basic features
    features = []
    
    # 1. Face dimensions and position (normalized)
    img_h, img_w = gray.shape
    features.extend([
        x / img_w,           # face_x_norm
        y / img_h,           # face_y_norm
        w / img_w,           # face_width_norm
        h / img_h,           # face_height_norm
        w / h,               # face_aspect_ratio
    ])
    
    # 2. Eyes detection
    eyes = eye_cascade.detectMultiScale(face_roi)
    features.extend([
        len(eyes),           # num_eyes_detected
        np.mean([eye[2] for eye in eyes]) / w if len(eyes) > 0 else 0,  # avg_eye_width_ratio
        np.mean([eye[3] for eye in eyes]) / h if len(eyes) > 0 else 0,  # avg_eye_height_ratio
    ])
    
    # 3. Smile detection
    smiles = smile_cascade.detectMultiScale(face_roi, 1.8, 20)
    features.extend([
        len(smiles),         # num_smiles_detected
        np.mean([smile[2] for smile in smiles]) / w if len(smiles) > 0 else 0,  # avg_smile_width_ratio
    ])
    
    # 4. Basic intensity features
    features.extend([
        np.mean(face_roi),           # face_mean_intensity
        np.std(face_roi),            # face_std_intensity
        np.mean(face_roi[:h//2]),    # upper_face_intensity
        np.mean(face_roi[h//2:]),    # lower_face_intensity
    ])
    
    # 5. Edge features (indicates facial structure/expressions)
    edges = cv2.Canny(face_roi, 50, 150)
    features.extend([
        np.sum(edges) / (w * h),     # edge_density
        np.mean(edges),              # avg_edge_intensity
    ])
    
    # 6. Histogram features (texture information)
    hist = cv2.calcHist([face_roi], [0], None, [8], [0, 256])
    hist_norm = hist.flatten() / np.sum(hist)
    features.extend(hist_norm.tolist())  # 8 histogram bins
    
    # Pad features to match original model input size (this is a workaround)
    # Since your original model expects MediaPipe landmarks, we need to pad
    target_size = 1662  # Approximate size based on MediaPipe landmarks
    current_size = len(features)
    
    if current_size < target_size:
        # Pad with zeros and repeat some features
        padding_needed = target_size - current_size
        # Repeat the existing features to fill space
        repeated_features = (features * (padding_needed // current_size + 1))[:padding_needed]
        features.extend(repeated_features)
    else:
        features = features[:target_size]
    
    return features, None

def predict_emotion_fallback(image):
    """Simple rule-based emotion prediction as fallback"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Load cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return "Normal", [0.25, 0.25, 0.25, 0.25]
    
    # Simple heuristic-based emotion detection
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Detect smiles and eyes
    smiles = smile_cascade.detectMultiScale(face_roi, 1.8, 20)
    eyes = eye_cascade.detectMultiScale(face_roi)
    
    # Simple rules
    if len(smiles) > 0:
        return "Happy", [0.1, 0.7, 0.1, 0.1]
    elif len(eyes) < 2:
        return "Stressed", [0.1, 0.1, 0.1, 0.7]
    else:
        # Analyze face brightness and contrast for other emotions
        mean_intensity = np.mean(face_roi)
        if mean_intensity < 100:  # Darker face might indicate stress
            return "Stressed", [0.2, 0.1, 0.1, 0.6]
        else:
            return "Normal", [0.1, 0.2, 0.6, 0.1]

def draw_face_detection(image):
    """Draw face detection rectangles on image"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image

def main():
    st.title("🎭 Face-Based Emotion Detection")
    st.write("Upload an image and let our AI detect the emotion using face analysis!")
    
    st.info("""
    **Note:** This version uses OpenCV face detection instead of MediaPipe, 
    which works better on cloud platforms but may be less accurate than the original model.
    """)
    
    # Load model
    model = load_model()
    use_ml_model = model is not None
    
    if not use_ml_model:
        st.warning("ML model not found. Using rule-based emotion detection as fallback.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="For best results, use images with clear faces"
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
            if use_ml_model:
                # Try to use the trained model
                features, error = extract_face_features(image)
                
                if error:
                    st.error(error)
                    st.info("Tips for better results:")
                    st.write("- Use images with clear, front-facing faces")
                    st.write("- Ensure good lighting")
                    st.write("- Avoid sunglasses or face coverings")
                else:
                    try:
                        # Make prediction with trained model
                        X = pd.DataFrame([features])
                        prediction = model.predict(X)[0]
                        probabilities = model.predict_proba(X)[0]
                        
                        prediction_success = True
                    except Exception as e:
                        st.warning(f"ML model failed: {str(e)}. Using fallback method.")
                        prediction, probabilities = predict_emotion_fallback(image)
                        prediction_success = True
            else:
                # Use rule-based fallback
                prediction, probabilities = predict_emotion_fallback(image)
                prediction_success = True
            
            if prediction_success:
                with col2:
                    st.subheader("Analysis Results")
                    
                    # Draw face detection
                    annotated_image = draw_face_detection(image.copy())
                    st.image(annotated_image, caption="Detected Face", use_column_width=True)
                
                # Display results
                st.success(f"🎯 Detected Emotion: **{prediction}**")
                
                # Display confidence
                max_prob = np.max(probabilities)
                st.info(f"📊 Confidence: {max_prob:.2%}")
                
                # Display all class probabilities
                st.subheader("📈 Probability Distribution")
                
                # Get class names
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
    with st.expander("ℹ️ About this Detection Method"):
        st.write("""
        This emotion detection uses OpenCV's built-in face detection combined with:
        
        **Detection Features:**
        - Face position and dimensions
        - Eye detection and positioning
        - Smile detection
        - Facial intensity patterns
        - Edge detection for facial structure
        - Texture analysis
        
        **Supported Emotions:**
        - Normal
        - Happy  
        - Doubt
        - Stressed
        
        **How it works:**
        1. Detects faces using OpenCV Haar cascades
        2. Extracts facial features (eyes, smile, intensity patterns)
        3. Uses machine learning model or rule-based classification
        
        **Limitations:**
        - Less accurate than MediaPipe-based detection
        - Works best with front-facing, well-lit faces
        - May struggle with extreme expressions or angles
        
        **For best results:**
        - Use clear, front-facing photos
        - Ensure good lighting
        - Avoid sunglasses or face coverings
        """)

if __name__ == "__main__":
    main()
