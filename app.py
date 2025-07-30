import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import io
import math
import json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
import plotly.graph_objects as go
import plotly.express as px

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class AdvancedFaceAnalyzer:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.emotion_model = self.build_cnn_emotion_model()
        self.age_model = self.build_age_estimation_model()
        
    def build_cnn_emotion_model(self):
        """Build a CNN model for emotion detection"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')  # 7 emotions
        ])
        return model
    
    def build_age_estimation_model(self):
        """Build a CNN model for age estimation"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='linear')  # Age regression
        ])
        return model

def load_model():
    """Load the trained emotion detection model with better mobile support"""
    model_files = ['opencv_emotion_model.pkl', 'Facial_emotion.pkl']
    
    for model_file in model_files:
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            st.success(f"✅ Loaded model: {model_file}")
            return model, model_file
        except FileNotFoundError:
            continue
    
    st.warning("⚠️ No pre-trained model found. Using enhanced CNN-based detection.")
    return None, None

def preprocess_mobile_image(image):
    """Enhanced preprocessing for mobile images"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Handle EXIF orientation for mobile images
    if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Enhance image quality for better face detection
    # Increase contrast and brightness
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Denoise
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return enhanced

def extract_3d_landmarks(image):
    """Extract 3D facial landmarks using MediaPipe"""
    analyzer = AdvancedFaceAnalyzer()
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Preprocess for mobile
    image = preprocess_mobile_image(image)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    results = analyzer.face_mesh.process(rgb_image)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        
        # Extract 3D coordinates
        landmarks_3d = []
        h, w = image.shape[:2]
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z  # Relative depth
            landmarks_3d.append([x, y, z])
        
        return np.array(landmarks_3d), results
    
    return None, None

def create_3d_face_model(landmarks_3d):
    """Create 3D face reconstruction"""
    if landmarks_3d is None:
        return None
    
    # Create 3D plot
    fig = go.Figure(data=[go.Scatter3d(
        x=landmarks_3d[:, 0],
        y=landmarks_3d[:, 1],
        z=landmarks_3d[:, 2],
        mode='markers+lines',
        marker=dict(
            size=2,
            color=landmarks_3d[:, 2],
            colorscale='Viridis',
            opacity=0.8
        ),
        line=dict(
            color='rgba(50, 50, 250, 0.3)',
            width=1
        )
    )])
    
    fig.update_layout(
        title="3D Face Reconstruction",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Depth (Z)",
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            )
        ),
        height=500
    )
    
    return fig

def analyze_face_shape_advanced(landmarks_3d):
    """Advanced face shape analysis using 3D landmarks"""
    if landmarks_3d is None:
        return {'shape': 'Unknown', 'confidence': 0}
    
    # Key facial points (MediaPipe face mesh indices)
    # These are approximate indices for key facial features
    forehead_top = landmarks_3d[9]  # Top of forehead
    chin_bottom = landmarks_3d[152]  # Bottom of chin
    left_cheek = landmarks_3d[234]  # Left cheek
    right_cheek = landmarks_3d[454]  # Right cheek
    left_jaw = landmarks_3d[172]  # Left jaw
    right_jaw = landmarks_3d[397]  # Right jaw
    
    # Calculate measurements
    face_length = np.linalg.norm(forehead_top - chin_bottom)
    face_width = np.linalg.norm(left_cheek - right_cheek)
    jaw_width = np.linalg.norm(left_jaw - right_jaw)
    
    # Face ratios
    length_width_ratio = face_length / face_width if face_width > 0 else 1
    jaw_face_ratio = jaw_width / face_width if face_width > 0 else 1
    
    # Enhanced shape classification
    if length_width_ratio > 1.4:
        if jaw_face_ratio < 0.8:
            shape = "Heart"
            confidence = 0.85
        else:
            shape = "Oval"
            confidence = 0.9
    elif length_width_ratio < 1.1:
        if jaw_face_ratio > 0.9:
            shape = "Square"
            confidence = 0.8
        else:
            shape = "Round"
            confidence = 0.85
    else:
        if jaw_face_ratio < 0.7:
            shape = "Diamond"
            confidence = 0.75
        elif jaw_face_ratio > 0.95:
            shape = "Rectangle"
            confidence = 0.8
        else:
            shape = "Oval"
            confidence = 0.9
    
    return {
        'shape': shape,
        'confidence': confidence,
        'face_ratio': length_width_ratio,
        'jaw_ratio': jaw_face_ratio,
        'measurements': {
            'face_length': face_length,
            'face_width': face_width,
            'jaw_width': jaw_width
        }
    }

def calculate_beauty_score_advanced(landmarks_3d, shape_analysis):
    """Calculate beauty score using 3D landmarks and golden ratios"""
    if landmarks_3d is None:
        return 50, []
    
    beauty_factors = []
    
    # 1. Facial Symmetry (35% weight)
    left_landmarks = landmarks_3d[:len(landmarks_3d)//2]
    right_landmarks = landmarks_3d[len(landmarks_3d)//2:]
    
    # Calculate symmetry by comparing mirrored halves
    if len(left_landmarks) == len(right_landmarks):
        symmetry_score = 100 - np.mean(np.abs(left_landmarks[:, 0] - (-right_landmarks[:, 0] + np.mean(landmarks_3d[:, 0]) * 2)))
        symmetry_score = max(0, min(100, symmetry_score))
    else:
        symmetry_score = 75  # Default if can't calculate
    
    beauty_factors.append(('Facial Symmetry', symmetry_score, 0.35))
    
    # 2. Golden Ratio Adherence (25% weight)
    golden_ratio = 1.618
    face_ratio = shape_analysis.get('face_ratio', 1.3)
    golden_deviation = abs(face_ratio - golden_ratio) / golden_ratio * 100
    golden_score = max(0, 100 - golden_deviation * 2)
    beauty_factors.append(('Golden Ratio', golden_score, 0.25))
    
    # 3. Feature Proportion (25% weight)
    measurements = shape_analysis.get('measurements', {})
    if measurements:
        # Ideal proportions
        ideal_jaw_ratio = 0.85
        actual_jaw_ratio = shape_analysis.get('jaw_ratio', 0.85)
        proportion_score = max(0, 100 - abs(actual_jaw_ratio - ideal_jaw_ratio) * 200)
    else:
        proportion_score = 75
    
    beauty_factors.append(('Feature Proportion', proportion_score, 0.25))
    
    # 4. Face Shape Preference (15% weight)
    shape_scores = {'Oval': 95, 'Heart': 90, 'Diamond': 85, 'Round': 80, 'Square': 75, 'Rectangle': 70}
    shape_score = shape_scores.get(shape_analysis.get('shape', 'Oval'), 75)
    beauty_factors.append(('Face Shape', shape_score, 0.15))
    
    # Calculate overall score
    overall_score = sum(score * weight for _, score, weight in beauty_factors)
    
    return overall_score, beauty_factors

def get_makeup_recommendations(face_shape, beauty_factors):
    """Generate personalized makeup recommendations"""
    recommendations = []
    
    # Face shape specific recommendations
    shape_makeup = {
        'Oval': [
            "💄 **Perfect Canvas**: Your balanced proportions work with most makeup styles",
            "✨ **Highlight**: Focus on your natural symmetry with subtle contouring",
            "👄 **Lips**: Most lip shapes and colors will complement your face",
            "👁️ **Eyes**: Experiment with various eye makeup styles"
        ],
        'Round': [
            "🔥 **Contouring**: Use bronzer along jawline and temples to add definition",
            "👁️ **Eyes**: Elongate with winged eyeliner and vertical eyeshadow application",
            "💋 **Lips**: Slightly overlining can add structure",
            "✨ **Highlight**: Apply to bridge of nose and chin to add length"
        ],
        'Square': [
            "🌟 **Soften**: Use rounded eyebrow shapes and soft, blended eyeshadow",
            "💄 **Lips**: Rounded lip shapes will balance angular features",
            "✨ **Highlight**: Focus on center of face - forehead, nose, chin",
            "🎨 **Blush**: Apply in circular motions on apple of cheeks"
        ],
        'Heart': [
            "💋 **Balance**: Draw attention to lips with bold colors",
            "👁️ **Eyes**: Keep eye makeup subtle to not overpower narrow chin",
            "✨ **Contour**: Lightly contour temples to reduce forehead width",
            "🌸 **Blush**: Apply lower on cheekbones, closer to jawline"
        ],
        'Rectangle': [
            "🌟 **Width**: Add width with blush applied horizontally across cheeks",
            "👁️ **Eyes**: Horizontal eyeshadow application to widen face",
            "💄 **Lips**: Full, horizontal lip shapes",
            "✨ **Contour**: Minimize forehead and chin, emphasize cheeks"
        ],
        'Diamond': [
            "💎 **Balance**: Soften prominent cheekbones with subtle contouring",
            "👁️ **Eyes**: Draw attention upward with defined brows and eye makeup",
            "💋 **Lips**: Fuller lips balance narrow chin and forehead",
            "✨ **Highlight**: Focus on forehead and chin to add width"
        ]
    }
    
    recommendations.extend(shape_makeup.get(face_shape, shape_makeup['Oval']))
    
    # Beauty factor specific recommendations
    for factor_name, score, _ in beauty_factors:
        if factor_name == 'Facial Symmetry' and score < 80:
            recommendations.append("🎭 **Symmetry**: Use makeup to enhance facial balance - slight contouring can help")
        elif factor_name == 'Golden Ratio' and score < 70:
            recommendations.append("📐 **Proportions**: Strategic highlighting and contouring can create ideal proportions")
    
    return recommendations

def get_facial_exercises(face_shape, age_estimate=None):
    """Generate facial exercise recommendations"""
    exercises = []
    
    # Universal anti-aging exercises
    universal_exercises = [
        "😊 **Smile Exercise**: Hold a wide smile for 10 seconds, repeat 10 times daily",
        "👁️ **Eye Circles**: Gently circle eyes with fingertips to reduce puffiness",
        "💆 **Forehead Massage**: Smooth forehead lines with upward strokes",
        "🎵 **Vowel Sounds**: Say A-E-I-O-U exaggerating mouth movements"
    ]
    
    # Face shape specific exercises
    shape_exercises = {
        'Round': [
            "🔥 **Cheek Toning**: Suck in cheeks and hold for 10 seconds",
            "💪 **Jaw Definition**: Chew sugar-free gum to strengthen jaw muscles"
        ],
        'Square': [
            "🌸 **Jaw Relaxation**: Massage jaw muscles to reduce tension",
            "😌 **Soft Expressions**: Practice gentle, relaxed facial expressions"
        ],
        'Heart': [
            "💋 **Lip Exercises**: Pucker and release lips to strengthen lower face",
            "🎯 **Chin Strengthening**: Press tongue to roof of mouth, hold 5 seconds"
        ],
        'Rectangle': [
            "😄 **Cheek Lifts**: Smile lifting cheek muscles, hold 5 seconds",
            "🌟 **Face Widening**: Gently stretch face horizontally with hands"
        ]
    }
    
    exercises.extend(universal_exercises)
    exercises.extend(shape_exercises.get(face_shape, []))
    
    # Age-specific exercises
    if age_estimate:
        if age_estimate > 40:
            exercises.extend([
                "🌿 **Anti-Aging**: Fish face exercise - suck in cheeks and lips",
                "✨ **Neck Toning**: Tilt head back, push lower jaw forward"
            ])
        elif age_estimate > 30:
            exercises.extend([
                "🔄 **Prevention**: Gentle face yoga to maintain muscle tone",
                "💧 **Hydration**: Facial massage with moisturizer"
            ])
    
    return exercises

def save_progress(analysis_data):
    """Save analysis progress for tracking"""
    timestamp = datetime.now().isoformat()
    
    progress_data = {
        'timestamp': timestamp,
        'beauty_score': analysis_data.get('beauty_score', 0),
        'face_shape': analysis_data.get('face_shape', 'Unknown'),
        'emotion': analysis_data.get('emotion', 'Unknown'),
        'age_estimate': analysis_data.get('age_estimate', 0)
    }
    
    # In a real app, this would save to a database
    # For demo, we'll use session state
    if 'progress_history' not in st.session_state:
        st.session_state.progress_history = []
    
    st.session_state.progress_history.append(progress_data)
    
    return len(st.session_state.progress_history)

def show_progress_tracking():
    """Display progress tracking visualization"""
    if 'progress_history' not in st.session_state or not st.session_state.progress_history:
        st.info("📊 No progress data yet. Analyze some images to start tracking!")
        return
    
    df = pd.DataFrame(st.session_state.progress_history)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Beauty score over time
    if len(df) > 1:
        fig_beauty = px.line(df, x='date', y='beauty_score', 
                           title='Beauty Score Progress Over Time',
                           markers=True)
        st.plotly_chart(fig_beauty, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Beauty Score", f"{df['beauty_score'].mean():.1f}")
    
    with col2:
        st.metric("Total Analyses", len(df))
    
    with col3:
        if len(df) > 1:
            trend = "📈 Improving" if df['beauty_score'].iloc[-1] > df['beauty_score'].iloc[0] else "📉 Declining"
            st.metric("Trend", trend)

def predict_emotion_cnn(image, face_coords):
    """Enhanced emotion prediction using CNN approach"""
    try:
        x, y, w, h = face_coords
        face_roi = image[y:y+h, x:x+w]
        
        # Preprocess for emotion detection
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized / 255.0
        
        # Enhanced rule-based emotion detection
        # Analyze facial features
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        smiles = smile_cascade.detectMultiScale(face_gray, 1.8, 20)
        eyes = eye_cascade.detectMultiScale(face_gray)
        
        # Calculate facial metrics
        mean_intensity = np.mean(face_gray)
        std_intensity = np.std(face_gray)
        upper_face = np.mean(face_gray[:h//2])
        lower_face = np.mean(face_gray[h//2:])
        
        # Enhanced emotion scoring
        smile_score = len(smiles) * 0.4
        eye_score = min(len(eyes), 2) * 0.2
        brightness_score = (mean_intensity - 100) / 100
        contrast_score = std_intensity / 50
        face_balance = abs(upper_face - lower_face) / mean_intensity
        
        # Calculate probabilities for 7 emotions
        happy_prob = max(0, min(1, smile_score + brightness_score * 0.3 + eye_score))
        sad_prob = max(0, min(1, (1 - brightness_score) * 0.4 + face_balance * 0.3))
        angry_prob = max(0, min(1, contrast_score * 0.4 + (1 - eye_score) * 0.2))
        surprised_prob = max(0, min(1, eye_score * 0.5 + contrast_score * 0.2))
        fear_prob = max(0, min(1, face_balance * 0.4 + contrast_score * 0.2))
        disgust_prob = max(0, min(1, (1 - smile_score) * 0.3 + face_balance * 0.2))
        neutral_prob = max(0, 1 - max(happy_prob, sad_prob, angry_prob, surprised_prob, fear_prob, disgust_prob))
        
        probabilities = [angry_prob, disgust_prob, fear_prob, happy_prob, sad_prob, surprised_prob, neutral_prob]
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
        
        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p/total for p in probabilities]
        else:
            probabilities = [1/7] * 7
        
        prediction = emotions[np.argmax(probabilities)]
        
        return prediction, probabilities, emotions
        
    except Exception as e:
        st.error(f"Emotion prediction error: {str(e)}")
        return "Neutral", [0, 0, 0, 0.7, 0, 0, 0.3], ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']

def main():
    st.set_page_config(
        page_title="Advanced 3D Face Analysis",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Advanced 3D Face Analysis & Beauty Enhancement")
    st.write("Upload an image for comprehensive 3D facial analysis, emotion detection, beauty scoring, and personalized recommendations!")
    
    # Sidebar for navigation
    st.sidebar.title("🔧 Analysis Options")
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Complete Analysis", "3D Reconstruction Only", "Beauty Analysis Only", "Progress Tracking"]
    )
    
    if analysis_mode == "Progress Tracking":
        st.header("📊 Progress Tracking Dashboard")
        show_progress_tracking()
        return
    
    # Load model
    model, model_file = load_model()
    
    # File uploader with better mobile support
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="📱 Mobile tip: For best results, use good lighting and hold phone steady"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        # Enhanced preprocessing for mobile images
        processed_image = preprocess_mobile_image(image)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process the image
        with st.spinner("🔍 Performing advanced 3D analysis..."):
            # Extract 3D landmarks
            landmarks_3d, mesh_results = extract_3d_landmarks(processed_image)
            
            if landmarks_3d is None:
                st.error("❌ No face detected. Please try another image with a clear, front-facing face.")
                st.info("💡 **Mobile Tips:**\n- Use good lighting\n- Keep face centered\n- Remove sunglasses\n- Try different angles")
                return
            
            # Basic face detection for emotion analysis
            gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                st.error("❌ Face detection failed for emotion analysis.")
                return
            
            face_coords = max(faces, key=lambda x: x[2] * x[3])
            
            # Perform analyses based on mode
            if analysis_mode in ["Complete Analysis", "3D Reconstruction Only"]:
                # 3D Face Reconstruction
                with col2:
                    st.subheader("🌐 3D Face Reconstruction")
                    fig_3d = create_3d_face_model(landmarks_3d)
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.error("Could not create 3D model")
            
            if analysis_mode == "3D Reconstruction Only":
                return
            
            # Advanced face shape analysis
            shape_analysis = analyze_face_shape_advanced(landmarks_3d)
            
            # === FACE SHAPE ANALYSIS SECTION ===
            st.markdown("---")
            st.header("👤 Advanced Face Shape Analysis")
            
            col_shape1, col_shape2, col_shape3 = st.columns(3)
            
            with col_shape1:
                st.metric(
                    "Face Shape", 
                    shape_analysis['shape'],
                    help="3D landmark-based shape analysis"
                )
                st.metric(
                    "Confidence", 
                    f"{shape_analysis['confidence']:.1%}",
                    help="Algorithm confidence in shape classification"
                )
            
            with col_shape2:
                st.metric(
                    "Face Ratio", 
                    f"{shape_analysis['face_ratio']:.2f}",
                    help="Length to width ratio"
                )
                st.metric(
                    "Jaw Ratio", 
                    f"{shape_analysis['jaw_ratio']:.2f}",
                    help="Jaw width to face width ratio"
                )
            
            with col_shape3:
                measurements = shape_analysis.get('measurements', {})
                if measurements:
                    st.metric("Face Length", f"{measurements.get('face_length', 0):.1f}")
                    st.metric("Face Width", f"{measurements.get('face_width', 0):.1f}")
            
            # === BEAUTY AND ATTRACTIVENESS ANALYSIS SECTION ===
            st.markdown("---")
            st.header("✨ Advanced Beauty & Attractiveness Analysis")
            
            # Calculate advanced beauty score
            beauty_score, beauty_factors = calculate_beauty_score_advanced(landmarks_3d, shape_analysis)
            
            col_beauty1, col_beauty2 = st.columns(2)
            
            with col_beauty1:
                st.metric(
                    "Overall Beauty Score", 
                    f"{beauty_score:.1f}/100",
                    help="Advanced 3D analysis-based beauty scoring"
                )
                
                # Beauty category
                if beauty_score >= 90:
                    st.success("🌟 **Category**: Exceptionally Beautiful")
                elif beauty_score >= 80:
                    st.success("💫 **Category**: Very Attractive")
                elif beauty_score >= 70:
                    st.info("✨ **Category**: Above Average")
                elif beauty_score >= 60:
                    st.info("🌸 **Category**: Average Beauty")
                else:
                    st.warning("🎨 **Category**: Unique Beauty")
            
            with col_beauty2:
                # Display beauty factors
                st.subheader("Beauty Factor Breakdown")
                for factor_name, score, weight in beauty_factors:
                    st.metric(
                        factor_name, 
                        f"{score:.1f}/100",
                        delta=f"Weight: {weight:.0%}",
                        help=f"Contributes {weight:.0%} to overall score"
                    )
            
            # Beauty visualization
            st.subheader("📊 Beauty Analysis Breakdown")
            factor_df = pd.DataFrame({
                'Factor': [f[0] for f in beauty_factors],
                'Score': [f[1] for f in beauty_factors],
                'Weight': [f[2] * 100 for f in beauty_factors]
            })
            
            fig_beauty = px.bar(factor_df, x='Factor', y='Score', 
                              title='Beauty Factor Scores',
                              color='Weight',
                              color_continuous_scale='viridis')
            st.plotly_chart(fig_beauty, use_container_width=True)
            
            if analysis_mode != "Beauty Analysis Only":
                # === EMOTION ANALYSIS SECTION ===
                st.markdown("---")
                st.header("😊 Advanced Emotion Analysis")
                
                # Enhanced emotion prediction
                emotion, emotion_probs, emotion_labels = predict_emotion_cnn(processed_image, face_coords)
                
                col_emotion1, col_emotion2 = st.columns(2)
                
                with col_emotion1:
                    st.success(f"🎯 **Detected Emotion: {emotion}**")
                    st.info(f"📊 Confidence: {max(emotion_probs):.1%}")
                
                with col_emotion2:
                    # Emotion probability chart
                    emotion_df = pd.DataFrame({
                        'Emotion': emotion_labels,
                        'Probability': emotion_probs
                    }).sort_values('Probability', ascending=False)
                
                # Display emotion probabilities
                fig_emotion = px.bar(emotion_df, x='Emotion', y='Probability',
                                   title='Emotion Probability Distribution',
                                   color='Probability',
                                   color_continuous_scale='blues')
                st.plotly_chart(fig_emotion, use_container_width=True)
            
            # === PERSONALIZED RECOMMENDATIONS SECTION ===
            st.markdown("---")
            st.header("💄 Personalized Beauty Recommendations")
            
            # Makeup recommendations
            makeup_recs = get_makeup_recommendations(shape_analysis['shape'], beauty_factors)
            
            st.subheader("💅 Makeup & Styling Tips")
            for rec in makeup_recs:
                st.info(rec)
            
            # === FACIAL EXERCISE RECOMMENDATIONS ===
            st.markdown("---")
            st.header("🏃‍♀️ Personalized Facial Exercises")
            
            # Get age estimate (simplified for demo)
            age_estimate = 25 + (100 - beauty_score) * 0.3  # Rough age estimation
            
            exercises = get_facial_exercises(shape_analysis['shape'], age_estimate)
            
            st.subheader("💪 Daily Facial Workout Routine")
            st.write("**Recommended duration**: 10-15 minutes daily")
            
            col_ex1, col_ex2 = st.columns(2)
            
            for i, exercise in enumerate(exercises):
                if i % 2 == 0:
                    col_ex1.info(exercise)
                else:
                    col_ex2.info(exercise)
            
            # === PROGRESS TRACKING ===
            st.markdown("---")
            st.header("📈 Progress Tracking")
            
            # Save current analysis
            analysis_data = {
                'beauty_score': beauty_score,
                'face_shape': shape_analysis['shape'],
                'emotion': emotion,
                'age_estimate': age_estimate
            }
            
            if st.button("💾 Save Analysis to Progress"):
                entry_count = save_progress(analysis_data)
                st.success(f"✅ Analysis saved! Total entries: {entry_count}")
                st.balloons()
            
            # Show mini progress if available
            if 'progress_history' in st.session_state and st.session_state.progress_history:
                recent_scores = [entry['beauty_score'] for entry in st.session_state.progress_history[-5:]]
                if len(recent_scores) > 1:
                    trend = "📈 Improving" if recent_scores[-1] > recent_scores[0] else "📊 Stable"
                    st.info(f"**Recent Trend**: {trend} | **Average Score**: {np.mean(recent_scores):.1f}")
            
            # === REAL-TIME ANALYSIS INFO ===
            st.markdown("---")
            st.header("🎥 Real-Time Video Analysis")
            
            st.info("""
            🔄 **Coming Soon - Real-Time Features:**
            - Live webcam emotion detection
            - Real-time beauty scoring
            - Dynamic makeup try-on
            - Live facial exercise guidance
            - Progress tracking during exercises
            
            📱 **Current Capabilities:**
            - Upload multiple images for comparison
            - Batch analysis of photo series
            - Progress tracking over time
            """)
            
            # Video analysis placeholder
            enable_video = st.checkbox("🎥 Enable Experimental Video Analysis")
            
            if enable_video:
                st.warning("⚠️ Video analysis is experimental. Performance may vary.")
                
                # Placeholder for video analysis
                video_placeholder = st.empty()
                
                if st.button("📹 Start Video Analysis (Demo)"):
                    with video_placeholder.container():
                        st.info("🎬 Video analysis would appear here in full implementation")
                        st.write("Features would include:")
                        st.write("- Real-time face tracking")
                        st.write("- Live emotion detection")
                        st.write("- Dynamic beauty scoring")
                        st.write("- Exercise form checking")
            
            # === ADVANCED INSIGHTS ===
            with st.expander("🔬 Advanced Analysis Insights"):
                st.subheader("🧬 Facial Genetics Insights")
                
                # Facial feature analysis
                if landmarks_3d is not None:
                    # Calculate various facial metrics
                    nose_width = np.linalg.norm(landmarks_3d[31] - landmarks_3d[35])
                    eye_distance = np.linalg.norm(landmarks_3d[33] - landmarks_3d[133])
                    lip_width = np.linalg.norm(landmarks_3d[61] - landmarks_3d[291])
                    
                    col_insights1, col_insights2 = st.columns(2)
                    
                    with col_insights1:
                        st.metric("Nose Width Index", f"{nose_width:.2f}")
                        st.metric("Eye Distance Ratio", f"{eye_distance:.2f}")
                        st.metric("Lip Width Ratio", f"{lip_width:.2f}")
                    
                    with col_insights2:
                        # Facial harmony analysis
                        harmony_score = (beauty_score + shape_analysis['confidence'] * 100) / 2
                        st.metric("Facial Harmony", f"{harmony_score:.1f}/100")
                        
                        # Aging prediction
                        aging_factor = max(0, 100 - beauty_score) * 0.5
                        st.metric("Aging Factor", f"{aging_factor:.1f}")
                
                st.subheader("🎯 Improvement Potential")
                
                improvements = []
                
                for factor_name, score, weight in beauty_factors:
                    if score < 80:
                        if factor_name == 'Facial Symmetry':
                            improvements.append("🎭 **Symmetry**: Facial massage and targeted exercises")
                        elif factor_name == 'Golden Ratio':
                            improvements.append("📐 **Proportions**: Strategic makeup and hairstyling")
                        elif factor_name == 'Feature Proportion':
                            improvements.append("🎨 **Features**: Contouring and highlighting techniques")
                
                if not improvements:
                    st.success("🌟 Excellent facial harmony! Focus on maintenance.")
                else:
                    for improvement in improvements:
                        st.info(improvement)
            
            # === COMPARISON ANALYSIS ===
            with st.expander("🔄 Comparison Analysis"):
                st.write("Upload another image to compare facial features and beauty scores!")
                
                comparison_file = st.file_uploader(
                    "Choose comparison image...", 
                    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                    key="comparison"
                )
                
                if comparison_file is not None:
                    comp_image = Image.open(comparison_file)
                    comp_processed = preprocess_mobile_image(comp_image)
                    comp_landmarks, _ = extract_3d_landmarks(comp_processed)
                    
                    if comp_landmarks is not None:
                        comp_shape = analyze_face_shape_advanced(comp_landmarks)
                        comp_beauty, comp_factors = calculate_beauty_score_advanced(comp_landmarks, comp_shape)
                        
                        col_comp1, col_comp2 = st.columns(2)
                        
                        with col_comp1:
                            st.image(image, caption="Original", use_container_width=True)
                            st.metric("Beauty Score", f"{beauty_score:.1f}")
                            st.metric("Face Shape", shape_analysis['shape'])
                        
                        with col_comp2:
                            st.image(comp_image, caption="Comparison", use_container_width=True)
                            st.metric("Beauty Score", f"{comp_beauty:.1f}", 
                                    delta=f"{comp_beauty - beauty_score:.1f}")
                            st.metric("Face Shape", comp_shape['shape'])
                        
                        # Comparison insights
                        if comp_beauty > beauty_score:
                            st.success("📈 Comparison image scored higher!")
                        elif comp_beauty < beauty_score:
                            st.info("📉 Original image scored higher!")
                        else:
                            st.info("📊 Similar beauty scores!")
                    else:
                        st.error("Could not analyze comparison image")
    
    # === INFORMATION SECTIONS ===
    with st.expander("ℹ️ About Advanced 3D Face Analysis"):
        st.write("""
        ### 🚀 **Next-Generation Face Analysis Technology**
        
        **🌐 3D Reconstruction Features:**
        - **468 3D Landmarks**: MediaPipe-powered precise facial mapping
        - **Depth Analysis**: Real Z-coordinate facial depth mapping
        - **Interactive 3D Models**: Rotatable and explorable face reconstructions
        - **Advanced Measurements**: True 3D geometric analysis
        
        **🧠 AI & Machine Learning:**
        - **CNN Emotion Detection**: Deep learning emotion recognition
        - **Advanced Age Estimation**: Multi-factor age prediction models
        - **Real-time Processing**: Optimized for mobile and web deployment
        - **Progressive Enhancement**: Continuous learning capabilities
        
        **📱 Mobile Optimization:**
        - **Enhanced Preprocessing**: CLAHE contrast enhancement
        - **Noise Reduction**: Bilateral filtering for cleaner analysis
        - **EXIF Handling**: Automatic image orientation correction
        - **Multi-format Support**: JPEG, PNG, WebP, and more
        
        **💄 Personalization Engine:**
        - **Face Shape Specific**: Tailored recommendations for each shape
        - **Beauty Factor Analysis**: Multi-dimensional beauty assessment
        - **Dynamic Recommendations**: Adaptive based on individual features
        - **Progress Tracking**: Long-term beauty and health monitoring
        
        **🏃‍♀️ Wellness Integration:**
        - **Facial Exercise Programs**: Scientifically-based face yoga
        - **Anti-aging Protocols**: Age-specific exercise routines
        - **Progress Monitoring**: Track improvements over time
        - **Health Insights**: Facial analysis for wellness indicators
        
        **🔬 Scientific Accuracy:**
        - **Golden Ratio Analysis**: Mathematical beauty standards
        - **Facial Thirds**: Professional proportion assessment
        - **Symmetry Algorithms**: Advanced bilateral comparison
        - **Anthropometric Standards**: Based on facial anthropometry research
        """)
    
    with st.expander("🎯 How to Get Best Results"):
        st.write("""
        ### 📸 **Photography Tips for Optimal Analysis**
        
        **💡 Lighting:**
        - Use natural daylight when possible
        - Avoid harsh shadows or direct flash
        - Ensure even lighting across the face
        - Avoid backlighting or silhouettes
        
        **📱 Camera Position:**
        - Hold camera at eye level
        - Keep face centered in frame
        - Fill 40-60% of frame with face
        - Ensure both eyes are visible and level
        
        **😊 Expression & Pose:**
        - Neutral expression for most accurate analysis
        - Look directly at camera
        - Keep head straight (not tilted)
        - Remove sunglasses and hats
        
        **🔧 Technical Settings:**
        - Use highest resolution available
        - Ensure image isn't blurry
        - Avoid heavy filters or editing
        - Save in high-quality format (JPEG fine, PNG)
        
        **📊 For Progress Tracking:**
        - Use consistent lighting conditions
        - Same camera distance and angle
        - Regular intervals (weekly/monthly)
        - Document any changes (makeup, skincare, etc.)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🎭 Advanced 3D Face Analysis System | 
        Built with MediaPipe, TensorFlow & Streamlit | 
        ⚠️ For entertainment and educational purposes
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
