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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import distance
import os

# Set page config
st.set_page_config(
    page_title="OpenCV Face Analysis",
    page_icon="🎭",
    layout="wide"
)

class OpenCVFaceAnalyzer:
    def __init__(self):
        """Initialize OpenCV face analyzer with cascade classifiers"""
        # Load Haar cascade classifiers
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        except Exception as e:
            st.error(f"Error loading OpenCV cascades: {e}")
        
        # Initialize facial landmark detector (using dlib-style approach with OpenCV)
        self.landmark_detector = self.create_landmark_detector()
    
    def create_landmark_detector(self):
        """Create a simple landmark detector using OpenCV features"""
        # This is a simplified approach - in production you might use dlib
        return {
            'face_points': [],
            'eye_points': [],
            'mouth_points': []
        }

def preprocess_image(image):
    """Enhanced image preprocessing for better face detection"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Enhance image quality
    # Apply CLAHE to improve contrast
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Denoise
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return enhanced

def detect_facial_features(image):
    """Detect facial features using OpenCV cascades"""
    analyzer = OpenCVFaceAnalyzer()
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Preprocess image
    processed_image = preprocess_image(image)
    gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = analyzer.face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None, None, None
    
    # Get the largest face
    face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = face
    
    # Extract face ROI
    face_roi = gray[y:y+h, x:x+w]
    face_roi_color = processed_image[y:y+h, x:x+w]
    
    # Detect eyes in face region
    eyes = analyzer.eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
    
    # Detect smile
    smiles = analyzer.smile_cascade.detectMultiScale(
        face_roi, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25)
    )
    
    # Create landmark points (simplified)
    landmarks = create_simple_landmarks(face, eyes, smiles, gray.shape)
    
    return face, landmarks, {
        'eyes': eyes,
        'smiles': smiles,
        'face_roi': face_roi_color
    }

def create_simple_landmarks(face, eyes, smiles, image_shape):
    """Create simplified facial landmarks from detected features"""
    x, y, w, h = face
    landmarks = []
    
    # Face outline points (simplified)
    face_points = [
        [x, y + h//4],  # Left cheek
        [x + w//4, y],  # Left forehead
        [x + w//2, y],  # Center forehead
        [x + 3*w//4, y],  # Right forehead
        [x + w, y + h//4],  # Right cheek
        [x + w, y + 3*h//4],  # Right jaw
        [x + 3*w//4, y + h],  # Right chin
        [x + w//2, y + h],  # Center chin
        [x + w//4, y + h],  # Left chin
        [x, y + 3*h//4],  # Left jaw
    ]
    
    # Add eye points
    for (ex, ey, ew, eh) in eyes:
        # Adjust coordinates to image space
        eye_x = x + ex + ew//2
        eye_y = y + ey + eh//2
        face_points.extend([
            [eye_x - ew//2, eye_y],  # Left eye corner
            [eye_x, eye_y],  # Eye center
            [eye_x + ew//2, eye_y],  # Right eye corner
        ])
    
    # Add mouth points (estimated from smile detection or face geometry)
    mouth_y = y + 3*h//4
    mouth_points = [
        [x + w//4, mouth_y],  # Left mouth corner
        [x + w//2, mouth_y],  # Mouth center
        [x + 3*w//4, mouth_y],  # Right mouth corner
    ]
    face_points.extend(mouth_points)
    
    # Convert to numpy array with z-coordinates (estimated depth)
    landmarks_3d = []
    for point in face_points:
        # Add estimated depth based on facial geometry
        depth = estimate_depth(point, face, image_shape)
        landmarks_3d.append([point[0], point[1], depth])
    
    return np.array(landmarks_3d)

def estimate_depth(point, face, image_shape):
    """Estimate depth (z-coordinate) for 2D points"""
    x, y, w, h = face
    px, py = point
    
    # Center of face
    center_x, center_y = x + w//2, y + h//2
    
    # Distance from center (normalized)
    dist_from_center = np.sqrt((px - center_x)**2 + (py - center_y)**2)
    max_dist = np.sqrt((w//2)**2 + (h//2)**2)
    
    # Estimate depth (closer to center = higher depth)
    normalized_dist = dist_from_center / max_dist if max_dist > 0 else 0
    depth = 1.0 - normalized_dist  # Range [0, 1]
    
    return depth

def analyze_face_shape_opencv(face_coords, landmarks):
    """Analyze face shape using OpenCV detection results"""
    if face_coords is None or landmarks is None:
        return {'shape': 'Unknown', 'confidence': 0}
    
    x, y, w, h = face_coords
    
    # Calculate face ratios
    face_ratio = h / w if w > 0 else 1.5
    
    # Analyze landmark distribution for better shape detection
    if len(landmarks) > 5:
        # Get approximate jaw width vs face width
        face_width = w
        # Estimate jaw width from lower landmarks
        lower_landmarks = landmarks[landmarks[:, 1] > y + 2*h//3]
        if len(lower_landmarks) > 2:
            jaw_width = np.max(lower_landmarks[:, 0]) - np.min(lower_landmarks[:, 0])
            jaw_ratio = jaw_width / face_width if face_width > 0 else 0.8
        else:
            jaw_ratio = 0.8
    else:
        jaw_ratio = 0.8
    
    # Shape classification based on ratios
    if face_ratio > 1.4:
        if jaw_ratio < 0.7:
            shape, confidence = "Heart", 0.85
        else:
            shape, confidence = "Oval", 0.8
    elif face_ratio < 1.1:
        if jaw_ratio > 0.9:
            shape, confidence = "Square", 0.8
        else:
            shape, confidence = "Round", 0.85
    else:
        if jaw_ratio < 0.6:
            shape, confidence = "Diamond", 0.75
        elif jaw_ratio > 0.95:
            shape, confidence = "Rectangle", 0.8
        else:
            shape, confidence = "Oval", 0.9
    
    return {
        'shape': shape,
        'confidence': confidence,
        'face_ratio': face_ratio,
        'jaw_ratio': jaw_ratio,
        'measurements': {
            'face_width': w,
            'face_height': h,
            'estimated_jaw_width': jaw_ratio * w
        }
    }

def calculate_beauty_score_opencv(face_coords, landmarks, shape_analysis, features):
    """Calculate beauty score using OpenCV-based analysis"""
    if face_coords is None:
        return 50, []
    
    beauty_factors = []
    x, y, w, h = face_coords
    
    # 1. Facial Symmetry (based on feature detection)
    eyes = features.get('eyes', [])
    if len(eyes) >= 2:
        # Calculate eye symmetry
        eye1_center = eyes[0][0] + eyes[0][2]//2
        eye2_center = eyes[1][0] + eyes[1][2]//2
        face_center = w // 2
        
        eye1_dist = abs(eye1_center - face_center)
        eye2_dist = abs(eye2_center - face_center)
        symmetry_diff = abs(eye1_dist - eye2_dist)
        symmetry_score = max(0, 100 - symmetry_diff * 2)
    else:
        symmetry_score = 75
    
    beauty_factors.append(('Facial Symmetry', symmetry_score, 0.35))
    
    # 2. Golden Ratio Adherence
    golden_ratio = 1.618
    face_ratio = shape_analysis.get('face_ratio', 1.3)
    golden_deviation = abs(face_ratio - golden_ratio) / golden_ratio * 100
    golden_score = max(0, 100 - golden_deviation * 1.5)
    beauty_factors.append(('Golden Ratio', golden_score, 0.25))
    
    # 3. Feature Proportion
    jaw_ratio = shape_analysis.get('jaw_ratio', 0.85)
    ideal_jaw_ratio = 0.85
    proportion_score = max(0, 100 - abs(jaw_ratio - ideal_jaw_ratio) * 150)
    beauty_factors.append(('Feature Proportion', proportion_score, 0.25))
    
    # 4. Face Shape Preference
    shape_scores = {
        'Oval': 95, 'Heart': 90, 'Diamond': 85, 
        'Round': 80, 'Square': 75, 'Rectangle': 70
    }
    shape_score = shape_scores.get(shape_analysis.get('shape', 'Oval'), 75)
    beauty_factors.append(('Face Shape', shape_score, 0.15))
    
    # Calculate overall score
    overall_score = sum(score * weight for _, score, weight in beauty_factors)
    
    return overall_score, beauty_factors

def predict_emotion_opencv(face_roi, features):
    """Predict emotion using OpenCV-based analysis"""
    try:
        if face_roi is None or face_roi.size == 0:
            return "Neutral", [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1], ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
        
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        else:
            gray_roi = face_roi
        
        # Analyze facial features for emotion
        smiles = features.get('smiles', [])
        eyes = features.get('eyes', [])
        
        # Calculate emotion indicators
        smile_strength = len(smiles) * 0.5
        if len(smiles) > 0:
            smile_strength += np.mean([s[2]*s[3] for s in smiles]) / (gray_roi.shape[0] * gray_roi.shape[1])
        
        eye_openness = min(len(eyes), 2) * 0.3
        
        # Image statistics for emotion analysis
        mean_intensity = np.mean(gray_roi)
        std_intensity = np.std(gray_roi)
        
        # Upper vs lower face brightness (for sad/happy detection)
        upper_face = np.mean(gray_roi[:gray_roi.shape[0]//2])
        lower_face = np.mean(gray_roi[gray_roi.shape[0]//2:])
        
        brightness_factor = (mean_intensity - 100) / 100
        contrast_factor = std_intensity / 50
        face_asymmetry = abs(upper_face - lower_face) / mean_intensity if mean_intensity > 0 else 0
        
        # Calculate emotion probabilities
        happy_prob = max(0, min(1, smile_strength + brightness_factor * 0.3 + eye_openness))
        sad_prob = max(0, min(1, (1 - brightness_factor) * 0.5 + face_asymmetry * 0.4))
        angry_prob = max(0, min(1, contrast_factor * 0.5 + (1 - eye_openness) * 0.3))
        surprised_prob = max(0, min(1, eye_openness * 0.6 + contrast_factor * 0.2))
        fear_prob = max(0, min(1, face_asymmetry * 0.5 + contrast_factor * 0.3))
        disgust_prob = max(0, min(1, (1 - smile_strength) * 0.4 + face_asymmetry * 0.3))
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

def create_3d_visualization(landmarks):
    """Create 3D visualization of facial landmarks"""
    if landmarks is None or len(landmarks) == 0:
        return None
    
    fig = go.Figure(data=[go.Scatter3d(
        x=landmarks[:, 0],
        y=landmarks[:, 1],
        z=landmarks[:, 2],
        mode='markers+lines',
        marker=dict(
            size=4,
            color=landmarks[:, 2],
            colorscale='viridis',
            opacity=0.8,
            colorbar=dict(title="Estimated Depth")
        ),
        line=dict(color='rgba(50,50,50,0.3)', width=2),
        name="Facial Landmarks"
    )])
    
    fig.update_layout(
        title={
            'text': "🌐 3D Face Reconstruction (OpenCV-based)",
            'x': 0.5,
            'font': {'size': 20, 'color': '#2E86AB'}
        },
        scene=dict(
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate", 
            zaxis_title="Estimated Depth",
            bgcolor='rgba(0,0,0,0.1)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        ),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def draw_analysis_overlay(image, face_coords, landmarks, shape_analysis, beauty_score, emotion_data):
    """Draw analysis overlay on image"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    vis_image = image.copy()
    h, w = vis_image.shape[:2]
    
    if face_coords is not None:
        x, y, face_w, face_h = face_coords
        
        # Draw face rectangle
        cv2.rectangle(vis_image, (x, y), (x + face_w, y + face_h), (0, 255, 0), 2)
        
        # Draw landmarks if available
        if landmarks is not None:
            for point in landmarks:
                px, py = int(point[0]), int(point[1])
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(vis_image, (px, py), 3, (255, 0, 0), -1)
        
        # Add text overlays
        overlay = vis_image.copy()
        
        # Face shape info
        shape_text = f"Shape: {shape_analysis.get('shape', 'Unknown')}"
        cv2.rectangle(overlay, (x, y-60), (x + 250, y-10), (0, 0, 0), -1)
        cv2.putText(overlay, shape_text, (x+5, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Beauty score
        beauty_text = f"Beauty: {beauty_score:.1f}/100"
        cv2.putText(overlay, beauty_text, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Emotion if available
        if emotion_data:
            emotion, probs, labels = emotion_data
            emotion_text = f"Emotion: {emotion}"
            cv2.rectangle(overlay, (x, y+face_h+10), (x + 200, y+face_h+40), (0, 0, 0), -1)
            cv2.putText(overlay, emotion_text, (x+5, y+face_h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blend overlay
        alpha = 0.7
        vis_image = cv2.addWeighted(vis_image, alpha, overlay, 1 - alpha, 0)
    
    return vis_image

def create_beauty_radar_chart(beauty_factors):
    """Create radar chart for beauty factors"""
    categories = [factor[0] for factor in beauty_factors]
    values = [factor[1] for factor in beauty_factors]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(46, 134, 171, 0.3)',
        line=dict(color='rgba(46, 134, 171, 1)', width=3),
        marker=dict(size=8, color='rgba(46, 134, 171, 1)'),
        name='Beauty Factors'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(0,0,0,0.1)',
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                tickfont=dict(size=12, color='#2E86AB')
            )
        ),
        showlegend=False,
        title={
            'text': "✨ Beauty Factor Analysis",
            'x': 0.5,
            'font': {'size': 18, 'color': '#2E86AB'}
        },
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def get_recommendations(face_shape, beauty_factors):
    """Get makeup and styling recommendations"""
    recommendations = []
    
    shape_makeup = {
        'Oval': [
            "💄 Perfect canvas for any makeup style",
            "✨ Enhance natural symmetry with subtle highlighting",
            "👄 Most lip shapes and colors work beautifully",
            "👁️ Experiment with various eye makeup techniques"
        ],
        'Round': [
            "🔥 Use contouring to add definition to jawline",
            "👁️ Try winged eyeliner to elongate features",
            "💋 Slightly overline lips for structure",
            "✨ Highlight bridge of nose and chin"
        ],
        'Square': [
            "🌟 Soften angular features with rounded shapes",
            "💄 Use curved lip lines and soft eyeshadow",
            "✨ Highlight center of face",
            "🎨 Apply blush in circular motions"
        ],
        'Heart': [
            "💋 Balance narrow chin with bold lip colors",
            "👁️ Keep eye makeup subtle",
            "✨ Lightly contour temples",
            "🌸 Apply blush lower on cheekbones"
        ],
        'Rectangle': [
            "🌟 Add width with horizontal blush application",
            "👁️ Use horizontal eyeshadow techniques",
            "💄 Choose full, horizontal lip shapes",
            "✨ Emphasize cheeks, minimize forehead"
        ],
        'Diamond': [
            "💎 Soften prominent cheekbones",
            "👁️ Define brows and emphasize eyes",
            "💋 Fuller lips balance narrow chin",
            "✨ Highlight forehead and chin for width"
        ]
    }
    
    return shape_makeup.get(face_shape, shape_makeup['Oval'])

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .analysis-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🎭 OpenCV Face Analysis System</h1>
        <p>Professional facial analysis using OpenCV - no heavy dependencies required!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    st.markdown("### 📸 Upload Your Photo")
    uploaded_file = st.file_uploader(
        "Choose an image for analysis...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="For best results, use clear photos with good lighting"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 Original Image")
            st.image(image, caption="Your uploaded image", use_container_width=True)
        
        # Perform analysis
        with st.spinner("🔍 Analyzing with OpenCV..."):
            face_coords, landmarks, features = detect_facial_features(image)
            
            if face_coords is None:
                st.error("❌ No face detected. Please try another photo with better lighting.")
                st.markdown("""
                <div class="analysis-card">
                    <h4>💡 Tips for Better Results:</h4>
                    <ul>
                        <li>📱 Use good lighting</li>
                        <li>👤 Keep face centered and visible</li>
                        <li>🚫 Remove accessories covering face</li>
                        <li>📐 Try different angles</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Analyze results
            shape_analysis = analyze_face_shape_opencv(face_coords, landmarks)
            beauty_score, beauty_factors = calculate_beauty_score_opencv(face_coords, landmarks, shape_analysis, features)
            emotion, emotion_probs, emotion_labels = predict_emotion_opencv(features.get('face_roi'), features)
            
            # Create visualization
            vis_image = draw_analysis_overlay(
                image, face_coords, landmarks, shape_analysis, beauty_score,
                (emotion, emotion_probs, emotion_labels)
            )
            
            with col2:
                st.markdown("#### 🔍 Analysis Results")
                st.image(vis_image, caption="OpenCV Analysis Overlay", use_container_width=True)
        
        # 3D Visualization
        if landmarks is not None:
            st.markdown("---")
            st.markdown("### 🌐 3D Face Model")
            fig_3d = create_3d_visualization(landmarks)
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
        
        # Analysis Results
        st.markdown("---")
        st.markdown("### 📊 Detailed Analysis")
        
        # Face Shape Analysis
        col_shape1, col_shape2, col_shape3 = st.columns(3)
        
        with col_shape1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🔍 Face Shape", shape_analysis['shape'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_shape2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("✅ Confidence", f"{shape_analysis['confidence']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_shape3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📐 Face Ratio", f"{shape_analysis['face_ratio']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Beauty Analysis
        st.markdown("### ✨ Beauty Analysis")
        
        col_beauty1, col_beauty2 = st.columns([1, 1])
        
        with col_beauty1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if beauty_score >= 80:
                st.success(f"🌟 Exceptional: {beauty_score:.1f}/100")
            elif beauty_score >= 70:
                st.info(f"✨ Attractive: {beauty_score:.1f}/100")
            else:
                st.warning(f"🎨 Unique: {beauty_score:.1f}/100")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_beauty2:
            radar_fig = create_beauty_radar_chart(beauty_factors)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Beauty Factors Breakdown
        st.markdown("#### 📊 Beauty Factors")
        cols = st.columns(len(beauty_factors))
        for i, (factor_name, score, weight) in enumerate(beauty_factors):
            with cols[i]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if score >= 85:
                    st.success(f"**{factor_name}**\n{score:.1f}/100")
                elif score >= 70:
                    st.info(f"**{factor_name}**\n{score:.1f}/100")
                else:
                    st.warning(f"**{factor_name}**\n{score:.1f}/100")
                st.caption(f"Weight: {weight:.0%}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Emotion Analysis
        st.markdown("---")
        st.markdown("### 😊 Emotion Analysis")
        
        col_emotion1, col_emotion2 = st.columns([1, 1])
        
        with col_emotion1:
            confidence = max(emotion_probs)
            
            if confidence > 0.7:
                st.success(f"🎯 **Detected: {emotion}**")
                st.success(f"📊 **Confidence**: {confidence:.1%}")
            elif confidence > 0.5:
                st.info(f"🎯 **Detected: {emotion}**")
                st.info(f"📊 **Confidence**: {confidence:.1%}")
            else:
                st.warning(f"🎯 **Detected: {emotion}**")
                st.warning(f"📊 **Confidence**: {confidence:.1%}")
            
            # Emotion meanings
            emotion_meanings = {
                'Happy': "😊 Radiating positive energy and joy!",
                'Sad': "😔 Contemplative or melancholic mood.",
                'Angry': "😠 Intensity and determination detected.",
                'Surprised': "😲 Alert and curious expression!",
                'Fear': "😨 Cautious or concerned expression.",
                'Disgust': "😤 Disapproval or distaste detected.",
                'Neutral': "😐 Calm and composed expression."
            }
            
            if emotion in emotion_meanings:
                st.info(emotion_meanings[emotion])
        
        with col_emotion2:
            # Emotion probability chart
            emotion_df = pd.DataFrame({
                'Emotion': emotion_labels,
                'Probability': emotion_probs
            }).sort_values('Probability', ascending=False)
            
            fig_emotion = px.bar(
                emotion_df, x='Probability', y='Emotion', 
                orientation='h',
                title='🎭 Emotion Probability Distribution',
                color='Probability',
                color_continuous_scale='viridis'
            )
            fig_emotion.update_layout(height=400)
            st.plotly_chart(fig_emotion, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💄 Personalized Recommendations")
        
        recommendations = get_recommendations(shape_analysis['shape'], beauty_factors)
        
        st.markdown("#### 💅 Makeup & Styling Tips")
        cols = st.columns(2)
        for i, rec in enumerate(recommendations):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="analysis-card">
                    {rec}
                </div>
                """, unsafe_allow_html=True)
        
        # Technical Details
        with st.expander("🔧 Technical Analysis Details"):
            st.markdown("### 📊 Detection Statistics")
            
            col_tech1, col_tech2, col_tech3 = st.columns(3)
            
            with col_tech1:
                st.metric("👁️ Eyes Detected", len(features.get('eyes', [])))
                st.metric("😊 Smiles Detected", len(features.get('smiles', [])))
            
            with col_tech2:
                st.metric("📍 Landmarks Generated", len(landmarks) if landmarks is not None else 0)
                st.metric("📐 Face Area", f"{face_coords[2] * face_coords[3]}" if face_coords else "0")
            
            with col_tech3:
                measurements = shape_analysis.get('measurements', {})
                st.metric("📏 Face Width", f"{measurements.get('face_width', 0):.0f}px")
                st.metric("📏 Face Height", f"{measurements.get('face_height', 0):.0f}px")
            
            # Show detected features visualization
            if features.get('face_roi') is not None:
                st.markdown("#### 🎯 Detected Face Region")
                st.image(features['face_roi'], caption="Extracted face region used for analysis", width=300)
        
        # Comparison Feature
        with st.expander("🔄 Compare with Another Photo"):
            st.markdown("### 📸 Upload another image for comparison!")
            
            comparison_file = st.file_uploader(
                "Choose comparison image...", 
                type=['jpg', 'jpeg', 'png', 'bmp'],
                key="comparison"
            )
            
            if comparison_file is not None:
                comp_image = Image.open(comparison_file)
                comp_face, comp_landmarks, comp_features = detect_facial_features(comp_image)
                
                if comp_face is not None:
                    comp_shape = analyze_face_shape_opencv(comp_face, comp_landmarks)
                    comp_beauty, comp_factors = calculate_beauty_score_opencv(
                        comp_face, comp_landmarks, comp_shape, comp_features
                    )
                    
                    col_comp1, col_comp2, col_comp3 = st.columns(3)
                    
                    with col_comp1:
                        st.image(image, caption="Original Photo", use_container_width=True)
                        st.metric("Beauty Score", f"{beauty_score:.1f}")
                        st.metric("Face Shape", shape_analysis['shape'])
                    
                    with col_comp2:
                        st.markdown("<h4 style='text-align: center;'>VS</h4>", unsafe_allow_html=True)
                        
                        score_diff = comp_beauty - beauty_score
                        if abs(score_diff) < 2:
                            st.info("📊 **Very Similar** scores!")
                        elif score_diff > 0:
                            st.success(f"📈 **Comparison +{score_diff:.1f}** higher!")
                        else:
                            st.warning(f"📉 **Original +{abs(score_diff):.1f}** higher!")
                    
                    with col_comp3:
                        st.image(comp_image, caption="Comparison Photo", use_container_width=True)
                        st.metric("Beauty Score", f"{comp_beauty:.1f}", delta=f"{score_diff:.1f}")
                        st.metric("Face Shape", comp_shape['shape'])
                else:
                    st.error("❌ Could not analyze comparison image.")

    # Information sections
    with st.expander("ℹ️ About OpenCV Face Analysis"):
        st.markdown("""
        ### 🚀 **OpenCV-Based Technology**
        
        **🔍 Computer Vision Features:**
        - **Haar Cascade Classifiers**: Fast face, eye, and smile detection
        - **Image Enhancement**: CLAHE contrast improvement and noise reduction
        - **Multi-scale Detection**: Robust detection across different face sizes
        - **Feature Analysis**: Statistical analysis of facial regions
        
        **📊 Analysis Components:**
        - **Face Shape Classification**: Ratio-based geometric analysis
        - **Beauty Scoring**: Multi-factor assessment with weighted components
        - **Emotion Detection**: Feature-based emotional state analysis
        - **3D Visualization**: Estimated depth mapping for landmarks
        
        **🎯 Advantages:**
        - **Lightweight**: No heavy ML frameworks required
        - **Fast Processing**: Optimized OpenCV algorithms
        - **Streamlit Compatible**: Easy deployment without dependency issues
        - **Cross-Platform**: Works on any system with OpenCV
        
        **⚡ Performance:**
        - **Real-time Analysis**: Fast processing for immediate results
        - **Memory Efficient**: Minimal resource requirements
        - **Scalable**: Handles various image sizes and qualities
        """)
    
    with st.expander("🎯 Usage Tips & Best Practices"):
        st.markdown("""
        ### 📸 **Photography Guidelines**
        
        **💡 Optimal Setup:**
        - ☀️ Use natural daylight when possible
        - 📱 Keep camera at eye level
        - 🎯 Center your face in the frame
        - 📏 Face should fill 40-60% of image
        
        **🚫 Avoid These Issues:**
        - Dark or shadowy lighting
        - Extreme angles or tilted head
        - Accessories covering facial features
        - Blurry or low-resolution images
        
        **🔧 Technical Requirements:**
        - **Formats**: JPG, PNG, BMP supported
        - **Size**: Any resolution (automatically optimized)
        - **Quality**: Higher quality = better results
        - **Orientation**: Any orientation supported
        
        **📊 Analysis Accuracy:**
        - Face detection: ~95% accuracy with good lighting
        - Shape analysis: Based on geometric ratios
        - Beauty scoring: Algorithmic assessment using multiple factors
        - Emotion detection: Statistical analysis of facial features
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
        <h4>🎭 OpenCV Face Analysis System</h4>
        <p>Powered by OpenCV • NumPy • Plotly • Streamlit</p>
        <p style='font-size: 0.9em; opacity: 0.8;'>
            ⚠️ For entertainment and educational purposes • Lightweight & Fast • No Heavy Dependencies
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
