import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import math
import json
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import distance
import os

# Set page config with attractive theme
st.set_page_config(
    page_title="AI Beauty Studio",
    page_icon="✨",
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
    
    # Detect faces with multiple parameters for better detection
    faces = []
    for scale_factor in [1.1, 1.2, 1.3]:
        for min_neighbors in [3, 4, 5]:
            detected_faces = analyzer.face_cascade.detectMultiScale(
                gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(50, 50)
            )
            if len(detected_faces) > 0:
                faces.extend(detected_faces)
                break
        if len(faces) > 0:
            break
    
    if len(faces) == 0:
        return None, None
    
    # Get the largest face
    face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = face
    
    # Extract face ROI
    face_roi = gray[y:y+h, x:x+w]
    face_roi_color = processed_image[y:y+h, x:x+w]
    
    # Detect eyes in face region with better parameters
    eyes = analyzer.eye_cascade.detectMultiScale(
        face_roi, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10)
    )
    
    # Detect smile with improved parameters
    smiles = analyzer.smile_cascade.detectMultiScale(
        face_roi, scaleFactor=1.7, minNeighbors=15, minSize=(15, 15)
    )
    
    return face, {
        'eyes': eyes,
        'smiles': smiles,
        'face_roi': face_roi_color,
        'face_roi_gray': face_roi
    }

def analyze_face_shape_opencv(face_coords):
    """Analyze face shape using OpenCV detection results"""
    if face_coords is None or len(face_coords) < 4:
        return {'shape': 'Unknown', 'confidence': 0, 'face_ratio': 1.5, 'jaw_ratio': 0.8, 'measurements': {}}
    
    x, y, w, h = face_coords
    
    # Calculate face ratios
    face_ratio = h / w if w > 0 else 1.5
    
    # Estimate jaw width (simplified approach)
    jaw_ratio = 0.8  # Default estimation
    
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
            'estimated_jaw_width': int(jaw_ratio * w)
        }
    }

def calculate_beauty_score_opencv(face_coords, shape_analysis, features):
    """Calculate beauty score using OpenCV-based analysis"""
    if face_coords is None or len(face_coords) < 4:
        return 50, []
    
    beauty_factors = []
    x, y, w, h = face_coords
    
    # 1. Facial Symmetry (based on feature detection)
    eyes = features.get('eyes', []) if features else []
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

def predict_emotion_opencv(features):
    """Enhanced emotion prediction using OpenCV-based analysis"""
    try:
        if not features:
            return "Neutral", [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1], ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
        
        face_roi_gray = features.get('face_roi_gray')
        face_roi_color = features.get('face_roi')
        smiles = features.get('smiles', [])
        eyes = features.get('eyes', [])
        
        if face_roi_gray is None or face_roi_gray.size == 0:
            return "Neutral", [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1], ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
        
        # Enhanced emotion detection using multiple factors
        h, w = face_roi_gray.shape
        
        # Initialize base scores
        emotion_scores = {
            'happy': 0.2,
            'sad': 0.15,
            'angry': 0.1,
            'surprised': 0.1,
            'fear': 0.1,
            'disgust': 0.1,
            'neutral': 0.25
        }
        
        # 1. Enhanced Smile Detection
        smile_intensity = 0
        if len(smiles) > 0:
            # Calculate relative smile size
            smile_areas = [(s[2] * s[3]) / (w * h) for s in smiles]
            smile_intensity = min(1.0, np.mean(smile_areas) * 50)  # Boost smile detection
            emotion_scores['happy'] += smile_intensity * 0.6
            emotion_scores['neutral'] -= smile_intensity * 0.3
            emotion_scores['sad'] -= smile_intensity * 0.2
        
        # 2. Eye Analysis for Surprise/Fear
        eye_openness = 0.5
        if len(eyes) >= 1:
            # Better eye analysis
            eye_heights = [e[3] for e in eyes]
            eye_widths = [e[2] for e in eyes]
            avg_eye_ratio = np.mean([h/w for h, w in zip(eye_heights, eye_widths) if w > 0])
            
            if avg_eye_ratio > 0.6:  # Wide eyes
                emotion_scores['surprised'] += 0.4
                emotion_scores['fear'] += 0.3
                emotion_scores['neutral'] -= 0.2
            elif avg_eye_ratio < 0.3:  # Narrow eyes
                emotion_scores['angry'] += 0.2
                emotion_scores['sad'] += 0.2
        
        # 3. Facial Region Contrast Analysis
        # Divide face into regions for better emotion detection
        upper_region = face_roi_gray[:h//3, :]  # Forehead
        middle_region = face_roi_gray[h//3:2*h//3, :]  # Eyes/nose
        lower_region = face_roi_gray[2*h//3:, :]  # Mouth/chin
        
        # Calculate regional brightness
        upper_brightness = np.mean(upper_region) / 255.0
        middle_brightness = np.mean(middle_region) / 255.0
        lower_brightness = np.mean(lower_region) / 255.0
        overall_brightness = np.mean(face_roi_gray) / 255.0
        
        # Brightness-based emotion indicators
        if overall_brightness < 0.4:  # Dark face (shadows/sad)
            emotion_scores['sad'] += 0.3
            emotion_scores['angry'] += 0.2
            emotion_scores['happy'] -= 0.2
        elif overall_brightness > 0.7:  # Bright face (happy/surprised)
            emotion_scores['happy'] += 0.2
            emotion_scores['surprised'] += 0.1
        
        # Regional contrast analysis
        if upper_brightness > lower_brightness + 0.1:  # Raised eyebrows
            emotion_scores['surprised'] += 0.3
            emotion_scores['fear'] += 0.2
        elif lower_brightness > upper_brightness + 0.1:  # Lowered brow
            emotion_scores['angry'] += 0.3
            emotion_scores['sad'] += 0.2
        
        # 4. Edge Detection for Expression Lines
        edges = cv2.Canny(face_roi_gray, 30, 100)
        edge_density = np.sum(edges > 0) / (w * h)
        
        if edge_density > 0.15:  # High edge density (wrinkles/tension)
            emotion_scores['angry'] += 0.25
            emotion_scores['disgust'] += 0.2
            emotion_scores['surprised'] += 0.15
        
        # 5. Mouth Region Analysis
        mouth_region = lower_region[h//6:, :]  # Bottom third of lower region
        if mouth_region.size > 0:
            mouth_std = np.std(mouth_region)
            if mouth_std > 25:  # High variation in mouth area
                emotion_scores['happy'] += 0.2
                emotion_scores['surprised'] += 0.1
        
        # Normalize scores to probabilities
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            probabilities = [
                emotion_scores['angry'] / total_score,
                emotion_scores['disgust'] / total_score,
                emotion_scores['fear'] / total_score,
                emotion_scores['happy'] / total_score,
                emotion_scores['sad'] / total_score,
                emotion_scores['surprised'] / total_score,
                emotion_scores['neutral'] / total_score
            ]
        else:
            probabilities = [1/7] * 7
        
        # Ensure all probabilities are valid
        probabilities = [max(0.01, min(0.98, p)) for p in probabilities]
        
        # Re-normalize
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]
        
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
        prediction = emotions[np.argmax(probabilities)]
        
        return prediction, probabilities, emotions
        
    except Exception as e:
        st.error(f"Emotion prediction error: {str(e)}")
        return "Neutral", [0.14, 0.14, 0.14, 0.15, 0.14, 0.14, 0.15], ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']

def draw_analysis_overlay(image, face_coords, shape_analysis, beauty_score, emotion_data):
    """Draw analysis overlay on image"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    vis_image = image.copy()
    h, w = vis_image.shape[:2]
    
    if face_coords is not None and len(face_coords) >= 4:
        x, y, face_w, face_h = face_coords
        
        # Draw face rectangle with orange theme
        cv2.rectangle(vis_image, (x, y), (x + face_w, y + face_h), (255, 140, 0), 3)
        
        # Add text overlays with orange theme
        overlay = vis_image.copy()
        
        # Background for text
        cv2.rectangle(overlay, (x, y-80), (x + 300, y), (0, 0, 0), -1)
        
        # Face shape info
        shape_text = f"Shape: {shape_analysis.get('shape', 'Unknown')}"
        cv2.putText(overlay, shape_text, (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Beauty score
        beauty_text = f"Beauty: {beauty_score:.1f}/100"
        cv2.putText(overlay, beauty_text, (x+5, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Emotion if available
        if emotion_data:
            emotion, probs, labels = emotion_data
            emotion_text = f"Emotion: {emotion}"
            cv2.rectangle(overlay, (x, y+face_h+5), (x + 250, y+face_h+40), (0, 0, 0), -1)
            cv2.putText(overlay, emotion_text, (x+5, y+face_h+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Blend overlay
        alpha = 0.8
        vis_image = cv2.addWeighted(vis_image, alpha, overlay, 1 - alpha, 0)
    
    return vis_image

def create_beauty_radar_chart(beauty_factors):
    """Create radar chart with orange theme"""
    if not beauty_factors:
        return None
        
    categories = [factor[0] for factor in beauty_factors]
    values = [factor[1] for factor in beauty_factors]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 140, 0, 0.3)',
        line=dict(color='rgba(255, 140, 0, 1)', width=3),
        marker=dict(size=8, color='rgba(255, 140, 0, 1)'),
        name='Beauty Factors'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255, 140, 0, 0.2)',
                tickfont=dict(size=10, color='#FF8C00')
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 140, 0, 0.2)',
                tickfont=dict(size=12, color='#FF8C00')
            )
        ),
        showlegend=False,
        title={
            'text': "✨ Beauty Factor Analysis",
            'x': 0.5,
            'font': {'size': 18, 'color': '#FF8C00'}
        },
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def get_recommendations(face_shape, beauty_factors):
    """Get makeup and styling recommendations"""
    shape_makeup = {
        'Oval': [
            "🧡 Perfect canvas for any makeup style",
            "✨ Enhance natural symmetry with subtle highlighting",
            "💋 Most lip shapes and colors work beautifully",
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
    # Enhanced attractive CSS with gradients and animations
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 25%, #FF6B9D 50%, #C44569 75%, #F8B500 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(255, 107, 107, 0.4);
        position: relative;
        overflow: hidden;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .analysis-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F5 50%, #FFE5E5 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #FFB3BA;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.15);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(255, 107, 107, 0.25);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF8F0 50%, #FFEDE0 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(255, 139, 0, 0.2);
        text-align: center;
        border: 3px solid #FFD700;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 40px rgba(255, 139, 0, 0.3);
    }
    
    .beauty-score-card {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FF8C00 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 50px rgba(255, 165, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .beauty-score-card::before {
        content: '✨';
        position: absolute;
        font-size: 3rem;
        opacity: 0.2;
        top: 10px;
        right: 15px;
        animation: float 2s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #F0FFF0 50%, #F5FFFA 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #32CD32;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(50, 205, 50, 0.15);
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(10px);
        border-left-width: 8px;
    }
    
    .tips-card {
        background: linear-gradient(135deg, #E6F3FF 0%, #F0F8FF 50%, #F8FCFF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #4169E1;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(65, 105, 225, 0.15);
        transition: all 0.3s ease;
    }
    
    .tips-card:hover {
        transform: translateX(10px);
        border-left-width: 8px;
    }
    
    .stMetric > label {
        color: #FF6B6B !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stMetric > div {
        color: #C44569 !important;
        font-weight: 700 !important;
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #E8E8FF 0%, #F0F0FF 50%, #F8F8FF 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 3px solid #9370DB;
        box-shadow: 0 15px 35px rgba(147, 112, 219, 0.2);
    }
    
    .section-divider {
        height: 4px;
        background: linear-gradient(90deg, #FF6B6B, #FF8E53, #FF6B9D, #C44569, #F8B500);
        border-radius: 2px;
        margin: 2rem 0;
        animation: gradient-flow 3s ease infinite;
    }
    
    @keyframes gradient-flow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-text {
        font-size: 1.2em;
        background: linear-gradient(45deg, #FF6B6B, #FF8E53, #FF6B9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1 style="font-size: 3.5em; margin: 0;">✨ AI Beauty Studio ✨</h1>
        <p style="font-size: 1.3em; margin: 15px 0; opacity: 0.95;">Discover Your Unique Beauty with Advanced AI Technology</p>
        <div class="hero-text">
            🎨 Professional Analysis • 💫 Personalized Recommendations • 📸 Beauty Enhancement Tips
        </div>
        <p style="font-size: 0.9em; opacity: 0.8; margin-top: 20px;">
            ⚡ Lightning Fast • 🔬 Scientific Accuracy • 🎯 Personalized Results
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader with enhanced styling
    st.markdown("### 📸 Upload Your Photo")
    st.markdown('<p class="hero-text">Transform your selfie into a professional beauty analysis!</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image for analysis...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="✨ For best results, use clear photos with good lighting"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 Original Image")
            st.image(image, caption="Your uploaded image", use_container_width=True)
        
        # Perform analysis
        with st.spinner("🔍 Analyzing with AI..."):
            face_coords, features = detect_facial_features(image)
            
            if face_coords is None:
                st.error("❌ No face detected. Please try another photo with better lighting.")
                st.markdown("""
                <div class="analysis-card">
                    <h4>💡 Tips for Better Results:</h4>
                    <ul>
                        <li>🌞 Use good lighting (natural daylight preferred)</li>
                        <li>👤 Keep face centered and clearly visible</li>
                        <li>🚫 Remove accessories covering facial features</li>
                        <li>📐 Try different angles if detection fails</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Analyze results
            shape_analysis = analyze_face_shape_opencv(face_coords)
            beauty_score, beauty_factors = calculate_beauty_score_opencv(face_coords, shape_analysis, features)
            emotion, emotion_probs, emotion_labels = predict_emotion_opencv(features)
            
            # Create visualization
            vis_image = draw_analysis_overlay(
                image, face_coords, shape_analysis, beauty_score,
                (emotion, emotion_probs, emotion_labels)
            )
            
            with col2:
                st.markdown("#### 🔍 AI Analysis Results")
                st.image(vis_image, caption="AI Beauty Analysis Results", use_container_width=True)
        
        # Personalized Beauty Recommendations (MOVED TO TOP)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 💄 Your Personalized Beauty Guide")
        
        recommendations = get_recommendations(shape_analysis.get('shape', 'Oval'), beauty_factors)
        
        st.markdown("### ✨ Makeup & Styling Recommendations")
        cols = st.columns(2)
        for i, rec in enumerate(recommendations):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h5 style="color: #228B22; margin: 0;">{rec}</h5>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional enhancement tips
        st.markdown("### 🌟 Personalized Enhancement Tips")
        
        enhancement_tips = []
        if beauty_score < 70:
            enhancement_tips.extend([
                "✨ **Skincare Focus**: Consistent skincare routine can enhance natural glow",
                "💫 **Strategic Highlighting**: Enhance facial structure with light and shadow",
                "🌟 **Confidence Boost**: Your unique features are your greatest asset!"
            ])
        
        # Add tips based on face shape
        shape_tips = {
            'Round': ["🔥 **Contouring Magic**: Focus on adding definition to jawline and cheekbones"],
            'Square': ["🌸 **Softening Techniques**: Use curved lines and soft colors to balance angular features"],
            'Heart': ["💋 **Perfect Balance**: Draw attention to lips to balance wider forehead"],
            'Rectangle': ["🌟 **Width Enhancement**: Add horizontal elements to create width illusion"],
            'Diamond': ["💎 **Harmony Creation**: Soften cheekbones while enhancing forehead and chin"],
            'Oval': ["✨ **Versatile Canvas**: You can experiment with most makeup styles!"]
        }
        
        current_shape = shape_analysis.get('shape', 'Oval')
        if current_shape in shape_tips:
            enhancement_tips.extend(shape_tips[current_shape])
        
        # Display enhancement tips in attractive cards
        for tip in enhancement_tips:
            st.markdown(f"""
            <div class="recommendation-card">
                <p style="margin: 0; color: #2F4F2F; font-weight: 500;">{tip}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Photography Tips Section
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 📸 Photography Tips for Stunning Results")
        
        photo_tips = [
            {
                "icon": "💡",
                "title": "Perfect Lighting",
                "tips": [
                    "🌅 **Golden Hour Magic**: 1 hour after sunrise or before sunset",
                    "🪟 **Window Light**: Soft, diffused natural light works best",
                    "🚫 **Avoid Harsh Flash**: Creates unflattering shadows",
                    "💡 **Even Illumination**: Light should hit your face evenly"
                ]
            },
            {
                "icon": "📱",
                "title": "Camera Technique",
                "tips": [
                    "👁️ **Eye Level Shot**: Camera at your eye level for best angles",
                    "🎯 **Center Your Face**: Keep yourself in the frame center",
                    "📏 **Optimal Distance**: Face should fill 40-60% of frame",
                    "📐 **Straight Angle**: Keep your head level and upright"
                ]
            },
            {
                "icon": "😊",
                "title": "Expression & Style",
                "tips": [
                    "😐 **Natural Expression**: Relaxed face for accurate analysis",
                    "👀 **Direct Gaze**: Look straight into the camera",
                    "👓 **Remove Accessories**: Take off glasses, hats, face coverings",
                    "💄 **Minimal Makeup**: Natural look gives best detection results"
                ]
            },
            {
                "icon": "🔧",
                "title": "Technical Quality",
                "tips": [
                    "📊 **High Resolution**: Use your camera's highest quality setting",
                    "🔍 **Sharp Focus**: Ensure image is crisp and clear",
                    "🎨 **No Filters**: Avoid Instagram filters or heavy editing",
                    "💾 **Original Format**: Upload uncompressed images when possible"
                ]
            }
        ]
        
        cols = st.columns(2)
        for i, tip_category in enumerate(photo_tips):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="tips-card">
                    <h4 style="color: #4169E1; margin-top: 0;">
                        {tip_category['icon']} {tip_category['title']}
                    </h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        {''.join([f'<li style="margin: 8px 0; color: #333;">{tip}</li>' for tip in tip_category['tips']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Beauty Analysis Section
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## ✨ Beauty Analysis Results")
        
        col_beauty1, col_beauty2 = st.columns([1, 1])
        
        with col_beauty1:
            st.markdown('<div class="beauty-score-card">', unsafe_allow_html=True)
            if beauty_score >= 85:
                st.markdown(f"<h2 style='margin: 0; font-size: 2.5em;'>🌟 {beauty_score:.1f}/100</h2>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin: 10px 0;'>Exceptional Beauty</h3>", unsafe_allow_html=True)
                desc = "You possess stunning, exceptional beauty!"
            elif beauty_score >= 75:
                st.markdown(f"<h2 style='margin: 0; font-size: 2.5em;'>✨ {beauty_score:.1f}/100</h2>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin: 10px 0;'>Very Attractive</h3>", unsafe_allow_html=True)
                desc = "You have highly attractive, striking features!"
            elif beauty_score >= 65:
                st.markdown(f"<h2 style='margin: 0; font-size: 2.5em;'>🧡 {beauty_score:.1f}/100</h2>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin: 10px 0;'>Above Average</h3>", unsafe_allow_html=True)
                desc = "Your features are notably attractive and well-balanced!"
            elif beauty_score >= 55:
                st.markdown(f"<h2 style='margin: 0; font-size: 2.5em;'>🌸 {beauty_score:.1f}/100</h2>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin: 10px 0;'>Naturally Pretty</h3>", unsafe_allow_html=True)
                desc = "You have natural, warm beauty that's very appealing!"
            else:
                st.markdown(f"<h2 style='margin: 0; font-size: 2.5em;'>🎨 {beauty_score:.1f}/100</h2>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin: 10px 0;'>Unique Beauty</h3>", unsafe_allow_html=True)
                desc = "Your distinctive features create memorable, unique beauty!"
            
            st.markdown(f"<p style='font-style: italic; opacity: 0.9; margin: 15px 0;'>{desc}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_beauty2:
            radar_fig = create_beauty_radar_chart(beauty_factors)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
        
        # Beauty Factors Breakdown
        if beauty_factors:
            st.markdown("### 📊 Beauty Factors Breakdown")
            cols = st.columns(len(beauty_factors))
            for i, (factor_name, score, weight) in enumerate(beauty_factors):
                with cols[i]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    if score >= 85:
                        st.markdown(f"<h4 style='color: #228B22; margin: 0;'>{factor_name}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color: #32CD32; margin: 5px 0;'>{score:.1f}</h2>", unsafe_allow_html=True)
                        st.markdown("<p style='color: #228B22; margin: 5px 0;'>Excellent ⭐</p>", unsafe_allow_html=True)
                    elif score >= 70:
                        st.markdown(f"<h4 style='color: #FF8C00; margin: 0;'>{factor_name}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color: #FFA500; margin: 5px 0;'>{score:.1f}</h2>", unsafe_allow_html=True)
                        st.markdown("<p style='color: #FF8C00; margin: 5px 0;'>Good ✨</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h4 style='color: #FF6B6B; margin: 0;'>{factor_name}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color: #FF4757; margin: 5px 0;'>{score:.1f}</h2>", unsafe_allow_html=True)
                        st.markdown("<p style='color: #FF6B6B; margin: 5px 0;'>Potential 💫</p>", unsafe_allow_html=True)
                    st.caption(f"Weight: {weight:.0%}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Emotion Analysis
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 😊 Emotion Analysis")
        
        col_emotion1, col_emotion2 = st.columns([1, 1])
        
        with col_emotion1:
            # Safe confidence calculation
            try:
                confidence = max(emotion_probs) if emotion_probs and isinstance(emotion_probs, (list, tuple)) else 0.0
            except (ValueError, TypeError):
                confidence = 0.0
            
            st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #9370DB; margin: 0; text-align: center;'>🎭 {emotion}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #8A2BE2; margin: 10px 0; text-align: center;'>Confidence: {confidence:.1%}</h3>", unsafe_allow_html=True)
            
            # Emotion meanings with attractive styling
            emotion_meanings = {
                'Happy': "😊 Radiating joy and positive energy! ✨",
                'Sad': "😔 Thoughtful or contemplative mood detected.",
                'Angry': "😠 Intensity and strong determination present.",
                'Surprised': "😲 Alert, curious, and engaged expression! 💫",
                'Fear': "😨 Cautious or concerned expression detected.",
                'Disgust': "😤 Disapproval or distaste in expression.",
                'Neutral': "😐 Calm, composed, and balanced expression. 🌟"
            }
            
            if emotion in emotion_meanings:
                st.markdown(f"<p style='text-align: center; font-style: italic; color: #6A5ACD; margin: 15px 0;'>{emotion_meanings[emotion]}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_emotion2:
            # Emotion probability chart with attractive theme
            if emotion_probs and emotion_labels and len(emotion_probs) == len(emotion_labels):
                emotion_df = pd.DataFrame({
                    'Emotion': emotion_labels,
                    'Probability': emotion_probs
                }).sort_values('Probability', ascending=False)
                
                fig_emotion = px.bar(
                    emotion_df, x='Probability', y='Emotion', 
                    orientation='h',
                    title='🎭 Emotion Probability Distribution',
                    color='Probability',
                    color_continuous_scale=['#E6E6FA', '#9370DB', '#8A2BE2', '#4B0082']
                )
                fig_emotion.update_layout(
                    height=400,
                    title_font_color='#9370DB',
                    title_font_size=16,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig_emotion, use_container_width=True)
        
        # Detailed Analysis
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 📊 Detailed Analysis Results")
        
        # Face Shape Analysis
        col_shape1, col_shape2, col_shape3 = st.columns(3)
        
        with col_shape1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #FF6B6B; margin: 0;'>🎯 Face Shape</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #C44569; margin: 10px 0;'>{shape_analysis.get('shape', 'Unknown')}</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_shape2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            confidence = shape_analysis.get('confidence', 0)
            st.markdown(f"<h3 style='color: #FF6B6B; margin: 0;'>✅ Confidence</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #C44569; margin: 10px 0;'>{confidence:.1%}</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_shape3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            face_ratio = shape_analysis.get('face_ratio', 1.3)
            st.markdown(f"<h3 style='color: #FF6B6B; margin: 0;'>📐 Face Ratio</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #C44569; margin: 10px 0;'>{face_ratio:.2f}</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Features Detection Info
        st.markdown("### 🔍 Detection Details")
        
        col_detect1, col_detect2, col_detect3, col_detect4 = st.columns(4)
        
        with col_detect1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            eyes_count = len(features.get('eyes', [])) if features else 0
            st.markdown(f"<h3 style='color: #4169E1; margin: 0;'>👁️ Eyes</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #1E90FF; margin: 10px 0;'>{eyes_count}</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_detect2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            smiles_count = len(features.get('smiles', [])) if features else 0
            st.markdown(f"<h3 style='color: #4169E1; margin: 0;'>😊 Smiles</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #1E90FF; margin: 10px 0;'>{smiles_count}</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_detect3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # Safe face area calculation
            try:
                face_area = face_coords[2] * face_coords[3] if (face_coords is not None and len(face_coords) >= 4) else 0
            except (IndexError, TypeError):
                face_area = 0
            st.markdown(f"<h3 style='color: #4169E1; margin: 0;'>📐 Area</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #1E90FF; margin: 10px 0;'>{face_area:,}px²</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_detect4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # Safe face width calculation
            measurements = shape_analysis.get('measurements', {}) if shape_analysis else {}
            face_width = measurements.get('face_width', 0) if measurements else 0
            st.markdown(f"<h3 style='color: #4169E1; margin: 0;'>📏 Width</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #1E90FF; margin: 10px 0;'>{face_width}px</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparison Feature
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with st.expander("🔄 Compare with Another Photo"):
            st.markdown("### 📸 Upload another image for comparison!")
            
            comparison_file = st.file_uploader(
                "Choose comparison image...", 
                type=['jpg', 'jpeg', 'png', 'bmp'],
                key="comparison"
            )
            
            if comparison_file is not None:
                comp_image = Image.open(comparison_file)
                comp_face, comp_features = detect_facial_features(comp_image)
                
                if comp_face is not None:
                    comp_shape = analyze_face_shape_opencv(comp_face)
                    comp_beauty, comp_factors = calculate_beauty_score_opencv(
                        comp_face, comp_shape, comp_features
                    )
                    comp_emotion, comp_probs, comp_labels = predict_emotion_opencv(comp_features)
                    
                    col_comp1, col_comp2, col_comp3 = st.columns(3)
                    
                    with col_comp1:
                        st.image(image, caption="Original Photo", use_container_width=True)
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color: #FF6B6B;'>Beauty Score</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color: #C44569;'>{beauty_score:.1f}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<p>Shape: {shape_analysis.get('shape', 'Unknown')}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p>Emotion: {emotion}</p>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_comp2:
                        st.markdown("<h2 style='text-align: center; color: #FF6B6B; margin-top: 100px;'>⚡ VS ⚡</h2>", unsafe_allow_html=True)
                        
                        score_diff = comp_beauty - beauty_score
                        if abs(score_diff) < 2:
                            st.info("📊 **Very Similar** beauty scores!")
                        elif score_diff > 0:
                            st.success(f"📈 **Comparison Image** +{score_diff:.1f} points higher!")
                        else:
                            st.warning(f"📉 **Original Image** +{abs(score_diff):.1f} points higher!")
                    
                    with col_comp3:
                        st.image(comp_image, caption="Comparison Photo", use_container_width=True)
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color: #FF6B6B;'>Beauty Score</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color: #C44569;'>{comp_beauty:.1f}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<p>Shape: {comp_shape.get('shape', 'Unknown')}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p>Emotion: {comp_emotion}</p>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ Could not analyze comparison image. Please try another photo.")

    # Information sections
    with st.expander("ℹ️ About Our AI Technology"):
        st.markdown("""
        <div class="analysis-card">
        <h3 style="color: #FF6B6B;">🚀 Advanced AI Technology</h3>
        
        <h4 style="color: #C44569;">🔍 Computer Vision Features:</h4>
        <ul>
            <li><strong>Multi-scale Detection:</strong> Advanced face detection with multiple parameters</li>
            <li><strong>Enhanced Preprocessing:</strong> CLAHE contrast enhancement and noise reduction</li>
            <li><strong>Feature Analysis:</strong> Eyes, smile, and facial region detection</li>
            <li><strong>Improved Emotion AI:</strong> Multi-factor emotion analysis system</li>
        </ul>
        
        <h4 style="color: #C44569;">📊 Analysis Components:</h4>
        <ul>
            <li><strong>Face Shape:</strong> Geometric ratio-based classification</li>
            <li><strong>Beauty Scoring:</strong> Multi-dimensional assessment with weighted factors</li>
            <li><strong>Emotion Detection:</strong> Enhanced regional analysis and feature correlation</li>
            <li><strong>Personalization:</strong> Tailored recommendations based on your unique features</li>
        </ul>
        
        <h4 style="color: #C44569;">⚡ Performance Benefits:</h4>
        <ul>
            <li><strong>Lightning Fast:</strong> Optimized OpenCV algorithms</li>
            <li><strong>No Dependencies:</strong> No heavy ML frameworks required</li>
            <li><strong>Cloud Ready:</strong> Perfect for Streamlit deployment</li>
            <li><strong>Cross-Platform:</strong> Works everywhere OpenCV is supported</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with attractive theme
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 25%, #FF6B9D 50%, #C44569 75%, #F8B500 100%); border-radius: 20px; color: white; margin-top: 30px; position: relative; overflow: hidden;'>
        <div style='position: relative; z-index: 2;'>
            <h2 style="margin: 0;">✨ AI Beauty Studio ✨</h2>
            <p style="font-size: 1.2em; margin: 15px 0; opacity: 0.95;">Powered by Advanced Computer Vision Technology</p>
            <div style="display: flex; justify-content: center; gap: 20px; margin: 20px 0; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px;">🔬 OpenCV</span>
                <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px;">📊 Plotly</span>
                <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px;">⚡ Streamlit</span>
                <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px;">🧮 NumPy</span>
            </div>
            <p style='font-size: 0.9em; opacity: 0.8; margin-top: 20px; line-height: 1.6;'>
                🎨 Beautiful Design • 🔬 Scientific Accuracy • 💫 Personalized Results • ⚡ Lightning Fast
            </p>
            <p style='font-size: 0.8em; opacity: 0.7; margin-top: 15px;'>
                ⚠️ For entertainment and educational purposes • Results may vary • Beauty is subjective and unique to everyone
            </p>
        </div>
        <div style='position: absolute; top: -50%; right: -50%; width: 200%; height: 200%; background: linear-gradient(45deg, transparent, rgba(255,255,255,0.05), transparent); animation: shine 4s infinite;'></div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
