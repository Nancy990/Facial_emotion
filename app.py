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

# Set page config with orange theme
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
        
        # 1. Smile detection score
        smile_score = len(smiles) * 0.3
        if len(smiles) > 0:
            # Calculate smile strength based on size
            smile_areas = [s[2] * s[3] for s in smiles]
            smile_score += np.mean(smile_areas) / (w * h) * 10
        
        # 2. Eye analysis
        eye_score = min(len(eyes), 2) * 0.2
        eye_openness = 1.0
        if len(eyes) >= 2:
            # Analyze eye dimensions for surprise/tiredness
            avg_eye_height = np.mean([e[3] for e in eyes])
            avg_eye_width = np.mean([e[2] for e in eyes])
            eye_ratio = avg_eye_height / avg_eye_width if avg_eye_width > 0 else 0.5
            eye_openness = min(2.0, eye_ratio * 2)
        
        # 3. Facial region analysis
        # Divide face into regions
        upper_face = face_roi_gray[:h//3, :]  # Forehead region
        middle_face = face_roi_gray[h//3:2*h//3, :]  # Eye region
        lower_face = face_roi_gray[2*h//3:, :]  # Mouth region
        
        # Calculate regional statistics
        upper_mean = np.mean(upper_face)
        middle_mean = np.mean(middle_face)
        lower_mean = np.mean(lower_face)
        
        # Overall face statistics
        face_mean = np.mean(face_roi_gray)
        face_std = np.std(face_roi_gray)
        
        # 4. Edge detection for expression lines
        edges = cv2.Canny(face_roi_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        
        # 5. Histogram analysis
        hist = cv2.calcHist([face_roi_gray], [0], None, [256], [0, 256])
        hist_mean = np.argmax(hist)
        
        # Calculate emotion probabilities based on features
        
        # Happy: Strong smile detection + bright face + eye openness
        happy_prob = (
            smile_score * 0.4 +
            (face_mean - 100) / 155 * 0.3 +
            eye_openness * 0.2 +
            (lower_mean / face_mean - 0.95) * 0.1
        )
        
        # Sad: Low brightness + droopy features + no smile
        sad_prob = (
            (1 - smile_score / 2) * 0.4 +
            (150 - face_mean) / 150 * 0.3 +
            (1 - eye_openness) * 0.2 +
            (upper_mean / lower_mean - 1) * 0.1
        )
        
        # Angry: High contrast + no smile + tense features
        angry_prob = (
            (face_std / 50) * 0.4 +
            (1 - smile_score / 2) * 0.3 +
            edge_density * 0.2 +
            (middle_mean / face_mean - 0.95) * 0.1
        )
        
        # Surprised: Wide eyes + high upper face brightness
        surprised_prob = (
            eye_openness * 0.5 +
            (upper_mean / face_mean - 0.95) * 0.3 +
            edge_density * 0.2
        )
        
        # Fear: Similar to surprised but with lower overall brightness
        fear_prob = (
            eye_openness * 0.3 +
            (130 - face_mean) / 130 * 0.4 +
            edge_density * 0.3
        )
        
        # Disgust: Nose wrinkle area + mouth tension
        disgust_prob = (
            edge_density * 0.4 +
            (middle_mean / lower_mean - 0.95) * 0.3 +
            (1 - smile_score / 2) * 0.3
        )
        
        # Neutral: Balanced features, no strong indicators
        neutral_prob = 1 - max(happy_prob, sad_prob, angry_prob, surprised_prob, fear_prob, disgust_prob)
        
        # Normalize probabilities
        probabilities = [angry_prob, disgust_prob, fear_prob, happy_prob, sad_prob, surprised_prob, max(0, neutral_prob)]
        
        # Ensure all probabilities are between 0 and 1
        probabilities = [max(0, min(1, p)) for p in probabilities]
        
        # Normalize to sum to 1
        total = sum(probabilities)
        if total > 0:
            probabilities = [p/total for p in probabilities]
        else:
            probabilities = [1/7] * 7
        
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
        prediction = emotions[np.argmax(probabilities)]
        
        return prediction, probabilities, emotions
        
    except Exception as e:
        st.error(f"Emotion prediction error: {str(e)}")
        return "Neutral", [0, 0, 0, 0.7, 0, 0, 0.3], ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']

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
    # Orange theme CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #FF8C00 0%, #FF6347 50%, #FF4500 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(255, 140, 0, 0.3);
    }
    .analysis-card {
        background: linear-gradient(135deg, #FFF8DC 0%, #FFEFD5 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FF8C00;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF8DC 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(255, 140, 0, 0.15);
        text-align: center;
        border: 2px solid #FFE4B5;
    }
    .stMetric > label {
        color: #FF8C00 !important;
        font-weight: bold !important;
    }
    .stMetric > div {
        color: #FF6347 !important;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FFF8DC 0%, #FFEFD5 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🎭 AI Face Analysis Studio</h1>
        <p>Discover your unique beauty with advanced OpenCV technology!</p>
        <p style="font-size: 0.9em; opacity: 0.9;">✨ Fast • Accurate • Beautiful Orange Theme ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    st.markdown("### 📸 Upload Your Photo")
    uploaded_file = st.file_uploader(
        "Choose an image for analysis...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="🧡 For best results, use clear photos with good lighting"
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
                st.image(vis_image, caption="OpenCV Analysis with Orange Theme", use_container_width=True)
        
        # Analysis Results
        st.markdown("---")
        st.markdown("### 📊 Detailed Analysis")
        
        # Face Shape Analysis
        col_shape1, col_shape2, col_shape3 = st.columns(3)
        
        with col_shape1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🎯 Face Shape", shape_analysis.get('shape', 'Unknown'))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_shape2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            confidence = shape_analysis.get('confidence', 0)
            st.metric("✅ Confidence", f"{confidence:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_shape3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            face_ratio = shape_analysis.get('face_ratio', 1.3)
            st.metric("📐 Face Ratio", f"{face_ratio:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Beauty Analysis
        st.markdown("### ✨ Beauty Analysis")
        
        col_beauty1, col_beauty2 = st.columns([1, 1])
        
        with col_beauty1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if beauty_score >= 85:
                st.success(f"🌟 Exceptional Beauty: {beauty_score:.1f}/100")
                desc = "You possess stunning, exceptional beauty!"
            elif beauty_score >= 75:
                st.success(f"✨ Very Attractive: {beauty_score:.1f}/100")
                desc = "You have highly attractive, striking features!"
            elif beauty_score >= 65:
                st.info(f"🧡 Above Average: {beauty_score:.1f}/100")
                desc = "Your features are notably attractive and well-balanced!"
            elif beauty_score >= 55:
                st.info(f"🌸 Naturally Pretty: {beauty_score:.1f}/100")
                desc = "You have natural, warm beauty that's very appealing!"
            else:
                st.warning(f"🎨 Unique Beauty: {beauty_score:.1f}/100")
                desc = "Your distinctive features create memorable, unique beauty!"
            
            st.markdown(f"<p style='text-align: center; font-style: italic; color: #FF8C00;'>{desc}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_beauty2:
            radar_fig = create_beauty_radar_chart(beauty_factors)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
        
        # Beauty Factors Breakdown
        if beauty_factors:
            st.markdown("#### 📊 Beauty Factors Breakdown")
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
        st.markdown("### 😊 Enhanced Emotion Analysis")
        
        col_emotion1, col_emotion2 = st.columns([1, 1])
        
        with col_emotion1:
            # Safe confidence calculation
            try:
                confidence = max(emotion_probs) if emotion_probs and isinstance(emotion_probs, (list, tuple)) else 0.0
            except (ValueError, TypeError):
                confidence = 0.0
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if confidence > 0.6:
                st.success(f"🎯 **Detected: {emotion}**")
                st.success(f"📊 **Confidence**: {confidence:.1%}")
            elif confidence > 0.4:
                st.info(f"🎯 **Detected: {emotion}**")
                st.info(f"📊 **Confidence**: {confidence:.1%}")
            else:
                st.warning(f"🎯 **Detected: {emotion}**")
                st.warning(f"📊 **Confidence**: {confidence:.1%}")
            
            # Emotion meanings with orange theme
            emotion_meanings = {
                'Happy': "😊 Radiating joy and positive energy! 🧡",
                'Sad': "😔 Thoughtful or contemplative mood detected.",
                'Angry': "😠 Intensity and strong determination present.",
                'Surprised': "😲 Alert, curious, and engaged expression! ✨",
                'Fear': "😨 Cautious or concerned expression detected.",
                'Disgust': "😤 Disapproval or distaste in expression.",
                'Neutral': "😐 Calm, composed, and balanced expression. 🧡"
            }
            
            if emotion in emotion_meanings:
                st.info(emotion_meanings[emotion])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_emotion2:
            # Emotion probability chart with orange theme
            if emotion_probs and emotion_labels and len(emotion_probs) == len(emotion_labels):
                emotion_df = pd.DataFrame({
                    'Emotion': emotion_labels,
                    'Probability': emotion_probs
                }).sort_values('Probability', ascending=False)
                
                fig_emotion = px.bar(
                    emotion_df, x='Probability', y='Emotion', 
                    orientation='h',
                    title='🎭 Emotion Analysis Results',
                    color='Probability',
                    color_continuous_scale=['#FFE4B5', '#FF8C00', '#FF4500']
                )
                fig_emotion.update_layout(
                    height=400,
                    title_font_color='#FF8C00',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_emotion, use_container_width=True)
        
        # Enhanced Features Detection Info
        st.markdown("---")
        st.markdown("### 🔍 Detection Details")
        
        col_detect1, col_detect2, col_detect3, col_detect4 = st.columns(4)
        
        with col_detect1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            eyes_count = len(features.get('eyes', [])) if features else 0
            st.metric("👁️ Eyes Detected", eyes_count)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_detect2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            smiles_count = len(features.get('smiles', [])) if features else 0
            st.metric("😊 Smiles Detected", smiles_count)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_detect3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # Safe face area calculation
            try:
                face_area = face_coords[2] * face_coords[3] if (face_coords is not None and len(face_coords) >= 4) else 0
            except (IndexError, TypeError):
                face_area = 0
            st.metric("📐 Face Area", f"{face_area:,}px²")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_detect4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # Safe face width calculation
            measurements = shape_analysis.get('measurements', {}) if shape_analysis else {}
            face_width = measurements.get('face_width', 0) if measurements else 0
            st.metric("📏 Face Width", f"{face_width}px")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💄 Personalized Beauty Recommendations")
        
        recommendations = get_recommendations(shape_analysis.get('shape', 'Oval'), beauty_factors)
        
        st.markdown("#### ✨ Makeup & Styling Tips")
        cols = st.columns(2)
        for i, rec in enumerate(recommendations):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="analysis-card">
                    <h5 style="color: #FF8C00;">{rec}</h5>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional tips based on beauty score
        st.markdown("#### 🌟 Personalized Enhancement Tips")
        
        enhancement_tips = []
        if beauty_score < 70:
            enhancement_tips.extend([
                "🧡 **Skincare Focus**: Consistent skincare routine can enhance natural glow",
                "✨ **Highlighting**: Strategic highlighting can enhance facial structure",
                "💫 **Confidence**: Your unique features are your greatest asset!"
            ])
        
        # Add tips based on face shape
        shape_tips = {
            'Round': ["🔥 **Contouring**: Focus on adding definition to jawline and cheekbones"],
            'Square': ["🌸 **Softening**: Use curved lines and soft colors to balance angular features"],
            'Heart': ["💋 **Balance**: Draw attention to lips to balance wider forehead"],
            'Rectangle': ["🌟 **Width**: Add horizontal elements to create width illusion"],
            'Diamond': ["💎 **Harmony**: Soften cheekbones while enhancing forehead and chin"],
            'Oval': ["✨ **Versatility**: You can experiment with most makeup styles!"]
        }
        
        current_shape = shape_analysis.get('shape', 'Oval')
        if current_shape in shape_tips:
            enhancement_tips.extend(shape_tips[current_shape])
        
        # Display enhancement tips
        for tip in enhancement_tips:
            st.markdown(f"""
            <div class="analysis-card">
                <p style="margin: 0; color: #333;">{tip}</p>
            </div>
            """, unsafe_allow_html=True)
        
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
                        st.metric("Beauty Score", f"{beauty_score:.1f}")
                        st.metric("Face Shape", shape_analysis.get('shape', 'Unknown'))
                        st.metric("Emotion", emotion)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_comp2:
                        st.markdown("<h4 style='text-align: center; color: #FF8C00; margin-top: 100px;'>⚡ VS ⚡</h4>", unsafe_allow_html=True)
                        
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
                        st.metric("Beauty Score", f"{comp_beauty:.1f}", delta=f"{score_diff:.1f}")
                        st.metric("Face Shape", comp_shape.get('shape', 'Unknown'))
                        st.metric("Emotion", comp_emotion)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ Could not analyze comparison image. Please try another photo.")

    # Information sections
    with st.expander("ℹ️ About Our AI Technology"):
        st.markdown("""
        <div class="analysis-card">
        <h3 style="color: #FF8C00;">🚀 Advanced OpenCV Technology</h3>
        
        <h4 style="color: #FF6347;">🔍 Computer Vision Features:</h4>
        <ul>
            <li><strong>Multi-scale Detection:</strong> Advanced face detection with multiple parameters</li>
            <li><strong>Enhanced Preprocessing:</strong> CLAHE contrast enhancement and noise reduction</li>
            <li><strong>Feature Analysis:</strong> Eyes, smile, and facial region detection</li>
            <li><strong>Improved Emotion AI:</strong> Multi-factor emotion analysis system</li>
        </ul>
        
        <h4 style="color: #FF6347;">📊 Analysis Components:</h4>
        <ul>
            <li><strong>Face Shape:</strong> Geometric ratio-based classification</li>
            <li><strong>Beauty Scoring:</strong> Multi-dimensional assessment with weighted factors</li>
            <li><strong>Emotion Detection:</strong> Enhanced regional analysis and feature correlation</li>
            <li><strong>Personalization:</strong> Tailored recommendations based on your unique features</li>
        </ul>
        
        <h4 style="color: #FF6347;">⚡ Performance Benefits:</h4>
        <ul>
            <li><strong>Lightning Fast:</strong> Optimized OpenCV algorithms</li>
            <li><strong>No Dependencies:</strong> No heavy ML frameworks required</li>
            <li><strong>Cloud Ready:</strong> Perfect for Streamlit deployment</li>
            <li><strong>Cross-Platform:</strong> Works everywhere OpenCV is supported</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🎯 Photography Tips for Best Results"):
        st.markdown("""
        <div class="analysis-card">
        <h3 style="color: #FF8C00;">📸 Professional Photo Guidelines</h3>
        
        <h4 style="color: #FF6347;">💡 Lighting Setup:</h4>
        <ul>
            <li>🌞 <strong>Natural Light:</strong> Soft daylight from a window works best</li>
            <li>📍 <strong>Even Illumination:</strong> Avoid harsh shadows on one side</li>
            <li>🚫 <strong>No Direct Flash:</strong> Creates unflattering harsh shadows</li>
            <li>⏰ <strong>Golden Hour:</strong> 1 hour after sunrise or before sunset is ideal</li>
        </ul>
        
        <h4 style="color: #FF6347;">📱 Camera Position:</h4>
        <ul>
            <li>👁️ <strong>Eye Level:</strong> Camera should be at your eye level</li>
            <li>🎯 <strong>Center Frame:</strong> Keep your face in the center</li>
            <li>📏 <strong>Proper Distance:</strong> Face should fill 40-60% of frame</li>
            <li>📐 <strong>Straight Angle:</strong> Keep your head upright and level</li>
        </ul>
        
        <h4 style="color: #FF6347;">😊 Expression Tips:</h4>
        <ul>
            <li>😐 <strong>Neutral Expression:</strong> Most accurate for analysis</li>
            <li>👀 <strong>Direct Gaze:</strong> Look straight at the camera</li>
            <li>🚫 <strong>Remove Accessories:</strong> Glasses, hats, or face coverings</li>
            <li>💄 <strong>Natural Look:</strong> Minimal makeup for best detection</li>
        </ul>
        
        <h4 style="color: #FF6347;">🔧 Technical Quality:</h4>
        <ul>
            <li>📊 <strong>High Resolution:</strong> Use your camera's highest setting</li>
            <li>🔍 <strong>Sharp Focus:</strong> Ensure the image is crisp and clear</li>
            <li>🎨 <strong>No Filters:</strong> Avoid Instagram filters or heavy editing</li>
            <li>💾 <strong>Original Format:</strong> Upload uncompressed images when possible</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with orange theme
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #FF8C00 0%, #FF6347 50%, #FF4500 100%); border-radius: 15px; color: white; margin-top: 20px;'>
        <h3>🎭 AI Face Analysis Studio</h3>
        <p style="font-size: 1.1em; margin: 10px 0;">Powered by OpenCV • NumPy • Plotly • Streamlit</p>
        <p style='font-size: 0.9em; opacity: 0.9; margin: 5px 0;'>
            ⚡ Lightning Fast • 🎨 Beautiful Orange Theme • 🔧 No Heavy Dependencies
        </p>
        <p style='font-size: 0.8em; opacity: 0.8; margin-top: 15px;'>
            ⚠️ For entertainment and educational purposes • Results may vary • Beauty is subjective and unique to everyone
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
