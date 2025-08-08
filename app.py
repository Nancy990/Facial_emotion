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

def draw_analysis_overlay(image, face_coords, shape_analysis, beauty_score):
    """Draw analysis overlay on image with orange theme"""
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
        cv2.rectangle(overlay, (x, y-60), (x + 280, y), (0, 0, 0), -1)
        
        # Face shape info
        shape_text = f"Shape: {shape_analysis.get('shape', 'Unknown')}"
        cv2.putText(overlay, shape_text, (x+5, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Beauty score
        beauty_text = f"Beauty: {beauty_score:.1f}/100"
        cv2.putText(overlay, beauty_text, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
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

def get_advanced_makeup_tips(face_shape, beauty_score):
    """Get advanced makeup and styling tips"""
    advanced_tips = []
    
    # Foundation & Base tips
    foundation_tips = {
        'Oval': "🎨 **Foundation**: Apply evenly for flawless canvas. Your balanced features need minimal correction.",
        'Round': "🎨 **Foundation**: Use slightly darker shade on sides of face to create definition and slim appearance.",
        'Square': "🎨 **Foundation**: Blend darker tones on corners of jawline and temples to soften angular features.",
        'Heart': "🎨 **Foundation**: Focus coverage on forehead area and use lighter tones on chin to balance proportions.",
        'Rectangle': "🎨 **Foundation**: Apply horizontally and avoid vertical highlighting to create width illusion.",
        'Diamond': "🎨 **Foundation**: Soften prominent cheekbones with careful blending and highlight forehead/chin."
    }
    
    # Contouring tips
    contour_tips = {
        'Oval': "📐 **Contouring**: Light contouring along cheekbones and jawline to enhance natural structure.",
        'Round': "📐 **Contouring**: Heavy contouring on sides of face, temples, and under cheekbones for definition.",
        'Square': "📐 **Contouring**: Focus on softening jaw angles with curved blending techniques.",
        'Heart': "📐 **Contouring**: Minimal forehead contouring, focus on building up chin area with highlighting.",
        'Rectangle': "📐 **Contouring**: Horizontal techniques only - contour top/bottom of face, not sides.",
        'Diamond': "📐 **Contouring**: Reduce cheekbone prominence while adding width to forehead and chin areas."
    }
    
    # Eye makeup tips
    eye_tips = {
        'Oval': "👁️ **Eye Makeup**: Any style works! Try bold colors, dramatic wings, or creative techniques.",
        'Round': "👁️ **Eye Makeup**: Elongate with winged liner, dark outer corners, and upward blending.",
        'Square': "👁️ **Eye Makeup**: Soft, rounded eyeshadow shapes and curved eyeliner to complement features.",
        'Heart': "👁️ **Eye Makeup**: Keep subtle to maintain balance - neutral tones and thin liner work best.",
        'Rectangle': "👁️ **Eye Makeup**: Horizontal eyeshadow placement and wider liner to add face width.",
        'Diamond': "👁️ **Eye Makeup**: Bold, defined brows and dramatic eye looks to draw attention upward."
    }
    
    # Lip tips
    lip_tips = {
        'Oval': "💋 **Lip Makeup**: Perfect for any lip shape - bold colors, glosses, matte finishes all work.",
        'Round': "💋 **Lip Makeup**: Slightly overline to add structure, use defined lip lines and medium tones.",
        'Square': "💋 **Lip Makeup**: Rounded, full lip shapes with soft colors to complement angular features.",
        'Heart': "💋 **Lip Makeup**: Bold, attention-grabbing colors and full shapes to balance narrow chin.",
        'Rectangle': "💋 **Lip Makeup**: Wide, horizontal lip shapes and bright colors to add facial width.",
        'Diamond': "💋 **Lip Makeup**: Full, plump lips with glossy finishes to balance narrow chin area."
    }
    
    advanced_tips.extend([
        foundation_tips.get(face_shape, foundation_tips['Oval']),
        contour_tips.get(face_shape, contour_tips['Oval']),
        eye_tips.get(face_shape, eye_tips['Oval']),
        lip_tips.get(face_shape, lip_tips['Oval'])
    ])
    
    # Add beauty score specific tips
    if beauty_score >= 85:
        advanced_tips.append("✨ **Enhancement**: Your features are naturally stunning - focus on enhancing rather than correcting!")
    elif beauty_score >= 70:
        advanced_tips.append("🌟 **Refinement**: Small adjustments can take your look from great to absolutely gorgeous!")
    else:
        advanced_tips.append("💫 **Transformation**: Strategic makeup techniques can dramatically enhance your unique beauty!")
    
    return advanced_tips

def main():
    # Enhanced attractive CSS with orange gradients and animations
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8E53 25%, #FFA500 50%, #FF8C00 75%, #FF7F50 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(255, 140, 0, 0.4);
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
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF8F0 50%, #FFE5D1 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #FFB366;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(255, 140, 0, 0.15);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(255, 140, 0, 0.25);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF8F0 50%, #FFEDE0 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(255, 140, 0, 0.2);
        text-align: center;
        border: 3px solid #FF8C00;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 40px rgba(255, 140, 0, 0.3);
    }
    
    .beauty-score-card {
        background: linear-gradient(135deg, #FF8C00 0%, #FFA500 50%, #FF7F50 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 50px rgba(255, 140, 0, 0.4);
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
        background: linear-gradient(135deg, #FFF8F0 0%, #FFEDE0 50%, #FFE5D1 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #FF8C00;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255, 140, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(10px);
        border-left-width: 8px;
    }
    
    .tips-card {
        background: linear-gradient(135deg, #FFF5E6 0%, #FFEBCC 50%, #FFE0B3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #FF6B35;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.15);
        transition: all 0.3s ease;
    }
    
    .tips-card:hover {
        transform: translateX(10px);
        border-left-width: 8px;
    }
    
    .stMetric > label {
        color: #FF6B35 !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stMetric > div {
        color: #FF8C00 !important;
        font-weight: 700 !important;
    }
    
    .section-divider {
        height: 4px;
        background: linear-gradient(90deg, #FF6B35, #FF8E53, #FFA500, #FF8C00, #FF7F50);
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
        background: linear-gradient(45deg, #FF6B35, #FF8E53, #FFA500);
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
            🎨 Professional Analysis • 💄 Makeup Recommendations • 📸 Beauty Enhancement Tips
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
            
            # Create visualization (removed emotion data)
            vis_image = draw_analysis_overlay(image, face_coords, shape_analysis, beauty_score)
            
            with col2:
                st.markdown("#### 🔍 AI Analysis Results")
                st.image(vis_image, caption="AI Beauty Analysis Results", use_container_width=True)
        
        # Beauty Analysis Section
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## ✨ Beauty Analysis Results")
        
        col_beauty1, col_beauty2 = st.columns([1, 1])
        
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
                        st.markdown(f"<h4 style='color: #FF6B35; margin: 0;'>{factor_name}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color: #FF8C00; margin: 5px 0;'>{score:.1f}</h2>", unsafe_allow_html=True)
                        st.markdown("<p style='color: #FF6B35; margin: 5px 0;'>Excellent ⭐</p>", unsafe_allow_html=True)
                    elif score >= 70:
                        st.markdown(f"<h4 style='color: #FF8C00; margin: 0;'>{factor_name}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color: #FFA500; margin: 5px 0;'>{score:.1f}</h2>", unsafe_allow_html=True)
                        st.markdown("<p style='color: #FF8C00; margin: 5px 0;'>Good ✨</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h4 style='color: #FF7F50; margin: 0;'>{factor_name}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color: #FF6347; margin: 5px 0;'>{score:.1f}</h2>", unsafe_allow_html=True)
                        st.markdown("<p style='color: #FF7F50; margin: 5px 0;'>Potential 💫</p>", unsafe_allow_html=True)
                    st.caption(f"Weight: {weight:.0%}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Personalized Beauty Recommendations (Main Focus)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 💄 Your Personalized Makeup & Styling Guide")
        
        recommendations = get_recommendations(shape_analysis.get('shape', 'Oval'), beauty_factors)
        advanced_tips = get_advanced_makeup_tips(shape_analysis.get('shape', 'Oval'), beauty_score)
        
        # Basic Recommendations
        st.markdown("### ✨ Basic Makeup & Styling Tips")
        cols = st.columns(2)
        for i, rec in enumerate(recommendations):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h5 style="color: #FF6B35; margin: 0;">{rec}</h5>
                </div>
                """, unsafe_allow_html=True)
        
        # Advanced Professional Tips
        st.markdown("### 🎨 Advanced Professional Techniques")
        cols = st.columns(1)
        for tip in advanced_tips:
            st.markdown(f"""
            <div class="tips-card">
                <p style="margin: 0; color: #8B4513; font-weight: 500;">{tip}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Face Shape Specific Guide
        current_shape = shape_analysis.get('shape', 'Oval')
        st.markdown(f"### 📐 Complete Guide for {current_shape} Face Shape")
        
        shape_guides = {
            'Oval': {
                'dos': [
                    "✅ **Experiment freely** - Your balanced proportions work with most styles",
                    "✅ **Try bold colors** - You can handle dramatic looks beautifully",
                    "✅ **Play with trends** - Most makeup trends will complement your features",
                    "✅ **Enhance naturally** - Light touches bring out your inherent beauty"
                ],
                'donts': [
                    "❌ **Don't over-contour** - Your natural balance doesn't need heavy correction",
                    "❌ **Avoid harsh lines** - Soft blending maintains your natural harmony",
                    "❌ **Don't hide features** - Your proportions are naturally appealing"
                ]
            },
            'Round': {
                'dos': [
                    "✅ **Create angles** - Use contouring to add definition and structure",
                    "✅ **Elongate features** - Winged eyeliner and vertical highlighting",
                    "✅ **Define cheekbones** - Strategic contouring creates facial depth",
                    "✅ **Use angular shapes** - Sharp lines complement your soft curves"
                ],
                'donts': [
                    "❌ **Avoid rounded shapes** - They can emphasize the circular appearance",
                    "❌ **Don't highlight cheek apples** - This can make face appear fuller",
                    "❌ **Skip horizontal lines** - They can make face appear wider"
                ]
            },
            'Square': {
                'dos': [
                    "✅ **Soften angles** - Use curved lines and rounded shapes",
                    "✅ **Blend thoroughly** - Soft gradients complement angular features",
                    "✅ **Add warmth** - Peachy and coral tones soften strong lines",
                    "✅ **Focus on eyes** - Draw attention upward with eye makeup"
                ],
                'donts': [
                    "❌ **Avoid sharp lines** - They can emphasize the angular jawline",
                    "❌ **Don't over-contour jaw** - It can make angles appear harsher",
                    "❌ **Skip geometric shapes** - They compete with your natural structure"
                ]
            },
            'Heart': {
                'dos': [
                    "✅ **Balance proportions** - Draw attention to lips and lower face",
                    "✅ **Use bold lip colors** - They help balance a wider forehead",
                    "✅ **Keep eye makeup subtle** - Don't compete with your natural eye area",
                    "✅ **Add chin definition** - Light highlighting balances proportions"
                ],
                'donts': [
                    "❌ **Don't emphasize forehead** - It can unbalance your proportions",
                    "❌ **Avoid heavy eye makeup** - It can overpower delicate features",
                    "❌ **Skip dark lip colors** - Very dark shades can shrink the chin area"
                ]
            },
            'Rectangle': {
                'dos': [
                    "✅ **Add width illusion** - Horizontal techniques and broader applications",
                    "✅ **Emphasize cheeks** - Blush application that extends outward",
                    "✅ **Use warm colors** - They add dimension and visual width",
                    "✅ **Create curves** - Rounded shapes soften the elongated appearance"
                ],
                'donts': [
                    "❌ **Avoid vertical emphasis** - It can make face appear longer",
                    "❌ **Don't contour sides** - This can make face appear narrower",
                    "❌ **Skip thin applications** - Broader strokes work better"
                ]
            },
            'Diamond': {
                'dos': [
                    "✅ **Balance cheekbones** - Soften prominent cheek area",
                    "✅ **Widen forehead/chin** - Add visual weight to narrower areas",
                    "✅ **Define brows** - Strong brows balance prominent cheekbones",
                    "✅ **Fuller lips** - Help balance the narrower chin area"
                ],
                'donts': [
                    "❌ **Don't over-highlight cheeks** - They're already prominent",
                    "❌ **Avoid narrow lip shapes** - They can emphasize the pointed chin",
                    "❌ **Skip forehead contouring** - You want to add width, not reduce it"
                ]
            }
        }
        
        shape_guide = shape_guides.get(current_shape, shape_guides['Oval'])
        
        col_dos, col_donts = st.columns(2)
        
        with col_dos:
            st.markdown("#### ✅ Do's for Your Face Shape")
            for do in shape_guide['dos']:
                st.markdown(f"""
                <div class="recommendation-card">
                    <p style="margin: 0; color: #2F4F2F; font-weight: 500;">{do}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_donts:
            st.markdown("#### ❌ Don'ts for Your Face Shape")
            for dont in shape_guide['donts']:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FFE5E5 0%, #FFD1D1 50%, #FFBABA 100%); padding: 1.5rem; border-radius: 15px; border-left: 6px solid #FF6B6B; margin: 1rem 0; box-shadow: 0 8px 25px rgba(255, 107, 107, 0.15); transition: all 0.3s ease;">
                    <p style="margin: 0; color: #8B0000; font-weight: 500;">{dont}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Product Recommendations
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 🛍️ Recommended Products & Colors")
        
        product_recommendations = {
            'Oval': {
                'colors': ['🧡 Coral', '💋 Classic Red', '🌸 Rose', '💜 Berry'],
                'products': ['✨ Highlighting palette', '🎨 Versatile eyeshadow', '💄 Statement lipstick', '🌟 Bronzer']
            },
            'Round': {
                'colors': ['🔥 Bold Red', '🍇 Plum', '🥉 Bronze', '🌰 Taupe'],
                'products': ['📐 Contouring kit', '👁️ Winged eyeliner', '✨ Face highlighter', '🎨 Angled brushes']
            },
            'Square': {
                'colors': ['🌸 Soft Pink', '🧡 Peach', '💛 Golden', '🌺 Coral'],
                'products': ['🎨 Blending brushes', '💄 Curved lip brush', '✨ Soft highlighter', '🌟 Cream blush']
            },
            'Heart': {
                'colors': ['💋 Bold Lip Colors', '🌰 Neutral Eyes', '🧡 Warm Blush', '✨ Subtle Highlight'],
                'products': ['💄 Lip liner', '👁️ Neutral palette', '🎨 Precision brushes', '✨ Chin highlight']
            },
            'Rectangle': {
                'colors': ['🔥 Warm Reds', '🧡 Orange Blush', '💛 Golden Tones', '🌸 Pink Hues'],
                'products': ['🎨 Wide brushes', '✨ Bronzing palette', '💄 Fuller lip products', '🌟 Horizontal applicators']
            },
            'Diamond': {
                'colors': ['💜 Rich Berry', '🧡 Warm Coral', '💛 Golden Brown', '🌸 Rose Gold'],
                'products': ['👁️ Brow kit', '💄 Lip plumper', '✨ Forehead highlighter', '🎨 Cheek minimizer']
            }
        }
        
        current_recs = product_recommendations.get(current_shape, product_recommendations['Oval'])
        
        col_colors, col_products = st.columns(2)
        
        with col_colors:
            st.markdown("### 🎨 Perfect Colors for You")
            for color in current_recs['colors']:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h5 style="color: #FF6B35; margin: 0;">{color}</h5>
                </div>
                """, unsafe_allow_html=True)
        
        with col_products:
            st.markdown("### 🛍️ Must-Have Products")
            for product in current_recs['products']:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h5 style="color: #FF6B35; margin: 0;">{product}</h5>
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
                    <h4 style="color: #FF6B35; margin-top: 0;">
                        {tip_category['icon']} {tip_category['title']}
                    </h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        {''.join([f'<li style="margin: 8px 0; color: #333;">{tip}</li>' for tip in tip_category['tips']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
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
                    
                    col_comp1, col_comp2, col_comp3 = st.columns(3)
                    
                    with col_comp1:
                        st.image(image, caption="Original Photo", use_container_width=True)
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color: #FF6B35;'>Beauty Score</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color: #FF8C00;'>{beauty_score:.1f}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<p>Shape: {shape_analysis.get('shape', 'Unknown')}</p>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_comp2:
                        st.markdown("<h2 style='text-align: center; color: #FF6B35; margin-top: 100px;'>⚡ VS ⚡</h2>", unsafe_allow_html=True)
                        
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
                        st.markdown(f"<h4 style='color: #FF6B35;'>Beauty Score</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color: #FF8C00;'>{comp_beauty:.1f}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<p>Shape: {comp_shape.get('shape', 'Unknown')}</p>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ Could not analyze comparison image. Please try another photo.")

    # Information sections
    with st.expander("ℹ️ About Our AI Technology"):
        st.markdown("""
        <div class="analysis-card">
        <h3 style="color: #FF6B35;">🚀 Advanced AI Technology</h3>
        
        <h4 style="color: #FF8C00;">🔍 Computer Vision Features:</h4>
        <ul>
            <li><strong>Multi-scale Detection:</strong> Advanced face detection with multiple parameters</li>
            <li><strong>Enhanced Preprocessing:</strong> CLAHE contrast enhancement and noise reduction</li>
            <li><strong>Feature Analysis:</strong> Eyes, smile, and facial region detection</li>
            <li><strong>Beauty Analysis:</strong> Multi-factor beauty scoring system</li>
        </ul>
        
        <h4 style="color: #FF8C00;">📊 Analysis Components:</h4>
        <ul>
            <li><strong>Face Shape:</strong> Geometric ratio-based classification</li>
            <li><strong>Beauty Scoring:</strong> Multi-dimensional assessment with weighted factors</li>
            <li><strong>Makeup Recommendations:</strong> Personalized styling based on face shape</li>
            <li><strong>Professional Tips:</strong> Advanced techniques for enhancement</li>
        </ul>
        
        <h4 style="color: #FF8C00;">⚡ Performance Benefits:</h4>
        <ul>
            <li><strong>Lightning Fast:</strong> Optimized OpenCV algorithms</li>
            <li><strong>No Dependencies:</strong> No heavy ML frameworks required</li>
            <li><strong>Cloud Ready:</strong> Perfect for Streamlit deployment</li>
            <li><strong>Cross-Platform:</strong> Works everywhere OpenCV is supported</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with attractive orange theme
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #FF6B35 0%, #FF8E53 25%, #FFA500 50%, #FF8C00 75%, #FF7F50 100%); border-radius: 20px; color: white; margin-top: 30px; position: relative; overflow: hidden;'>
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
                🎨 Beautiful Design • 🔬 Scientific Accuracy • 💄 Makeup Expertise • ⚡ Lightning Fast
            </p>
            <p style='font-size: 0.8em; opacity: 0.7; margin-top: 15px;'>
                ⚠️ For entertainment and educational purposes • Results may vary • Beauty is subjective and unique to everyone
            </p>
        </div>
        <div style='position: absolute; top: -50%; right: -50%; width: 200%; height: 200%; background: linear-gradient(45deg, transparent, rgba(255,255,255,0.05), transparent); animation: shine 4s infinite;'></div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()beauty1:
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
        
        with col_
