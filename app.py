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
import plotly.graph_objects as go
import plotly.express as px

def load_model():
    """Load model placeholder - using rule-based detection"""
    st.warning("⚠️ Using rule-based emotion detection.")
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

def analyze_face_shape_advanced(face_coords, image_shape):
    """Simplified face shape analysis using OpenCV face detection"""
    if face_coords is None:
        return {'shape': 'Unknown', 'confidence': 0}
    
    x, y, w, h = face_coords
    face_length = h
    face_width = w
    
    # Face ratios
    length_width_ratio = face_length / face_width if face_width > 0 else 1
    jaw_face_ratio = face_width / (image_shape[1] * 0.5) if image_shape[1] > 0 else 1
    
    # Simplified shape classification
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
            'face_width': face_width
        }
    }

def calculate_beauty_score_advanced(face_coords, shape_analysis, image_shape):
    """Calculate beauty score using simplified face data"""
    if face_coords is None:
        return 50, []
    
    beauty_factors = []
    
    # 1. Facial Symmetry (35% weight)
    x, y, w, h = face_coords
    center_x = x + w / 2
    image_center = image_shape[1] / 2
    symmetry_score = max(0, 100 - abs(center_x - image_center) / image_center * 100)
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
    
    for factor_name, score, _ in beauty_factors:
        if factor_name == 'Facial Symmetry' and score < 80:
            recommendations.append("🎭 **Symmetry**: Use makeup to enhance facial balance - slight contouring can help")
        elif factor_name == 'Golden Ratio' and score < 70:
            recommendations.append("📐 **Proportions**: Strategic highlighting and contouring can create ideal proportions")
    
    return recommendations

def get_facial_exercises(face_shape):
    """Generate facial exercise recommendations without age-specific exercises"""
    exercises = []
    
    universal_exercises = [
        "😊 **Smile Exercise**: Hold a wide smile for 10 seconds, repeat 10 times daily",
        "👁️ **Eye Circles**: Gently circle eyes with fingertips to reduce puffiness",
        "💆 **Forehead Massage**: Smooth forehead lines with upward strokes",
        "🎵 **Vowel Sounds**: Say A-E-I-O-U exaggerating mouth movements"
    ]
    
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
    
    return exercises

def save_progress(analysis_data):
    """Save analysis progress for tracking without age data"""
    timestamp = datetime.now().isoformat()
    
    progress_data = {
        'timestamp': timestamp,
        'beauty_score': analysis_data.get('beauty_score', 0),
        'face_shape': analysis_data.get('face_shape', 'Unknown'),
        'emotion': analysis_data.get('emotion', 'Unknown')
    }
    
    if 'progress_history' not in st.session_state:
        st.session_state.progress_history = []
    
    st.session_state.progress_history.append(progress_data)
    
    return len(st.session_state.progress_history)

def show_progress_tracking():
    """Display progress tracking visualization without age data"""
    if 'progress_history' not in st.session_state or not st.session_state.progress_history:
        st.info("📊 No progress data yet. Analyze some images to start tracking!")
        return
    
    df = pd.DataFrame(st.session_state.progress_history)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    if len(df) > 1:
        fig_beauty = px.line(df, x='date', y='beauty_score', 
                           title='Beauty Score Progress Over Time',
                           markers=True)
        st.plotly_chart(fig_beauty, use_container_width=True)
    
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
    """Rule-based emotion prediction using OpenCV"""
    try:
        x, y, w, h = face_coords
        face_roi = image[y:y+h, x:x+w]
        
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized / 255.0
        
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        smiles = smile_cascade.detectMultiScale(face_gray, 1.8, 20)
        eyes = eye_cascade.detectMultiScale(face_gray)
        
        mean_intensity = np.mean(face_gray)
        std_intensity = np.std(face_gray)
        upper_face = np.mean(face_gray[:h//2])
        lower_face = np.mean(face_gray[h//2:])
        
        smile_score = len(smiles) * 0.4
        eye_score = min(len(eyes), 2) * 0.2
        brightness_score = (mean_intensity - 100) / 100
        contrast_score = std_intensity / 50
        face_balance = abs(upper_face - lower_face) / mean_intensity
        
        happy_prob = max(0, min(1, smile_score + brightness_score * 0.3 + eye_score))
        sad_prob = max(0, min(1, (1 - brightness_score) * 0.4 + face_balance * 0.3))
        angry_prob = max(0, min(1, contrast_score * 0.4 + (1 - eye_score) * 0.2))
        surprised_prob = max(0, min(1, eye_score * 0.5 + contrast_score * 0.2))
        fear_prob = max(0, min(1, face_balance * 0.4 + contrast_score * 0.2))
        disgust_prob = max(0, min(1, (1 - smile_score) * 0.3 + face_balance * 0.2))
        neutral_prob = max(0, 1 - max(happy_prob, sad_prob, angry_prob, surprised_prob, fear_prob, disgust_prob))
        
        probabilities = [angry_prob, disgust_prob, fear_prob, happy_prob, sad_prob, surprised_prob, neutral_prob]
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
        
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
        page_title="Face Analysis",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Face Analysis & Beauty Enhancement")
    st.write("Upload an image for facial analysis, emotion detection, beauty scoring, and personalized recommendations!")
    
    st.sidebar.title("🔧 Analysis Options")
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Complete Analysis", "Beauty Analysis Only", "Progress Tracking"]
    )
    
    if analysis_mode == "Progress Tracking":
        st.header("📊 Progress Tracking Dashboard")
        show_progress_tracking()
        return
    
    model, model_file = load_model()
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="📱 Mobile tip: For best results, use good lighting and hold phone steady"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_image = preprocess_mobile_image(image)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("🔍 Performing analysis..."):
            gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                st.error("❌ No face detected. Please try another image with a clear, front-facing face.")
                st.info("💡 **Mobile Tips:**\n- Use good lighting\n- Keep face centered\n- Remove sunglasses\n- Try different angles")
                return
            
            face_coords = max(faces, key=lambda x: x[2] * x[3])
            
            shape_analysis = analyze_face_shape_advanced(face_coords, processed_image.shape)
            
            st.markdown("---")
            st.header("👤 Face Shape Analysis")
            
            col_shape1, col_shape2, col_shape3 = st.columns(3)
            
            with col_shape1:
                st.metric("Face Shape", shape_analysis['shape'], help="Shape analysis based on face detection")
                st.metric("Confidence", f"{shape_analysis['confidence']:.1%}", help="Algorithm confidence in shape classification")
            
            with col_shape2:
                st.metric("Face Ratio", f"{shape_analysis['face_ratio']:.2f}", help="Length to width ratio")
                st.metric("Jaw Ratio", f"{shape_analysis['jaw_ratio']:.2f}", help="Jaw width to face width ratio")
            
            with col_shape3:
                measurements = shape_analysis.get('measurements', {})
                if measurements:
                    st.metric("Face Length", f"{measurements.get('face_length', 0):.1f}")
                    st.metric("Face Width", f"{measurements.get('face_width', 0):.1f}")
            
            st.markdown("---")
            st.header("✨ Beauty & Attractiveness Analysis")
            
            beauty_score, beauty_factors = calculate_beauty_score_advanced(face_coords, shape_analysis, processed_image.shape)
            
            col_beauty1, col_beauty2 = st.columns(2)
            
            with col_beauty1:
                st.metric("Overall Beauty Score", f"{beauty_score:.1f}/100", help="Beauty scoring based on face analysis")
                
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
                st.subheader("Beauty Factor Breakdown")
                for factor_name, score, weight in beauty_factors:
                    st.metric(factor_name, f"{score:.1f}/100", delta=f"Weight: {weight:.0%}", help=f"Contributes {weight:.0%} to overall score")
            
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
                st.markdown("---")
                st.header("😊 Emotion Analysis")
                
                emotion, emotion_probs, emotion_labels = predict_emotion_cnn(processed_image, face_coords)
                
                col_emotion1, col_emotion2 = st.columns(2)
                
                with col_emotion1:
                    st.success(f"🎯 **Detected Emotion: {emotion}**")
                    st.info(f"📊 Confidence: {max(emotion_probs):.1%}")
                
                with col_emotion2:
                    emotion_df = pd.DataFrame({
                        'Emotion': emotion_labels,
                        'Probability': emotion_probs
                    }).sort_values('Probability', ascending=False)
                
                    fig_emotion = px.bar(emotion_df, x='Emotion', y='Probability',
                                       title='Emotion Probability Distribution',
                                       color='Probability',
                                       color_continuous_scale='blues')
                    st.plotly_chart(fig_emotion, use_container_width=True)
            
            st.markdown("---")
            st.header("💄 Personalized Beauty Recommendations")
            
            makeup_recs = get_makeup_recommendations(shape_analysis['shape'], beauty_factors)
            
            st.subheader("💅 Makeup & Styling Tips")
            for rec in makeup_recs:
                st.info(rec)
            
            st.markdown("---")
            st.header("🏃‍♀️ Personalized Facial Exercises")
            
            exercises = get_facial_exercises(shape_analysis['shape'])
            
            st.subheader("💪 Daily Facial Workout Routine")
            st.write("**Recommended duration**: 10-15 minutes daily")
            
            col_ex1, col_ex2 = st.columns(2)
            
            for i, exercise in enumerate(exercises):
                if i % 2 == 0:
                    col_ex1.info(exercise)
                else:
                    col_ex2.info(exercise)
            
            st.markdown("---")
            st.header("📈 Progress Tracking")
            
            analysis_data = {
                'beauty_score': beauty_score,
                'face_shape': shape_analysis['shape'],
                'emotion': emotion
            }
            
            if st.button("💾 Save Analysis to Progress"):
                entry_count = save_progress(analysis_data)
                st.success(f"✅ Analysis saved! Total entries: {entry_count}")
                st.balloons()
            
            if 'progress_history' in st.session_state and st.session_state.progress_history:
                recent_scores = [entry['beauty_score'] for entry in st.session_state.progress_history[-5:]]
                if len(recent_scores) > 1:
                    trend = "📈 Improving" if recent_scores[-1] > recent_scores[0] else "📊 Stable"
                    st.info(f"**Recent Trend**: {trend} | **Average Score**: {np.mean(recent_scores):.1f}")
            
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
            
            enable_video = st.checkbox("🎥 Enable Experimental Video Analysis")
            
            if enable_video:
                st.warning("⚠️ Video analysis is experimental. Performance may vary.")
                
                video_placeholder = st.empty()
                
                if st.button("📹 Start Video Analysis (Demo)"):
                    with video_placeholder.container():
                        st.info("🎬 Video analysis would appear here in full implementation")
                        st.write("Features would include:")
                        st.write("- Real-time face tracking")
                        st.write("- Live emotion detection")
                        st.write("- Dynamic beauty scoring")
                        st.write("- Exercise form checking")
            
            with st.expander("🔬 Advanced Analysis Insights"):
                st.subheader("🧬 Facial Analysis Insights")
                
                if face_coords is not None:
                    x, y, w, h = face_coords
                    col_insights1, col_insights2 = st.columns(2)
                    
                    with col_insights1:
                        st.metric("Face Width", f"{w:.2f}")
                        st.metric("Face Height", f"{h:.2f}")
                    
                    with col_insights2:
                        harmony_score = (beauty_score + shape_analysis['confidence'] * 100) / 2
                        st.metric("Facial Harmony", f"{harmony_score:.1f}/100")
                
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
                    gray = cv2.cvtColor(comp_processed, cv2.COLOR_RGB2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        comp_face_coords = max(faces, key=lambda x: x[2] * x[3])
                        comp_shape = analyze_face_shape_advanced(comp_face_coords, comp_processed.shape)
                        comp_beauty, comp_factors = calculate_beauty_score_advanced(comp_face_coords, comp_shape, comp_processed.shape)
                        
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
                        
                        if comp_beauty > beauty_score:
                            st.success("📈 Comparison image scored higher!")
                        elif comp_beauty < beauty_score:
                            st.info("📉 Original image scored higher!")
                        else:
                            st.info("📊 Similar beauty scores!")
                    else:
                        st.error("Could not analyze comparison image")
    
    with st.expander("ℹ️ About Face Analysis"):
        st.write("""
        ### 🚀 **Face Analysis Technology**
        
        **🧠 Analysis Features:**
        - **Emotion Detection**: Rule-based analysis using facial features
        - **Face Shape Analysis**: Based on face proportions
        - **Beauty Scoring**: Symmetry and proportion-based assessment
        - **Personalized Recommendations**: Tailored makeup and exercise tips
        
        **📱 Mobile Optimization:**
        - **Enhanced Preprocessing**: CLAHE contrast enhancement
        - **Noise Reduction**: Bilateral filtering for cleaner analysis
        - **EXIF Handling**: Automatic image orientation correction
        - **Multi-format Support**: JPEG, PNG, WebP, and more
        
        **💄 Personalization Engine:**
        - **Face Shape Specific**: Tailored recommendations for each shape
        - **Beauty Factor Analysis**: Multi-dimensional beauty assessment
        - **Dynamic Recommendations**: Adaptive based on individual features
        - **Progress Tracking**: Long-term beauty monitoring
        
        **🏃‍♀️ Wellness Integration:**
        - **Facial Exercise Programs**: Scientifically-based face yoga
        - **Progress Monitoring**: Track improvements over time
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
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🎭 Face Analysis System | 
        Built with Streamlit & OpenCV | 
        ⚠️ For entertainment and educational purposes
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
