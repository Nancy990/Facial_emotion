import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import io
import math

def load_model():
    """Load the trained emotion detection model"""
    # Try to load the new OpenCV-optimized model first
    model_files = ['opencv_emotion_model.pkl', 'Facial_emotion.pkl']
    
    for model_file in model_files:
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            st.success(f"✅ Loaded model: {model_file}")
            return model, model_file
        except FileNotFoundError:
            continue
    
    st.error("❌ No model file found. Please ensure you have either 'opencv_emotion_model.pkl' or 'Facial_emotion.pkl' in the directory.")
    return None, None

def extract_face_landmarks(face_roi):
    """Extract basic facial landmarks using contour analysis"""
    # Convert to binary for contour detection
    _, binary = cv2.threshold(face_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Get the largest contour (presumably the face outline)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Approximate key points
        landmarks = {
            'jaw_width': w,
            'face_height': h,
            'forehead_y': y,
            'chin_y': y + h,
            'left_cheek': x,
            'right_cheek': x + w,
            'face_center': (x + w//2, y + h//2)
        }
        
        return landmarks
    
    return None

def analyze_face_shape(face_roi):
    """Analyze face shape based on measurements and ratios"""
    h, w = face_roi.shape
    
    # Calculate key ratios
    face_ratio = h / w  # Height to width ratio
    
    # Analyze face regions for width variations
    upper_third = face_roi[:h//3]
    middle_third = face_roi[h//3:2*h//3]
    lower_third = face_roi[2*h//3:]
    
    # Calculate width at different levels (using edge detection)
    def get_width_at_level(region):
        edges = cv2.Canny(region, 50, 150)
        rows_with_edges = np.sum(edges > 0, axis=1)
        return np.mean(rows_with_edges) if len(rows_with_edges) > 0 else 0
    
    upper_width = get_width_at_level(upper_third)
    middle_width = get_width_at_level(middle_third)
    lower_width = get_width_at_level(lower_third)
    
    # Calculate jawline angle (simplified)
    edges = cv2.Canny(face_roi, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    jaw_angle = 90  # Default
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:5]:  # Check first 5 lines
            angle = theta * 180 / np.pi
            if 45 < angle < 135:  # Filter for relevant angles
                angles.append(angle)
        if angles:
            jaw_angle = np.mean(angles)
    
    # Face shape classification logic
    if face_ratio > 1.3:
        if upper_width > middle_width and middle_width > lower_width:
            shape = "Heart"
            confidence = 0.8
        elif lower_width > middle_width and middle_width > upper_width:
            shape = "Pear"
            confidence = 0.7
        else:
            shape = "Oval"
            confidence = 0.9
    elif face_ratio < 1.1:
        if abs(upper_width - lower_width) < 0.1 * middle_width:
            shape = "Round"
            confidence = 0.8
        else:
            shape = "Square"
            confidence = 0.7
    else:
        if jaw_angle < 80:
            shape = "Diamond"
            confidence = 0.6
        elif abs(upper_width - lower_width) < 0.15 * middle_width:
            shape = "Rectangle"
            confidence = 0.7
        else:
            shape = "Oval"
            confidence = 0.8
    
    return {
        'shape': shape,
        'confidence': confidence,
        'face_ratio': face_ratio,
        'upper_width': upper_width,
        'middle_width': middle_width,
        'lower_width': lower_width,
        'jaw_angle': jaw_angle
    }

def estimate_age(face_roi, face_measurements):
    """Estimate age based on facial features and measurements"""
    h, w = face_roi.shape
    
    # Analyze skin texture and wrinkles
    # More edges and texture variation typically indicate older age
    edges = cv2.Canny(face_roi, 30, 100)
    edge_density = np.sum(edges > 0) / (h * w)
    
    # Analyze contrast and uniformity
    std_intensity = np.std(face_roi)
    mean_intensity = np.mean(face_roi)
    
    # Analyze different face regions
    eye_region = face_roi[:h//3]  # Upper third
    cheek_region = face_roi[h//3:2*h//3]  # Middle third
    
    eye_texture = np.std(eye_region)
    cheek_smoothness = 255 - np.std(cheek_region)  # Inverse of texture
    
    # Age estimation factors
    texture_factor = min(edge_density * 1000, 50)  # 0-50 years from texture
    contrast_factor = min(std_intensity / 2, 30)   # 0-30 years from contrast
    smoothness_factor = max(0, 40 - cheek_smoothness / 5)  # Younger faces are smoother
    
    # Face ratio can indicate age (faces change shape with age)
    ratio_factor = 0
    if 'face_ratio' in face_measurements:
        # Very young or very old faces have different proportions
        if face_measurements['face_ratio'] < 1.2 or face_measurements['face_ratio'] > 1.4:
            ratio_factor = 10
    
    # Calculate estimated age
    estimated_age = texture_factor + contrast_factor + smoothness_factor + ratio_factor
    estimated_age = max(5, min(estimated_age, 80))  # Clamp between 5-80
    
    # Age range and confidence
    if estimated_age < 18:
        age_range = "Child/Teen (5-17)"
        confidence = 0.6
    elif estimated_age < 30:
        age_range = "Young Adult (18-29)"
        confidence = 0.7
    elif estimated_age < 45:
        age_range = "Adult (30-44)"
        confidence = 0.8
    elif estimated_age < 60:
        age_range = "Middle-aged (45-59)"
        confidence = 0.7
    else:
        age_range = "Senior (60+)"
        confidence = 0.6
    
    return {
        'estimated_age': int(estimated_age),
        'age_range': age_range,
        'confidence': confidence,
        'texture_score': texture_factor,
        'contrast_score': contrast_factor,
        'smoothness_score': smoothness_factor
    }

def get_face_measurements(face_roi):
    """Get detailed face measurements and proportions"""
    h, w = face_roi.shape
    
    # Basic measurements
    measurements = {
        'face_width_px': w,
        'face_height_px': h,
        'face_area_px': h * w,
        'aspect_ratio': h / w,
    }
    
    # Proportional measurements (using golden ratio and facial proportions)
    # Standard facial proportions
    measurements['forehead_height'] = h * 0.33  # Top third
    measurements['mid_face_height'] = h * 0.33   # Middle third
    measurements['lower_face_height'] = h * 0.34 # Bottom third
    
    # Width measurements at different levels
    measurements['forehead_width'] = w * 0.9     # Typically narrower
    measurements['cheek_width'] = w              # Widest part
    measurements['jaw_width'] = w * 0.85         # Typically narrower than cheeks
    
    # Eye region analysis
    eye_region = face_roi[:h//3]
    measurements['eye_region_contrast'] = np.std(eye_region)
    
    # Symmetry analysis
    left_half = face_roi[:, :w//2]
    right_half = cv2.flip(face_roi[:, w//2:], 1)  # Flip right half
    
    # Resize right half to match left half if needed
    if right_half.shape[1] != left_half.shape[1]:
        right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
    
    symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
    measurements['symmetry_score'] = max(0, 100 - symmetry_diff)  # Higher is more symmetric
    
    return measurements

def draw_face_analysis(image, face_coords, shape_info, age_info, measurements):
    """Draw comprehensive face analysis on image"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    x, y, w, h = face_coords
    
    # Draw main face rectangle
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Draw face divisions (thirds)
    third_h = h // 3
    cv2.line(image, (x, y + third_h), (x + w, y + third_h), (0, 255, 0), 1)
    cv2.line(image, (x, y + 2 * third_h), (x + w, y + 2 * third_h), (0, 255, 0), 1)
    
    # Draw center line
    cv2.line(image, (x + w//2, y), (x + w//2, y + h), (0, 255, 0), 1)
    
    # Add labels
    cv2.putText(image, f"Shape: {shape_info['shape']}", (x, y-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(image, f"Age: {age_info['age_range']}", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw measurement points
    # Forehead center
    cv2.circle(image, (x + w//2, y + third_h//2), 3, (255, 255, 0), -1)
    # Cheek points
    cv2.circle(image, (x + w//4, y + third_h + third_h//2), 3, (255, 255, 0), -1)
    cv2.circle(image, (x + 3*w//4, y + third_h + third_h//2), 3, (255, 255, 0), -1)
    # Chin point
    cv2.circle(image, (x + w//2, y + h - 10), 3, (255, 255, 0), -1)
    
    return image

def extract_opencv_features(image):
    """Extract comprehensive OpenCV-based features from an image"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Load cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, "No face detected in the image. Please upload an image with a clear face.", None, None
    
    # Use the largest face
    face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = face
    
    # Extract face region
    face_roi = gray[y:y+h, x:x+w]
    
    # Extract comprehensive features
    features = []
    
    # 1. Face geometry features
    img_h, img_w = gray.shape
    features.extend([
        x / img_w,                    # face_x_norm
        y / img_h,                    # face_y_norm
        w / img_w,                    # face_width_norm
        h / img_h,                    # face_height_norm
        w / h,                        # face_aspect_ratio
        (x + w/2) / img_w,           # face_center_x_norm
        (y + h/2) / img_h,           # face_center_y_norm
    ])
    
    # 2. Eyes features
    eyes = eye_cascade.detectMultiScale(face_roi)
    num_eyes = len(eyes)
    features.extend([
        num_eyes,                     # num_eyes_detected
        np.mean([eye[0] for eye in eyes]) / w if num_eyes > 0 else 0,  # avg_eye_x_ratio
        np.mean([eye[1] for eye in eyes]) / h if num_eyes > 0 else 0,  # avg_eye_y_ratio
        np.mean([eye[2] for eye in eyes]) / w if num_eyes > 0 else 0,  # avg_eye_width_ratio
        np.mean([eye[3] for eye in eyes]) / h if num_eyes > 0 else 0,  # avg_eye_height_ratio
    ])
    
    # 3. Smile features
    smiles = smile_cascade.detectMultiScale(face_roi, 1.8, 20)
    num_smiles = len(smiles)
    features.extend([
        num_smiles,                   # num_smiles_detected
        np.mean([smile[0] for smile in smiles]) / w if num_smiles > 0 else 0,  # avg_smile_x_ratio
        np.mean([smile[1] for smile in smiles]) / h if num_smiles > 0 else 0,  # avg_smile_y_ratio
        np.mean([smile[2] for smile in smiles]) / w if num_smiles > 0 else 0,  # avg_smile_width_ratio
        np.mean([smile[3] for smile in smiles]) / h if num_smiles > 0 else 0,  # avg_smile_height_ratio
    ])
    
    # 4. Intensity and contrast features
    features.extend([
        np.mean(face_roi),            # face_mean_intensity
        np.std(face_roi),             # face_std_intensity
        np.min(face_roi),             # face_min_intensity
        np.max(face_roi),             # face_max_intensity
        np.mean(face_roi[:h//3]),     # upper_face_intensity
        np.mean(face_roi[h//3:2*h//3]), # middle_face_intensity
        np.mean(face_roi[2*h//3:]),   # lower_face_intensity
        np.mean(face_roi[:, :w//3]),  # left_face_intensity
        np.mean(face_roi[:, w//3:2*w//3]), # center_face_intensity
        np.mean(face_roi[:, 2*w//3:]),# right_face_intensity
    ])
    
    # 5. Edge and texture features
    edges = cv2.Canny(face_roi, 50, 150)
    features.extend([
        np.sum(edges) / (w * h),      # edge_density
        np.mean(edges),               # avg_edge_intensity
        np.std(edges),                # edge_std
        np.sum(edges[:h//2]) / np.sum(edges) if np.sum(edges) > 0 else 0,  # upper_edge_ratio
        np.sum(edges[h//2:]) / np.sum(edges) if np.sum(edges) > 0 else 0,  # lower_edge_ratio
    ])
    
    # 6. Gradient features
    grad_x = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
    features.extend([
        np.mean(np.abs(grad_x)),      # mean_gradient_x
        np.mean(np.abs(grad_y)),      # mean_gradient_y
        np.std(grad_x),               # std_gradient_x
        np.std(grad_y),               # std_gradient_y
    ])
    
    # 7. Histogram features
    hist = cv2.calcHist([face_roi], [0], None, [16], [0, 256])
    hist_norm = hist.flatten() / np.sum(hist)
    features.extend(hist_norm.tolist())  # 16 histogram bins
    
    # 8. Local features (grid-based)
    grid_h, grid_w = 4, 4
    cell_h, cell_w = max(1, h // grid_h), max(1, w // grid_w)
    
    for i in range(grid_h):
        for j in range(grid_w):
            start_y, end_y = i * cell_h, min((i + 1) * cell_h, h)
            start_x, end_x = j * cell_w, min((j + 1) * cell_w, w)
            cell = face_roi[start_y:end_y, start_x:end_x]
            
            if cell.size > 0:
                features.extend([
                    np.mean(cell),        # cell_mean
                    np.std(cell),         # cell_std
                ])
            else:
                features.extend([0, 0])
    
    return features, None, (x, y, w, h), face_roi

def predict_emotion_enhanced_fallback(image):
    """Enhanced rule-based emotion prediction"""
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
    
    # Enhanced heuristic-based emotion detection
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Detect features
    smiles = smile_cascade.detectMultiScale(face_roi, 1.8, 20)
    eyes = eye_cascade.detectMultiScale(face_roi)
    
    # Calculate additional metrics
    mean_intensity = np.mean(face_roi)
    std_intensity = np.std(face_roi)
    upper_face = np.mean(face_roi[:h//2])
    lower_face = np.mean(face_roi[h//2:])
    
    # Enhanced rules
    smile_score = len(smiles) * 0.3
    eye_score = min(len(eyes), 2) * 0.2
    brightness_score = (mean_intensity - 100) / 100  # Normalized brightness
    contrast_score = std_intensity / 50  # Normalized contrast
    face_balance = abs(upper_face - lower_face) / mean_intensity
    
    # Calculate emotion probabilities
    happy_prob = max(0, smile_score + brightness_score * 0.2 + eye_score)
    stressed_prob = max(0, face_balance * 0.5 + (1 - brightness_score) * 0.3)
    doubt_prob = max(0, contrast_score * 0.3 + (1 - eye_score) * 0.2)
    normal_prob = 1 - max(happy_prob, stressed_prob, doubt_prob)
    
    # Normalize probabilities
    total = happy_prob + stressed_prob + doubt_prob + normal_prob
    if total > 0:
        probabilities = [doubt_prob/total, happy_prob/total, normal_prob/total, stressed_prob/total]
    else:
        probabilities = [0.25, 0.25, 0.25, 0.25]
    
    # Determine prediction
    emotions = ['Doubt', 'Happy', 'Normal', 'Stressed']
    prediction = emotions[np.argmax(probabilities)]
    
    return prediction, probabilities

def main():
    st.title("🎭 Advanced Face Analysis & Emotion Detection")
    st.write("Upload an image for comprehensive facial analysis including emotion, face shape, age estimation, and detailed measurements!")
    
    # Load model
    model, model_file = load_model()
    
    if model is None:
        st.info("💡 **Tip:** For better accuracy, retrain the model specifically for OpenCV features using the training script.")
        st.stop()
    
    # Show model info
    if model_file == 'opencv_emotion_model.pkl':
        st.success("🎯 Using OpenCV-optimized model for better accuracy!")
    else:
        st.warning("⚠️ Using original MediaPipe model. Consider retraining for better accuracy with OpenCV features.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="For best results, use images with clear, front-facing faces"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process the image
        with st.spinner("🔍 Analyzing facial features..."):
            # Extract features
            try:
                result = extract_opencv_features(image)
                
                if result[1] is not None:  # Error case
                    features, error = result[:2]
                    face_coords, face_roi = None, None
                else:  # Success case
                    features, error, face_coords, face_roi = result
            except ValueError as ve:
                st.error(f"Feature extraction error: {str(ve)}")
                features, error, face_coords, face_roi = None, "Feature extraction failed", None, None
            except Exception as e:
                st.error(f"Unexpected error during feature extraction: {str(e)}")
                features, error, face_coords, face_roi = None, "Unexpected error occurred", None, None
            
            if error:
                st.error(error)
                st.info("💡 **Tips for better results:**")
                st.write("- Use images with clear, front-facing faces")
                st.write("- Ensure good lighting")
                st.write("- Avoid sunglasses or face coverings")
                st.write("- Try images where the face takes up a good portion of the frame")
            else:
                # Perform comprehensive face analysis
                face_measurements = get_face_measurements(face_roi)
                shape_analysis = analyze_face_shape(face_roi)
                age_analysis = estimate_age(face_roi, face_measurements)
                
                try:
                    # Make prediction with trained model
                    X = pd.DataFrame([features])
                    prediction = model.predict(X)[0]
                    probabilities = model.predict_proba(X)[0]
                    
                    model_success = True
                    method_used = "Machine Learning Model"
                    
                except Exception as e:
                    st.warning(f"⚠️ ML model failed: {str(e)}. Using enhanced fallback method.")
                    prediction, probabilities = predict_emotion_enhanced_fallback(image)
                    model_success = False
                    method_used = "Enhanced Rule-Based Detection"
                
                with col2:
                    st.subheader("🔬 Analysis Results")
                    
                    # Draw comprehensive face analysis
                    if face_coords is not None:
                        annotated_image = draw_face_analysis(
                            image.copy(), face_coords, shape_analysis, age_analysis, face_measurements
                        )
                        st.image(annotated_image, caption="Face Analysis Overlay", use_container_width=True)
                
                # === EMOTION ANALYSIS SECTION ===
                st.markdown("---")
                st.header("😊 Emotion Analysis")
                
                # Display emotion results
                st.success(f"🎯 **Detected Emotion: {prediction}**")
                
                # Show method used
                if model_success:
                    st.info(f"📊 Method: {method_used} | Confidence: {np.max(probabilities):.2%}")
                else:
                    st.info(f"📊 Method: {method_used}")
                
                # Display all class probabilities
                st.subheader("📈 Emotion Probability Distribution")
                
                # Get class names
                if model_file == 'opencv_emotion_model.pkl':
                    try:
                        class_names = model.classes_.tolist() if hasattr(model, 'classes_') else ['Doubt', 'Happy', 'Normal', 'Stressed']
                    except:
                        class_names = ['Doubt', 'Happy', 'Normal', 'Stressed']
                else:
                    class_names = ['Doubt', 'Happy', 'Normal', 'Stressed']
                
                # Ensure we have the right number of probabilities
                if len(probabilities) != len(class_names):
                    st.warning("⚠️ Probability mismatch. Using generic labels.")
                    class_names = [f'Class_{i}' for i in range(len(probabilities))]
                
                # Create a dataframe for better visualization
                prob_df = pd.DataFrame({
                    'Emotion': class_names,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)
                
                # Display as bar chart
                st.bar_chart(prob_df.set_index('Emotion'))
                
                # === FACE SHAPE ANALYSIS SECTION ===
                st.markdown("---")
                st.header("👤 Face Shape Analysis")
                
                col_shape1, col_shape2 = st.columns(2)
                
                with col_shape1:
                    st.metric(
                        "Face Shape", 
                        shape_analysis['shape'],
                        help="Determined by analyzing face proportions and measurements"
                    )
                    st.metric(
                        "Shape Confidence", 
                        f"{shape_analysis['confidence']:.1%}",
                        help="How confident the algorithm is in the shape classification"
                    )
                    st.metric(
                        "Face Ratio (H:W)", 
                        f"{shape_analysis['face_ratio']:.2f}",
                        help="Height to width ratio - key indicator of face shape"
                    )
                
                with col_shape2:
                    st.metric("Jaw Angle", f"{shape_analysis['jaw_angle']:.1f}°")
                    st.metric("Symmetry Score", f"{face_measurements['symmetry_score']:.1f}/100")
                    st.metric("Face Area", f"{face_measurements['face_area_px']:,} pixels")
                
                # Face shape characteristics
                st.subheader("🔍 Shape Characteristics")
                shape_chars = {
                    "Oval": "Balanced proportions, slightly longer than wide, considered ideal face shape",
                    "Round": "Width and height are similar, soft curved lines, youthful appearance",
                    "Square": "Strong jawline, width and height similar, angular features",
                    "Rectangle": "Longer than wide, strong jawline, high forehead",
                    "Heart": "Wide forehead, narrow chin, prominent cheekbones",
                    "Pear": "Narrow forehead, wide jawline, fuller lower face",
                    "Diamond": "Wide cheekbones, narrow forehead and chin"
                }
                
                if shape_analysis['shape'] in shape_chars:
                    st.info(f"**{shape_analysis['shape']} Face:** {shape_chars[shape_analysis['shape']]}")
                
                # === AGE ESTIMATION SECTION ===
                st.markdown("---")
                st.header("⏰ Age Estimation")
                
                col_age1, col_age2 = st.columns(2)
                
                with col_age1:
                    st.metric(
                        "Estimated Age", 
                        f"{age_analysis['estimated_age']} years",
                        help="Estimated based on facial texture, contrast, and features"
                    )
                    st.metric(
                        "Age Range", 
                        age_analysis['age_range'],
                        help="Broader age category for the estimation"
                    )
                    st.metric(
                        "Age Confidence", 
                        f"{age_analysis['confidence']:.1%}",
                        help="Confidence level in the age estimation"
                    )
                
                with col_age2:
                    st.metric("Skin Texture Score", f"{age_analysis['texture_score']:.1f}")
                    st.metric("Facial Contrast", f"{age_analysis['contrast_score']:.1f}")
                    st.metric("Skin Smoothness", f"{age_analysis['smoothness_score']:.1f}")
                
                # Age estimation disclaimer
                st.warning("⚠️ **Disclaimer:** Age estimation is approximate and based on visible facial characteristics. Actual age may vary significantly.")
                
                # === DETAILED MEASUREMENTS SECTION ===
                st.markdown("---")
                st.header("📏 Detailed Face Measurements")
                
                col_meas1, col_meas2, col_meas3 = st.columns(3)
                
                with col_meas1:
                    st.subheader("📐 Basic Dimensions")
                    st.metric("Face Width", f"{face_measurements['face_width_px']} px")
                    st.metric("Face Height", f"{face_measurements['face_height_px']} px")
                    st.metric("Aspect Ratio", f"{face_measurements['aspect_ratio']:.2f}")
                
                with col_meas2:
                    st.subheader("📊 Proportional Analysis")
                    st.metric("Forehead Height", f"{face_measurements['forehead_height']:.0f} px")
                    st.metric("Mid-face Height", f"{face_measurements['mid_face_height']:.0f} px")
                    st.metric("Lower Face Height", f"{face_measurements['lower_face_height']:.0f} px")
                
                with col_meas3:
                    st.subheader("🎯 Width Analysis")
                    st.metric("Forehead Width", f"{face_measurements['forehead_width']:.0f} px")
                    st.metric("Cheek Width", f"{face_measurements['cheek_width']:.0f} px")
                    st.metric("Jaw Width", f"{face_measurements['jaw_width']:.0f} px")
                
                # Measurement Analysis
                st.subheader("🔬 Measurement Analysis")
                
                # Golden ratio analysis
                golden_ratio = 1.618
                face_golden_ratio = face_measurements['face_height_px'] / face_measurements['face_width_px']
                golden_ratio_deviation = abs(face_golden_ratio - golden_ratio) / golden_ratio * 100
                
                col_analysis1, col_analysis2 = st.columns(2)
                
                with col_analysis1:
                    st.metric(
                        "Golden Ratio Adherence", 
                        f"{100 - golden_ratio_deviation:.1f}%",
                        help="How closely your face matches the golden ratio (1.618)"
                    )
                    
                    # Facial thirds analysis
                    ideal_third = face_measurements['face_height_px'] / 3
                    forehead_dev = abs(face_measurements['forehead_height'] - ideal_third) / ideal_third * 100
                    midface_dev = abs(face_measurements['mid_face_height'] - ideal_third) / ideal_third * 100
                    lowerface_dev = abs(face_measurements['lower_face_height'] - ideal_third) / ideal_third * 100
                    
                    avg_proportion_score = 100 - (forehead_dev + midface_dev + lowerface_dev) / 3
                    st.metric(
                        "Facial Thirds Balance", 
                        f"{max(0, avg_proportion_score):.1f}%",
                        help="How well your face divides into equal thirds"
                    )
                
                with col_analysis2:
                    st.metric(
                        "Eye Region Contrast", 
                        f"{face_measurements['eye_region_contrast']:.1f}",
                        help="Contrast level in the eye region"
                    )
                    st.metric(
                        "Overall Symmetry", 
                        f"{face_measurements['symmetry_score']:.1f}/100",
                        help="How symmetric your left and right face sides are"
                    )
                
                # === COMPREHENSIVE FEATURE ANALYSIS ===
                if features and model_success:
                    with st.expander("🔍 Advanced Feature Analysis"):
                        st.write("**Comprehensive facial features detected and analyzed:**")
                        
                        # Parse key features
                        if len(features) >= 20:
                            col_feat1, col_feat2, col_feat3 = st.columns(3)
                            
                            with col_feat1:
                                st.subheader("👁️ Eyes & Expression")
                                st.metric("Eyes Detected", int(features[7]))
                                st.metric("Smiles Detected", int(features[12]))
                                st.metric("Eye Position Ratio", f"{features[9]:.3f}")
                                st.metric("Smile Position Ratio", f"{features[14]:.3f}")
                            
                            with col_feat2:
                                st.subheader("💡 Lighting & Texture")
                                st.metric("Face Brightness", f"{features[17]:.1f}")
                                st.metric("Face Contrast", f"{features[18]:.1f}")
                                st.metric("Edge Density", f"{features[27]:.4f}")
                                st.metric("Texture Variation", f"{features[30]:.3f}" if len(features) > 30 else "N/A")
                            
                            with col_feat3:
                                st.subheader("📍 Positioning")
                                st.metric("Face X Position", f"{features[0]:.3f}")
                                st.metric("Face Y Position", f"{features[1]:.3f}")
                                st.metric("Face Width Ratio", f"{features[2]:.3f}")
                                st.metric("Face Height Ratio", f"{features[3]:.3f}")
                        
                        st.write(f"**Total ML features analyzed:** {len(features)}")
                        
                        # Feature importance (if available)
                        if hasattr(model, 'feature_importances_'):
                            st.subheader("🎯 Most Important Features")
                            importance_df = pd.DataFrame({
                                'Feature': [f'Feature_{i}' for i in range(len(model.feature_importances_))],
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            st.bar_chart(importance_df.set_index('Feature'))
                
                # === BEAUTY AND ATTRACTIVENESS ANALYSIS ===
                st.markdown("---")
                st.header("✨ Beauty & Attractiveness Analysis")
                
                # Calculate beauty score based on various factors
                beauty_factors = []
                
                # Golden ratio adherence (weight: 25%)
                golden_score = max(0, 100 - golden_ratio_deviation)
                beauty_factors.append(('Golden Ratio', golden_score, 0.25))
                
                # Symmetry (weight: 30%)
                symmetry_score = face_measurements['symmetry_score']
                beauty_factors.append(('Facial Symmetry', symmetry_score, 0.30))
                
                # Proportional balance (weight: 20%)
                proportion_score = max(0, avg_proportion_score)
                beauty_factors.append(('Proportional Balance', proportion_score, 0.20))
                
                # Skin smoothness (weight: 15%)
                smoothness_score = max(0, 100 - age_analysis['texture_score'] * 2)
                beauty_factors.append(('Skin Smoothness', smoothness_score, 0.15))
                
                # Face shape preference (weight: 10%)
                shape_scores = {'Oval': 95, 'Heart': 85, 'Round': 80, 'Square': 75, 'Rectangle': 70, 'Diamond': 85, 'Pear': 70}
                shape_score = shape_scores.get(shape_analysis['shape'], 70)
                beauty_factors.append(('Face Shape', shape_score, 0.10))
                
                # Calculate overall beauty score
                overall_beauty_score = sum(score * weight for _, score, weight in beauty_factors)
                
                col_beauty1, col_beauty2 = st.columns(2)
                
                with col_beauty1:
                    st.metric(
                        "Overall Attractiveness Score", 
                        f"{overall_beauty_score:.1f}/100",
                        help="Composite score based on facial harmony, symmetry, and proportions"
                    )
                    
                    # Beauty score interpretation
                    if overall_beauty_score >= 90:
                        beauty_category = "Exceptionally Attractive"
                        beauty_color = "success"
                    elif overall_beauty_score >= 80:
                        beauty_category = "Very Attractive"
                        beauty_color = "success"
                    elif overall_beauty_score >= 70:
                        beauty_category = "Above Average"
                        beauty_color = "info"
                    elif overall_beauty_score >= 60:
                        beauty_category = "Average"
                        beauty_color = "info"
                    else:
                        beauty_category = "Unique Beauty"
                        beauty_color = "warning"
                    
                    if beauty_color == "success":
                        st.success(f"Beauty Category: {beauty_category}")
                    elif beauty_color == "info":
                        st.info(f"Beauty Category: {beauty_category}")
                    else:
                        st.warning(f"Beauty Category: {beauty_category}")
                
                with col_beauty2:
                    # Display individual beauty factors
                    st.subheader("Beauty Factor Breakdown")
                    for factor_name, score, weight in beauty_factors:
                        st.metric(
                            factor_name, 
                            f"{score:.1f}/100",
                            delta=f"Weight: {weight:.0%}"
                        )
                
                # Beauty tips based on analysis
                st.subheader("💄 Personalized Beauty Tips")
                
                tips = []
                
                if golden_ratio_deviation > 20:
                    tips.append("💡 **Proportion Tip:** Consider hairstyles that balance your face shape")
                
                if face_measurements['symmetry_score'] < 70:
                    tips.append("📸 **Photo Tip:** Try different angles to find your most symmetric side")
                
                if age_analysis['texture_score'] > 25:
                    tips.append("🧴 **Skincare Tip:** Focus on moisturizing and gentle exfoliation for smoother skin texture")
                
                if shape_analysis['shape'] == 'Round':
                    tips.append("💇 **Style Tip:** Angular hairstyles and defined eyebrows can add structure")
                elif shape_analysis['shape'] == 'Square':
                    tips.append("💇 **Style Tip:** Soft, curved hairstyles can balance strong jawlines")
                elif shape_analysis['shape'] == 'Heart':
                    tips.append("💇 **Style Tip:** Hairstyles with volume at the chin level can balance a wider forehead")
                
                if not tips:
                    tips.append("✨ **Great news:** Your facial features show excellent natural harmony!")
                
                for tip in tips:
                    st.info(tip)
                
                # Beauty disclaimer
                st.warning("⚠️ **Important:** Beauty is subjective and cultural. This analysis is based on mathematical ratios and should be taken as fun insights, not definitive judgments.")
    
    # === INFORMATION SECTIONS ===
    # Model information and tips
    with st.expander("ℹ️ About This Advanced Analysis System"):
        st.write("""
        ### 🔬 **Comprehensive Face Analysis Technology**
        
        This system provides multi-dimensional facial analysis using advanced computer vision:
        
        **🎯 Analysis Features:**
        - **Emotion Detection**: AI-powered emotion recognition from facial expressions
        - **Face Shape Classification**: 7 major face shapes with confidence scoring
        - **Age Estimation**: Age range prediction based on facial characteristics
        - **Detailed Measurements**: Precise facial proportions and dimensions
        - **Beauty Analysis**: Mathematical beauty scoring based on golden ratios
        - **Symmetry Assessment**: Left-right facial symmetry evaluation
        
        **📊 Technical Capabilities:**
        - **80+ Facial Features**: Comprehensive feature extraction
        - **Multi-Model Approach**: ML models with intelligent fallback systems
        - **Real-time Processing**: Fast analysis suitable for web applications
        - **Golden Ratio Analysis**: Mathematical beauty proportion assessment
        - **Facial Thirds**: Professional facial proportion analysis
        
        **🎨 Face Shape Classifications:**
        - **Oval**: Balanced, slightly longer than wide (considered ideal)
        - **Round**: Similar width and height, soft features
        - **Square**: Strong jawline, angular features
        - **Rectangle**: Longer face with strong jaw
        - **Heart**: Wide forehead, narrow chin
        - **Pear**: Narrow forehead, wide jaw
        - **Diamond**: Wide cheekbones, narrow forehead and chin
        
        **⏰ Age Estimation Factors:**
        - **Skin Texture**: Smoothness and fine line analysis
        - **Facial Contrast**: Light/dark variation patterns
        - **Feature Positioning**: How facial features change with age
        - **Edge Detection**: Wrinkle and texture pattern recognition
        
        **✨ Beauty Score Components:**
        - **Golden Ratio (25%)**: 1.618 ratio adherence
        - **Symmetry (30%)**: Left-right facial balance
        - **Proportions (20%)**: Facial thirds and feature balance
        - **Skin Quality (15%)**: Smoothness and texture
        - **Face Shape (10%)**: Shape preference scoring
        
        **💡 For Best Results:**
        - Use clear, well-lit photos with front-facing pose
        - Ensure face occupies 30-60% of image frame
        - Avoid heavy shadows, sunglasses, or obstructions
        - Natural lighting works better than artificial lighting
        
        **🔒 Privacy & Ethics:**
        - All processing is done locally/server-side
        - No images are stored permanently
        - Analysis is for entertainment and educational purposes
        - Beauty standards are subjective and culturally variable
        """)
    
    # Training and improvement information
    with st.expander("🎓 Model Training & Improvement"):
        st.write("""
        ### 📚 **Enhancing Analysis Accuracy**
        
        **🔄 Current Model Status:**
        - Emotion detection uses trained ML models with rule-based fallbacks
        - Face shape analysis uses geometric algorithms
        - Age estimation combines multiple facial characteristics
        - Beauty scoring uses mathematical ratios and proportions
        
        **📈 Potential Improvements:**
        
        **For Emotion Detection:**
        - Retrain with more diverse emotion datasets
        - Implement deep learning CNN models
        - Add micro-expression detection
        - Include cultural emotion variations
        
        **For Face Shape Analysis:**
        - Machine learning classification models
        - 3D facial landmark detection
        - More precise contour analysis
        - Professional facial measurement standards
        
        **For Age Estimation:**
        - Deep learning age regression models
        - Larger diverse age datasets
        - Multi-ethnic training data
        - Gender-specific age patterns
        
        **For Beauty Analysis:**
        - Cultural beauty standard variations
        - Time-period beauty trend analysis
        - Individual preference learning
        - More sophisticated feature weighting
        
        **🚀 Advanced Features (Future):**
        - 3D face reconstruction and analysis
        - Real-time video analysis
        - Makeup and styling recommendations
        - Facial exercise suggestions
        - Progress tracking over time
        
        **🛠️ Technical Enhancements:**
        - GPU acceleration for faster processing
        - Mobile app deployment
        - Batch processing capabilities
        - API endpoints for integration
        - Cloud-based model serving
        """)

if __name__ == "__main__":
    main()
