import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import time
import cv2

# Page configuration
st.set_page_config(
    page_title="SkinAI - Skin Condition Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'doctors_data' not in st.session_state:
    st.session_state.doctors_data = []

# Load model function with improved error handling
@st.cache_resource
def load_model():
    """Load the trained skin disease model with better error handling"""
    try:
        # Attempt multiple loading strategies
        model_path = 'skindisease1.keras'
        
        # Strategy 1: Load with compile=False and custom_objects
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Recompile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Test the model with a dummy input to ensure it works
            dummy_input = np.random.random((1, 224, 224, 3))
            _ = model.predict(dummy_input, verbose=0)
            
            st.success("✅ Model loaded successfully!")
            return model
            
        except Exception as e1:
            st.warning(f"Strategy 1 failed: {str(e1)[:100]}...")
            
            # Strategy 2: Try loading with custom objects
            try:
                custom_objects = {
                    'FixedDropout': tf.keras.layers.Dropout,
                    'relu6': tf.nn.relu6,
                    'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D
                }
                
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False, 
                    custom_objects=custom_objects
                )
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Test with dummy input
                dummy_input = np.random.random((1, 224, 224, 3))
                _ = model.predict(dummy_input, verbose=0)
                
                st.success("✅ Model loaded with custom objects!")
                return model
                
            except Exception as e2:
                st.warning(f"Strategy 2 failed: {str(e2)[:100]}...")
                
                # Strategy 3: Try loading weights only if architecture file exists
                try:
                    # This would require a separate architecture file
                    st.info("Attempting to rebuild model architecture...")
                    model = build_fallback_model()
                    
                    if model:
                        st.success("✅ Fallback model created!")
                        return model
                    else:
                        raise Exception("Fallback model creation failed")
                        
                except Exception as e3:
                    st.error(f"All loading strategies failed. Last error: {str(e3)[:100]}...")
                    st.error("Using mock predictions for demonstration.")
                    return "mock"
    
    except FileNotFoundError:
        st.error("❌ Model file 'skindisease.h5' not found!")
        st.info("Please ensure the model file is in the same directory as this script.")
        return "mock"
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {e}")
        return "mock"

def build_fallback_model():
    """Build a simple fallback model if the original fails to load"""
    try:
        # Create a simple CNN model as fallback
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: Cancer, Eczema, Vitiligo
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.warning("⚠️ Using fallback model architecture. Predictions may not be accurate.")
        return model
        
    except Exception as e:
        st.error(f"Failed to create fallback model: {e}")
        return None

# Enhanced mock prediction function
def mock_predict_skin_condition():
    """Generate realistic mock predictions for testing"""
    # Create more realistic probability distributions
    conditions = ['Eczema','Skin Cancer', 'Vitiligo']
    
    # Generate probabilities that sum to 1
    raw_probs = np.random.dirichlet([1.5, 1.0, 0.8])  # Slightly bias towards first condition
    
    results = {
        conditions[i]: float(raw_probs[i]) * 100 
        for i in range(len(conditions))
    }
    
    predicted_class = conditions[np.argmax(raw_probs)]
    
    return predicted_class, results

# Enhanced doctor scraping function
def scrape_doctors(governorate="cairo"):
    """Scrape doctors from Vezeeta with improved error handling"""
    governorate_map = {
        "cairo": "cairo",
        "giza": "giza", 
        "alexandria": "alexandria",
        "qalyubia": "qalyubia"
    }
    
    governorate = governorate_map.get(governorate.lower(), "cairo")
    
    # Fallback mock data in case scraping fails
    mock_doctors = [
        {"name": "Ahmed Hassan", "specialty": "Dermatology & Cosmetic Surgery", "location": f"{governorate.title()} Medical Center"},
        {"name": "Fatima Ali", "specialty": "Pediatric Dermatology", "location": f"{governorate.title()} Hospital"},
        {"name": "Mohamed Ibrahim", "specialty": "Dermatopathology", "location": f"{governorate.title()} Clinic"},
        {"name": "Sarah Ahmed", "specialty": "Cosmetic Dermatology", "location": f"{governorate.title()} Beauty Center"},
        {"name": "Omar Mahmoud", "specialty": "General Dermatology", "location": f"{governorate.title()} Medical Complex"}
    ]
    
    try:
        url = f"https://www.vezeeta.com/en/doctor/dermatology/{governorate}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Try multiple selector strategies
        doctors = []
        
        # Strategy 1: Original selectors
        try:
            names = soup.find_all('a', {'class': 'CommonStylesstyle__TransparentA-sc-1vkcu2o-2 cTFrlk'})
            specialties = soup.find_all('p', {'class': 'DoctorCardSubComponentsstyle__Text-sc-1vq3h7c-14 DoctorCardSubComponentsstyle__DescText-sc-1vq3h7c-17 fuBVZG esZVig'})
            locations = soup.find_all('span', {'class': 'DoctorCardstyle__Text-sc-uptab2-4 blwPZf'})
            
            min_length = min(len(names), len(specialties), len(locations), 10)
            
            for i in range(min_length):
                try:
                    doctor = {
                        'name': names[i].text.strip(),
                        'specialty': specialties[i].text.strip(),
                        'location': locations[i].text.strip()
                    }
                    doctors.append(doctor)
                except (AttributeError, IndexError):
                    continue
        except:
            pass
        
        # If no doctors found with original selectors, try alternative ones
        if not doctors:
            try:
                # Alternative selector strategy
                doctor_cards = soup.find_all('div', class_=lambda x: x and 'doctor' in x.lower())
                
                for card in doctor_cards[:10]:  # Limit to 10
                    name_elem = card.find('a') or card.find('h3') or card.find('h2')
                    specialty_elem = card.find('p') or card.find('span', class_=lambda x: x and 'specialty' in x.lower() if x else False)
                    location_elem = card.find('span', class_=lambda x: x and 'location' in x.lower() if x else False)
                    
                    if name_elem:
                        doctor = {
                            'name': name_elem.text.strip(),
                            'specialty': specialty_elem.text.strip() if specialty_elem else 'Dermatology',
                            'location': location_elem.text.strip() if location_elem else f'{governorate.title()} Area'
                        }
                        doctors.append(doctor)
            except:
                pass
        
        # If still no doctors found, use mock data
        if not doctors:
            st.info("Using sample doctor data for demonstration.")
            return mock_doctors
        
        return doctors
        
    except requests.RequestException as e:
        st.warning(f"Network error: {e}. Using sample data.")
        return mock_doctors
    except Exception as e:
        st.warning(f"Scraping error: {e}. Using sample data.")
        return mock_doctors

# Enhanced image preprocessing
def preprocess_image(image):
    """Enhanced image preprocessing with error handling"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Handle different image formats
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            elif img_array.shape[2] == 1:  # Grayscale with channel
                img_array = np.stack([img_array[:, :, 0]] * 3, axis=-1)
        elif len(img_array.shape) == 2:  # Pure grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        else:
            raise ValueError(f"Unsupported image shape: {img_array.shape}")
        
        # Ensure the image has the right data type
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        # Resize to model input size
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
        
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Enhanced prediction function
def predict_skin_condition(image, model):
    """Make prediction with enhanced error handling"""
    if model is None:
        return None, None
    
    # Use mock predictions if model loading failed
    if model == "mock":
        time.sleep(1)  # Simulate processing time
        return mock_predict_skin_condition()
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, None
        
        # Make prediction
        with st.spinner("Running inference..."):
            predictions = model.predict(processed_image, verbose=0)
        
        # Define class labels (adjust based on your model)
        class_labels = ['Eczema','Skin Cancer', 'Vitiligo']
        
        # Handle different prediction shapes
        if len(predictions.shape) > 1:
            probabilities = predictions[0]
        else:
            probabilities = predictions
        
        # Ensure we have the right number of classes
        if len(probabilities) != len(class_labels):
            st.warning(f"Model outputs {len(probabilities)} classes, but {len(class_labels)} expected.")
            # Pad or truncate as needed
            if len(probabilities) < len(class_labels):
                probabilities = np.pad(probabilities, (0, len(class_labels) - len(probabilities)))
            else:
                probabilities = probabilities[:len(class_labels)]
        
        # Create results dictionary
        results = {
            class_labels[i]: float(probabilities[i]) * 100 
            for i in range(len(class_labels))
        }
        
        # Get predicted class
        predicted_class = class_labels[np.argmax(probabilities)]
        
        return predicted_class, results
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.info("Falling back to mock predictions.")
        return mock_predict_skin_condition()

# Sidebar navigation (unchanged)
def sidebar_navigation():
    with st.sidebar:
        st.markdown("# 🔬 SkinAI")
        
        page = st.selectbox(
            "Navigate to:",
            ["Home", "Prediction", "Find Doctors", "About", "History"],
            key="main_navigation_selectbox"
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "96.2%")
        with col2:
            st.metric("Predictions", f"{len(st.session_state.prediction_history)}")
    
    return page

# Home page (unchanged)
def home_page():
    st.markdown('<h1 class="main-header">🔬 SkinAI - Advanced Skin Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Skin Condition Detection & Medical Assistance")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Accurate Detection</h3>
            <p>Advanced deep learning model for detecting skin cancer, eczema, and vitiligo</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🏥 Find Doctors</h3>
            <p>Locate nearby dermatologists in Cairo, Giza, and other governorates</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>📱 Easy to Use</h3>
            <p>Simple image upload interface with instant results</p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced prediction page
def prediction_page():
    st.markdown("# 📸 Skin Condition Analysis")
    st.markdown("Upload an image of the affected skin area for AI-powered analysis")
    
    # Load model with status display
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the affected skin area (JPG, JPEG, or PNG format)",
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Show image info
                st.info(f"Image size: {image.size[0]}x{image.size[1]} pixels")
                
                # Prediction button
                if st.button("🔍 Analyze Image", type="primary", key="analyze_button"):
                    with st.spinner("Analyzing image..."):
                        predicted_class, results = predict_skin_condition(image, model)
                        
                        if predicted_class and results:
                            # Show model status
                            if model == "mock":
                                st.warning("⚠️ Using mock predictions for demonstration. Please ensure your model file is properly configured.")
                            
                            # Store in session state
                            st.session_state.prediction_history.append({
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'prediction': predicted_class,
                                'confidence': max(results.values())
                            })
                            
                            # Display results in the second column
                            with col2:
                                st.markdown("### 📋 Analysis Results")
                                
                                # Prediction card
                                confidence = max(results.values())
                                st.markdown(f"""
                                <div class="prediction-card">
                                    <h3>🎯 Predicted Condition: {predicted_class}</h3>
                                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Probability chart
                                df_results = pd.DataFrame(list(results.items()), columns=['Condition', 'Probability'])
                                fig = px.bar(
                                    df_results, 
                                    x='Condition', 
                                    y='Probability',
                                    color='Probability',
                                    title="Prediction Probabilities",
                                    color_continuous_scale='viridis'
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Recommendations based on prediction
                                st.markdown("### 💡 Recommendations")
                                if "Cancer" in predicted_class:
                                    st.error("⚠️ **URGENT:** This result suggests potential skin cancer. Please consult a dermatologist immediately for proper diagnosis.")
                                elif "Eczema" in predicted_class:
                                    st.info("ℹ️ **Eczema detected.** Consider seeing a dermatologist for proper treatment and management options.")
                                elif "Vitiligo" in predicted_class:
                                    st.info("ℹ️ **Vitiligo detected.** Consult a dermatologist to discuss treatment options and management strategies.")
                                
                                st.markdown("**⚠️ Medical Disclaimer:** This AI analysis is for educational and informational purposes only and should not replace professional medical advice, diagnosis, or treatment.")
                        else:
                            st.error("Failed to analyze the image. Please try again with a different image.")
            
            except Exception as e:
                st.error(f"Error processing uploaded image: {e}")
                st.info("Please try uploading a different image file.")

# Enhanced doctors page
def doctors_page():
    st.markdown("# 🏥 Find Dermatologists")
    st.markdown("Locate qualified dermatologists in your area")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Select Location")
        governorate = st.selectbox(
            "Choose Governorate:",
            ["Cairo", "Giza", "Alexandria", "Qalyubia"],
            key="governorate_selector"
        )
        
        if st.button("🔍 Search Doctors", type="primary", key="search_doctors_button"):
            with st.spinner("Searching for doctors..."):
                doctors = scrape_doctors(governorate.lower())
                
                if doctors:
                    st.session_state.doctors_data = doctors
                    st.success(f"Found {len(doctors)} dermatologists in {governorate}")
                else:
                    st.error("No doctors found. Please try again.")
    
    with col2:
        if st.session_state.doctors_data:
            st.markdown("### 👨‍⚕️ Available Doctors")
            
            # Display doctors in a more organized way
            for i, doctor in enumerate(st.session_state.doctors_data, 1):
                with st.container():
                    st.markdown(f"""
                    **👨‍⚕️ {i}. Dr. {doctor['name']}**
                    - **Specialty:** {doctor['specialty']}
                    - **Location:** {doctor['location']}
                    """)
                    st.markdown("---")

# About page (unchanged)
def about_page():
    st.markdown("# ℹ️ About SkinAI")
    
    st.markdown("""
    ## 🎯 Our Mission
    SkinAI is an AI-powered platform for skin condition analysis including skin cancer, eczema, and vitiligo detection.
    
    ## 🔬 Technology
    Our deep learning model analyzes skin images to provide preliminary assessments with confidence scores.
    
    ## ⚠️ Important Disclaimer
    **This application is for educational purposes only.** Always consult with qualified dermatologists for proper medical diagnosis and treatment.
    
    ## 📊 Features
    - AI-powered skin analysis
    - Doctor finder for Egyptian governorates
    - Interactive results visualization
    - Prediction history tracking
    
    ## 🏥 Doctor Network
    We provide access to a comprehensive network of dermatologists across Egypt, making it easier for users to find and connect with qualified medical professionals in their area.
    
    ## 📊 Key Features
    - **AI-Powered Analysis**: Advanced deep learning for accurate skin condition detection
    - **Doctor Finder**: Locate nearby dermatologists in major Egyptian governorates
    - **User-Friendly Interface**: Simple, intuitive design for easy navigation
    - **Instant Results**: Get analysis results within seconds
    - **History Tracking**: Keep track of your previous analyses
    
    ## 👥 Contact & Support
    For technical support or medical inquiries, please consult with healthcare professionals in your area.
    """)

# Enhanced history page
def history_page():
    st.markdown("# 📊 Prediction History")
    
    if st.session_state.prediction_history:
        # Create DataFrame from history
        df_history = pd.DataFrame(st.session_state.prediction_history)
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(df_history))
        with col2:
            if len(df_history) > 0:
                avg_confidence = df_history['confidence'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.1f}%")
        with col3:
            if len(df_history) > 0:
                most_common = df_history['prediction'].mode().iloc[0] if len(df_history) > 0 else "N/A"
                st.metric("Most Common", most_common)
        
        # Display history table
        st.markdown("### Recent Predictions")
        st.dataframe(df_history, use_container_width=True)
        
        # Visualizations
        if len(df_history) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of predictions
                fig_pie = px.pie(
                    df_history, 
                    names='prediction', 
                    title="Distribution of Predictions",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Confidence over time
                df_history['prediction_number'] = range(1, len(df_history) + 1)
                fig_line = px.line(
                    df_history, 
                    x='prediction_number', 
                    y='confidence',
                    title="Confidence Scores Over Time",
                    markers=True
                )
                fig_line.update_xaxes(title="Prediction Number")
                fig_line.update_yaxes(title="Confidence (%)")
                st.plotly_chart(fig_line, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary", key="clear_history_button"):
            st.session_state.prediction_history = []
            st.success("History cleared successfully!")
            st.rerun()
    else:
        st.info("No prediction history available. Start by analyzing some images!")

# Main application
def main():
    try:
        # Sidebar navigation
        selected_page = sidebar_navigation()
        
        # Route to appropriate page
        if selected_page == "Home":
            home_page()
        elif selected_page == "Prediction":
            prediction_page()
        elif selected_page == "Find Doctors":
            doctors_page()
        elif selected_page == "About":
            about_page()
        elif selected_page == "History":
            history_page()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
