import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import logging
import os
from typing import Tuple, Dict, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinConditionModel:
    """
    Skin condition prediction model wrapper
    """
    
    def __init__(self, model_path: Optional[str] = "skindisease.h5"):
        self.model_path = model_path
        self.model = None
        self.class_names = ['Skin Cancer', 'Eczema', 'Vitiligo']
        self.input_shape = (224, 224, 3)
        self.loaded = False
        
        # Condition information for user guidance
        self.condition_info = {
            'Skin Cancer': {
                'description': 'Abnormal growth of skin cells, often caused by UV exposure',
                'urgency': 'HIGH',
                'color': 'red',
                'recommendations': [
                    'Consult a dermatologist immediately',
                    'Avoid further sun exposure',
                    'Document changes in the affected area',
                    'Schedule regular skin checks'
                ]
            },
            'Eczema': {
                'description': 'Inflammatory skin condition causing itchy, red patches',
                'urgency': 'MEDIUM',
                'color': 'orange',
                'recommendations': [
                    'See a dermatologist for proper diagnosis',
                    'Use gentle, fragrance-free moisturizers',
                    'Avoid known triggers (stress, certain foods)',
                    'Consider topical treatments'
                ]
            },
            'Vitiligo': {
                'description': 'Condition causing loss of skin color in patches',
                'urgency': 'MEDIUM',
                'color': 'blue',
                'recommendations': [
                    'Consult a dermatologist for treatment options',
                    'Use sunscreen to protect affected areas',
                    'Consider phototherapy treatments',
                    'Explore cosmetic options if needed'
                ]
            }
        }
    
    def load_model(self) -> bool:
        """
        Load the trained skin disease model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
                self.loaded = True
                logger.info("Skin disease model loaded successfully")
                return True
            else:
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _create_dummy_model(self):
        """
        Create a dummy model for demonstration purposes
        Replace this with your actual trained model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.warning("Using dummy model - replace with your trained model")
        return model
    
    def preprocess_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Preprocess image for model prediction
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            np.ndarray: Preprocessed image array or None if error
        """
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Handle different image modes
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Resize to model input size
            img_resized = cv2.resize(img_array, self.input_shape[:2])
            
            # Normalize pixel values to [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            return img_batch
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image: Image.Image) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        """
        Make prediction on the uploaded image
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            Tuple[str, Dict]: Predicted class and probability scores
        """
        if not self.loaded:
            if not self.load_model():
                return None, None
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None, None
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            probabilities = predictions[0]
            
            # Create results dictionary
            results = {
                self.class_names[i]: float(probabilities[i]) * 100 
                for i in range(len(self.class_names))
            }
            
            # Get predicted class
            predicted_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_idx]
            
            return predicted_class, results
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None, None
    
    def get_condition_info(self, condition: str) -> Dict:
        """
        Get detailed information about a condition
        
        Args:
            condition (str): Name of the condition
            
        Returns:
            Dict: Condition information
        """
        return self.condition_info.get(condition, {})
    
    def validate_image(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Validate uploaded image
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            # Check image size
            width, height = image.size
            if width < 50 or height < 50:
                return False, "Image too small. Please upload an image at least 50x50 pixels."
            
            if width > 4000 or height > 4000:
                return False, "Image too large. Please upload an image smaller than 4000x4000 pixels."
            
            # Check file size (approximate)
            img_array = np.array(image)
            size_mb = img_array.nbytes / (1024 * 1024)
            if size_mb > 10:
                return False, "Image file too large. Please upload an image smaller than 10MB."
            
            # Check if image has valid channels
            if len(img_array.shape) not in [2, 3]:
                return False, "Invalid image format. Please upload a valid RGB or grayscale image."
            
            return True, "Image is valid"
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
    
    def get_model_info(self) -> Dict:
        """
        Get model information and statistics
        
        Returns:
            Dict: Model information
        """
        return {
            'classes': self.class_names,
            'input_shape': self.input_shape,
            'loaded': self.loaded,
            'model_path': self.model_path,
            'architecture': 'Convolutional Neural Network',
            'accuracy': '94.2%',  # Replace with actual accuracy
            'training_samples': '15,000+',  # Replace with actual number
            'version': '1.0.0'
        }


# Utility functions for image processing
def enhance_image_quality(image: Image.Image) -> Image.Image:
    """
    Enhance image quality for better predictions
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Enhanced image
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(img_array.shape) == 3:
            # For color images, convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_array)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return Image.fromarray(enhanced)
        
    except Exception as e:
        logger.warning(f"Could not enhance image: {e}")
        return image


def create_confidence_visualization(results: Dict[str, float]) -> Dict:
    """
    Create data for confidence visualization
    
    Args:
        results (Dict): Prediction results
        
    Returns:
        Dict: Visualization data
    """
    colors = {
        'Skin Cancer': '#ff4444',
        'Eczema': '#ff8800',
        'Vitiligo': '#4488ff'
    }
    
    viz_data = {
        'labels': list(results.keys()),
        'values': list(results.values()),
        'colors': [colors.get(label, '#888888') for label in results.keys()]
    }
    
    return viz_data


# Example usage and testing
if __name__ == "__main__":
    # Test the model utilities
    print("🔬 Testing Skin Condition Model Utilities")
    print("=" * 50)
    
    # Initialize model
    model = SkinConditionModel()
    
    # Test model loading
    if model.load_model():
        print("✅ Model loaded successfully")
        
        # Display model info
        info = model.get_model_info()
        print(f"📊 Model Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    else:
        print("❌ Failed to load model")
    
    # Test condition info
    print(f"\n📋 Condition Information:")
    for condition in model.class_names:
        info = model.get_condition_info(condition)
        print(f"\n{condition}:")
        print(f"   Description: {info.get('description', 'N/A')}")
        print(f"   Urgency: {info.get('urgency', 'N/A')}")
