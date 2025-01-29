# prediction_service.py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger('PredictionService')

class PredictionService:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.input_shape = (1, 12, 5)  # Expected input shape
        self._load_resources()
        self._verify_model()

    def _load_resources(self):
        """Load the model and scalers with enhanced error handling"""
        try:
            # Print paths for debugging
            logger.info(f"Looking for model in: {self.model_dir}")
            
            model_path = os.path.join(self.model_dir, 'model.keras')
            scaler_X_path = os.path.join(self.model_dir, 'scaler_X.pkl')
            scaler_y_path = os.path.join(self.model_dir, 'scaler_y.pkl')
            
            # Log file existence
            logger.info(f"Model exists: {os.path.exists(model_path)}")
            logger.info(f"Scaler X exists: {os.path.exists(scaler_X_path)}")
            logger.info(f"Scaler Y exists: {os.path.exists(scaler_y_path)}")

            if not all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_y_path]):
                raise FileNotFoundError(f"Missing files in {self.model_dir}")

            # Load model using direct path
            self.model = tf.keras.models.load_model(model_path)
            
            # Load scalers
            with open(scaler_X_path, 'rb') as f:
                self.scaler_X = pickle.load(f)
            
            with open(scaler_y_path, 'rb') as f:
                self.scaler_y = pickle.load(f)
                
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            raise

    def _verify_model(self):
        """Verify model can make predictions with dummy data"""
        try:
            dummy_input = np.zeros(self.input_shape)
            self.model.predict(dummy_input, verbose=0)
            logger.info("Model verification successful")
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            raise RuntimeError(f"Model verification failed: {str(e)}")

    def _validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters"""
        validators = {
            'base_price': lambda x: isinstance(x, (int, float)) and x > 0,
            'total_price': lambda x: isinstance(x, (int, float)) and x > 0,
            'is_featured_sku': lambda x: isinstance(x, int) and x in (0, 1),
            'is_display_sku': lambda x: isinstance(x, int) and x in (0, 1),
            'sku_id': lambda x: isinstance(x, int) and x > 0
        }
        
        return all(validators[k](v) for k, v in kwargs.items())

    def predict(self, base_price: float, total_price: float, is_featured_sku: int,
                is_display_sku: int, sku_id: int) -> int:
        """Make prediction with input validation"""
        try:
            # Validate inputs
            inputs = locals()
            inputs.pop('self')
            if not self._validate_inputs(**inputs):
                raise ValueError("Invalid input parameters")

            # Prepare and scale input
            new_data = np.array([[base_price, total_price, is_featured_sku, is_display_sku, sku_id]])
            sequence = self._prepare_sequence(new_data)
            
            # Make prediction
            prediction_scaled = self.model.predict(sequence, verbose=0)
            prediction = self.scaler_y.inverse_transform(prediction_scaled)
            
            return max(0, int(round(prediction[0][0])))  # Ensure non-negative prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction error: {str(e)}")

    def _prepare_sequence(self, new_data: np.ndarray) -> np.ndarray:
        """Prepare input sequence with validation"""
        if new_data.shape[1] != 5:
            raise ValueError(f"Expected 5 features, got {new_data.shape[1]}")
            
        dummy_history = np.zeros((11, 5))
        sequence = np.vstack([dummy_history, new_data])
        sequence_scaled = self.scaler_X.transform(sequence)
        return sequence_scaled.reshape(self.input_shape)
