"""
AI models for trading prediction and decision making.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime

from core.logger import get_logger
from core.config import config

logger = get_logger(__name__)


class TradingModel(ABC):
    """Abstract base class for trading models."""
    
    @abstractmethod
    def train(self, features: np.ndarray, targets: np.ndarray) -> bool:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> bool:
        """Save the model."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> bool:
        """Load the model."""
        pass


class SimpleMLPredictor(TradingModel):
    """
    Simple machine learning predictor using ensemble methods.
    
    Uses Random Forest and Gradient Boosting for trading signal prediction.
    This serves as a baseline model before implementing LSTM/Transformer models.
    """
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, features: np.ndarray, targets: np.ndarray) -> bool:
        """
        Train the model on features and targets.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target labels (n_samples,) - 0: HOLD, 1: BUY, 2: SELL
            
        Returns:
            True if training successful
        """
        try:
            logger.info(f"Training {self.model_type} model with {len(features)} samples")
            
            # Validate inputs
            if len(features) != len(targets):
                raise ValueError("Features and targets must have same length")
            
            if len(features) < 100:
                logger.warning("Very few samples for training. Consider gathering more data.")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42, stratify=targets
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained successfully. Test accuracy: {accuracy:.4f}")
            
            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            features_scaled = self.scaler.transform(features)
            predictions = self.model.predict(features_scaled)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            features_scaled = self.scaler.transform(features)
            probabilities = self.model.predict_proba(features_scaled)
            return probabilities
            
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {e}")
            return np.array([])
    
    def save(self, filepath: str) -> bool:
        """Save the model and scaler."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'feature_importance': self.feature_importance,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """Load the model and scaler."""
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            self.feature_importance = model_data.get('feature_importance')
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        return self.feature_importance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'n_features': self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None,
            'has_feature_importance': self.feature_importance is not None
        }


class LSTMPredictor(TradingModel):
    """
    LSTM-based predictor for time series forecasting.
    
    This is a placeholder for future LSTM implementation using TensorFlow/PyTorch.
    For now, it falls back to the SimpleMLPredictor.
    """
    
    def __init__(self, sequence_length: int = 60, hidden_size: int = 50):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.fallback_model = SimpleMLPredictor("gradient_boosting")
        
        logger.warning("LSTM model not yet implemented. Using SimpleMLPredictor as fallback.")
    
    def train(self, features: np.ndarray, targets: np.ndarray) -> bool:
        """Train using fallback model."""
        return self.fallback_model.train(features, targets)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict using fallback model."""
        return self.fallback_model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get probabilities using fallback model."""
        return self.fallback_model.predict_proba(features)
    
    def save(self, filepath: str) -> bool:
        """Save using fallback model."""
        return self.fallback_model.save(filepath)
    
    def load(self, filepath: str) -> bool:
        """Load using fallback model."""
        return self.fallback_model.load(filepath)


def create_trading_labels(
    prices: np.ndarray, 
    returns_threshold: float = 0.01,
    lookforward_periods: int = 5
) -> np.ndarray:
    """
    Create trading labels from price data.
    
    Args:
        prices: Array of prices
        returns_threshold: Minimum return to consider significant
        lookforward_periods: How many periods to look forward
        
    Returns:
        Array of labels: 0=HOLD, 1=BUY, 2=SELL
    """
    labels = np.zeros(len(prices))
    
    for i in range(len(prices) - lookforward_periods):
        current_price = prices[i]
        future_prices = prices[i+1:i+1+lookforward_periods]
        
        # Calculate max future return and min future return
        max_future_return = (np.max(future_prices) - current_price) / current_price
        min_future_return = (np.min(future_prices) - current_price) / current_price
        
        # Label based on significant moves
        if max_future_return > returns_threshold:
            labels[i] = 1  # BUY signal
        elif abs(min_future_return) > returns_threshold and min_future_return < 0:
            labels[i] = 2  # SELL signal
        else:
            labels[i] = 0  # HOLD signal
    
    return labels.astype(int)


def prepare_features_for_training(feature_vectors: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature vectors for training by handling missing values and scaling.
    
    Args:
        feature_vectors: List of feature vectors
        
    Returns:
        Tuple of (features_array, valid_indices)
    """
    # Convert to numpy array
    features = np.array(feature_vectors)
    
    # Handle missing values
    # Replace NaN with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Find rows with all valid data
    valid_mask = ~np.isnan(features).any(axis=1)
    valid_indices = np.where(valid_mask)[0]
    
    logger.info(f"Prepared {len(features)} feature vectors, {len(valid_indices)} valid")
    
    return features[valid_indices], valid_indices