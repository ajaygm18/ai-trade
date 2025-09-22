"""
AI Decision Engine - The brain of the trading system.
"""

import asyncio
import numpy as np
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from core.logger import get_logger, trade_log
from core.event_bus import EventBus, Event, EventType, event_bus
from core.config import config
from .models import TradingModel, SimpleMLPredictor, LSTMPredictor, create_trading_labels

logger = get_logger(__name__)


class TradingAction(Enum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradingSignal:
    """Trading signal structure."""
    
    symbol: str
    action: TradingAction
    confidence: float
    timestamp: datetime
    features: List[float]
    model_predictions: Dict[str, float]
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'action': self.action.name,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'features': self.features,
            'model_predictions': self.model_predictions,
            'reasoning': self.reasoning
        }


class AIDecisionEngine:
    """
    AI Decision Engine - The brain of the trading system.
    
    Responsibilities:
    - Subscribe to feature-ready events
    - Load and manage AI models
    - Generate trading signals based on features
    - Publish proposed trade events
    - Continuous learning and model updating
    """
    
    def __init__(self, event_bus_instance: Optional[EventBus] = None):
        self.event_bus = event_bus_instance or event_bus
        self.models: Dict[str, TradingModel] = {}
        self.feature_history: Dict[str, List[List[float]]] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.signal_history: List[TradingSignal] = []
        
        self._running = False
        self._models_loaded = False
        self._training_data_size = 1000  # Minimum data points for training
        self._retrain_interval = 24 * 60 * 60  # Retrain every 24 hours
        self._last_training_time: Optional[datetime] = None
        
        # Subscribe to events
        self.event_bus.subscribe(EventType.FEATURES_READY, self._handle_features_ready)
        
        logger.info("AIDecisionEngine initialized")
    
    async def start(self) -> bool:
        """Start the AI decision engine."""
        try:
            self._running = True
            
            # Load or initialize models
            await self._initialize_models()
            
            logger.info("AIDecisionEngine started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting AIDecisionEngine: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the AI decision engine."""
        self._running = False
        
        # Save models before stopping
        await self._save_models()
        
        logger.info("AIDecisionEngine stopped")
    
    async def _initialize_models(self) -> None:
        """Initialize AI models."""
        try:
            # Initialize different model types
            self.models['random_forest'] = SimpleMLPredictor("random_forest")
            self.models['gradient_boosting'] = SimpleMLPredictor("gradient_boosting")
            self.models['lstm'] = LSTMPredictor()
            
            # Try to load existing models
            model_dir = config.ai_model.model_path
            for model_name in self.models.keys():
                model_path = f"{model_dir}/{model_name}_model.pkl"
                if self.models[model_name].load(model_path):
                    logger.info(f"Loaded existing {model_name} model")
                else:
                    logger.info(f"No existing {model_name} model found, will train new one")
            
            self._models_loaded = True
            logger.info("Models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def _handle_features_ready(self, event: Event) -> None:
        """Handle features ready events."""
        try:
            if not self._running or not self._models_loaded:
                return
            
            data = event.data
            symbol = data['symbol']
            features = data['features']
            timestamp = datetime.fromisoformat(data['timestamp'])
            
            # Store feature history
            if symbol not in self.feature_history:
                self.feature_history[symbol] = []
            self.feature_history[symbol].append(features)
            
            # Keep only recent history to manage memory
            max_history = 2000
            if len(self.feature_history[symbol]) > max_history:
                self.feature_history[symbol] = self.feature_history[symbol][-max_history:]
            
            # Store price history for training
            if 'raw_data' in data and 'price' in data['raw_data']:
                price = data['raw_data']['price']['close']
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                self.price_history[symbol].append(price)
                
                if len(self.price_history[symbol]) > max_history:
                    self.price_history[symbol] = self.price_history[symbol][-max_history:]
            
            # Generate trading signal
            signal = await self._generate_trading_signal(symbol, features, timestamp, data)
            
            if signal:
                # Store signal history
                self.signal_history.append(signal)
                if len(self.signal_history) > 1000:  # Keep last 1000 signals
                    self.signal_history = self.signal_history[-1000:]
                
                # Publish trading signal
                await self._publish_trading_signal(signal)
            
            # Check if we should retrain models
            await self._check_retrain_models()
            
        except Exception as e:
            logger.error(f"Error handling features ready event: {e}")
    
    async def _generate_trading_signal(
        self, 
        symbol: str, 
        features: List[float], 
        timestamp: datetime,
        raw_data: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Generate a trading signal based on features."""
        try:
            # Convert features to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Get predictions from all models
            model_predictions = {}
            model_probabilities = {}
            
            for model_name, model in self.models.items():
                if hasattr(model, 'is_trained') and model.is_trained:
                    try:
                        # Get prediction
                        prediction = model.predict(features_array)
                        probabilities = model.predict_proba(features_array)
                        
                        model_predictions[model_name] = int(prediction[0])
                        model_probabilities[model_name] = probabilities[0].tolist()
                        
                    except Exception as e:
                        logger.warning(f"Error getting prediction from {model_name}: {e}")
                        continue
            
            if not model_predictions:
                logger.debug(f"No trained models available for {symbol}")
                return None
            
            # Ensemble prediction (voting)
            action, confidence = self._ensemble_prediction(model_predictions, model_probabilities)
            
            # Create reasoning
            reasoning = self._create_reasoning(features, model_predictions, raw_data)
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                timestamp=timestamp,
                features=features,
                model_predictions=model_predictions,
                reasoning=reasoning
            )
            
            logger.debug(f"Generated signal for {symbol}: {action.name} (confidence: {confidence:.3f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal for {symbol}: {e}")
            return None
    
    def _ensemble_prediction(
        self, 
        model_predictions: Dict[str, int], 
        model_probabilities: Dict[str, List[float]]
    ) -> Tuple[TradingAction, float]:
        """Combine predictions from multiple models using ensemble voting."""
        
        # Count votes for each action
        action_votes = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
        
        # Weight models differently (can be made configurable)
        model_weights = {
            'random_forest': 0.3,
            'gradient_boosting': 0.4,
            'lstm': 0.3
        }
        
        total_weight = 0
        weighted_probabilities = np.zeros(3)
        
        for model_name, prediction in model_predictions.items():
            weight = model_weights.get(model_name, 0.1)
            action_votes[prediction] += weight
            total_weight += weight
            
            # Add weighted probabilities
            if model_name in model_probabilities:
                probs = np.array(model_probabilities[model_name])
                if len(probs) == 3:  # Ensure we have probabilities for all 3 classes
                    weighted_probabilities += probs * weight
        
        # Normalize probabilities
        if total_weight > 0:
            weighted_probabilities /= total_weight
        
        # Get the action with highest vote
        best_action = max(action_votes.keys(), key=lambda k: action_votes[k])
        
        # Calculate confidence based on the probability of the chosen action
        confidence = weighted_probabilities[best_action] if len(weighted_probabilities) > best_action else 0.5
        
        # Apply minimum confidence threshold
        min_confidence = 0.6
        if confidence < min_confidence:
            best_action = 0  # Default to HOLD for low confidence
            confidence = 0.5
        
        return TradingAction(best_action), confidence
    
    def _create_reasoning(
        self, 
        features: List[float], 
        model_predictions: Dict[str, int],
        raw_data: Dict[str, Any]
    ) -> str:
        """Create human-readable reasoning for the trading signal."""
        
        reasoning_parts = []
        
        # Market structure reasoning
        if 'raw_data' in raw_data and 'market_structure' in raw_data['raw_data']:
            structure = raw_data['raw_data']['market_structure']
            trend = structure.get('trend', 'unknown')
            reasoning_parts.append(f"Market trend: {trend}")
            
            if structure.get('structure_broken', False):
                reasoning_parts.append("Market structure broken")
        
        # Pattern reasoning
        if 'raw_data' in raw_data and 'ict_patterns' in raw_data['raw_data']:
            patterns = raw_data['raw_data']['ict_patterns']
            
            if patterns.get('order_blocks', {}).get('unmitigated', 0) > 0:
                reasoning_parts.append("Unmitigated order blocks present")
            
            if patterns.get('fair_value_gaps', {}).get('unfilled', 0) > 0:
                reasoning_parts.append("Unfilled fair value gaps detected")
        
        # Model consensus
        if model_predictions:
            unique_predictions = set(model_predictions.values())
            if len(unique_predictions) == 1:
                reasoning_parts.append("Strong model consensus")
            else:
                reasoning_parts.append("Mixed model signals")
        
        # Technical indicator reasoning (simplified)
        if len(features) > 20:  # Ensure we have enough features
            # RSI-like signal (assuming RSI is around index 13 in features)
            if 13 < len(features):
                rsi_value = features[13]
                if rsi_value > 70:
                    reasoning_parts.append("Overbought conditions")
                elif rsi_value < 30:
                    reasoning_parts.append("Oversold conditions")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "Model-based signal"
    
    async def _publish_trading_signal(self, signal: TradingSignal) -> None:
        """Publish a trading signal as a proposed trade event."""
        
        # Log the signal for audit purposes
        trade_log(
            action=f"SIGNAL_{signal.action.name}",
            symbol=signal.symbol,
            quantity=0,  # Will be determined by position sizing
            price=0.0,  # Current market price
            confidence=signal.confidence,
            reasoning=signal.reasoning
        )
        
        # Publish event
        await self.event_bus.publish(Event(
            event_type=EventType.PROPOSED_TRADE,
            data=signal.to_dict(),
            source='AIDecisionEngine'
        ))
        
        logger.info(f"Published {signal.action.name} signal for {signal.symbol} (confidence: {signal.confidence:.3f})")
    
    async def _check_retrain_models(self) -> None:
        """Check if models should be retrained based on new data."""
        try:
            current_time = datetime.now()
            
            # Check if enough time has passed since last training
            if (self._last_training_time and 
                (current_time - self._last_training_time).total_seconds() < self._retrain_interval):
                return
            
            # Check if we have enough data for retraining
            symbols_with_data = []
            for symbol, features in self.feature_history.items():
                if (len(features) >= self._training_data_size and 
                    symbol in self.price_history and 
                    len(self.price_history[symbol]) >= self._training_data_size):
                    symbols_with_data.append(symbol)
            
            if not symbols_with_data:
                logger.debug("Not enough data for model retraining")
                return
            
            logger.info(f"Retraining models with data from {len(symbols_with_data)} symbols")
            await self._retrain_models(symbols_with_data)
            
            self._last_training_time = current_time
            
        except Exception as e:
            logger.error(f"Error checking model retraining: {e}")
    
    async def _retrain_models(self, symbols: List[str]) -> None:
        """Retrain models with accumulated data."""
        try:
            # Combine data from all symbols
            all_features = []
            all_prices = []
            
            for symbol in symbols:
                features = self.feature_history[symbol]
                prices = self.price_history[symbol]
                
                # Ensure same length
                min_length = min(len(features), len(prices))
                all_features.extend(features[:min_length])
                all_prices.extend(prices[:min_length])
            
            if len(all_features) < self._training_data_size:
                logger.warning("Insufficient data for retraining")
                return
            
            # Create training labels
            price_array = np.array(all_prices)
            labels = create_trading_labels(price_array)
            
            # Prepare features
            features_array = np.array(all_features)
            
            # Ensure same length after label creation
            min_length = min(len(features_array), len(labels))
            features_array = features_array[:min_length]
            labels = labels[:min_length]
            
            logger.info(f"Retraining with {len(features_array)} samples")
            
            # Retrain each model
            for model_name, model in self.models.items():
                try:
                    success = model.train(features_array, labels)
                    if success:
                        logger.info(f"Successfully retrained {model_name} model")
                    else:
                        logger.warning(f"Failed to retrain {model_name} model")
                        
                except Exception as e:
                    logger.error(f"Error retraining {model_name} model: {e}")
            
            # Save retrained models
            await self._save_models()
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    async def _save_models(self) -> None:
        """Save all models to disk."""
        try:
            model_dir = config.ai_model.model_path
            os.makedirs(model_dir, exist_ok=True)
            
            for model_name, model in self.models.items():
                model_path = f"{model_dir}/{model_name}_model.pkl"
                if model.save(model_path):
                    logger.debug(f"Saved {model_name} model")
                    
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def get_signal_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[TradingSignal]:
        """Get signal history."""
        signals = self.signal_history
        
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        return signals[-limit:] if signals else []
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'get_model_info'):
                status[model_name] = model.get_model_info()
            else:
                status[model_name] = {'status': 'unknown'}
        
        return status
    
    async def force_retrain(self, symbols: Optional[List[str]] = None) -> bool:
        """Force retrain models immediately."""
        try:
            if symbols is None:
                symbols = list(self.feature_history.keys())
            
            await self._retrain_models(symbols)
            return True
            
        except Exception as e:
            logger.error(f"Error force retraining: {e}")
            return False