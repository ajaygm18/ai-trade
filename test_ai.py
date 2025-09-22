#!/usr/bin/env python3
"""
Test script for the AI Decision Engine.
"""

import asyncio
import sys
from pathlib import Path
import os
import numpy as np
from datetime import datetime, timedelta

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
os.chdir(src_path)

from core.logger import get_logger
from core.event_bus import EventBus, Event, EventType
from ai.engine import AIDecisionEngine, TradingAction, TradingSignal
from ai.models import SimpleMLPredictor, create_trading_labels, prepare_features_for_training

logger = get_logger(__name__)


def create_sample_features(n_samples: int = 500) -> tuple:
    """Create sample feature data for testing."""
    np.random.seed(42)
    
    # Create realistic feature vectors (55 features as per feature engine)
    n_features = 55
    
    # Price features (5)
    prices = 100 + np.cumsum(np.random.normal(0, 0.5, n_samples))
    volumes = np.random.lognormal(10, 0.5, n_samples)
    
    features = []
    for i in range(n_samples):
        # Basic price features
        feature_vector = [
            prices[i] * 0.995,  # open
            prices[i] * 1.002,  # high
            prices[i] * 0.998,  # low
            prices[i],          # close
            volumes[i]          # volume
        ]
        
        # Add technical indicator features (30)
        feature_vector.extend([
            prices[i] * (1 + np.random.normal(0, 0.01)),  # SMA 20
            np.random.choice([-1, 0, 1]),  # SMA 20 signal
            prices[i] * (1 + np.random.normal(0, 0.02)),  # SMA 50
            np.random.choice([-1, 0, 1]),  # SMA 50 signal
            prices[i] * (1 + np.random.normal(0, 0.03)),  # SMA 200
            np.random.choice([-1, 0, 1]),  # SMA 200 signal
            prices[i] * (1 + np.random.normal(0, 0.005)), # EMA 12
            np.random.choice([-1, 0, 1]),  # EMA 12 signal
            prices[i] * (1 + np.random.normal(0, 0.01)),  # EMA 26
            np.random.choice([-1, 0, 1]),  # EMA 26 signal
            prices[i] * (1 + np.random.normal(0, 0.02)),  # EMA 50
            np.random.choice([-1, 0, 1]),  # EMA 50 signal
            np.random.uniform(20, 80),     # RSI
            np.random.choice([-1, 0, 1]),  # RSI signal
            # Add more features to reach 55 total
        ] + [np.random.normal(0, 1) for _ in range(35)])
        
        features.append(feature_vector)
    
    return np.array(features), prices


async def test_ai_models():
    """Test AI models functionality."""
    logger.info("Testing AI models...")
    
    # Create sample data
    features, prices = create_sample_features(1000)
    labels = create_trading_labels(prices)
    
    # Ensure same length
    min_length = min(len(features), len(labels))
    features = features[:min_length]
    labels = labels[:min_length]
    
    logger.info(f"Created {len(features)} feature samples with {features.shape[1]} features")
    logger.info(f"Label distribution: HOLD={np.sum(labels==0)}, BUY={np.sum(labels==1)}, SELL={np.sum(labels==2)}")
    
    # Test SimpleMLPredictor
    logger.info("Testing Random Forest model...")
    rf_model = SimpleMLPredictor("random_forest")
    
    # Train model
    success = rf_model.train(features, labels)
    if success:
        logger.info("✅ Random Forest trained successfully")
        
        # Test prediction
        test_features = features[-10:]
        predictions = rf_model.predict(test_features)
        probabilities = rf_model.predict_proba(test_features)
        
        logger.info(f"Test predictions: {predictions}")
        logger.info(f"Test probabilities shape: {probabilities.shape}")
        
        # Test feature importance
        importance = rf_model.get_feature_importance()
        if importance is not None:
            top_features = np.argsort(importance)[-5:]  # Top 5 features
            logger.info(f"Top 5 important features: {top_features}")
    
    # Test Gradient Boosting
    logger.info("Testing Gradient Boosting model...")
    gb_model = SimpleMLPredictor("gradient_boosting")
    
    success = gb_model.train(features, labels)
    if success:
        logger.info("✅ Gradient Boosting trained successfully")
    
    return True


async def test_ai_decision_engine():
    """Test AI Decision Engine functionality."""
    logger.info("Testing AI Decision Engine...")
    
    # Create event bus
    event_bus = EventBus()
    await event_bus.start()
    
    # Create AI Decision Engine
    ai_engine = AIDecisionEngine(event_bus)
    await ai_engine.start()
    
    # Generate sample features data
    features, prices = create_sample_features(100)
    
    # Simulate features ready events
    logger.info("Simulating features ready events...")
    
    for i in range(10):  # Send 10 feature events
        timestamp = datetime.now() - timedelta(days=10-i)
        
        # Create sample feature event
        feature_event = Event(
            event_type=EventType.FEATURES_READY,
            data={
                'symbol': 'TEST',
                'timestamp': timestamp.isoformat(),
                'features': features[i].tolist(),
                'raw_data': {
                    'price': {
                        'open': prices[i] * 0.995,
                        'high': prices[i] * 1.002,
                        'low': prices[i] * 0.998,
                        'close': prices[i],
                        'volume': 100000
                    },
                    'market_structure': {
                        'trend': 'uptrend',
                        'structure_broken': False,
                        'structure_strength': 0.7
                    },
                    'ict_patterns': {
                        'order_blocks': {'unmitigated': 2},
                        'fair_value_gaps': {'unfilled': 1}
                    }
                }
            },
            source='test'
        )
        
        await event_bus.publish(feature_event)
        await asyncio.sleep(0.1)  # Small delay
    
    # Wait for processing
    await asyncio.sleep(1.0)
    
    # Check results
    signal_history = ai_engine.get_signal_history()
    logger.info(f"Generated {len(signal_history)} signals")
    
    if signal_history:
        latest_signal = signal_history[-1]
        logger.info(f"Latest signal: {latest_signal.action.name} for {latest_signal.symbol}")
        logger.info(f"Confidence: {latest_signal.confidence:.3f}")
        logger.info(f"Reasoning: {latest_signal.reasoning}")
    
    # Check model status
    model_status = ai_engine.get_model_status()
    logger.info(f"Model status: {model_status}")
    
    # Test with enough data for training
    logger.info("Testing model training with more data...")
    
    # Add more feature history to trigger training
    for i in range(1000):
        ai_engine.feature_history['TEST'] = features.tolist()
        ai_engine.price_history['TEST'] = prices.tolist()
    
    # Force retrain
    success = await ai_engine.force_retrain(['TEST'])
    if success:
        logger.info("✅ Model retraining successful")
    
    await ai_engine.stop()
    await event_bus.stop()
    
    return True


async def test_trading_signals():
    """Test trading signal creation and handling."""
    logger.info("Testing trading signal functionality...")
    
    # Create a sample signal
    signal = TradingSignal(
        symbol='TEST',
        action=TradingAction.BUY,
        confidence=0.85,
        timestamp=datetime.now(),
        features=[1.0, 2.0, 3.0],
        model_predictions={'rf': 1, 'gb': 1},
        reasoning="Strong bullish signals"
    )
    
    # Test serialization
    signal_dict = signal.to_dict()
    logger.info(f"Signal serialization: {signal_dict}")
    
    # Test different actions
    for action in TradingAction:
        test_signal = TradingSignal(
            symbol='TEST',
            action=action,
            confidence=0.7,
            timestamp=datetime.now(),
            features=[],
            model_predictions={},
            reasoning=f"Test {action.name} signal"
        )
        logger.info(f"Created {action.name} signal")
    
    return True


async def test_all():
    """Run all AI tests."""
    logger.info("🧠 Starting AI Decision Engine tests...")
    
    try:
        # Test AI models
        await test_ai_models()
        
        # Test trading signals
        await test_trading_signals()
        
        # Test AI decision engine
        await test_ai_decision_engine()
        
        logger.info("✅ All AI tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ AI tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_all())
    if result:
        logger.info("✅ AI Decision Engine test PASSED")
    else:
        logger.error("❌ AI Decision Engine test FAILED")