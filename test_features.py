#!/usr/bin/env python3
"""
Test script for the features module.
"""

import asyncio
import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
os.chdir(src_path)

from core.logger import get_logger
from features.engine import FeatureEngine
from features.indicators import TechnicalIndicators
from features.market_structure import MarketStructureAnalyzer
from features.patterns import ICTPatterns

logger = get_logger(__name__)


def create_sample_data(days: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Generate realistic price data with trends and patterns
    np.random.seed(42)
    
    # Start with a base price
    base_price = 100.0
    prices = [base_price]
    
    # Generate price movements with some trend
    for i in range(1, days):
        # Add some trend and noise
        trend = 0.001 * np.sin(i / 20)  # Gentle trend
        noise = np.random.normal(0, 0.02)  # 2% daily volatility
        change = trend + noise
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate open, high, low based on close
        open_price = close * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(np.random.lognormal(10, 0.5))  # Log-normal volume distribution
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


async def test_features():
    """Test the features module."""
    logger.info("Starting features module test...")
    
    # Create sample data
    df = create_sample_data(200)
    logger.info(f"Created sample data with {len(df)} data points")
    
    # Test Technical Indicators
    logger.info("Testing technical indicators...")
    tech_indicators = TechnicalIndicators()
    indicators = tech_indicators.calculate_all_indicators(df)
    
    logger.info(f"Calculated {len(indicators)} technical indicators:")
    for name, indicator in indicators.items():
        latest_value = indicator.values[-1] if len(indicator.values) > 0 else None
        latest_signal = indicator.signals[-1] if indicator.signals is not None and len(indicator.signals) > 0 else None
        
        # Handle numpy arrays and scalars
        if latest_value is not None:
            if hasattr(latest_value, '__len__') and len(latest_value) > 1:
                value_str = f"[{', '.join([f'{v:.4f}' for v in latest_value[:3]])}...]"  # Show first 3 elements
            else:
                value_str = f"{float(latest_value):.4f}"
        else:
            value_str = "N/A"
            
        logger.info(f"  {name}: value={value_str}, signal={latest_signal}")
    
    # Test Market Structure Analysis
    logger.info("Testing market structure analysis...")
    structure_analyzer = MarketStructureAnalyzer()
    market_structure = structure_analyzer.analyze_structure(df)
    
    logger.info(f"Market structure analysis:")
    logger.info(f"  Trend: {market_structure.trend.value}")
    logger.info(f"  Swing highs: {len(market_structure.swing_highs)}")
    logger.info(f"  Swing lows: {len(market_structure.swing_lows)}")
    logger.info(f"  Higher highs: {len(market_structure.higher_highs)}")
    logger.info(f"  Higher lows: {len(market_structure.higher_lows)}")
    logger.info(f"  Lower highs: {len(market_structure.lower_highs)}")
    logger.info(f"  Lower lows: {len(market_structure.lower_lows)}")
    logger.info(f"  Structure strength: {market_structure.structure_strength:.4f}")
    logger.info(f"  Structure broken: {market_structure.current_structure_break is not None}")
    
    # Test ICT Patterns
    logger.info("Testing ICT pattern recognition...")
    ict_patterns = ICTPatterns()
    patterns = ict_patterns.find_all_patterns(df)
    
    logger.info(f"ICT patterns found:")
    for pattern_type, pattern_list in patterns.items():
        logger.info(f"  {pattern_type}: {len(pattern_list)} patterns")
        if pattern_list:
            # Show details of first pattern as example
            example = pattern_list[0]
            logger.info(f"    Example: {example}")
    
    # Test Feature Engine
    logger.info("Testing feature engine...")
    feature_engine = FeatureEngine()
    
    # Manually add data to buffer for testing
    feature_engine._data_buffer['TEST'] = df
    
    # Process features
    await feature_engine._process_features('TEST')
    
    # Get features
    features_data = await feature_engine.get_features_for_symbol('TEST')
    if features_data:
        feature_vector = features_data['features']
        logger.info(f"Feature vector length: {len(feature_vector)}")
        logger.info(f"Feature vector (first 10): {feature_vector[:10]}")
        
        # Check serialized data
        raw_data = features_data['raw_data']
        logger.info(f"Raw data keys: {list(raw_data.keys())}")
        
        indicators_summary = raw_data['indicators']
        logger.info(f"Indicators summary: {len(indicators_summary)} indicators")
        
        structure_summary = raw_data['market_structure']
        logger.info(f"Market structure summary: {structure_summary}")
        
        patterns_summary = raw_data['ict_patterns']
        logger.info(f"ICT patterns summary: {patterns_summary}")
    
    logger.info("✅ Features module test completed successfully!")
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_features())
        if result:
            logger.info("✅ Features test PASSED")
        else:
            logger.error("❌ Features test FAILED")
    except Exception as e:
        logger.error(f"❌ Features test failed with error: {e}")
        import traceback
        traceback.print_exc()