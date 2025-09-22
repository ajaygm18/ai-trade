"""
Main Feature Engine that coordinates all feature extraction and pattern recognition.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from core.logger import get_logger
from core.event_bus import EventBus, Event, EventType, event_bus
from core.config import config
from .indicators import TechnicalIndicators
from .market_structure import MarketStructureAnalyzer, MarketStructure
from .patterns import ICTPatterns

logger = get_logger(__name__)


class FeatureEngine:
    """
    Main Feature Engine that coordinates all feature extraction.
    
    Responsibilities:
    - Subscribe to market data events
    - Calculate technical indicators
    - Analyze market structure
    - Detect ICT/SMC patterns
    - Publish feature-ready events
    """
    
    def __init__(self, event_bus_instance: Optional[EventBus] = None):
        self.event_bus = event_bus_instance or event_bus
        self.tech_indicators = TechnicalIndicators()
        self.structure_analyzer = MarketStructureAnalyzer()
        self.ict_patterns = ICTPatterns()
        
        self._data_buffer: Dict[str, pd.DataFrame] = {}
        self._running = False
        self._min_data_points = 200  # Minimum data points for reliable analysis
        
        # Subscribe to events
        self.event_bus.subscribe(EventType.NEW_MARKET_DATA, self._handle_market_data)
        self.event_bus.subscribe(EventType.HISTORICAL_DATA_LOADED, self._handle_historical_data)
        
        logger.info("FeatureEngine initialized")
    
    async def start(self) -> bool:
        """Start the feature engine."""
        try:
            self._running = True
            logger.info("FeatureEngine started")
            return True
        except Exception as e:
            logger.error(f"Error starting FeatureEngine: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the feature engine."""
        self._running = False
        logger.info("FeatureEngine stopped")
    
    async def _handle_market_data(self, event: Event) -> None:
        """Handle new market data events."""
        try:
            if not self._running:
                return
            
            data = event.data
            symbol = data['symbol']
            
            # Convert to DataFrame row
            new_row = pd.DataFrame([{
                'timestamp': pd.to_datetime(data['timestamp']),
                'open': data['open'],
                'high': data['high'], 
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume']
            }])
            new_row.set_index('timestamp', inplace=True)
            
            # Update data buffer
            if symbol not in self._data_buffer:
                self._data_buffer[symbol] = new_row
            else:
                self._data_buffer[symbol] = pd.concat([self._data_buffer[symbol], new_row])
                # Keep only last N points to manage memory
                max_points = 1000
                if len(self._data_buffer[symbol]) > max_points:
                    self._data_buffer[symbol] = self._data_buffer[symbol].tail(max_points)
            
            # Process features if we have enough data
            if len(self._data_buffer[symbol]) >= self._min_data_points:
                await self._process_features(symbol)
            
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
    
    async def _handle_historical_data(self, event: Event) -> None:
        """Handle historical data loaded events."""
        try:
            logger.info("Processing historical data for feature extraction")
            # Process features for all symbols that have sufficient data
            for symbol in self._data_buffer:
                if len(self._data_buffer[symbol]) >= self._min_data_points:
                    await self._process_features(symbol)
            
        except Exception as e:
            logger.error(f"Error handling historical data: {e}")
    
    async def _process_features(self, symbol: str) -> None:
        """Process all features for a symbol."""
        try:
            df = self._data_buffer[symbol].copy()
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Missing required columns for {symbol}")
                return
            
            logger.debug(f"Processing features for {symbol} with {len(df)} data points")
            
            # Calculate technical indicators
            indicators = self.tech_indicators.calculate_all_indicators(df)
            
            # Analyze market structure
            market_structure = self.structure_analyzer.analyze_structure(df)
            
            # Find ICT patterns
            ict_patterns = self.ict_patterns.find_all_patterns(df)
            
            # Create feature vector
            feature_vector = self._create_feature_vector(
                df.iloc[-1], indicators, market_structure, ict_patterns
            )
            
            # Publish features ready event
            await self.event_bus.publish(Event(
                event_type=EventType.FEATURES_READY,
                data={
                    'symbol': symbol,
                    'timestamp': df.index[-1].isoformat(),
                    'features': feature_vector,
                    'raw_data': {
                        'price': df.iloc[-1].to_dict(),
                        'indicators': self._serialize_indicators(indicators),
                        'market_structure': self._serialize_market_structure(market_structure),
                        'ict_patterns': self._serialize_ict_patterns(ict_patterns)
                    }
                },
                source='FeatureEngine'
            ))
            
            logger.debug(f"Published features for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing features for {symbol}: {e}")
    
    def _create_feature_vector(
        self, 
        latest_data: pd.Series,
        indicators: Dict[str, Any],
        market_structure: MarketStructure,
        ict_patterns: Dict[str, List[Any]]
    ) -> List[float]:
        """Create a numerical feature vector from all analysis results."""
        
        features = []
        
        # Price features
        features.extend([
            latest_data['open'],
            latest_data['high'],
            latest_data['low'],
            latest_data['close'],
            float(latest_data['volume'])
        ])
        
        # Technical indicator features
        features.extend(self._extract_indicator_features(indicators))
        
        # Market structure features
        features.extend(self._extract_structure_features(market_structure))
        
        # ICT pattern features
        features.extend(self._extract_pattern_features(ict_patterns))
        
        # Replace any NaN values with 0
        features = [0.0 if pd.isna(x) else float(x) for x in features]
        
        return features
    
    def _extract_indicator_features(self, indicators: Dict[str, Any]) -> List[float]:
        """Extract numerical features from technical indicators."""
        features = []
        
        # Moving average features
        for ma_name in ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'ema_50']:
            if ma_name in indicators:
                indicator = indicators[ma_name]
                latest_value = indicator.values[-1] if len(indicator.values) > 0 else 0.0
                latest_signal = indicator.signals[-1] if indicator.signals is not None and len(indicator.signals) > 0 else 0.0
                features.append(float(latest_value))
                features.append(float(latest_signal))
            else:
                features.extend([0.0, 0.0])
        
        # Oscillator features
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            latest_value = rsi.values[-1] if len(rsi.values) > 0 else 50.0
            latest_signal = rsi.signals[-1] if rsi.signals is not None and len(rsi.signals) > 0 else 0.0
            features.append(float(latest_value))
            features.append(float(latest_signal))
        else:
            features.extend([50.0, 0.0])
        
        if 'stoch' in indicators:
            stoch = indicators['stoch']
            if len(stoch.values) > 0 and stoch.values.ndim > 1:
                features.extend([stoch.values[-1, 0], stoch.values[-1, 1]])  # K and D values
            else:
                features.extend([50.0, 50.0])
            features.append(float(stoch.signals[-1]) if stoch.signals is not None and len(stoch.signals) > 0 else 0.0)
        else:
            features.extend([50.0, 50.0, 0.0])
        
        if 'macd' in indicators:
            macd = indicators['macd']
            if len(macd.values) > 0 and macd.values.ndim > 1:
                features.extend([macd.values[-1, 0], macd.values[-1, 1], macd.values[-1, 2]])  # MACD, Signal, Histogram
            else:
                features.extend([0.0, 0.0, 0.0])
            features.append(float(macd.signals[-1]) if macd.signals is not None and len(macd.signals) > 0 else 0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Volatility features
        if 'bbands' in indicators:
            bbands = indicators['bbands']
            if len(bbands.values) > 0 and bbands.values.ndim > 1:
                features.extend([bbands.values[-1, 0], bbands.values[-1, 1], bbands.values[-1, 2]])  # Upper, Middle, Lower
            else:
                features.extend([0.0, 0.0, 0.0])
            features.append(float(bbands.signals[-1]) if bbands.signals is not None and len(bbands.signals) > 0 else 0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        if 'atr' in indicators:
            atr = indicators['atr']
            features.append(atr.values[-1] if len(atr.values) > 0 else 0.0)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_structure_features(self, structure: MarketStructure) -> List[float]:
        """Extract numerical features from market structure."""
        features = []
        
        # Trend state (one-hot encoding)
        trend_states = ['uptrend', 'downtrend', 'consolidation', 'unknown']
        for state in trend_states:
            features.append(1.0 if structure.trend.value == state else 0.0)
        
        # Structure counts
        features.extend([
            float(len(structure.swing_highs)),
            float(len(structure.swing_lows)),
            float(len(structure.higher_highs)),
            float(len(structure.higher_lows)),
            float(len(structure.lower_highs)),
            float(len(structure.lower_lows))
        ])
        
        # Structure strength
        features.append(structure.structure_strength)
        
        # Structure break indicator
        features.append(1.0 if structure.current_structure_break is not None else 0.0)
        
        return features
    
    def _extract_pattern_features(self, patterns: Dict[str, List[Any]]) -> List[float]:
        """Extract numerical features from ICT patterns."""
        features = []
        
        # Order block features
        order_blocks = patterns.get('order_blocks', [])
        recent_obs = [ob for ob in order_blocks if not ob.mitigated][-5:]  # Last 5 unmitigated
        
        # Count of recent order blocks by type
        bullish_obs = len([ob for ob in recent_obs if ob.type == 'bullish'])
        bearish_obs = len([ob for ob in recent_obs if ob.type == 'bearish'])
        features.extend([float(bullish_obs), float(bearish_obs)])
        
        # Average strength of recent order blocks
        if recent_obs:
            avg_ob_strength = np.mean([ob.strength for ob in recent_obs])
        else:
            avg_ob_strength = 0.0
        features.append(avg_ob_strength)
        
        # Fair value gap features
        fvgs = patterns.get('fair_value_gaps', [])
        recent_fvgs = [fvg for fvg in fvgs if not fvg.filled][-5:]  # Last 5 unfilled
        
        # Count of recent FVGs by type
        bullish_fvgs = len([fvg for fvg in recent_fvgs if fvg.type == 'bullish'])
        bearish_fvgs = len([fvg for fvg in recent_fvgs if fvg.type == 'bearish'])
        features.extend([float(bullish_fvgs), float(bearish_fvgs)])
        
        # Average strength of recent FVGs
        if recent_fvgs:
            avg_fvg_strength = np.mean([fvg.strength for fvg in recent_fvgs])
        else:
            avg_fvg_strength = 0.0
        features.append(avg_fvg_strength)
        
        # Liquidity grab features
        liquidity_grabs = patterns.get('liquidity_grabs', [])
        recent_grabs = liquidity_grabs[-5:]  # Last 5
        
        # Count of recent liquidity grabs by type
        buy_side_grabs = len([lg for lg in recent_grabs if lg.type == 'buy_side'])
        sell_side_grabs = len([lg for lg in recent_grabs if lg.type == 'sell_side'])
        features.extend([float(buy_side_grabs), float(sell_side_grabs)])
        
        # Average strength of recent liquidity grabs
        if recent_grabs:
            avg_grab_strength = np.mean([lg.strength for lg in recent_grabs])
        else:
            avg_grab_strength = 0.0
        features.append(avg_grab_strength)
        
        return features
    
    def _serialize_indicators(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize indicators for JSON transmission."""
        serialized = {}
        for name, indicator in indicators.items():
            serialized[name] = {
                'name': indicator.name,
                'latest_value': float(indicator.values[-1]) if len(indicator.values) > 0 else None,
                'latest_signal': int(indicator.signals[-1]) if indicator.signals is not None and len(indicator.signals) > 0 else None,
                'metadata': indicator.metadata
            }
        return serialized
    
    def _serialize_market_structure(self, structure: MarketStructure) -> Dict[str, Any]:
        """Serialize market structure for JSON transmission."""
        return {
            'trend': structure.trend.value,
            'swing_highs_count': len(structure.swing_highs),
            'swing_lows_count': len(structure.swing_lows),
            'higher_highs_count': len(structure.higher_highs),
            'higher_lows_count': len(structure.higher_lows),
            'lower_highs_count': len(structure.lower_highs),
            'lower_lows_count': len(structure.lower_lows),
            'structure_strength': structure.structure_strength,
            'structure_broken': structure.current_structure_break is not None
        }
    
    def _serialize_ict_patterns(self, patterns: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Serialize ICT patterns for JSON transmission."""
        return {
            'order_blocks': {
                'total': len(patterns.get('order_blocks', [])),
                'bullish': len([ob for ob in patterns.get('order_blocks', []) if ob.type == 'bullish']),
                'bearish': len([ob for ob in patterns.get('order_blocks', []) if ob.type == 'bearish']),
                'unmitigated': len([ob for ob in patterns.get('order_blocks', []) if not ob.mitigated])
            },
            'fair_value_gaps': {
                'total': len(patterns.get('fair_value_gaps', [])),
                'bullish': len([fvg for fvg in patterns.get('fair_value_gaps', []) if fvg.type == 'bullish']),
                'bearish': len([fvg for fvg in patterns.get('fair_value_gaps', []) if fvg.type == 'bearish']),
                'unfilled': len([fvg for fvg in patterns.get('fair_value_gaps', []) if not fvg.filled])
            },
            'liquidity_grabs': {
                'total': len(patterns.get('liquidity_grabs', [])),
                'buy_side': len([lg for lg in patterns.get('liquidity_grabs', []) if lg.type == 'buy_side']),
                'sell_side': len([lg for lg in patterns.get('liquidity_grabs', []) if lg.type == 'sell_side'])
            }
        }
    
    def get_feature_vector_length(self) -> int:
        """Get the expected length of the feature vector."""
        # This should match the length calculated in _create_feature_vector
        # Price features: 5
        # Indicator features: varies, but approximately 30
        # Structure features: 12
        # Pattern features: 8
        return 55  # Approximate, should be calculated exactly
    
    async def get_features_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current features for a specific symbol."""
        if symbol not in self._data_buffer:
            return None
        
        if len(self._data_buffer[symbol]) < self._min_data_points:
            return None
        
        try:
            df = self._data_buffer[symbol].copy()
            
            # Calculate all features
            indicators = self.tech_indicators.calculate_all_indicators(df)
            market_structure = self.structure_analyzer.analyze_structure(df)
            ict_patterns = self.ict_patterns.find_all_patterns(df)
            
            feature_vector = self._create_feature_vector(
                df.iloc[-1], indicators, market_structure, ict_patterns
            )
            
            return {
                'symbol': symbol,
                'timestamp': df.index[-1].isoformat(),
                'features': feature_vector,
                'raw_data': {
                    'price': df.iloc[-1].to_dict(),
                    'indicators': self._serialize_indicators(indicators),
                    'market_structure': self._serialize_market_structure(market_structure),
                    'ict_patterns': self._serialize_ict_patterns(ict_patterns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}")
            return None