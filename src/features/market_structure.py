"""
Market structure analysis for ICT/SMC concepts.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from core.logger import get_logger

logger = get_logger(__name__)


class TrendState(Enum):
    """Market trend states."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    CONSOLIDATION = "consolidation"
    UNKNOWN = "unknown"


@dataclass
class SwingPoint:
    """Swing point structure."""
    
    index: int
    timestamp: pd.Timestamp
    price: float
    type: str  # 'high' or 'low'
    strength: float = 0.0  # How significant the swing is
    
    def __repr__(self):
        return f"SwingPoint({self.type}={self.price:.2f} @ {self.timestamp})"


@dataclass
class MarketStructure:
    """Market structure analysis results."""
    
    trend: TrendState
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    higher_highs: List[SwingPoint]
    higher_lows: List[SwingPoint]
    lower_highs: List[SwingPoint]
    lower_lows: List[SwingPoint]
    current_structure_break: Optional[SwingPoint] = None
    structure_strength: float = 0.0


class MarketStructureAnalyzer:
    """
    Market structure analyzer following ICT/SMC principles.
    
    Identifies swing points, market structure, and trend changes.
    """
    
    def __init__(self, swing_lookback: int = 5, min_swing_strength: float = 0.001):
        self.swing_lookback = swing_lookback
        self.min_swing_strength = min_swing_strength
    
    def analyze_structure(self, df: pd.DataFrame) -> MarketStructure:
        """
        Analyze market structure for the given data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            MarketStructure object with analysis results
        """
        logger.debug("Analyzing market structure")
        
        try:
            # Find swing points
            swing_highs = self._find_swing_highs(df)
            swing_lows = self._find_swing_lows(df)
            
            # Classify structure patterns
            higher_highs = self._find_higher_highs(swing_highs)
            higher_lows = self._find_higher_lows(swing_lows)
            lower_highs = self._find_lower_highs(swing_highs)
            lower_lows = self._find_lower_lows(swing_lows)
            
            # Determine overall trend
            trend = self._determine_trend(higher_highs, higher_lows, lower_highs, lower_lows)
            
            # Check for structure breaks
            structure_break = self._find_structure_break(df, swing_highs, swing_lows, trend)
            
            # Calculate structure strength
            structure_strength = self._calculate_structure_strength(
                higher_highs, higher_lows, lower_highs, lower_lows
            )
            
            logger.debug(f"Market structure: {trend.value}, {len(swing_highs)} highs, {len(swing_lows)} lows")
            
            return MarketStructure(
                trend=trend,
                swing_highs=swing_highs,
                swing_lows=swing_lows,
                higher_highs=higher_highs,
                higher_lows=higher_lows,
                lower_highs=lower_highs,
                lower_lows=lower_lows,
                current_structure_break=structure_break,
                structure_strength=structure_strength
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return MarketStructure(
                trend=TrendState.UNKNOWN,
                swing_highs=[],
                swing_lows=[],
                higher_highs=[],
                higher_lows=[],
                lower_highs=[],
                lower_lows=[]
            )
    
    def _find_swing_highs(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing high points."""
        swing_highs = []
        
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            current_high = df.iloc[i]['high']
            
            # Check if current high is higher than surrounding points
            is_swing_high = True
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and df.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                # Calculate strength based on price difference
                left_min = df.iloc[i - self.swing_lookback:i]['high'].min()
                right_min = df.iloc[i + 1:i + self.swing_lookback + 1]['high'].min()
                strength = (current_high - max(left_min, right_min)) / current_high
                
                if strength >= self.min_swing_strength:
                    swing_highs.append(SwingPoint(
                        index=i,
                        timestamp=df.iloc[i].name,
                        price=current_high,
                        type='high',
                        strength=strength
                    ))
        
        return swing_highs
    
    def _find_swing_lows(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing low points."""
        swing_lows = []
        
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            current_low = df.iloc[i]['low']
            
            # Check if current low is lower than surrounding points
            is_swing_low = True
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and df.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                # Calculate strength based on price difference
                left_max = df.iloc[i - self.swing_lookback:i]['low'].max()
                right_max = df.iloc[i + 1:i + self.swing_lookback + 1]['low'].max()
                strength = (min(left_max, right_max) - current_low) / current_low
                
                if strength >= self.min_swing_strength:
                    swing_lows.append(SwingPoint(
                        index=i,
                        timestamp=df.iloc[i].name,
                        price=current_low,
                        type='low',
                        strength=strength
                    ))
        
        return swing_lows
    
    def _find_higher_highs(self, swing_highs: List[SwingPoint]) -> List[SwingPoint]:
        """Find higher highs pattern."""
        if len(swing_highs) < 2:
            return []
        
        higher_highs = []
        for i in range(1, len(swing_highs)):
            if swing_highs[i].price > swing_highs[i-1].price:
                higher_highs.append(swing_highs[i])
        
        return higher_highs
    
    def _find_higher_lows(self, swing_lows: List[SwingPoint]) -> List[SwingPoint]:
        """Find higher lows pattern."""
        if len(swing_lows) < 2:
            return []
        
        higher_lows = []
        for i in range(1, len(swing_lows)):
            if swing_lows[i].price > swing_lows[i-1].price:
                higher_lows.append(swing_lows[i])
        
        return higher_lows
    
    def _find_lower_highs(self, swing_highs: List[SwingPoint]) -> List[SwingPoint]:
        """Find lower highs pattern."""
        if len(swing_highs) < 2:
            return []
        
        lower_highs = []
        for i in range(1, len(swing_highs)):
            if swing_highs[i].price < swing_highs[i-1].price:
                lower_highs.append(swing_highs[i])
        
        return lower_highs
    
    def _find_lower_lows(self, swing_lows: List[SwingPoint]) -> List[SwingPoint]:
        """Find lower lows pattern."""
        if len(swing_lows) < 2:
            return []
        
        lower_lows = []
        for i in range(1, len(swing_lows)):
            if swing_lows[i].price < swing_lows[i-1].price:
                lower_lows.append(swing_lows[i])
        
        return lower_lows
    
    def _determine_trend(
        self, 
        higher_highs: List[SwingPoint], 
        higher_lows: List[SwingPoint],
        lower_highs: List[SwingPoint], 
        lower_lows: List[SwingPoint]
    ) -> TrendState:
        """Determine the overall trend based on structure patterns."""
        
        # Score different patterns
        uptrend_score = len(higher_highs) + len(higher_lows)
        downtrend_score = len(lower_highs) + len(lower_lows)
        
        # Require minimum evidence for trend determination
        min_points = 2
        
        if uptrend_score >= min_points and uptrend_score > downtrend_score:
            return TrendState.UPTREND
        elif downtrend_score >= min_points and downtrend_score > uptrend_score:
            return TrendState.DOWNTREND
        elif uptrend_score == downtrend_score and uptrend_score > 0:
            return TrendState.CONSOLIDATION
        else:
            return TrendState.UNKNOWN
    
    def _find_structure_break(
        self, 
        df: pd.DataFrame, 
        swing_highs: List[SwingPoint], 
        swing_lows: List[SwingPoint],
        trend: TrendState
    ) -> Optional[SwingPoint]:
        """Find the most recent market structure break."""
        
        if len(df) < 10:
            return None
        
        recent_price = df['close'].iloc[-1]
        
        # Look for structure breaks in the last portion of data
        recent_highs = [h for h in swing_highs if h.index >= len(df) - 20]
        recent_lows = [l for l in swing_lows if l.index >= len(df) - 20]
        
        if trend == TrendState.UPTREND:
            # Look for break of recent higher low
            if recent_lows:
                last_higher_low = recent_lows[-1]
                if recent_price < last_higher_low.price:
                    return last_higher_low
        
        elif trend == TrendState.DOWNTREND:
            # Look for break of recent lower high
            if recent_highs:
                last_lower_high = recent_highs[-1]
                if recent_price > last_lower_high.price:
                    return last_lower_high
        
        return None
    
    def _calculate_structure_strength(
        self,
        higher_highs: List[SwingPoint],
        higher_lows: List[SwingPoint],
        lower_highs: List[SwingPoint],
        lower_lows: List[SwingPoint]
    ) -> float:
        """Calculate the strength of the current market structure."""
        
        total_points = len(higher_highs) + len(higher_lows) + len(lower_highs) + len(lower_lows)
        
        if total_points == 0:
            return 0.0
        
        # Calculate consistency score
        uptrend_points = len(higher_highs) + len(higher_lows)
        downtrend_points = len(lower_highs) + len(lower_lows)
        
        max_points = max(uptrend_points, downtrend_points)
        consistency = max_points / total_points
        
        # Factor in the strength of individual swing points
        all_swings = higher_highs + higher_lows + lower_highs + lower_lows
        avg_strength = np.mean([s.strength for s in all_swings]) if all_swings else 0
        
        # Combine consistency and average strength
        structure_strength = (consistency * 0.7) + (avg_strength * 0.3)
        
        return min(structure_strength, 1.0)  # Cap at 1.0
    
    def is_structure_broken(self, df: pd.DataFrame, structure: MarketStructure) -> bool:
        """Check if market structure is broken."""
        return structure.current_structure_break is not None
    
    def get_key_levels(self, structure: MarketStructure) -> Dict[str, float]:
        """Get key support and resistance levels from structure."""
        levels = {}
        
        if structure.swing_highs:
            # Recent resistance levels
            recent_highs = sorted(structure.swing_highs, key=lambda x: x.index)[-3:]
            levels['resistance'] = [h.price for h in recent_highs]
        
        if structure.swing_lows:
            # Recent support levels
            recent_lows = sorted(structure.swing_lows, key=lambda x: x.index)[-3:]
            levels['support'] = [l.price for l in recent_lows]
        
        return levels