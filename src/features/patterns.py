"""
ICT (Inner Circle Trader) pattern recognition implementation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OrderBlock:
    """Order Block pattern."""
    
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    price_high: float
    price_low: float
    type: str  # 'bullish' or 'bearish'
    strength: float
    mitigated: bool = False
    
    @property
    def price_range(self) -> float:
        return self.price_high - self.price_low
    
    def __repr__(self):
        return f"OrderBlock({self.type}, {self.price_low:.2f}-{self.price_high:.2f})"


@dataclass
class FairValueGap:
    """Fair Value Gap (Imbalance) pattern."""
    
    index: int
    timestamp: datetime
    gap_high: float
    gap_low: float
    type: str  # 'bullish' or 'bearish'
    strength: float
    filled: bool = False
    
    @property
    def gap_size(self) -> float:
        return self.gap_high - self.gap_low
    
    def __repr__(self):
        return f"FVG({self.type}, {self.gap_low:.2f}-{self.gap_high:.2f})"


@dataclass
class LiquidityGrab:
    """Liquidity Grab (Stop Hunt) pattern."""
    
    index: int
    timestamp: datetime
    grab_price: float
    reversal_price: float
    type: str  # 'buy_side' or 'sell_side'
    strength: float
    volume: float
    
    @property
    def grab_distance(self) -> float:
        return abs(self.reversal_price - self.grab_price)
    
    def __repr__(self):
        return f"LiquidityGrab({self.type}, {self.grab_price:.2f})"


class ICTPatterns:
    """
    ICT (Inner Circle Trader) pattern recognition.
    
    Implements the core ICT concepts:
    - Order Blocks
    - Fair Value Gaps (Imbalances)
    - Liquidity Grabs (Stop Hunts)
    """
    
    def __init__(self, min_gap_size: float = 0.001, min_block_size: float = 0.002):
        self.min_gap_size = min_gap_size
        self.min_block_size = min_block_size
    
    def find_all_patterns(self, df: pd.DataFrame) -> Dict[str, List[Any]]:
        """
        Find all ICT patterns in the given data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing all found patterns
        """
        logger.debug("Finding all ICT patterns")
        
        try:
            patterns = {
                'order_blocks': self.find_order_blocks(df),
                'fair_value_gaps': self.find_fair_value_gaps(df),
                'liquidity_grabs': self.find_liquidity_grabs(df)
            }
            
            total_patterns = sum(len(p) for p in patterns.values())
            logger.debug(f"Found {total_patterns} ICT patterns")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error finding ICT patterns: {e}")
            return {'order_blocks': [], 'fair_value_gaps': [], 'liquidity_grabs': []}
    
    def find_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Find Order Block patterns.
        
        Order Blocks are specific candles that represent areas where institutions
        placed large orders, causing significant price movement.
        """
        order_blocks = []
        
        for i in range(2, len(df) - 2):
            current = df.iloc[i]
            prev1 = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            next1 = df.iloc[i+1]
            next2 = df.iloc[i+2]
            
            # Bullish Order Block: Look for strong upward move after consolidation
            if self._is_bullish_order_block(prev2, prev1, current, next1, next2):
                strength = self._calculate_order_block_strength(df, i, 'bullish')
                if strength > 0:
                    order_blocks.append(OrderBlock(
                        start_index=i-1,
                        end_index=i,
                        start_time=prev1.name,
                        end_time=current.name,
                        price_high=max(prev1['high'], current['high']),
                        price_low=min(prev1['low'], current['low']),
                        type='bullish',
                        strength=strength
                    ))
            
            # Bearish Order Block: Look for strong downward move after consolidation
            elif self._is_bearish_order_block(prev2, prev1, current, next1, next2):
                strength = self._calculate_order_block_strength(df, i, 'bearish')
                if strength > 0:
                    order_blocks.append(OrderBlock(
                        start_index=i-1,
                        end_index=i,
                        start_time=prev1.name,
                        end_time=current.name,
                        price_high=max(prev1['high'], current['high']),
                        price_low=min(prev1['low'], current['low']),
                        type='bearish',
                        strength=strength
                    ))
        
        return order_blocks
    
    def find_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Find Fair Value Gap (FVG) patterns.
        
        FVGs occur when there's a gap between the high/low of consecutive candles,
        indicating an imbalance in price discovery.
        """
        fvgs = []
        
        for i in range(1, len(df) - 1):
            prev_candle = df.iloc[i-1]
            current_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG: Previous low > Next high
            if prev_candle['low'] > next_candle['high']:
                gap_size = prev_candle['low'] - next_candle['high']
                if gap_size / current_candle['close'] >= self.min_gap_size:
                    strength = self._calculate_gap_strength(gap_size, current_candle['close'])
                    fvgs.append(FairValueGap(
                        index=i,
                        timestamp=current_candle.name,
                        gap_high=prev_candle['low'],
                        gap_low=next_candle['high'],
                        type='bullish',
                        strength=strength
                    ))
            
            # Bearish FVG: Previous high < Next low
            elif prev_candle['high'] < next_candle['low']:
                gap_size = next_candle['low'] - prev_candle['high']
                if gap_size / current_candle['close'] >= self.min_gap_size:
                    strength = self._calculate_gap_strength(gap_size, current_candle['close'])
                    fvgs.append(FairValueGap(
                        index=i,
                        timestamp=current_candle.name,
                        gap_high=next_candle['low'],
                        gap_low=prev_candle['high'],
                        type='bearish',
                        strength=strength
                    ))
        
        return fvgs
    
    def find_liquidity_grabs(self, df: pd.DataFrame) -> List[LiquidityGrab]:
        """
        Find Liquidity Grab patterns.
        
        Liquidity grabs occur when price briefly moves beyond key levels
        to trigger stops, then quickly reverses.
        """
        liquidity_grabs = []
        window = 10  # Look for reversal within this window
        
        for i in range(window, len(df) - window):
            current = df.iloc[i]
            
            # Look for local extremes
            recent_high = df.iloc[i-window:i+1]['high'].max()
            recent_low = df.iloc[i-window:i+1]['low'].min()
            
            # Buy-side liquidity grab (spike above resistance then reversal down)
            if current['high'] == recent_high:
                reversal_found = False
                for j in range(i+1, min(i+window+1, len(df))):
                    if df.iloc[j]['close'] < current['high'] * 0.995:  # 0.5% reversal
                        strength = self._calculate_liquidity_grab_strength(df, i, j, 'buy_side')
                        if strength > 0:
                            liquidity_grabs.append(LiquidityGrab(
                                index=i,
                                timestamp=current.name,
                                grab_price=current['high'],
                                reversal_price=df.iloc[j]['close'],
                                type='buy_side',
                                strength=strength,
                                volume=current['volume']
                            ))
                            reversal_found = True
                            break
            
            # Sell-side liquidity grab (spike below support then reversal up)
            if current['low'] == recent_low:
                reversal_found = False
                for j in range(i+1, min(i+window+1, len(df))):
                    if df.iloc[j]['close'] > current['low'] * 1.005:  # 0.5% reversal
                        strength = self._calculate_liquidity_grab_strength(df, i, j, 'sell_side')
                        if strength > 0:
                            liquidity_grabs.append(LiquidityGrab(
                                index=i,
                                timestamp=current.name,
                                grab_price=current['low'],
                                reversal_price=df.iloc[j]['close'],
                                type='sell_side',
                                strength=strength,
                                volume=current['volume']
                            ))
                            reversal_found = True
                            break
        
        return liquidity_grabs
    
    def _is_bullish_order_block(self, prev2, prev1, current, next1, next2) -> bool:
        """Check if pattern matches bullish order block criteria."""
        # Look for strong bullish candle with follow-through
        is_strong_bull = (current['close'] > current['open'] and 
                         current['close'] > prev1['high'])
        
        # Check for continuation
        has_continuation = (next1['close'] > current['close'] or 
                          next2['close'] > current['close'])
        
        # Ensure significant move
        move_size = (current['close'] - prev1['close']) / prev1['close']
        is_significant = move_size > self.min_block_size
        
        return is_strong_bull and has_continuation and is_significant
    
    def _is_bearish_order_block(self, prev2, prev1, current, next1, next2) -> bool:
        """Check if pattern matches bearish order block criteria."""
        # Look for strong bearish candle with follow-through
        is_strong_bear = (current['close'] < current['open'] and 
                         current['close'] < prev1['low'])
        
        # Check for continuation
        has_continuation = (next1['close'] < current['close'] or 
                          next2['close'] < current['close'])
        
        # Ensure significant move
        move_size = (prev1['close'] - current['close']) / prev1['close']
        is_significant = move_size > self.min_block_size
        
        return is_strong_bear and has_continuation and is_significant
    
    def _calculate_order_block_strength(self, df: pd.DataFrame, index: int, ob_type: str) -> float:
        """Calculate the strength of an order block."""
        current = df.iloc[index]
        prev = df.iloc[index-1]
        
        # Volume strength
        avg_volume = df['volume'].rolling(20).mean().iloc[index]
        volume_strength = min(current['volume'] / avg_volume, 3.0) / 3.0
        
        # Price move strength
        if ob_type == 'bullish':
            price_move = (current['close'] - prev['low']) / prev['low']
        else:
            price_move = (prev['high'] - current['close']) / prev['high']
        
        move_strength = min(price_move / 0.02, 1.0)  # Normalize to 2% move = 1.0
        
        # Combine factors
        strength = (volume_strength * 0.4) + (move_strength * 0.6)
        return strength
    
    def _calculate_gap_strength(self, gap_size: float, price: float) -> float:
        """Calculate the strength of a fair value gap."""
        gap_percentage = gap_size / price
        # Normalize: 0.5% gap = 0.5 strength, 1% gap = 1.0 strength
        return min(gap_percentage / 0.01, 1.0)
    
    def _calculate_liquidity_grab_strength(
        self, df: pd.DataFrame, grab_index: int, reversal_index: int, grab_type: str
    ) -> float:
        """Calculate the strength of a liquidity grab."""
        grab_candle = df.iloc[grab_index]
        reversal_candle = df.iloc[reversal_index]
        
        # Volume spike strength
        avg_volume = df['volume'].rolling(10).mean().iloc[grab_index]
        volume_strength = min(grab_candle['volume'] / avg_volume, 2.0) / 2.0
        
        # Reversal speed (faster reversal = stronger grab)
        time_to_reversal = reversal_index - grab_index
        speed_strength = max(1.0 - (time_to_reversal / 10.0), 0.1)
        
        # Price reversal magnitude
        if grab_type == 'buy_side':
            reversal_magnitude = (grab_candle['high'] - reversal_candle['close']) / grab_candle['high']
        else:
            reversal_magnitude = (reversal_candle['close'] - grab_candle['low']) / grab_candle['low']
        
        magnitude_strength = min(reversal_magnitude / 0.01, 1.0)  # 1% reversal = 1.0
        
        # Combine factors
        strength = (volume_strength * 0.3) + (speed_strength * 0.3) + (magnitude_strength * 0.4)
        return strength
    
    def check_pattern_mitigation(
        self, patterns: List, current_price: float, pattern_type: str
    ) -> List:
        """Check if any patterns have been mitigated by current price."""
        for pattern in patterns:
            if pattern_type == 'order_block' and not pattern.mitigated:
                if pattern.type == 'bullish' and current_price <= pattern.price_low:
                    pattern.mitigated = True
                elif pattern.type == 'bearish' and current_price >= pattern.price_high:
                    pattern.mitigated = True
            
            elif pattern_type == 'fair_value_gap' and not pattern.filled:
                if (pattern.type == 'bullish' and 
                    pattern.gap_low <= current_price <= pattern.gap_high):
                    pattern.filled = True
                elif (pattern.type == 'bearish' and 
                      pattern.gap_low <= current_price <= pattern.gap_high):
                    pattern.filled = True
        
        return patterns