"""
Technical indicators implementation using modern TA libraries.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import ta
from dataclasses import dataclass

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IndicatorValues:
    """Container for indicator values."""
    
    name: str
    values: np.ndarray
    signals: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class TechnicalIndicators:
    """
    Technical indicators implementation following the blueprint specifications.
    
    Implements classical technical indicators with modern calculation methods.
    """
    
    def __init__(self):
        self.indicators: Dict[str, IndicatorValues] = {}
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, IndicatorValues]:
        """
        Calculate all technical indicators for the given dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of indicator names to IndicatorValues
        """
        logger.debug("Calculating all technical indicators")
        
        try:
            # Moving Averages
            self.indicators['sma_20'] = self._sma(df, period=20)
            self.indicators['sma_50'] = self._sma(df, period=50)
            self.indicators['sma_200'] = self._sma(df, period=200)
            self.indicators['ema_12'] = self._ema(df, period=12)
            self.indicators['ema_26'] = self._ema(df, period=26)
            self.indicators['ema_50'] = self._ema(df, period=50)
            
            # Oscillators
            self.indicators['rsi'] = self._rsi(df, period=14)
            self.indicators['stoch'] = self._stochastic(df)
            self.indicators['macd'] = self._macd(df)
            
            # Volatility
            self.indicators['bbands'] = self._bollinger_bands(df)
            self.indicators['atr'] = self._atr(df, period=14)
            
            # Volume
            self.indicators['volume_sma'] = self._volume_sma(df, period=20)
            self.indicators['obv'] = self._obv(df)
            
            # Momentum
            self.indicators['momentum'] = self._momentum(df, period=10)
            self.indicators['roc'] = self._rate_of_change(df, period=10)
            
            logger.debug(f"Calculated {len(self.indicators)} technical indicators")
            return self.indicators.copy()
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _sma(self, df: pd.DataFrame, period: int) -> IndicatorValues:
        """Simple Moving Average."""
        values = ta.trend.SMAIndicator(close=df['close'], window=period).sma_indicator()
        
        # Generate signals
        signals = np.where(df['close'] > values, 1, -1)  # 1 = bullish, -1 = bearish
        
        return IndicatorValues(
            name=f'SMA_{period}',
            values=values.values,
            signals=signals,
            metadata={'period': period, 'type': 'trend'}
        )
    
    def _ema(self, df: pd.DataFrame, period: int) -> IndicatorValues:
        """Exponential Moving Average."""
        values = ta.trend.EMAIndicator(close=df['close'], window=period).ema_indicator()
        
        # Generate signals
        signals = np.where(df['close'] > values, 1, -1)
        
        return IndicatorValues(
            name=f'EMA_{period}',
            values=values.values,
            signals=signals,
            metadata={'period': period, 'type': 'trend'}
        )
    
    def _rsi(self, df: pd.DataFrame, period: int = 14) -> IndicatorValues:
        """Relative Strength Index."""
        values = ta.momentum.RSIIndicator(close=df['close'], window=period).rsi()
        
        # Generate signals: overbought > 70, oversold < 30
        signals = np.where(values > 70, -1, np.where(values < 30, 1, 0))
        
        return IndicatorValues(
            name='RSI',
            values=values.values,
            signals=signals,
            metadata={'period': period, 'type': 'oscillator', 'overbought': 70, 'oversold': 30}
        )
    
    def _stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> IndicatorValues:
        """Stochastic Oscillator."""
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'], 
            low=df['low'], 
            close=df['close'],
            window=k_period,
            smooth_window=d_period
        )
        
        k_values = stoch.stoch()
        d_values = stoch.stoch_signal()
        
        # Generate signals: K > D = bullish, K < D = bearish
        signals = np.where(k_values > d_values, 1, -1)
        
        # Combine K and D values
        values = np.column_stack([k_values.values, d_values.values])
        
        return IndicatorValues(
            name='STOCH',
            values=values,
            signals=signals,
            metadata={'k_period': k_period, 'd_period': d_period, 'type': 'oscillator'}
        )
    
    def _macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> IndicatorValues:
        """MACD (Moving Average Convergence Divergence)."""
        macd_indicator = ta.trend.MACD(
            close=df['close'],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal
        )
        
        macd_line = macd_indicator.macd()
        macd_signal = macd_indicator.macd_signal()
        macd_histogram = macd_indicator.macd_diff()
        
        # Generate signals: MACD > Signal = bullish
        signals = np.where(macd_line > macd_signal, 1, -1)
        
        # Combine all MACD components
        values = np.column_stack([
            macd_line.values,
            macd_signal.values,
            macd_histogram.values
        ])
        
        return IndicatorValues(
            name='MACD',
            values=values,
            signals=signals,
            metadata={'fast': fast, 'slow': slow, 'signal': signal, 'type': 'trend'}
        )
    
    def _bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2) -> IndicatorValues:
        """Bollinger Bands."""
        bb = ta.volatility.BollingerBands(
            close=df['close'],
            window=period,
            window_dev=std
        )
        
        upper = bb.bollinger_hband()
        middle = bb.bollinger_mavg()
        lower = bb.bollinger_lband()
        
        # Generate signals: close > upper = overbought, close < lower = oversold
        signals = np.where(
            df['close'] > upper, -1,
            np.where(df['close'] < lower, 1, 0)
        )
        
        # Combine bands
        values = np.column_stack([upper.values, middle.values, lower.values])
        
        return IndicatorValues(
            name='BBANDS',
            values=values,
            signals=signals,
            metadata={'period': period, 'std': std, 'type': 'volatility'}
        )
    
    def _atr(self, df: pd.DataFrame, period: int = 14) -> IndicatorValues:
        """Average True Range."""
        values = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period
        ).average_true_range()
        
        # ATR doesn't generate directional signals, but can indicate volatility
        signals = np.zeros(len(values))
        
        return IndicatorValues(
            name='ATR',
            values=values.values,
            signals=signals,
            metadata={'period': period, 'type': 'volatility'}
        )
    
    def _volume_sma(self, df: pd.DataFrame, period: int = 20) -> IndicatorValues:
        """Volume Simple Moving Average."""
        values = df['volume'].rolling(window=period).mean()
        
        # Generate signals: volume > average = high activity
        signals = np.where(df['volume'] > values, 1, 0)
        
        return IndicatorValues(
            name='VOLUME_SMA',
            values=values.values,
            signals=signals,
            metadata={'period': period, 'type': 'volume'}
        )
    
    def _obv(self, df: pd.DataFrame) -> IndicatorValues:
        """On-Balance Volume."""
        values = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'],
            volume=df['volume']
        ).on_balance_volume()
        
        # Generate signals based on OBV trend
        obv_change = values.diff()
        signals = np.where(obv_change > 0, 1, np.where(obv_change < 0, -1, 0))
        
        return IndicatorValues(
            name='OBV',
            values=values.values,
            signals=signals,
            metadata={'type': 'volume'}
        )
    
    def _momentum(self, df: pd.DataFrame, period: int = 10) -> IndicatorValues:
        """Price Momentum."""
        values = df['close'] / df['close'].shift(period) - 1
        
        # Generate signals: positive momentum = bullish
        signals = np.where(values > 0, 1, -1)
        
        return IndicatorValues(
            name='MOMENTUM',
            values=values.values,
            signals=signals,
            metadata={'period': period, 'type': 'momentum'}
        )
    
    def _rate_of_change(self, df: pd.DataFrame, period: int = 10) -> IndicatorValues:
        """Rate of Change."""
        values = ta.momentum.ROCIndicator(
            close=df['close'],
            window=period
        ).roc()
        
        # Generate signals: positive ROC = bullish
        signals = np.where(values > 0, 1, -1)
        
        return IndicatorValues(
            name='ROC',
            values=values.values,
            signals=signals,
            metadata={'period': period, 'type': 'momentum'}
        )
    
    def get_indicator(self, name: str) -> Optional[IndicatorValues]:
        """Get a specific indicator by name."""
        return self.indicators.get(name)
    
    def get_latest_values(self, name: str, count: int = 1) -> Optional[np.ndarray]:
        """Get the latest N values for an indicator."""
        indicator = self.get_indicator(name)
        if indicator is not None and len(indicator.values) >= count:
            return indicator.values[-count:]
        return None
    
    def get_latest_signal(self, name: str) -> Optional[int]:
        """Get the latest signal for an indicator."""
        indicator = self.get_indicator(name)
        if indicator is not None and indicator.signals is not None and len(indicator.signals) > 0:
            return int(indicator.signals[-1])
        return None