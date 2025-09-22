"""
Features module for technical analysis and pattern recognition.
"""

from .engine import FeatureEngine
from .indicators import TechnicalIndicators
from .market_structure import MarketStructureAnalyzer
from .patterns import ICTPatterns

# For compatibility, create SMCPatterns as alias
SMCPatterns = ICTPatterns

__all__ = ["FeatureEngine", "TechnicalIndicators", "MarketStructureAnalyzer", "ICTPatterns", "SMCPatterns"]