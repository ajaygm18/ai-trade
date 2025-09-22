"""
AI Trading System

A comprehensive AI-powered trading system implementing ICT/SMC concepts
with machine learning for the Indian stock market.
"""

__version__ = "1.0.0"
__author__ = "AI Trading System"
__email__ = "dev@aitrade.com"

from .core.config import Config
from .core.event_bus import EventBus
from .core.logger import get_logger

__all__ = ["Config", "EventBus", "get_logger"]