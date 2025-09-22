"""
Core module initialization.
"""

from .config import Config, config
from .event_bus import EventBus, Event
from .logger import get_logger

__all__ = ["Config", "config", "EventBus", "Event", "get_logger"]