"""
Event-driven architecture implementation for the trading system.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from loguru import logger


class EventType(Enum):
    """Event types for the trading system."""
    
    # Data events
    NEW_MARKET_DATA = "new_market_data"
    HISTORICAL_DATA_LOADED = "historical_data_loaded"
    
    # Feature events
    FEATURES_READY = "features_ready"
    PATTERN_DETECTED = "pattern_detected"
    
    # AI events
    PROPOSED_TRADE = "proposed_trade"
    MODEL_PREDICTION = "model_prediction"
    
    # Risk events
    VALIDATED_ORDER = "validated_order"
    RISK_VIOLATION = "risk_violation"
    
    # Execution events
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    TRADE_EXECUTED = "trade_executed"
    
    # Portfolio events
    POSITION_UPDATED = "position_updated"
    PNL_UPDATED = "pnl_updated"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ERROR_OCCURRED = "error_occurred"
    HEALTH_CHECK = "health_check"


@dataclass
class Event:
    """Base event class."""
    
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = None
    event_id: str = None
    source: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        event_type = EventType(data['event_type'])
        timestamp = datetime.fromisoformat(data['timestamp'])
        return cls(
            event_type=event_type,
            data=data['data'],
            timestamp=timestamp,
            event_id=data.get('event_id'),
            source=data.get('source')
        )


class EventBus:
    """Event bus for handling system-wide events."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._running = False
        self._event_queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler {handler.__name__} to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                logger.info(f"Unsubscribed handler {handler.__name__} from {event_type.value}")
            except ValueError:
                logger.warning(f"Handler {handler.__name__} not found for {event_type.value}")
    
    async def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        if not self._running:
            logger.warning("Event bus not running, event will be queued")
        
        await self._event_queue.put(event)
        logger.debug(f"Published event {event.event_type.value} with ID {event.event_id}")
    
    def publish_sync(self, event: Event) -> None:
        """Synchronously publish an event."""
        try:
            asyncio.create_task(self.publish(event))
        except RuntimeError:
            # No event loop running, handle synchronously
            self._handle_event_sync(event)
    
    def _handle_event_sync(self, event: Event) -> None:
        """Handle event synchronously."""
        self._event_history.append(event)
        
        if event.event_type in self._subscribers:
            for handler in self._subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        logger.warning(f"Async handler {handler.__name__} called synchronously")
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler {handler.__name__}: {e}")
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._handle_event_async(event)
                self._event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event_async(self, event: Event) -> None:
        """Handle event asynchronously."""
        self._event_history.append(event)
        
        if event.event_type in self._subscribers:
            for handler in self._subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler {handler.__name__}: {e}")
    
    async def start(self) -> None:
        """Start the event bus."""
        if self._running:
            logger.warning("Event bus already running")
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop the event bus."""
        if not self._running:
            return
        
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining events
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                await self._handle_event_async(event)
                self._event_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        logger.info("Event bus stopped")
    
    def get_event_history(self, event_type: Optional[EventType] = None, 
                         limit: Optional[int] = None) -> List[Event]:
        """Get event history."""
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        logger.info("Event history cleared")


# Global event bus instance
event_bus = EventBus()