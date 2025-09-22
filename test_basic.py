#!/usr/bin/env python3
"""
Simple test script to verify basic functionality.
"""

import asyncio
import sys
from pathlib import Path
import os

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Change to src directory for relative imports
os.chdir(src_path)

from core.config import config
from core.logger import get_logger
from core.event_bus import EventBus, Event, EventType
from data.providers import YFinanceProvider
from data.storage import InMemoryStorage

logger = get_logger(__name__)


async def test_basic_functionality():
    """Test basic system functionality."""
    logger.info("Starting basic functionality test...")
    
    # Test configuration
    logger.info(f"Config loaded: {config.data.symbols[:3]}...")
    
    # Test event bus
    event_bus = EventBus()
    await event_bus.start()
    
    events_received = []
    
    def event_handler(event):
        events_received.append(event)
        logger.info(f"Received event: {event.event_type.value}")
    
    event_bus.subscribe(EventType.NEW_MARKET_DATA, event_handler)
    
    # Test event publishing
    test_event = Event(
        event_type=EventType.NEW_MARKET_DATA,
        data={"symbol": "TEST", "price": 100.0},
        source="test"
    )
    
    await event_bus.publish(test_event)
    await asyncio.sleep(0.1)  # Let event process
    
    # Test data provider
    provider = YFinanceProvider()
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    logger.info("Testing data provider...")
    data = await provider.get_historical_data("AAPL", start_date, end_date)
    
    if data:
        logger.info(f"Retrieved {len(data)} data points for AAPL")
        logger.info(f"Latest data: {data[-1].close}")
    else:
        logger.warning("No data retrieved from provider")
    
    # Test storage
    storage = InMemoryStorage()
    if data:
        await storage.store_market_data(data)
        stored_data = await storage.get_latest_data("AAPL", limit=1)
        if stored_data:
            logger.info(f"Storage test successful: {stored_data[0].close}")
    
    await event_bus.stop()
    
    logger.info(f"Test completed. Events received: {len(events_received)}")
    return len(events_received) > 0 and len(data) > 0


if __name__ == "__main__":
    try:
        result = asyncio.run(test_basic_functionality())
        if result:
            logger.info("✅ Basic functionality test PASSED")
        else:
            logger.error("❌ Basic functionality test FAILED")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()