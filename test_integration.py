#!/usr/bin/env python3
"""
Integration test for the complete AI Trading System.
"""

import asyncio
import sys
from pathlib import Path
import os
from datetime import datetime

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
os.chdir(src_path)

from main import TradingSystem
from core.logger import get_logger
from core.event_bus import Event, EventType

logger = get_logger(__name__)


async def test_integration():
    """Test the complete system integration."""
    logger.info("🔄 Starting integration test...")
    
    # Create trading system
    trading_system = TradingSystem()
    
    try:
        # Start the system
        logger.info("Starting trading system...")
        if not await trading_system.start():
            logger.error("Failed to start trading system")
            return False
        
        # Let the system run for a short time to process data
        logger.info("Letting system process data for 30 seconds...")
        
        # Monitor system status
        initial_status = trading_system.get_system_status()
        logger.info(f"Initial system status: {initial_status}")
        
        # Wait and monitor
        await asyncio.sleep(30)
        
        # Check final status
        final_status = trading_system.get_system_status()
        logger.info(f"Final system status: {final_status}")
        
        # Check if we have any data in the data handler
        if trading_system.data_handler:
            symbols = trading_system.data_handler.symbols
            logger.info(f"Tracked symbols: {symbols}")
            
            # Check if we have data for first symbol
            if symbols:
                symbol = symbols[0]
                latest_data = await trading_system.data_handler.get_latest_data(symbol, limit=1)
                if latest_data:
                    logger.info(f"Latest data for {symbol}: {latest_data[0].close}")
                else:
                    logger.warning(f"No data available for {symbol}")
        
        # Check AI engine status
        if trading_system.ai_engine:
            model_status = trading_system.ai_engine.get_model_status()
            logger.info(f"AI model status: {model_status}")
            
            signal_history = trading_system.ai_engine.get_signal_history(limit=5)
            logger.info(f"Generated signals: {len(signal_history)}")
            
            if signal_history:
                for i, signal in enumerate(signal_history):
                    logger.info(f"Signal {i+1}: {signal.action.name} {signal.symbol} "
                               f"(confidence: {signal.confidence:.3f})")
        
        logger.info("✅ Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Shutdown the system
        await trading_system.shutdown()


async def test_event_flow():
    """Test the event flow through the system."""
    logger.info("🔄 Testing event flow...")
    
    trading_system = TradingSystem()
    
    try:
        # Start system
        await trading_system.start()
        
        # Track events
        events_received = []
        
        def event_tracker(event):
            events_received.append(event.event_type.value)
            logger.info(f"Event tracked: {event.event_type.value}")
        
        # Subscribe to all major events
        trading_system.event_bus.subscribe(EventType.NEW_MARKET_DATA, event_tracker)
        trading_system.event_bus.subscribe(EventType.FEATURES_READY, event_tracker) 
        trading_system.event_bus.subscribe(EventType.PROPOSED_TRADE, event_tracker)
        
        # Wait for events to flow
        await asyncio.sleep(20)
        
        # Check events
        logger.info(f"Events received: {len(events_received)}")
        for event_type in set(events_received):
            count = events_received.count(event_type)
            logger.info(f"  {event_type}: {count} times")
        
        return len(events_received) > 0
        
    finally:
        await trading_system.shutdown()


async def test_performance():
    """Test system performance metrics."""
    logger.info("🔄 Testing system performance...")
    
    trading_system = TradingSystem()
    
    try:
        start_time = datetime.now()
        
        # Start system
        await trading_system.start()
        startup_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Startup time: {startup_time:.2f} seconds")
        
        # Run for a bit and measure
        run_start = datetime.now()
        await asyncio.sleep(15)
        run_time = (datetime.now() - run_start).total_seconds()
        
        # Check memory usage and event counts
        event_history = trading_system.event_bus.get_event_history()
        logger.info(f"Events processed in {run_time:.2f}s: {len(event_history)}")
        
        if len(event_history) > 0:
            events_per_second = len(event_history) / run_time
            logger.info(f"Event processing rate: {events_per_second:.2f} events/second")
        
        return True
        
    finally:
        shutdown_start = datetime.now()
        await trading_system.shutdown()
        shutdown_time = (datetime.now() - shutdown_start).total_seconds()
        logger.info(f"Shutdown time: {shutdown_time:.2f} seconds")


async def run_all_tests():
    """Run all integration tests."""
    logger.info("🧪 Starting comprehensive integration tests...")
    
    tests = [
        ("Integration Test", test_integration),
        ("Event Flow Test", test_event_flow),
        ("Performance Test", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
        
        # Wait between tests
        await asyncio.sleep(2)
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    try:
        result = asyncio.run(run_all_tests())
        if result:
            logger.info("🎉 All integration tests PASSED!")
            sys.exit(0)
        else:
            logger.error("💥 Some integration tests FAILED!")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)