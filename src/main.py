"""
Main integration module for the AI Trading System.
"""

import asyncio
import signal
import sys
from typing import Optional
from datetime import datetime

from core.logger import get_logger, system_log
from core.event_bus import EventBus, Event, EventType, event_bus
from core.config import config
from data.handler import DataHandler
from features.engine import FeatureEngine
from ai.engine import AIDecisionEngine

logger = get_logger(__name__)


class TradingSystem:
    """
    Main trading system that orchestrates all components.
    
    This is the primary coordinator that starts and manages:
    - Data Handler
    - Feature Engine  
    - AI Decision Engine
    - Risk Management (to be implemented)
    - Execution Handler (to be implemented)
    - Portfolio Manager (to be implemented)
    - Monitoring Dashboard (to be implemented)
    """
    
    def __init__(self):
        self.event_bus = event_bus
        self.data_handler: Optional[DataHandler] = None
        self.feature_engine: Optional[FeatureEngine] = None
        self.ai_engine: Optional[AIDecisionEngine] = None
        
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("TradingSystem initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def start(self) -> bool:
        """Start the complete trading system."""
        try:
            system_log("STARTUP", {"timestamp": datetime.now().isoformat()})
            logger.info("🚀 Starting AI Trading System...")
            
            # Start event bus
            await self.event_bus.start()
            logger.info("✅ Event bus started")
            
            # Initialize components
            self.data_handler = DataHandler(event_bus_instance=self.event_bus)
            self.feature_engine = FeatureEngine(event_bus_instance=self.event_bus)
            self.ai_engine = AIDecisionEngine(event_bus_instance=self.event_bus)
            
            # Start components in order
            if not await self.data_handler.start():
                raise Exception("Failed to start data handler")
            logger.info("✅ Data handler started")
            
            if not await self.feature_engine.start():
                raise Exception("Failed to start feature engine")
            logger.info("✅ Feature engine started")
            
            if not await self.ai_engine.start():
                raise Exception("Failed to start AI engine")
            logger.info("✅ AI decision engine started")
            
            self._running = True
            
            # Subscribe to proposed trades for logging
            self.event_bus.subscribe(EventType.PROPOSED_TRADE, self._handle_proposed_trade)
            
            logger.info("🎯 AI Trading System fully operational!")
            system_log("SYSTEM_READY", {
                "symbols": config.data.symbols,
                "components": ["DataHandler", "FeatureEngine", "AIDecisionEngine"]
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            await self.shutdown()
            return False
    
    async def run(self) -> None:
        """Run the trading system until shutdown."""
        if not self._running:
            logger.error("System not started. Call start() first.")
            return
        
        try:
            logger.info("📊 Trading system running. Press Ctrl+C to stop...")
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
        except asyncio.CancelledError:
            logger.info("Run loop cancelled")
        except Exception as e:
            logger.error(f"Error in main run loop: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the trading system."""
        if not self._running:
            return
        
        logger.info("🛑 Shutting down AI Trading System...")
        system_log("SHUTDOWN", {"timestamp": datetime.now().isoformat()})
        
        self._running = False
        self._shutdown_event.set()
        
        # Stop components in reverse order
        try:
            if self.ai_engine:
                await self.ai_engine.stop()
                logger.info("✅ AI engine stopped")
            
            if self.feature_engine:
                await self.feature_engine.stop()
                logger.info("✅ Feature engine stopped")
            
            if self.data_handler:
                await self.data_handler.stop()
                logger.info("✅ Data handler stopped")
            
            await self.event_bus.stop()
            logger.info("✅ Event bus stopped")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("🏁 AI Trading System shutdown complete")
    
    async def _handle_proposed_trade(self, event: Event) -> None:
        """Handle proposed trade events for monitoring."""
        try:
            data = event.data
            logger.info(f"📈 Proposed Trade: {data['action']} {data['symbol']} "
                       f"(confidence: {data['confidence']:.3f})")
            
            # In the future, this will be handled by Risk Management and Execution Handler
            # For now, just log it
            
        except Exception as e:
            logger.error(f"Error handling proposed trade: {e}")
    
    def get_system_status(self) -> dict:
        """Get current system status."""
        return {
            "running": self._running,
            "components": {
                "data_handler": self.data_handler.is_running if self.data_handler else False,
                "feature_engine": self._running,  # Feature engine doesn't have is_running method
                "ai_engine": self._running,  # AI engine doesn't have is_running method
            },
            "event_bus_running": hasattr(self.event_bus, '_running') and self.event_bus._running,
            "symbols": config.data.symbols if self.data_handler else [],
            "timestamp": datetime.now().isoformat()
        }


async def main():
    """Main entry point for the trading system."""
    
    # Create and start the trading system
    trading_system = TradingSystem()
    
    if await trading_system.start():
        # Run the system
        await trading_system.run()
    else:
        logger.error("Failed to start trading system")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)