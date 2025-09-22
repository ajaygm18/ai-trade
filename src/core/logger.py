"""
Centralized logging configuration for the trading system.
"""

import sys
import os
from pathlib import Path
from typing import Optional
from loguru import logger
from datetime import datetime

from .config import config


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "1 day",
    retention: str = "30 days"
) -> None:
    """
    Setup centralized logging for the trading system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path. If None, uses default path
        rotation: Log rotation interval
        retention: Log retention period
    """
    # Remove default logger
    logger.remove()
    
    # Console logging
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # File logging
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"trading_system_{datetime.now().strftime('%Y%m%d')}.log"
    
    logger.add(
        str(log_file),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
               "{name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        compression="zip"
    )
    
    # Audit logging (for SEBI compliance)
    audit_dir = Path("logs/audit")
    audit_dir.mkdir(exist_ok=True, parents=True)
    
    logger.add(
        str(audit_dir / "audit_{time:YYYY-MM-DD}.log"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | AUDIT | {message}",
        rotation="1 day",
        retention=f"{config.compliance.audit_log_retention_days} days",
        filter=lambda record: "AUDIT" in record["extra"]
    )
    
    # Error logging
    error_dir = Path("logs/errors")
    error_dir.mkdir(exist_ok=True, parents=True)
    
    logger.add(
        str(error_dir / "error_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
               "{name}:{function}:{line} | {message} | {exception}",
        rotation="1 day",
        retention="90 days"
    )


def get_logger(name: str) -> "logger":
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


def audit_log(message: str, **kwargs) -> None:
    """
    Log an audit message for SEBI compliance.
    
    Args:
        message: Audit message
        **kwargs: Additional context
    """
    logger.bind(AUDIT=True).info(f"{message} | Context: {kwargs}")


def trade_log(action: str, symbol: str, quantity: int, price: float, **kwargs) -> None:
    """
    Log a trade action for audit purposes.
    
    Args:
        action: Trade action (BUY, SELL, etc.)
        symbol: Trading symbol
        quantity: Trade quantity
        price: Trade price
        **kwargs: Additional context
    """
    audit_log(
        f"TRADE | Action: {action} | Symbol: {symbol} | "
        f"Quantity: {quantity} | Price: {price}",
        **kwargs
    )


def risk_log(event: str, details: dict) -> None:
    """
    Log a risk management event.
    
    Args:
        event: Risk event type
        details: Event details
    """
    audit_log(f"RISK | Event: {event} | Details: {details}")


def system_log(event: str, details: dict) -> None:
    """
    Log a system event.
    
    Args:
        event: System event type
        details: Event details
    """
    audit_log(f"SYSTEM | Event: {event} | Details: {details}")


# Initialize logging on module import
setup_logger(log_level=config.monitoring.log_level)


# Export logger instance
__all__ = ["get_logger", "audit_log", "trade_log", "risk_log", "system_log"]