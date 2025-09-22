"""
Core configuration management for the AI Trading System.
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BrokerConfig(BaseSettings):
    """Broker API configuration."""
    
    name: str = Field(default="kite", env="BROKER_NAME")
    api_key: str = Field(default="", env="BROKER_API_KEY")
    api_secret: str = Field(default="", env="BROKER_API_SECRET")
    access_token: str = Field(default="", env="BROKER_ACCESS_TOKEN")
    user_id: str = Field(default="", env="BROKER_USER_ID")


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    influxdb_url: str = Field(default="http://localhost:8086", env="INFLUXDB_URL")
    influxdb_token: str = Field(default="", env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="trading_org", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="trading_data", env="INFLUXDB_BUCKET")
    
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    postgres_url: str = Field(default="", env="POSTGRES_URL")


class TradingConfig(BaseSettings):
    """Trading configuration."""
    
    initial_capital: float = Field(default=100000.0, env="INITIAL_CAPITAL")
    max_position_size: float = Field(default=0.1, env="MAX_POSITION_SIZE")
    max_daily_loss: float = Field(default=0.02, env="MAX_DAILY_LOSS")
    max_drawdown: float = Field(default=0.1, env="MAX_DRAWDOWN")
    risk_free_rate: float = Field(default=0.06, env="RISK_FREE_RATE")
    
    stop_loss_pct: float = Field(default=0.02, env="STOP_LOSS_PCT")
    take_profit_pct: float = Field(default=0.04, env="TAKE_PROFIT_PCT")
    position_sizing: str = Field(default="kelly", env="POSITION_SIZING")


class AIModelConfig(BaseSettings):
    """AI model configuration."""
    
    model_path: str = Field(default="models/", env="MODEL_PATH")
    lstm_sequence_length: int = Field(default=60, env="LSTM_SEQUENCE_LENGTH")
    lstm_hidden_size: int = Field(default=50, env="LSTM_HIDDEN_SIZE")
    transformer_heads: int = Field(default=8, env="TRANSFORMER_HEADS")
    rl_environment: str = Field(default="TradingEnv-v1", env="RL_ENVIRONMENT")


class DataConfig(BaseSettings):
    """Data configuration."""
    
    provider: str = Field(default="yfinance", env="DATA_PROVIDER")
    symbols: List[str] = Field(
        default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
        env="SYMBOLS"
    )
    timeframe: str = Field(default="1d", env="TIMEFRAME")
    lookback_days: int = Field(default=252, env="LOOKBACK_DAYS")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Parse symbols from comma-separated string if provided as env var
        symbols_env = os.getenv("SYMBOLS")
        if symbols_env:
            self.symbols = [s.strip() for s in symbols_env.split(",")]


class RiskConfig(BaseSettings):
    """Risk management configuration."""
    
    max_orders_per_minute: int = Field(default=10, env="MAX_ORDERS_PER_MINUTE")


class MonitoringConfig(BaseSettings):
    """Monitoring and logging configuration."""
    
    dashboard_port: int = Field(default=8501, env="DASHBOARD_PORT")
    api_port: int = Field(default=8000, env="API_PORT")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")


class ComplianceConfig(BaseSettings):
    """SEBI compliance configuration."""
    
    algo_id: Optional[str] = Field(env="ALGO_ID", default=None)
    static_ip: Optional[str] = Field(env="STATIC_IP", default=None)
    audit_log_retention_days: int = Field(default=90, env="AUDIT_LOG_RETENTION_DAYS")


class Config(BaseSettings):
    """Main configuration class."""
    
    debug: bool = Field(default=False, env="DEBUG")
    
    # Sub-configurations
    broker: BrokerConfig = BrokerConfig()
    database: DatabaseConfig = DatabaseConfig()
    trading: TradingConfig = TradingConfig()
    ai_model: AIModelConfig = AIModelConfig()
    data: DataConfig = DataConfig()
    risk: RiskConfig = RiskConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    compliance: ComplianceConfig = ComplianceConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instance
config = Config()