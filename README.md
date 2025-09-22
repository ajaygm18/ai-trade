# AI Trading System

A sophisticated AI-powered trading system for the Indian stock market, implementing ICT (Inner Circle Trader) and SMC (Smart Money Concepts) pattern recognition with deep learning models.

## Features

- **Advanced Pattern Recognition**: ICT/SMC concepts including Order Blocks, Fair Value Gaps, Liquidity Grabs
- **AI Decision Engine**: LSTM/Transformer models for prediction + Reinforcement Learning for decision making
- **SEBI Compliance**: Full regulatory compliance for Indian algorithmic trading
- **Real-time Data Processing**: Live market data ingestion and processing
- **Risk Management**: Pre-trade risk controls and portfolio management
- **Event-Driven Architecture**: Scalable, modular design for production deployment

## System Components

1. **Data Handler** - Real-time and historical market data ingestion
2. **Feature Engine** - Technical analysis and pattern recognition
3. **AI Decision Engine** - Machine learning models for trading decisions
4. **Risk Management Module** - Pre-trade risk controls and compliance
5. **Execution Handler** - Broker API integration with SEBI compliance
6. **Portfolio Manager** - Position tracking and P&L calculation
7. **Monitoring & Logging** - Dashboard and comprehensive audit trails

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the system
python -m src.main
```

## Configuration

Create a `.env` file with your broker API credentials and configuration:

```env
# Broker Configuration
BROKER_API_KEY=your_api_key
BROKER_API_SECRET=your_api_secret
BROKER_ACCESS_TOKEN=your_access_token

# Database Configuration
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_token
INFLUXDB_ORG=your_org
INFLUXDB_BUCKET=trading_data

# Trading Configuration
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.1
MAX_DAILY_LOSS=0.02
RISK_FREE_RATE=0.06
```

## Architecture

The system follows an event-driven architecture with the following event flow:

```
Market Data → Feature Engine → AI Decision Engine → Risk Management → Execution Handler
     ↓              ↓              ↓                   ↓              ↓
Time Series DB → Features → Trading Signals → Risk Checks → Broker Orders
```

## Compliance

This system is designed to comply with SEBI regulations for algorithmic trading in India:

- Unique Algo ID tagging on all orders
- Comprehensive audit trails
- Pre-trade risk controls
- Secure API access with OAuth/2FA
- Static IP whitelisting support

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational and research purposes. Trading involves substantial risk and is not suitable for all investors. Past performance does not guarantee future results.