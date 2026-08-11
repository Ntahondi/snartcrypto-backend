# SmartCrypto AI Trading System

An AI-powered cryptocurrency trading system built with FastAPI and machine learning.

## Project Structure

```
snartcrypto/
├── models/                          # AI Model Assets
├── src/                             # Source Code
│   ├── api/                         # FastAPI Application
│   ├── core/                        # Core Logic
│   ├── services/                    # Business Logic
│   ├── data/                        # Data Management
│   └── utils/                       # Utilities
├── tests/                           # Test Suite
├── scripts/                         # Utility Scripts
├── storage/                         # Data Storage
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── main.py                         # Application Entry Point
├── config.yaml                     # Configuration
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py

python scripts/train_trainer.py --force-full

```

3. Or using Docker:
```bash
docker-compose up --build
```

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API documentation.
