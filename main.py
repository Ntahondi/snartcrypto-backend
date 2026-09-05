#!/usr/bin/env python3
"""
SmartCrypto - AI-Powered Cryptocurrency Trading API
Enhanced with Derivatives, Order Book, and Stationary Features
Version: 3.0.0
"""
import sys
import io

# Force Windows console to use UTF-8 with safe error replacement
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
import os
import uvicorn
import secrets
from fastapi import FastAPI, HTTPException, Depends, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import logging
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

from src.api.security import verify_snailguard_request_shield

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔥 FIX: Suppress TensorFlow Warnings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress other warnings
import warnings
warnings.filterwarnings('ignore')

from src.core.config import get_settings, Settings
from src.api.endpoints import router as api_router, set_market_analyzer, websocket_live_stream
from src.services.market_analyzer import MarketAnalyzer
from src.utils.safe_logger import SafeLogger

# Setup logging
# Setup safe logging
logger = SafeLogger.setup_logging(
    name="smartcrypto",
    console_level="INFO",    # Production: "WARNING"
    file_level="DEBUG"
)
# Global services
market_analyzer_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global market_analyzer_instance
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STARTUP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("🚀 Starting SmartCrypto API v3.0.0...")
    
    try:
        settings = get_settings()
        market_analyzer_instance = MarketAnalyzer(settings)
        await market_analyzer_instance.initialize()
        
        # Set market analyzer for API endpoints
        try:
            set_market_analyzer(market_analyzer_instance)
        except NameError:
            logger.warning("⚠️ set_market_analyzer not found - skipping")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 FIX: Initialize Portfolio Manager (Only once)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if market_analyzer_instance.portfolio_manager is None:
            try:
                from src.core.trading_profiles import get_profile
                from src.services.portfolio_manager import PortfolioManager
                
                profile_name = getattr(settings, 'TRADING_PROFILE', 'test')
                default_profile = get_profile(profile_name)
                
                market_analyzer_instance.portfolio_manager = PortfolioManager(
                    initial_capital=settings.INITIAL_CAPITAL,
                    profile=default_profile,
                    history_manager=market_analyzer_instance.signal_generator.history_manager if market_analyzer_instance.signal_generator else None
                )
                logger.info(f"💰 Portfolio Manager initialized with profile: {default_profile.trading_style.value} (Max hold: {default_profile.max_holding_hours}h)")
            except Exception as e:
                logger.error(f"❌ Failed to initialize portfolio manager: {e}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 FIX: Order Book Monitor - ONLY in market_analyzer
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Remove duplicate OrderBook Monitor initialization here
        # market_analyzer already starts it in start_real_time_analysis()
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 FIX: Start real-time analysis (includes OrderBook)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        asyncio.create_task(market_analyzer_instance.start_real_time_analysis())
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 FIX: Auto-Retraining - Run in background with reduced logging
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if getattr(settings, 'AUTO_RETRAIN', True):
            try:
                from src.services.model_trainer import ModelTrainer
                trainer = ModelTrainer(settings)
                asyncio.create_task(trainer.start_auto_retraining())
                logger.info("🔄 Auto-retraining scheduler started")
                market_analyzer_instance.model_trainer = trainer
            except Exception as e:
                logger.error(f"❌ Failed to start auto-retraining: {e}")
        else:
            logger.info("⏭️ Auto-retraining disabled in settings")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # LOG SYSTEM STATUS (Reduced verbosity)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        logger.info("✅ SmartCrypto API v3.0.0 started successfully")
        logger.info(f"📊 Monitoring symbols: {settings.SYMBOLS}")
        
        if market_analyzer_instance.signal_generator:
            logger.info(f"📡 Signal generator: {market_analyzer_instance.signal_generator.model_loaded}")
            
            # Log features summary (not detailed)
            if hasattr(market_analyzer_instance.signal_generator, 'feature_columns'):
                features = market_analyzer_instance.signal_generator.feature_columns
                logger.info(f"📊 Features: {len(features)} stationary")
        
        if market_analyzer_instance.portfolio_manager:
            logger.info("💰 Portfolio Manager: ACTIVE")
            
            # Start portfolio monitoring (only once)
            if not hasattr(market_analyzer_instance, '_portfolio_started'):
                asyncio.create_task(market_analyzer_instance.portfolio_manager.start_monitoring(market_analyzer_instance))
                market_analyzer_instance._portfolio_started = True
            
            try:
                portfolio_value = market_analyzer_instance.portfolio_manager.get_portfolio_value()
                open_positions = len(market_analyzer_instance.portfolio_manager.open_positions)
                logger.info(f"💼 Portfolio: ${portfolio_value:.2f}, {open_positions} open positions")
            except:
                pass
        
    except Exception as e:
        logger.error(f"❌ Failed to start SmartCrypto API: {e}")
        raise
    
    yield  # App runs here
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SHUTDOWN
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if market_analyzer_instance:
        await market_analyzer_instance.cleanup()
    logger.info("🔴 SmartCrypto API shutdown complete")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CREATE FASTAPI APPLICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = FastAPI(
    title="SnartCrypto AI Trading API",
    description="AI-Powered Cryptocurrency Trading with Derivatives, Order Book & Stationary Features",
    version="3.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECURE API DOCUMENTATION (HTTP BASIC AUTH)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
docs_security = HTTPBasic()


async def verify_docs_auth(credentials: HTTPBasicCredentials = Depends(docs_security)) -> str:
    """Verify HTTP Basic Auth credentials before serving Swagger UI / ReDoc / OpenAPI."""
    current_settings = get_settings()
    
    # If docs auth is disabled in settings, allow access
    if not getattr(current_settings, "DOCS_AUTH_ENABLED", True):
        return credentials.username
        
    docs_username = getattr(current_settings, "DOCS_USERNAME", "admin")
    docs_password = getattr(current_settings, "DOCS_PASSWORD", "")
    
    # If no password configured, block access for safety in production
    if not docs_password:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API documentation access is locked. Configure DOCS_PASSWORD in .env to unlock.",
        )
        
    is_correct_user = secrets.compare_digest(credentials.username.encode("utf-8"), docs_username.encode("utf-8"))
    is_correct_pass = secrets.compare_digest(credentials.password.encode("utf-8"), docs_password.encode("utf-8"))
    
    if not (is_correct_user and is_correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Incorrect credentials for API Documentation",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


settings_for_docs = get_settings()
if getattr(settings_for_docs, "ENABLE_API_DOCS", True):

    @app.get("/docs", include_in_schema=False)
    async def get_swagger_ui(username: str = Depends(verify_docs_auth)):
        """Protected Swagger UI documentation."""
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{app.title} - Swagger UI",
            swagger_favicon_url="https://snartcrypto.snartpace.com/favicon.png",
        )

    @app.get("/redoc", include_in_schema=False)
    async def get_redoc_ui(username: str = Depends(verify_docs_auth)):
        """Protected ReDoc documentation."""
        return get_redoc_html(
            openapi_url="/openapi.json",
            title=f"{app.title} - ReDoc",
            redoc_favicon_url="https://snartcrypto.snartpace.com/favicon.png",
        )

    @app.get("/openapi.json", include_in_schema=False)
    async def get_openapi_schema(username: str = Depends(verify_docs_auth)):
        """Protected OpenAPI schema specification."""
        return get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://snartcrypto.snartpace.com",
        "http://localhost:57247",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIVE REAL-TIME WEBSOCKET STREAMING (ROOT & API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.websocket("/ws/live")
@app.websocket("/ws")
async def root_websocket_live_stream(websocket: WebSocket) -> None:
    """Sub-second live streaming feed at root URL."""
    await websocket_live_stream(websocket)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROOT ENDPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.get("/")
async def root():
    """Root endpoint with API information"""
    global market_analyzer_instance
    
    portfolio_available = market_analyzer_instance and market_analyzer_instance.portfolio_manager is not None
    auto_training_enabled = market_analyzer_instance and hasattr(market_analyzer_instance, 'model_trainer')
    
    # Get feature info
    feature_info = {}
    if market_analyzer_instance and market_analyzer_instance.signal_generator:
        if hasattr(market_analyzer_instance.signal_generator, 'feature_columns'):
            features = market_analyzer_instance.signal_generator.feature_columns
            feature_info = {
                "total_features": len(features),
                "derivatives_features": len([f for f in features if any(x in f for x in ['funding', 'oi_'])]),
                "orderbook_features": len([f for f in features if any(x in f for x in ['buy_pressure', 'order_imbalance'])]),
                "stationary": True,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    
    return {
        "message": "Welcome to SmartCrypto AI Trading API v3.0.0",
        "version": "3.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "features": {
            "ai_trading_signals": market_analyzer_instance and market_analyzer_instance.signal_generator is not None,
            "real_time_analysis": True,
            "derivatives_data": feature_info.get("derivatives_features", 0) > 0 if feature_info else False,
            "order_book_analysis": feature_info.get("orderbook_features", 0) > 0 if feature_info else False,
            "stationary_features": feature_info.get("stationary", False),
            "portfolio_tracking": portfolio_available,
            "auto_model_training": auto_training_enabled,
            "risk_management": portfolio_available,
            "backtesting": True,
            "forward_testing": True,
        },
        "signal_filters": {
            "min_confidence": getattr(get_settings(), 'MIN_CONFIDENCE', 0.5),
            "min_strength": getattr(get_settings(), 'MIN_SIGNAL_STRENGTH', 0.3),
            "timeframe_alignment": "2 of 3 timeframes must agree"
        },
        "risk_management": {
            "stop_loss": "ATR-based (1.5 × ATR) or Fixed 2%",
            "take_profit": "ATR-based (3.0 × ATR) or Fixed 4%",
            "position_sizing": "Kelly Criterion (0.1-0.2 fraction)",
            "max_holding_hours": getattr(get_settings(), 'MAX_HOLDING_HOURS', 48)
        },
        "endpoints": {
            "signals": "/api/v1/signals",
            "signal_for_symbol": "/api/v1/signals/{symbol}",
            "portfolio": "/api/v1/portfolio/*",
            "backtest": "/api/v1/backtest/*",
            "health": "/api/v1/health",
            "status": "/api/v1/status",
            "system_status": "/system-status",
            "retrain": "/api/v1/retrain",
            "training_status": "/api/v1/training/status",
            "forward_test": "/api/v1/forward-test"
        }
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYSTEM STATUS ENDPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.get("/system-status")
async def system_status():
    """Comprehensive system status with all components"""
    global market_analyzer_instance
    
    status = {
        "api": "operational",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "3.0.0"
    }
    
    if market_analyzer_instance:
        # Signal Generator Status
        signal_gen_status = {
            "available": market_analyzer_instance.signal_generator is not None,
            "model_loaded": market_analyzer_instance.signal_generator.model_loaded if market_analyzer_instance.signal_generator else False,
        }
        
        if market_analyzer_instance.signal_generator:
            sig_gen = market_analyzer_instance.signal_generator
            if hasattr(sig_gen, 'feature_columns'):
                features = sig_gen.feature_columns
                signal_gen_status["features"] = {
                    "total": len(features),
                    "derivatives": len([f for f in features if any(x in f for x in ['funding', 'oi_'])]),
                    "orderbook": len([f for f in features if any(x in f for x in ['buy_pressure', 'order_imbalance'])]),
                    "stationary": True
                }
            
            if hasattr(sig_gen, 'model4_engine') and sig_gen.model4_engine:
                signal_gen_status["model4_strategy_layer"] = {
                    "loaded": sig_gen.model4_engine.is_loaded,
                    "detectors_count": len(sig_gen.model4_engine.detectors),
                    "detectors": list(sig_gen.model4_engine.detectors.keys()),
                    "role": "Strategy Intelligence / Confirmation Layer"
                }

            if hasattr(sig_gen, 'performance_metrics'):
                perf = sig_gen.performance_metrics
                signal_gen_status["performance"] = {
                    "total_predictions": perf.get('total_predictions', 0),
                    "successful_predictions": perf.get('successful_predictions', 0)
                }
        
        # Portfolio Status
        portfolio_status = {
            "available": market_analyzer_instance.portfolio_manager is not None,
            "open_positions": len(market_analyzer_instance.portfolio_manager.open_positions) if market_analyzer_instance.portfolio_manager else 0,
        }
        
        if market_analyzer_instance.portfolio_manager:
            try:
                portfolio_status["portfolio_value"] = market_analyzer_instance.portfolio_manager.get_portfolio_value()
                portfolio_status["available_capital"] = market_analyzer_instance.portfolio_manager.available_capital
            except:
                pass
        
        # Training Status
        training_status = {
            "auto_training_enabled": getattr(get_settings(), 'AUTO_RETRAIN', True),
            "is_training": False,
            "last_training_time": None
        }
        
        if hasattr(market_analyzer_instance, 'model_trainer') and market_analyzer_instance.model_trainer:
            trainer = market_analyzer_instance.model_trainer
            training_status.update({
                "is_training": trainer.is_training,
                "last_training_time": trainer.last_training_time.isoformat() if trainer.last_training_time else None,
            })
        
        # Order Book Monitor Status
        orderbook_status = {
            "available": hasattr(market_analyzer_instance, 'orderbook_monitor'),
            "active": market_analyzer_instance.orderbook_monitor.is_running if hasattr(market_analyzer_instance, 'orderbook_monitor') else False
        }
        
        status.update({
            "market_analyzer": {
                "is_running": market_analyzer_instance.is_running,
                "symbols_monitored": len(market_analyzer_instance.market_data),
                "active_signals": len(market_analyzer_instance.latest_signals),
                "websocket_connections": len(market_analyzer_instance.ws_connections),
            },
            "signal_generator": signal_gen_status,
            "portfolio_manager": portfolio_status,
            "model_trainer": training_status,
            "orderbook_monitor": orderbook_status,
            "history_system": {
                "available": (market_analyzer_instance.signal_generator and 
                            hasattr(market_analyzer_instance.signal_generator, 'history_manager')),
                "status": "active" if (market_analyzer_instance.signal_generator and 
                                    hasattr(market_analyzer_instance.signal_generator, 'history_manager')) else "unavailable"
            }
        })
    
    return status


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PORTFOLIO ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.get("/api/v1/portfolio/overview")
async def portfolio_overview():
    """Get comprehensive portfolio overview"""
    global market_analyzer_instance
    if not market_analyzer_instance or not market_analyzer_instance.portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    pm = market_analyzer_instance.portfolio_manager
    if hasattr(pm, 'get_portfolio_model'):
        try:
            return pm.get_portfolio_model()
        except Exception:
            pass
    if hasattr(pm, 'get_summary'):
        return pm.get_summary()
    if hasattr(pm, 'get_portfolio_value'):
        return {
            "portfolio_value": pm.get_portfolio_value(),
            "available_capital": getattr(pm, 'available_capital', 0.0),
            "open_positions": len(getattr(pm, 'open_positions', {})),
        }
    return {"error": "Portfolio manager methods not fully implemented"}


@app.get("/api/v1/portfolio/positions")
async def portfolio_positions():
    """Get all open positions"""
    global market_analyzer_instance
    if not market_analyzer_instance or not market_analyzer_instance.portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    pm = market_analyzer_instance.portfolio_manager
    if hasattr(pm, 'get_positions_model'):
        try:
            return pm.get_positions_model()
        except Exception:
            pass
    if hasattr(pm, 'get_positions'):
        return pm.get_positions()
    if hasattr(pm, 'open_positions'):
        return [
            p if isinstance(p, dict) else (p.__dict__ if hasattr(p, '__dict__') else str(p))
            for p in pm.open_positions.values()
        ]
    return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKTEST ENDPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.post(
    "/api/v1/backtest/run",
    dependencies=[Depends(verify_snailguard_request_shield)],
)
async def run_backtest(symbol: str, days: int = 30):
    """Run backtest for a specific symbol"""
    global market_analyzer_instance
    if not market_analyzer_instance:
        raise HTTPException(status_code=503, detail="Market analyzer not available")
    
    try:
        from src.services.backtester import Backtester
        backtester = Backtester(market_analyzer_instance)
        result = await backtester.run_backtest(symbol, days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING MANAGEMENT ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.get("/api/v1/training/status")
async def training_status_endpoint():
    """Get model training status"""
    global market_analyzer_instance
    if not market_analyzer_instance or not hasattr(market_analyzer_instance, 'model_trainer'):
        return {"error": "Model trainer not available"}
    
    return market_analyzer_instance.model_trainer.get_training_status()


@app.post(
    "/api/v1/retrain",
    dependencies=[Depends(verify_snailguard_request_shield)],
)
async def trigger_training(force: bool = False):
    """Manually trigger model training"""
    global market_analyzer_instance
    if not market_analyzer_instance:
        return {"error": "Market analyzer not available"}
    
    try:
        success = await market_analyzer_instance.trigger_retraining()
        return {
            "success": success,
            "message": "Training triggered successfully" if success else "Training failed or not needed"
        }
    except Exception as e:
        return {"error": f"Training error: {str(e)}"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORWARD TEST ENDPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.get("/api/v1/forward-test/{symbol}")
async def forward_test(symbol: str, days: int = 30):
    """Run forward test for a specific symbol"""
    global market_analyzer_instance
    if not market_analyzer_instance:
        raise HTTPException(status_code=503, detail="Market analyzer not available")
    
    try:
        from src.services.forward_tester import ForwardTester
        tester = ForwardTester(market_analyzer_instance)
        result = await tester.run_forward_test(symbol, days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forward test failed: {str(e)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# START SERVER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server"""
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning"  # 🔥 Changed from "info" to "warning"
    )


if __name__ == "__main__":
    start_server()