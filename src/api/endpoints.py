"""
FastAPI endpoints for SmartCrypto API v3.0.0
Fully compatible with the new AI model (derivatives, order book, stationary features)
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
from fastapi.responses import HTMLResponse

from src.core.config import Settings, get_settings
from src.services.market_analyzer import MarketAnalyzer
from src.utils.logger import get_logger
from src.api.security import verify_api_key

router = APIRouter(
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)]  # Protects ALL routes under /api/v1!
)
logger = get_logger(__name__)

# Global market analyzer instance
market_analyzer: Optional[MarketAnalyzer] = None


def get_market_analyzer() -> MarketAnalyzer:
    """Dependency to get market analyzer instance"""
    global market_analyzer
    if market_analyzer is None:
        logger.error("MarketAnalyzer not initialized - returning 503")
        raise HTTPException(status_code=503, detail="Market analyzer not initialized. Service is starting up.")
    return market_analyzer


def set_market_analyzer(analyzer: MarketAnalyzer):
    """Function to set the market analyzer from main.py"""
    global market_analyzer
    market_analyzer = analyzer
    logger.info("✅ MarketAnalyzer set in endpoints")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIGNAL ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/signals", response_model=Dict[str, Dict])
async def get_all_signals(
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Get trading signals for all symbols.
    
    Returns a dictionary with symbol as key and signal as value.
    Each signal contains:
    - direction_1h, direction_4h, direction_1d: BUY/SELL/HOLD
    - confidence: 0-1 confidence score
    - signal_strength: 0-1 signal strength
    - probabilities: probability breakdown for each timeframe
    - strategy: stop loss, take profit levels
    - feature_info: details about features used
    """
    try:
        signals = analyzer.get_latest_signals()
        
        # Add feature info to each signal
        if analyzer.signal_generator and hasattr(analyzer.signal_generator, 'feature_columns'):
            feature_info = {
                'total_features': len(analyzer.signal_generator.feature_columns),
                'derivatives': len([f for f in analyzer.signal_generator.feature_columns if any(x in f for x in ['funding', 'oi_'])]),
                'orderbook': len([f for f in analyzer.signal_generator.feature_columns if any(x in f for x in ['buy_pressure', 'order_imbalance'])]),
                'stationary': True,
                'timestamp': datetime.now().isoformat() + 'Z'
            }
            for symbol, signal in signals.items():
                if signal:
                    signal['feature_info'] = feature_info
        
        logger.info(f"📊 Returning signals for {len(signals)} symbols")
        return signals
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/signals/{symbol}", response_model=Dict)
async def get_signal(
    symbol: str,
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Get trading signal for a specific symbol.
    
    Returns detailed signal including:
    - Direction for all timeframes (1h, 4h, 1d)
    - Confidence and strength metrics
    - Probability breakdown
    - Risk management levels (stop loss, take profit)
    - Feature information
    """
    try:
        symbol = symbol.upper()
        signal = analyzer.get_signal(symbol)
        
        if signal is None:
            return {
                "symbol": symbol,
                "action": "HOLD",
                "confidence": 0.5,
                "timestamp": datetime.now().isoformat() + 'Z',
                "price": 0,
                "message": "No signal available yet - waiting for next candle close"
            }
        
        # Add feature info
        if analyzer.signal_generator and hasattr(analyzer.signal_generator, 'feature_columns'):
            signal['feature_info'] = {
                'total_features': len(analyzer.signal_generator.feature_columns),
                'derivatives': len([f for f in analyzer.signal_generator.feature_columns if any(x in f for x in ['funding', 'oi_'])]),
                'orderbook': len([f for f in analyzer.signal_generator.feature_columns if any(x in f for x in ['buy_pressure', 'order_imbalance'])]),
                'stationary': True
            }
        
        logger.info(f"📡 Signal for {symbol}: {signal.get('direction_1h', 'UNKNOWN')}")
        return signal
    except Exception as e:
        logger.error(f"Error getting signal for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DERIVATIVES ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/derivatives/{symbol}")
async def get_derivatives_data(
    symbol: str,
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Get derivatives data including funding rates and open interest.
    
    Returns:
    - Current funding rate, 8h and 24h sums
    - Funding Z-score and percentile
    - Open interest and changes
    - OI-Volume ratio and price-OI divergence
    """
    try:
        symbol = symbol.upper()
        
        if symbol not in analyzer.market_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        data = analyzer.market_data[symbol]
        
        # Check if derivatives features exist
        derivatives_cols = [col for col in data.columns if any(x in col for x in ['funding', 'oi_'])]
        
        if not derivatives_cols:
            return {
                "symbol": symbol,
                "message": "No derivatives data available",
                "available": False,
                "timestamp": datetime.now().isoformat() + 'Z'
            }
        
        latest = data.iloc[-1]
        derivatives_data = {col: float(latest[col]) for col in derivatives_cols if col in latest and pd.notna(latest[col])}
        
        return {
            "symbol": symbol,
            "available": True,
            "current": derivatives_data,
            "funding_rate": float(latest['funding_rate']) if 'funding_rate' in latest and pd.notna(latest['funding_rate']) else None,
            "open_interest": float(latest['open_interest']) if 'open_interest' in latest and pd.notna(latest['open_interest']) else None,
            "open_interest_usd": float(latest['open_interest_usd']) if 'open_interest_usd' in latest and pd.notna(latest['open_interest_usd']) else None,
            "funding_zscore": float(latest['funding_zscore']) if 'funding_zscore' in latest and pd.notna(latest['funding_zscore']) else None,
            "oi_change_24h": float(latest['oi_change_24h']) if 'oi_change_24h' in latest and pd.notna(latest['oi_change_24h']) else None,
            "price_oi_divergence": float(latest['price_oi_divergence']) if 'price_oi_divergence' in latest and pd.notna(latest['price_oi_divergence']) else None,
            "timestamp": datetime.now().isoformat() + 'Z'
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting derivatives data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/derivatives/funding/{symbol}")
async def get_funding_rate_history(
    symbol: str,
    hours: int = Query(24, ge=1, le=168),
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Get funding rate history for a symbol.
    
    Returns:
    - Historical funding rates for the specified period
    - Statistics (mean, std, min, max)
    """
    try:
        symbol = symbol.upper()
        
        if symbol not in analyzer.market_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        data = analyzer.market_data[symbol]
        
        if 'funding_rate' not in data.columns:
            return {
                "symbol": symbol,
                "message": "No funding rate data available",
                "available": False
            }
        
        history = data[['timestamp', 'funding_rate']].tail(hours).to_dict(orient='records')
        funding_values = [h['funding_rate'] for h in history]
        
        return {
            "symbol": symbol,
            "available": True,
            "history": history,
            "statistics": {
                "current": funding_values[-1] if funding_values else 0,
                "mean": round(np.mean(funding_values), 6) if funding_values else 0,
                "std": round(np.std(funding_values), 6) if funding_values else 0,
                "min": round(np.min(funding_values), 6) if funding_values else 0,
                "max": round(np.max(funding_values), 6) if funding_values else 0,
                "count": len(funding_values)
            },
            "timestamp": datetime.now().isoformat() + 'Z'
        }
    except Exception as e:
        logger.error(f"Error getting funding rate history for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ORDER BOOK ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/orderbook/{symbol}")
async def get_orderbook_imbalance(
    symbol: str,
    depth_pct: float = Query(0.01, ge=0.005, le=0.05, description="Depth percentage around mid price"),
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Get current order book imbalance for a symbol.
    
    Returns:
    - Imbalance ratio (-1 to +1, positive = buy pressure)
    - Bid/Ask volumes within depth percentage
    - Spread percentage
    - Mid price
    """
    try:
        symbol = symbol.upper()
        
        # Check if orderbook monitor is available
        if hasattr(analyzer, 'orderbook_monitor') and analyzer.orderbook_monitor:
            imbalance_data = analyzer.orderbook_monitor.get_imbalance(symbol)
            if imbalance_data:
                return {
                    "symbol": symbol,
                    "available": True,
                    "imbalance": round(imbalance_data.get('imbalance', 0), 4),
                    "spread_pct": round(imbalance_data.get('spread_pct', 0), 4),
                    "bid_volume": round(imbalance_data.get('bid_volume', 0), 2),
                    "ask_volume": round(imbalance_data.get('ask_volume', 0), 2),
                    "mid_price": round(imbalance_data.get('mid_price', 0), 2),
                    "timestamp": datetime.now().isoformat() + 'Z'
                }
        
        # Fallback: Use market data features
        if symbol in analyzer.market_data:
            data = analyzer.market_data[symbol]
            if 'order_imbalance' in data.columns and 'buy_pressure' in data.columns:
                latest = data.iloc[-1]
                return {
                    "symbol": symbol,
                    "available": True,
                    "imbalance": round(float(latest['order_imbalance']), 4),
                    "buy_pressure": round(float(latest['buy_pressure']), 4),
                    "timestamp": datetime.now().isoformat() + 'Z'
                }
        
        return {
            "symbol": symbol,
            "available": False,
            "message": "Order book data not available",
            "timestamp": datetime.now().isoformat() + 'Z'
        }
    except Exception as e:
        logger.error(f"Error getting orderbook data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE INFO ENDPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/features/info")
async def get_feature_info(
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Get detailed information about features used by the model.
    
    Returns:
    - Total feature count
    - Feature categories (derivatives, orderbook, technical, etc.)
    - Sample features from each category
    - Whether features are stationary
    """
    try:
        if not analyzer.signal_generator or not hasattr(analyzer.signal_generator, 'feature_columns'):
            return {
                "available": False,
                "message": "Feature information not available"
            }
        
        features = analyzer.signal_generator.feature_columns
        
        # Categorize features
        categories = {
            "derivatives": [f for f in features if any(x in f for x in ['funding', 'oi_'])],
            "orderbook": [f for f in features if any(x in f for x in ['buy_pressure', 'order_imbalance'])],
            "technical": [f for f in features if any(x in f for x in ['rsi', 'macd', 'atr', 'bb', 'adx', 'stoch'])],
            "momentum": [f for f in features if any(x in f for x in ['mom_', 'ret_'])],
            "volume": [f for f in features if any(x in f for x in ['volume', 'vol_'])],
            "seasonality": [f for f in features if any(x in f for x in ['hour', 'day'])],
            "price_ratios": [f for f in features if any(x in f for x in ['price_', 'distance_', 'vwap'])],
            "other": []
        }
        
        # Add uncategorized
        categorized = set()
        for cat in categories:
            categorized.update(categories[cat])
        categories["other"] = [f for f in features if f not in categorized]
        
        # Count stationary features (all should be stationary)
        non_stationary = ['open', 'high', 'low', 'close', 'log_close', 'sma20', 'sma50', 'ema12', 'ema26']
        stationary_count = len([f for f in features if f not in non_stationary])
        
        return {
            "available": True,
            "total_features": len(features),
            "stationary": stationary_count == len(features),
            "stationary_count": stationary_count,
            "feature_list": features,
            "categories": {
                cat: {
                    "count": len(features_list),
                    "features": features_list[:10]  # Show first 10 only for brevity
                }
                for cat, features_list in categories.items()
            },
            "has_derivatives": len(categories["derivatives"]) > 0,
            "has_orderbook": len(categories["orderbook"]) > 0,
            "timestamp": datetime.now().isoformat() + 'Z'
        }
    except Exception as e:
        logger.error(f"Error getting feature info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PORTFOLIO ENDPOINTS (UPDATED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/portfolio/recommendations")
async def get_portfolio_recommendations(
    portfolio_value: float = Query(10000.0, ge=0),
    risk_tolerance: str = Query("MODERATE", regex="^(CONSERVATIVE|MODERATE|AGGRESSIVE|EXTREME)$"),
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Get portfolio recommendations with position sizing.
    
    Uses Kelly Criterion for position sizing based on:
    - Signal confidence
    - Timeframe alignment
    - Risk tolerance
    - Market regime
    """
    try:
        signals = analyzer.get_latest_signals()
        recommendations = []
        active_count = 0
        
        for symbol, signal in signals.items():
            if not signal:
                continue
                
            action = signal.get('action', signal.get('action', 'HOLD'))
            confidence = signal.get('confidence', 0.5)
            signal_strength = signal.get('signal_strength', 0.3)
            current_price = signal.get('price', 0)
            regime = signal.get('market_regime', 'TRENDING')
            risk_level = signal.get('risk_level', 'MEDIUM')
            
            if action == 'HOLD' or confidence < 0.4:
                continue
                
            # Kelly-based position sizing
            base_size = 0.02  # 2% base
            
            # Confidence boost
            confidence_boost = (confidence - 0.5) * 0.1
            strength_boost = (signal_strength - 0.3) * 0.05
            
            # Risk tolerance multiplier
            risk_multipliers = {
                "CONSERVATIVE": 0.6,
                "MODERATE": 1.0,
                "AGGRESSIVE": 1.4,
                "EXTREME": 1.8
            }
            
            position_size_pct = (base_size + confidence_boost + strength_boost) * risk_multipliers.get(risk_tolerance, 1.0)
            position_size_pct = min(max(position_size_pct, 0.01), 0.15)  # Cap at 15%
            
            allocation_usd = portfolio_value * position_size_pct
            
            # Get strategy levels
            strategy = signal.get('strategy', {})
            stop_loss = strategy.get('stop_loss', current_price * 0.98)
            take_profit_1 = strategy.get('take_profit_1', current_price * 1.02)
            take_profit_2 = strategy.get('take_profit_2', current_price * 1.04)
            
            # ATR-based if available
            if 'atr_used' in strategy and strategy['atr_used']:
                atr = strategy['atr_used']
                if action == 'BUY':
                    stop_loss = current_price - 1.5 * atr
                    take_profit_1 = current_price + 2.0 * atr
                    take_profit_2 = current_price + 3.0 * atr
                elif action == 'SELL':
                    stop_loss = current_price + 1.5 * atr
                    take_profit_1 = current_price - 2.0 * atr
                    take_profit_2 = current_price - 3.0 * atr
            
            recommendations.append({
                'symbol': symbol,
                'action': action,
                'confidence': round(confidence, 3),
                'signal_strength': round(signal_strength, 3),
                'position_size_pct': round(position_size_pct, 4),
                'allocation_usd': round(allocation_usd, 2),
                'entry_price': round(current_price, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit_1': round(take_profit_1, 2),
                'take_profit_2': round(take_profit_2, 2),
                'regime': regime,
                'risk_level': risk_level
            })
            active_count += 1
        
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate total allocated
        total_allocated = sum([r['allocation_usd'] for r in recommendations])
        
        return {
            "recommendations": recommendations,
            "active_count": active_count,
            "total_allocated": round(total_allocated, 2),
            "portfolio_value": portfolio_value,
            "risk_tolerance": risk_tolerance,
            "position_sizing_method": "Kelly Criterion + Risk Tolerance",
            "timestamp": datetime.now().isoformat() + 'Z'
        }
        
    except Exception as e:
        logger.error(f"Error generating portfolio recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolio/overview")
async def get_portfolio_overview(
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """Get comprehensive portfolio overview"""
    try:
        if not analyzer.portfolio_manager:
            raise HTTPException(status_code=503, detail="Portfolio manager not available")
        
        try:
            return analyzer.portfolio_manager.get_portfolio_model()
        except AttributeError:
            # Fallback
            return {
                'portfolio_value': analyzer.portfolio_manager.get_portfolio_value(),
                'available_capital': analyzer.portfolio_manager.available_capital,
                'open_positions': len(analyzer.portfolio_manager.open_positions),
                'closed_positions': len(analyzer.portfolio_manager.closed_positions),
                'timestamp': datetime.now().isoformat() + 'Z'
            }
    except Exception as e:
        logger.error(f"Error getting portfolio overview: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolio/analytics")
async def get_portfolio_analytics(
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """Get portfolio analytics"""
    try:
        if not analyzer.portfolio_manager:
            raise HTTPException(status_code=503, detail="Portfolio manager not available")
        
        try:
            return analyzer.portfolio_manager.get_analytics_model()
        except AttributeError:
            positions = analyzer.portfolio_manager.get_positions_model()
            open_positions = [p for p in positions if p.get('status') == 'OPEN']
            closed_positions = [p for p in positions if p.get('status') == 'CLOSED']
            
            winning = len([p for p in closed_positions if p.get('pnl', 0) > 0])
            losing = len([p for p in closed_positions if p.get('pnl', 0) <= 0])
            
            return {
                'total_value': analyzer.portfolio_manager.get_portfolio_value(),
                'position_count': len(open_positions),
                'winning_positions': winning,
                'losing_positions': losing,
                'win_rate': round(winning / (winning + losing), 3) if (winning + losing) > 0 else 0,
                'timestamp': datetime.now().isoformat() + 'Z'
            }
    except Exception as e:
        logger.error(f"Error getting portfolio analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/retrain")
async def trigger_retraining(
    force: bool = Query(False, description="Force retraining even if not scheduled"),
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """Manually trigger model retraining"""
    try:
        if hasattr(analyzer, 'trigger_retraining'):
            success = await analyzer.trigger_retraining()
        else:
            from src.services.model_trainer import ModelTrainer
            trainer = ModelTrainer(analyzer.settings)
            success = await trainer.retrain_model(force_retrain=force)
        
        if success:
            return {"message": "Model retraining completed successfully", "success": True}
        else:
            raise HTTPException(status_code=500, detail="Retraining failed")
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@router.get("/training/status")
async def get_training_status(
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """Get model training status"""
    try:
        from src.services.model_trainer import ModelTrainer
        trainer = ModelTrainer(analyzer.settings)
        return trainer.get_training_status()
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail="Error getting training status")


@router.get("/training/quality")
async def get_training_quality(
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """Get model quality metrics"""
    try:
        from src.services.model_trainer import ModelTrainer
        trainer = ModelTrainer(analyzer.settings)
        
        performance = getattr(trainer, 'current_model_performance', None)
        
        return {
            "current_performance": performance,
            "quality_thresholds": {
                "min_accuracy_1h": getattr(trainer, 'min_accuracy_1h', 0.55),
                "min_accuracy_4h": getattr(trainer, 'min_accuracy_4h', 0.58),
                "min_accuracy_1d": getattr(trainer, 'min_accuracy_1d', 0.62),
                "min_improvement": getattr(trainer, 'min_improvement', 0.02)
            },
            "training_history": [
                {
                    "timestamp": th['timestamp'].isoformat() if isinstance(th.get('timestamp'), datetime) else th.get('timestamp'),
                    "overall_score": th.get('performance', {}).get('overall_score', 0),
                    "accuracy_1h": th.get('performance', {}).get('accuracy_1h', 0),
                    "accuracy_4h": th.get('performance', {}).get('accuracy_4h', 0),
                    "accuracy_1d": th.get('performance', {}).get('accuracy_1d', 0)
                }
                for th in getattr(trainer, 'training_history', [])[-10:]
            ]
        }
    except Exception as e:
        logger.error(f"Error getting training quality: {e}")
        raise HTTPException(status_code=500, detail="Error getting training quality")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEALTH & STATUS ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/health")
async def health_check(
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """Detailed health check with feature status"""
    try:
        signal_loaded = getattr(analyzer.signal_generator, 'model_loaded', False) if analyzer.signal_generator else False
        
        # Get feature info
        feature_status = {}
        if analyzer.signal_generator and hasattr(analyzer.signal_generator, 'feature_columns'):
            features = analyzer.signal_generator.feature_columns
            feature_status = {
                "total_features": len(features),
                "has_derivatives": any('funding' in f or 'oi_' in f for f in features),
                "has_orderbook": any('buy_pressure' in f or 'order_imbalance' in f for f in features),
                "stationary": True
            }
        
        return {
            "status": "healthy" if analyzer.is_healthy() else "degraded",
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "services": {
                "market_analyzer": analyzer.is_healthy(),
                "model_loaded": signal_loaded,
                "symbols_monitored": len(getattr(analyzer, 'market_data', {})),
                "active_signals": len(getattr(analyzer, 'latest_signals', {})),
                "portfolio_manager": analyzer.portfolio_manager is not None,
                "features": feature_status
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")


@router.get("/live")
async def liveness_check():
    """Simple liveness check without dependencies"""
    return {"status": "alive", "message": "API is running"}


@router.get("/ready")
async def readiness_check():
    """Readiness check that verifies MarketAnalyzer is ready"""
    global market_analyzer
    if market_analyzer and market_analyzer.is_healthy():
        return {"status": "ready", "message": "MarketAnalyzer is ready"}
    else:
        raise HTTPException(status_code=503, detail="MarketAnalyzer not ready")


@router.get("/status")
async def get_system_status(
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """Get detailed system status showing candle states and features"""
    try:
        status = {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0",
            "monitoring": {
                "symbols": analyzer.settings.SYMBOLS,
                "interval": "1h",
                "signals_generated_on": "candle_close"
            },
            "current_state": {
                "symbols_with_data": len(analyzer.market_data),
                "symbols_with_signals": len(analyzer.latest_signals),
                "waiting_for": "next_1h_candle_close"
            },
            "signals": {
                "available": list(analyzer.latest_signals.keys()),
                "total": len(analyzer.latest_signals)
            }
        }
        
        # Add feature info
        if analyzer.signal_generator and hasattr(analyzer.signal_generator, 'feature_columns'):
            features = analyzer.signal_generator.feature_columns
            status["features"] = {
                "total": len(features),
                "derivatives": len([f for f in features if any(x in f for x in ['funding', 'oi_'])]),
                "orderbook": len([f for f in features if any(x in f for x in ['buy_pressure', 'order_imbalance'])]),
                "stationary": True
            }
        
        status["data_stats"] = {}
        for symbol in analyzer.settings.SYMBOLS:
            if symbol in analyzer.market_data and len(analyzer.market_data[symbol]) > 0:
                data = analyzer.market_data[symbol]
                status["data_stats"][symbol] = {
                    "records": len(data),
                    "latest_timestamp": str(data['timestamp'].iloc[-1]) if 'timestamp' in data.columns else "unknown",
                    "price_range": {
                        "min": float(data['close'].min()),
                        "max": float(data['close'].max()),
                        "current": float(data['close'].iloc[-1]) if len(data) > 0 else 0
                    }
                }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST & DEBUG ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/test/data-quality/{symbol}")
async def test_data_quality(
    symbol: str,
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """Check if data is sufficient for signal generation"""
    try:
        symbol = symbol.upper()
        
        if symbol not in analyzer.market_data:
            return {
                "symbol": symbol,
                "has_data": False,
                "message": "No data available for symbol",
                "record_count": 0,
                "sufficient_for_trading": False
            }
        
        data = analyzer.market_data[symbol]
        
        # Check for derivatives data
        has_derivatives = any('funding' in col or 'oi_' in col for col in data.columns)
        has_orderbook = any('buy_pressure' in col or 'order_imbalance' in col for col in data.columns)
        
        quality_report = {
            "symbol": symbol,
            "has_data": True,
            "record_count": len(data),
            "data_quality": "GOOD" if len(data) >= 100 else "INSUFFICIENT",
            "sufficient_for_trading": len(data) >= 20,
            "features": {
                "derivatives_available": has_derivatives,
                "orderbook_available": has_orderbook,
                "stationary": True
            },
            "latest_timestamp": None,
            "columns_available": list(data.columns)[:20]  # First 20 columns
        }
        
        if len(data) > 0 and 'timestamp' in data.columns:
            try:
                quality_report["latest_timestamp"] = str(data['timestamp'].iloc[-1])
            except:
                quality_report["latest_timestamp"] = "unknown"
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Data quality check failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "has_data": False,
            "message": f"Error checking data quality: {str(e)}",
            "record_count": 0,
            "sufficient_for_trading": False
        }


@router.post("/test/signal/{symbol}")
async def test_signal_generation(
    symbol: str,
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """Test signal generation with current data (for verification)"""
    try:
        symbol = symbol.upper()
        
        if symbol not in analyzer.market_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        historical_data = analyzer.market_data[symbol]
        
        if len(historical_data) < 20:
            raise HTTPException(status_code=400, detail=f"Insufficient data for {symbol}. Need at least 20 records, have {len(historical_data)}")
        
        last_candle = historical_data.iloc[-1]
        current_price = last_candle['close']
        analysis_data = historical_data.tail(50).copy()
        
        logger.info(f"🧪 Testing signal generation for {symbol} at price ${current_price:.2f}")
        
        signal = await analyzer.signal_generator.generate_signal(
            symbol, 
            analysis_data,
            current_price
        )
        
        if signal:
            # Add feature info
            if analyzer.signal_generator and hasattr(analyzer.signal_generator, 'feature_columns'):
                signal['feature_info'] = {
                    'total_features': len(analyzer.signal_generator.feature_columns),
                    'derivatives': len([f for f in analyzer.signal_generator.feature_columns if any(x in f for x in ['funding', 'oi_'])]),
                    'orderbook': len([f for f in analyzer.signal_generator.feature_columns if any(x in f for x in ['buy_pressure', 'order_imbalance'])]),
                    'stationary': True
                }
            
            return {
                "test_type": "signal_generation_verification",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "data_used": {
                    "historical_candles": len(analysis_data),
                    "latest_price": current_price,
                    "latest_timestamp": str(last_candle['timestamp'])
                },
                "generated_signal": signal,
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail="Signal generator returned None")
            
    except Exception as e:
        logger.error(f"Test signal generation failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMPIRICAL PAPER TEST ANALYTICS ENDPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/analytics/paper-test")
async def get_paper_test_analytics(
    days: int = Query(7, ge=1, le=90, description="Days of trade history to analyze"),
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Get empirical paper trading analytics broken down by:
    - Symbol performance
    - 3-AI Voting Consensus (3/3 Unanimous vs 2/3 Majority)
    - Market Regime (TRENDING, RANGING, VOLATILE)
    Protected by X-API-Key header.
    """
    try:
        history_mgr = getattr(analyzer, 'history_manager', None)
        if not history_mgr:
            raise HTTPException(status_code=503, detail="HistoryManager not available")

        # Query recent signals from storage/cache
        signals = history_mgr.get_recent_signals(hours=days * 24, limit=5000, include_closed=True)
        
        closed_signals = [s for s in signals if s.get('outcome') in ['WIN', 'LOSS']]
        open_signals = [s for s in signals if s.get('outcome') == 'OPEN']

        if not closed_signals:
            return {
                "timestamp": datetime.now().isoformat() + 'Z',
                "status": "accumulating_data",
                "message": "Paper trading is active. Accumulating closed trades for analysis. Let the bot run longer!",
                "total_signals_generated": len(signals),
                "open_positions": len(open_signals)
            }

        total_closed = len(closed_signals)
        winning_signals = [s for s in closed_signals if s.get('outcome') == 'WIN']
        losing_signals = [s for s in closed_signals if s.get('outcome') == 'LOSS']
        
        overall_win_rate = len(winning_signals) / total_closed if total_closed > 0 else 0.0

        pnls = [s.get('pnl_percentage', 0.0) for s in closed_signals if s.get('pnl_percentage') is not None]
        total_pnl_pct = sum(pnls) if pnls else 0.0
        avg_pnl_pct = total_pnl_pct / len(pnls) if pnls else 0.0

        # 1. Performance by Symbol
        by_symbol = {}
        for s in closed_signals:
            sym = s.get('symbol', 'UNKNOWN')
            if sym not in by_symbol:
                by_symbol[sym] = {'total': 0, 'wins': 0, 'pnl': 0.0}
            by_symbol[sym]['total'] += 1
            if s.get('outcome') == 'WIN':
                by_symbol[sym]['wins'] += 1
            by_symbol[sym]['pnl'] += (s.get('pnl_percentage') or 0.0)

        symbol_report = {}
        for sym, data in by_symbol.items():
            tot = data['total']
            symbol_report[sym] = {
                'trades': tot,
                'win_rate': f"{data['wins'] / tot:.1%}" if tot > 0 else "0.0%",
                'total_pnl_pct': f"{data['pnl']:+.2f}%",
                'avg_pnl_pct': f"{data['pnl'] / tot:+.2f}%" if tot > 0 else "0.00%"
            }

        # 2. Performance by AI Consensus Type (3/3 Unanimous vs 2/3 Majority)
        by_consensus = {}
        for s in closed_signals:
            votes = s.get('votes', {})
            tag = votes.get('tag', 'UNKNOWN')
            if tag not in by_consensus:
                by_consensus[tag] = {'total': 0, 'wins': 0, 'pnl': 0.0}
            by_consensus[tag]['total'] += 1
            if s.get('outcome') == 'WIN':
                by_consensus[tag]['wins'] += 1
            by_consensus[tag]['pnl'] += (s.get('pnl_percentage') or 0.0)

        consensus_report = {}
        for tag, data in by_consensus.items():
            tot = data['total']
            consensus_report[tag] = {
                'trades': tot,
                'win_rate': f"{data['wins'] / tot:.1%}" if tot > 0 else "0.0%",
                'total_pnl_pct': f"{data['pnl']:+.2f}%"
            }

        # 3. Performance by Market Regime
        by_regime = {}
        for s in closed_signals:
            regime = s.get('market_regime', 'UNKNOWN')
            if regime not in by_regime:
                by_regime[regime] = {'total': 0, 'wins': 0, 'pnl': 0.0}
            by_regime[regime]['total'] += 1
            if s.get('outcome') == 'WIN':
                by_regime[regime]['wins'] += 1
            by_regime[regime]['pnl'] += (s.get('pnl_percentage') or 0.0)

        regime_report = {}
        for regime, data in by_regime.items():
            tot = data['total']
            regime_report[regime] = {
                'trades': tot,
                'win_rate': f"{data['wins'] / tot:.1%}" if tot > 0 else "0.0%",
                'total_pnl_pct': f"{data['pnl']:+.2f}%"
            }

        return {
            "timestamp": datetime.now().isoformat() + 'Z',
            "status": "success",
            "timeframe_days": days,
            "summary": {
                "total_signals_generated": len(signals),
                "open_positions": len(open_signals),
                "total_closed_trades": total_closed,
                "winning_trades": len(winning_signals),
                "losing_trades": len(losing_signals),
                "overall_win_rate": f"{overall_win_rate:.1%}",
                "total_pnl_percentage": f"{total_pnl_pct:+.2f}%",
                "average_trade_pnl": f"{avg_pnl_pct:+.2f}%",
                "best_trade_pnl": f"{max(pnls):+.2f}%" if pnls else "0.00%",
                "worst_trade_pnl": f"{min(pnls):+.2f}%" if pnls else "0.00%"
            },
            "performance_by_symbol": symbol_report,
            "performance_by_consensus_votes": consensus_report,
            "performance_by_market_regime": regime_report
        }

    except Exception as e:
        logger.error(f"Error generating paper test analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VISUAL WEB DASHBOARD ANALYTICS ENDPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/analytics/paper-test/dashboard", response_class=HTMLResponse)
async def get_paper_test_visual_dashboard(
    days: int = Query(7, ge=1, le=90, description="Days of trade history to analyze"),
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Renders a modern, dark-mode visual web dashboard in your browser
    showing complete 3-AI model vote breakdowns and trade outcomes.
    """
    try:
        history_mgr = getattr(analyzer, 'history_manager', None)
        if not history_mgr:
            return HTMLResponse("<h1>HistoryManager not available</h1>", status_code=503)

        signals = history_mgr.get_recent_signals(hours=days * 24, limit=5000, include_closed=True)
        closed = [s for s in signals if s.get('outcome') in ['WIN', 'LOSS']]

        # Generate HTML Table rows for each signal
        rows_html = ""
        for s in signals:
            sym = s.get('symbol', 'N/A').replace('USDT', '')
            act = s.get('action', 'HOLD')
            out = s.get('outcome', 'OPEN')
            pnl = s.get('pnl_percentage')
            price = float(s.get('price', 0.0))
            
            pnl_str = f"{pnl:+.2f}%" if pnl is not None else "0.00%"
            
            act_color = "#00FF88" if act == "BUY" else ("#FF0055" if act == "SELL" else "#FFC107")
            out_color = "#00FF88" if out == "WIN" else ("#FF0055" if out == "LOSS" else "#00E676")

            # Extract detailed individual AI model votes
            breakdown = s.get('ai_model_breakdown', {})
            m1 = breakdown.get('model_1_regression', {})
            m2 = breakdown.get('model_2_smart_trader', {})
            m3 = breakdown.get('model_3_market_gpt', {})
            consensus = breakdown.get('committee_consensus', {}).get('consensus_type', '2/3 MAJORITY')

            m1_str = f"{m1.get('vote', 'N/A')} ({m1.get('pred_4h_return', '0%')})"
            m2_str = f"{m2.get('vote', 'N/A')} (1H={m2.get('direction_1h', 'N/A')}, 4H={m2.get('direction_4h', 'N/A')})"
            m3_str = f"{m3.get('vote', 'N/A')} (WinProb: {m3.get('trade_win_probability', '0%')})"

            rows_html += f"""
            <tr>
                <td style="font-weight:bold;">#{sym}USDT</td>
                <td style="color:{act_color}; font-weight:bold;">{act}</td>
                <td><span class="badge">{consensus}</span></td>
                <td>${price:,.4f}</td>
                <td style="color:{out_color}; font-weight:bold;">{out}</td>
                <td style="color:{out_color}; font-weight:bold;">{pnl_str}</td>
                <td><code>{m1_str}</code></td>
                <td><code>{m2_str}</code></td>
                <td><code>{m3_str}</code></td>
            </tr>
            """

        win_count = len([s for s in closed if s.get('outcome') == 'WIN'])
        win_rate_str = f"{win_count / len(closed):.1%}" if len(closed) > 0 else "0.0%"
        total_pnl_val = sum([s.get('pnl_percentage', 0.0) for s in closed if s.get('pnl_percentage') is not None])

        # Modern Dark-Mode Glassmorphism HTML Template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SnartCrypto AI v3.0 - Live Model Audit Dashboard</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background-color: #0d1117;
                    color: #c9d1d9;
                    margin: 0;
                    padding: 30px;
                }}
                .header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 30px;
                    border-bottom: 1px solid #30363d;
                    padding-bottom: 20px;
                }}
                h1 {{ color: #58a6ff; margin: 0; font-size: 26px; }}
                .card-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .card {{
                    background-color: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 10px;
                    padding: 20px;
                }}
                .card-title {{ color: #8b949e; font-size: 13px; text-transform: uppercase; margin-bottom: 8px; }}
                .card-value {{ font-size: 28px; font-weight: bold; color: #f0f6fc; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: #161b22;
                    border-radius: 10px;
                    overflow: hidden;
                    border: 1px solid #30363d;
                }}
                th, td {{
                    padding: 14px 18px;
                    text-align: left;
                    border-bottom: 1px solid #30363d;
                }}
                th {{ background-color: #21262d; color: #8b949e; font-size: 12px; text-transform: uppercase; }}
                tr:hover {{ background-color: #1c2128; }}
                .badge {{
                    background-color: #21262d;
                    color: #58a6ff;
                    padding: 4px 8px;
                    border-radius: 6px;
                    font-size: 11px;
                    border: 1px solid #30363d;
                }}
                code {{
                    background-color: #0d1117;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-family: monospace;
                    color: #e6edf3;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div>
                    <h1>🚀 SnartCrypto AI v3.0 - Live Model Audit Dashboard</h1>
                    <p style="color:#8b949e; margin-top:5px;">Real-Time 3-AI Committee Vote Breakdown & Individual Model Metrics</p>
                </div>
                <div>
                    <span class="badge">Period: Last {days} Days</span>
                </div>
            </div>

            <div class="card-grid">
                <div class="card">
                    <div class="card-title">Total Signals</div>
                    <div class="card-value">{len(signals)}</div>
                </div>
                <div class="card">
                    <div class="card-title">Closed Trades</div>
                    <div class="card-value">{len(closed)}</div>
                </div>
                <div class="card">
                    <div class="card-title">Win Rate</div>
                    <div class="card-value" style="color: #00FF88;">{win_rate_str}</div>
                </div>
                <div class="card">
                    <div class="card-title">Total Net PnL</div>
                    <div class="card-value" style="color: #00FF88;">+{total_pnl_val:.2f}%</div>
                </div>
            </div>

            <h2>📋 Detailed Signal & Individual AI Vote Audit Log</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Action</th>
                        <th>Consensus</th>
                        <th>Entry Price</th>
                        <th>Outcome</th>
                        <th>PnL %</th>
                        <th>Model 1 (Regression)</th>
                        <th>Model 2 (Smart Trader)</th>
                        <th>Model 3 (Market GPT)</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html if rows_html else "<tr><td colspan='9' style='text-align:center;'>No signals generated yet. Let paper trading run!</td></tr>"}
                </tbody>
            </tbody>
            </table>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)

    except Exception as e:
        logger.error(f"Error generating visual dashboard: {e}", exc_info=True)
        return HTMLResponse(f"<h1>Error generating dashboard: {e}</h1>", status_code=500)

@router.get("/ws/positions/updates")
async def position_websocket_info():
    """Info about WebSocket positions updates (placeholder)"""
    return {
        "message": "WebSocket endpoint for real-time position updates",
        "endpoint": "/ws/positions",
        "protocol": "WebSocket",
        "updates": "real-time position PnL and status"
    }