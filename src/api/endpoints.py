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

from src.core.config import Settings, get_settings
from src.services.market_analyzer import MarketAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

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
# BACKTESTING ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/backtest/run")
async def run_backtest(
    symbol: str,
    days: int = Query(30, ge=1, le=365),
    initial_capital: float = Query(10000, ge=100),
    position_size_pct: float = Query(0.05, ge=0.01, le=0.20),
    confidence_threshold: float = Query(0.5, ge=0.1, le=0.9),
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Run a backtest for a specific symbol.
    
    Simulates trading over historical data with realistic execution.
    Returns:
    - Portfolio performance (returns, Sharpe, drawdown)
    - Trade statistics (win rate, profit factor, total trades)
    - Detailed trade history
    """
    try:
        symbol = symbol.upper()
        
        if symbol not in analyzer.market_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Get historical data
        data = analyzer.market_data[symbol].tail(days * 24).copy()
        
        if len(data) < 50:
            raise HTTPException(status_code=400, detail=f"Insufficient data. Need at least 50 candles, have {len(data)}")
        
        # Run backtest
        try:
            from src.services.backtester import Backtester
        except ImportError:
            # Fallback: Use simple backtest
            return await _run_simple_backtest(data, symbol, days, initial_capital)
        
        backtester = Backtester(
            model=analyzer.signal_generator.model,
            scaler=analyzer.signal_generator.scaler,
            transformer=analyzer.signal_generator.power_transformer,
            feature_cols=analyzer.signal_generator.feature_columns,
            initial_capital=initial_capital,
            commission=0.001,
            slippage=0.0005,
            confidence_threshold=confidence_threshold,
            position_size_pct=position_size_pct
        )
        
        result = backtester.run_backtest(data, symbol)
        
        return {
            "symbol": symbol,
            "period_days": days,
            "initial_capital": initial_capital,
            "backtest": result,
            "timestamp": datetime.now().isoformat() + 'Z'
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


async def _run_simple_backtest(data, symbol, days, initial_capital):
    """Simple backtest fallback if Backtester not available"""
    # Simple buy-and-hold comparison
    start_price = data['close'].iloc[0]
    end_price = data['close'].iloc[-1]
    buy_hold_return = (end_price / start_price - 1) * 100
    
    return {
        "symbol": symbol,
        "period_days": days,
        "initial_capital": initial_capital,
        "buy_hold_return": round(buy_hold_return, 2),
        "message": "Full backtester not available - showing buy-and-hold only",
        "timestamp": datetime.now().isoformat() + 'Z'
    }


@router.get("/backtest/summary")
async def get_backtest_summary(
    symbol: str,
    days: int = Query(30, ge=1, le=365),
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Get a quick backtest summary without running full simulation.
    
    Returns:
    - Price change and volatility
    - Data quality metrics
    """
    try:
        symbol = symbol.upper()
        
        if symbol not in analyzer.market_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        data = analyzer.market_data[symbol].tail(days * 24).copy()
        
        if len(data) < 50:
            raise HTTPException(status_code=400, detail=f"Insufficient data. Need at least 50 candles, have {len(data)}")
        
        price_change = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
        volatility = data['close'].pct_change().std() * 100
        max_price = data['close'].max()
        min_price = data['close'].min()
        
        return {
            "symbol": symbol,
            "period_days": days,
            "price_change_pct": round(price_change, 2),
            "volatility_pct": round(volatility, 2),
            "max_price": round(max_price, 2),
            "min_price": round(min_price, 2),
            "data_points": len(data),
            "start_price": round(data['close'].iloc[0], 2),
            "end_price": round(data['close'].iloc[-1], 2),
            "timestamp": datetime.now().isoformat() + 'Z'
        }
    except Exception as e:
        logger.error(f"Error getting backtest summary for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORWARD TESTING ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/forward-test/{symbol}")
async def run_forward_test(
    symbol: str,
    days: int = Query(30, ge=1, le=90),
    initial_capital: float = Query(10000, ge=100),
    analyzer: MarketAnalyzer = Depends(get_market_analyzer)
):
    """
    Run a forward test on recent data.
    
    Simulates how the model would perform on the most recent data.
    This is the closest you can get to live performance.
    """
    try:
        symbol = symbol.upper()
        
        if symbol not in analyzer.market_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Use last N days for forward test
        data = analyzer.market_data[symbol].tail(days * 24).copy()
        
        if len(data) < 50:
            raise HTTPException(status_code=400, detail=f"Insufficient data. Need at least 50 candles, have {len(data)}")
        
        # Use simple simulation
        from src.services.forward_tester import ForwardTester
        
        tester = ForwardTester(analyzer)
        result = await tester.run_forward_test(symbol, data, initial_capital)
        
        return {
            "symbol": symbol,
            "period_days": days,
            "initial_capital": initial_capital,
            "forward_test": result,
            "timestamp": datetime.now().isoformat() + 'Z'
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running forward test for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Forward test failed: {str(e)}")


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

@router.get("/ws/positions/updates")
async def position_websocket_info():
    """Info about WebSocket positions updates (placeholder)"""
    return {
        "message": "WebSocket endpoint for real-time position updates",
        "endpoint": "/ws/positions",
        "protocol": "WebSocket",
        "updates": "real-time position PnL and status"
    }