"""
History Manager Service
High-level business logic for signal history, patterns, and performance
Integrated with DataStorage (SQLite DB)
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.data.storage import DataStorage
from src.utils.safe_logger import SafeLogger

logger = SafeLogger.get_logger(__name__)


class HistoryManager:
    """
    History Manager Service - handles:
    - Signal management with business logic
    - Pattern visualization (frontend-ready)
    - Pattern similarity matching
    - Performance calculations
    - Summary reports
    """
    
    def __init__(self, storage_path: str = "data/", use_db: bool = True):
        """Initialize with storage layer"""
        self.storage = DataStorage(storage_path, use_db)
        self._load_caches()
        logger.info(f"📊 HistoryManager initialized (mode: {'DB' if use_db else 'Files'})")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CACHE MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def _load_caches(self):
        """Load caches for fast access"""
        self.signals_cache = {}
        self.pattern_drawings_cache = {}
        
        # Load signals
        signals = self.storage.get_signals(hours=8760, limit=10000)  # 1 year
        for signal in signals:
            if signal.get('signal_id'):
                self.signals_cache[signal['signal_id']] = signal
        
        # Load pattern drawings
        drawings = self.storage.get_pattern_drawings(hours=8760, limit=5000)
        for drawing in drawings:
            if drawing.get('pattern_id'):
                self.pattern_drawings_cache[drawing['pattern_id']] = drawing
        
        logger.info(f"📚 Cached {len(self.signals_cache)} signals and {len(self.pattern_drawings_cache)} drawings")
    
    def refresh_cache(self):
        """Refresh caches from storage"""
        self._load_caches()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SIGNAL MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def save_signal(self, signal: Dict, outcome: Optional[str] = None, 
                   pnl: Optional[float] = None) -> str:
        """
        Save a trading signal with business logic
        
        Returns:
            signal_id: The ID of the saved signal
        """
        try:
            if 'signal_id' not in signal:
                signal['signal_id'] = f"{signal.get('symbol', 'UNKNOWN')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            
            signal_record = {
                "signal_id": signal.get('signal_id'),
                "timestamp": signal.get('timestamp', datetime.utcnow().isoformat()),
                "symbol": signal.get('symbol'),
                "action": signal.get('action'),
                "price": signal.get('price'),
                "confidence": signal.get('confidence', 0),
                "signal_strength": signal.get('signal_strength', 0),
                "strategy": signal.get('strategy', {}),
                "analysis": signal.get('analysis', {}),
                "direction_1h": signal.get('direction_1h', 'HOLD'),
                "direction_4h": signal.get('direction_4h', 'HOLD'),
                "direction_1d": signal.get('direction_1d', 'HOLD'),
                "outcome": outcome or "OPEN",
                "pnl_percentage": pnl,
                "entry_price": signal.get('price'),
                "exit_price": None,
                "exit_time": None,
                "position_id": None,
                "stop_loss": signal.get('strategy', {}).get('stop_loss'),
                "take_profit": signal.get('strategy', {}).get('take_profit_2'),
                "max_holding_hours": signal.get('strategy', {}).get('max_holding_hours', 8)
            }
            
            if self.storage.save_signal(signal_record):
                self.signals_cache[signal_record['signal_id']] = signal_record
                
                if outcome in ['WIN', 'LOSS']:
                    self._update_performance(signal_record)
                
                logger.info(f"💾 Signal saved: {signal['symbol']} {signal['action']} | Outcome: {outcome or 'OPEN'}")
                return signal_record['signal_id']
            
            return ""
            
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return ""
    
    def update_signal_outcome(self, signal_id: str, outcome: str, 
                             pnl: Optional[float] = None,
                             pnl_percentage: Optional[float] = None,
                             exit_price: Optional[float] = None,
                             exit_time: Optional[Any] = None,
                             position_id: Optional[str] = None):
        """Update signal outcome with safe datetime parsing"""
        try:
            signal = self.signals_cache.get(signal_id)
            if not signal:
                signal = self.storage.get_signal(signal_id)
                if not signal:
                    logger.error(f"Signal {signal_id} not found")
                    return
            
            # Safe datetime handling
            if exit_time is None:
                exit_time_str = datetime.utcnow().isoformat() + 'Z'
            elif hasattr(exit_time, 'isoformat'):
                exit_time_str = exit_time.isoformat() + 'Z'
            else:
                exit_time_str = str(exit_time)

            signal['outcome'] = outcome
            signal['pnl'] = pnl
            signal['pnl_percentage'] = pnl_percentage
            signal['exit_price'] = exit_price
            signal['exit_time'] = exit_time_str
            if position_id:
                signal['position_id'] = position_id
            
            if self.storage.update_signal(signal):
                self.signals_cache[signal_id] = signal
                self._update_performance(signal)
                
                pnl_display = f"{pnl_percentage:.2f}%" if pnl_percentage is not None else "0.00%"
                logger.info(f"📊 Updated signal {signal_id}: {signal['symbol']} {signal['action']} → {outcome} (PnL: {pnl_display})")
            
        except Exception as e:
            logger.error(f"Error updating signal outcome: {e}", exc_info=True)
    
    def get_recent_signals(self, symbol: Optional[str] = None, 
                          hours: int = 24, limit: int = 100,
                          include_closed: bool = True) -> List[Dict]:
        """Get recent signals with filters"""
        return self.storage.get_signals(symbol, hours, limit, include_closed)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PERFORMANCE MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def _update_performance(self, signal: Dict):
        """Update performance metrics"""
        symbol = signal.get('symbol')
        outcome = signal.get('outcome')
        
        if not symbol or outcome not in ['WIN', 'LOSS']:
            return
        
        current = self.storage.get_performance(symbol)
        
        total_signals = current.get('total_signals', 0) + 1
        total_wins = current.get('total_wins', 0) + (1 if outcome == 'WIN' else 0)
        total_losses = current.get('total_losses', 0) + (1 if outcome == 'LOSS' else 0)
        win_rate = total_wins / total_signals if total_signals > 0 else 0
        total_pnl = current.get('total_pnl', 0) + (signal.get('pnl_percentage') or 0)
        avg_pnl = total_pnl / total_signals if total_signals > 0 else 0
        
        recent = self.storage.get_signals(symbol, hours=8760, limit=1000, include_closed=True)
        closed = [s for s in recent if s.get('outcome') in ['WIN', 'LOSS'] and s.get('pnl_percentage') is not None]
        pnls = [s['pnl_percentage'] for s in closed]
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        
        timeframe_accuracy = current.get('timeframe_accuracy', {})
        for tf in ['1h', '4h', '1d']:
            direction = signal.get(f'direction_{tf}')
            if direction:
                if tf not in timeframe_accuracy:
                    timeframe_accuracy[tf] = {}
                if direction not in timeframe_accuracy[tf]:
                    timeframe_accuracy[tf][direction] = {"correct": 0, "total": 0}
                timeframe_accuracy[tf][direction]['total'] += 1
                if outcome == 'WIN':
                    timeframe_accuracy[tf][direction]['correct'] += 1
        
        metrics = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'total_signals': total_signals,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'sharpe_ratio': self._calculate_sharpe_ratio(pnls),
            'max_drawdown': self._calculate_max_drawdown(pnls),
            'timeframe_accuracy': timeframe_accuracy,
            'signal_type_accuracy': current.get('signal_type_accuracy', {})
        }
        
        self.storage.save_performance_metrics(metrics)
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from returns"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_arr = np.array(returns)
        avg_return = np.mean(returns_arr)
        std_return = np.std(returns_arr)
        
        if std_return == 0:
            return 0.0
        
        return float((avg_return - risk_free_rate) / std_return)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns) / 100)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(abs(np.min(drawdown) * 100))
    
    def get_symbol_performance(self, symbol: str, days: int = 30) -> Dict:
        """Get performance metrics for a specific symbol"""
        signals = self.storage.get_signals(symbol, hours=days * 24, limit=1000)
        
        if not signals:
            return {}
        
        closed = [s for s in signals if s.get('outcome') in ['WIN', 'LOSS']]
        open_signals = [s for s in signals if s.get('outcome') == 'OPEN']
        
        if not closed:
            return {
                "total_signals": len(signals),
                "open_signals": len(open_signals),
                "win_rate": 0,
                "avg_pnl": 0,
                "total_closed": 0
            }
        
        wins = len([s for s in closed if s.get('outcome') == 'WIN'])
        total_closed = len(closed)
        win_rate = wins / total_closed if total_closed > 0 else 0
        
        pnls = [s.get('pnl_percentage', 0) for s in closed if s.get('pnl_percentage') is not None]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        total_pnl = sum(pnls) if pnls else 0
        
        return {
            "total_signals": len(signals),
            "open_signals": len(open_signals),
            "closed_signals": total_closed,
            "wins": wins,
            "losses": total_closed - wins,
            "win_rate": round(win_rate, 4),
            "avg_pnl": round(avg_pnl, 4),
            "total_pnl": round(total_pnl, 4),
            "best_trade": round(max(pnls) if pnls else 0, 4),
            "worst_trade": round(min(pnls) if pnls else 0, 4)
        }
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary"""
        performance = self.storage.get_performance()
        patterns = self.storage.get_pattern_stats()
        
        return {
            "total_signals": len(self.signals_cache),
            "open_signals": len([s for s in self.signals_cache.values() if s.get('outcome') == 'OPEN']),
            "closed_signals": len([s for s in self.signals_cache.values() if s.get('outcome') in ['WIN', 'LOSS']]),
            "symbol_count": len(set(s.get('symbol') for s in self.signals_cache.values() if s.get('symbol'))),
            "overall_win_rate": performance.get('win_rate', 0),
            "total_pnl": performance.get('total_pnl', 0),
            "avg_pnl": performance.get('avg_pnl', 0),
            "best_trade": performance.get('best_trade', 0),
            "worst_trade": performance.get('worst_trade', 0),
            "sharpe_ratio": performance.get('sharpe_ratio', 0),
            "max_drawdown": performance.get('max_drawdown', 0),
            "pattern_drawings": len(self.pattern_drawings_cache),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PATTERN VISUALIZATION & SIMILARITY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def save_pattern_drawing(self, signal: Dict, candle_data: Optional[List[Dict]] = None) -> Dict:
        """Generate and save pattern drawing for frontend"""
        try:
            drawing_record = {
                "pattern_id": f"pat_{signal.get('signal_id', datetime.utcnow().timestamp())}",
                "signal_id": signal.get('signal_id'),
                "symbol": signal.get('symbol'),
                "pattern_type": signal.get('analysis', {}).get('detected_pattern', 'UNKNOWN'),
                "action": signal.get('action', 'HOLD'),
                "price": signal.get('price', 0),
                "confidence": signal.get('confidence', 0),
                "signal_strength": signal.get('signal_strength', 0),
                "timestamp": signal.get('timestamp', datetime.utcnow().isoformat()),
                "created_at": datetime.utcnow().isoformat() + 'Z'
            }
            
            if self.storage.save_pattern_drawing(drawing_record):
                self.pattern_drawings_cache[drawing_record['pattern_id']] = drawing_record
                logger.info(f"🎨 Pattern drawing saved: {signal['symbol']} - {drawing_record['pattern_type']}")
                return drawing_record
            
            return {}
            
        except Exception as e:
            logger.error(f"Error saving pattern drawing: {e}")
            return {}
    
    def get_similar_patterns(self, current_signal: Dict, limit: int = 5) -> List[Dict]:
        """Find similar historical patterns"""
        try:
            recent = self.storage.get_signals(hours=720, limit=1000)
            similar = []
            
            for signal in recent:
                if signal.get('symbol') != current_signal.get('symbol'):
                    continue
                if signal.get('action') != current_signal.get('action'):
                    continue
                if signal.get('outcome') not in ['WIN', 'LOSS']:
                    continue
                
                score = self._calculate_similarity(current_signal, signal)
                if score > 0.5:
                    similar.append({
                        **signal,
                        "similarity_score": round(score, 3),
                        "days_ago": (datetime.utcnow() - datetime.fromisoformat(signal['timestamp'].replace('Z', ''))).days
                    })
            
            similar.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []
    
    def _calculate_similarity(self, signal1: Dict, signal2: Dict) -> float:
        """Calculate similarity between two signals"""
        score = 0.0
        factors = 0
        
        if 'confidence' in signal1 and 'confidence' in signal2:
            conf_diff = abs(signal1['confidence'] - signal2['confidence'])
            score += 1.0 - min(conf_diff, 0.5)
            factors += 1
        
        risk1 = signal1.get('analysis', {}).get('risk_level')
        risk2 = signal2.get('analysis', {}).get('risk_level')
        if risk1 and risk2 and risk1 == risk2:
            score += 1.0
            factors += 1
        
        regime1 = signal1.get('analysis', {}).get('market_regime')
        regime2 = signal2.get('analysis', {}).get('market_regime')
        if regime1 and regime2 and regime1 == regime2:
            score += 1.0
            factors += 1
        
        if 'signal_strength' in signal1 and 'signal_strength' in signal2:
            strength_diff = abs(signal1['signal_strength'] - signal2['signal_strength'])
            score += 1.0 - min(strength_diff, 0.5)
            factors += 1
        
        return score / factors if factors > 0 else 0.0

    def health_check(self) -> Dict:
        storage_health = self.storage.health_check()
        return {
            **storage_health,
            "cache_size": {
                "signals": len(self.signals_cache),
                "drawings": len(self.pattern_drawings_cache)
            },
            "storage_mode": "database" if self.storage.use_db else "files"
        }