#!/usr/bin/env python3
"""
Paper Trading Monitor - Track real positions with virtual $10 starting balance
Runs alongside the main app without modifying existing files
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_settings
from src.services.history_manager import HistoryManager
from src.services.portfolio_manager import PortfolioManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def utc_now() -> datetime:
    """Get current UTC datetime with timezone"""
    return datetime.now(timezone.utc)


@dataclass
class PaperTrade:
    """Paper trade tracking"""
    symbol: str
    action: str  # BUY or SELL
    entry_price: float
    entry_time: datetime
    quantity: float
    stop_loss: float
    take_profit: float
    status: str = "OPEN"  # OPEN, CLOSED_WIN, CLOSED_LOSS, CLOSED_TIMEOUT
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    holding_hours: float = 0.0
    signal_id: Optional[str] = None
    position_id: Optional[str] = None


class PaperTradingMonitor:
    def __init__(self, initial_capital: float = 10.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.total_pnl = 0.0
        self.open_trades: Dict[str, PaperTrade] = {}
        self.closed_trades: List[PaperTrade] = []
        self.trade_history: List[Dict] = []
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'best_symbol': None,
            'worst_symbol': None,
            'symbol_performance': defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0})
        }
        
        # File paths
        self.paper_trades_file = Path("storage/paper_trades.json")
        self.paper_summary_file = Path("storage/paper_summary.json")
        
        # Create storage directory
        self.paper_trades_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing paper trades
        self._load_paper_trades()
        
        print(f"[Paper Trading] Initialized with ${initial_capital:.2f}")
        print(f"[Paper Trading] Strategy: SCALPER (max 4-hour holds)")
        print(f"[Paper Trading] Tracking: {len(self.open_trades)} open, {len(self.closed_trades)} closed")
    
    def _load_paper_trades(self):
        """Load existing paper trades from file"""
        try:
            if self.paper_trades_file.exists():
                with open(self.paper_trades_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Load open trades
                    for trade_data in data.get('open_trades', []):
                        trade = PaperTrade(
                            symbol=trade_data['symbol'],
                            action=trade_data['action'],
                            entry_price=trade_data['entry_price'],
                            entry_time=datetime.fromisoformat(trade_data['entry_time']),
                            quantity=trade_data['quantity'],
                            stop_loss=trade_data['stop_loss'],
                            take_profit=trade_data['take_profit'],
                            status=trade_data.get('status', 'OPEN'),
                            signal_id=trade_data.get('signal_id'),
                            position_id=trade_data.get('position_id')
                        )
                        self.open_trades[trade.symbol] = trade
                    
                    # Load closed trades
                    for trade_data in data.get('closed_trades', []):
                        trade = PaperTrade(
                            symbol=trade_data['symbol'],
                            action=trade_data['action'],
                            entry_price=trade_data['entry_price'],
                            entry_time=datetime.fromisoformat(trade_data['entry_time']),
                            quantity=trade_data['quantity'],
                            stop_loss=trade_data['stop_loss'],
                            take_profit=trade_data['take_profit'],
                            status=trade_data['status'],
                            exit_price=trade_data.get('exit_price'),
                            exit_time=datetime.fromisoformat(trade_data['exit_time']) if trade_data.get('exit_time') else None,
                            pnl=trade_data.get('pnl', 0.0),
                            pnl_percentage=trade_data.get('pnl_percentage', 0.0),
                            holding_hours=trade_data.get('holding_hours', 0.0),
                            signal_id=trade_data.get('signal_id'),
                            position_id=trade_data.get('position_id')
                        )
                        self.closed_trades.append(trade)
                    
                    # Restore capital
                    self.current_capital = data.get('current_capital', self.initial_capital)
                    self.total_pnl = data.get('total_pnl', 0.0)
                    
                print(f"[Paper Trading] Loaded {len(self.open_trades)} open and {len(self.closed_trades)} closed paper trades")
        except Exception as e:
            print(f"[Paper Trading] Error loading paper trades: {e}")
    
    def _save_paper_trades(self):
        """Save paper trades to file"""
        try:
            data = {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'total_pnl': self.total_pnl,
                'open_trades': [
                    {
                        'symbol': t.symbol,
                        'action': t.action,
                        'entry_price': t.entry_price,
                        'entry_time': t.entry_time.isoformat(),
                        'quantity': t.quantity,
                        'stop_loss': t.stop_loss,
                        'take_profit': t.take_profit,
                        'status': t.status,
                        'signal_id': t.signal_id,
                        'position_id': t.position_id
                    }
                    for t in self.open_trades.values()
                ],
                'closed_trades': [
                    {
                        'symbol': t.symbol,
                        'action': t.action,
                        'entry_price': t.entry_price,
                        'entry_time': t.entry_time.isoformat(),
                        'quantity': t.quantity,
                        'stop_loss': t.stop_loss,
                        'take_profit': t.take_profit,
                        'status': t.status,
                        'exit_price': t.exit_price,
                        'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                        'pnl': t.pnl,
                        'pnl_percentage': t.pnl_percentage,
                        'holding_hours': t.holding_hours,
                        'signal_id': t.signal_id,
                        'position_id': t.position_id
                    }
                    for t in self.closed_trades
                ],
                'last_updated': utc_now().isoformat()
            }
            
            with open(self.paper_trades_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[Paper Trading] Error saving paper trades: {e}")
    
    def sync_with_app(self, history_manager: HistoryManager, portfolio_manager: PortfolioManager):
        """Sync paper trades with app's actual positions"""
        # Get actual positions from portfolio manager
        app_positions = portfolio_manager.get_positions_model()
        
        # Get signals from history
        signals = list(history_manager.signals_cache.values())
        
        # Find new signals that aren't yet paper trades
        new_signals = []
        for signal in signals:
            signal_id = signal.get('signal_id')
            if signal_id:
                # Check if already in paper trades
                exists = False
                for trade in self.open_trades.values():
                    if trade.signal_id == signal_id:
                        exists = True
                        break
                if not exists:
                    # Check if it's in closed trades
                    for trade in self.closed_trades:
                        if trade.signal_id == signal_id:
                            exists = True
                            break
                if not exists:
                    new_signals.append(signal)
        
        # Create paper trades for new signals (scalper strategy)
        for signal in new_signals[:10]:  # Limit to 10 new trades per sync
            self._create_paper_trade(signal)
        
        # Update existing paper trades with current prices
        self._update_paper_trades(portfolio_manager)
        
        # Save state
        self._save_paper_trades()
    
    def _create_paper_trade(self, signal: Dict) -> Optional[PaperTrade]:
        """Create a paper trade from a signal using scalper strategy"""
        symbol = signal.get('symbol')
        action = signal.get('action')
        price = signal.get('price', 0)
        confidence = signal.get('confidence', 0)
        
        # Only trade if confidence is high (scalper uses slightly lower threshold)
        if confidence < 0.55:
            return None
        
        # Get strategy details
        strategy = signal.get('strategy', {})
        stop_loss = strategy.get('stop_loss', price * 0.98 if action == 'BUY' else price * 1.02)
        take_profit = strategy.get('take_profit_2', price * 1.04 if action == 'BUY' else price * 0.96)
        
        # Calculate quantity based on capital (use 10% of capital per trade for scalper)
        position_size = min(0.10, self.current_capital * 0.10 / price)
        
        # Check if we have enough capital
        position_value = position_size * price
        if position_value > self.current_capital * 0.9:  # Leave 10% buffer
            position_size = (self.current_capital * 0.9) / price
        
        if position_size <= 0 or self.current_capital < 1.0:
            return None
        
        # Create paper trade
        trade = PaperTrade(
            symbol=symbol,
            action=action,
            entry_price=price,
            entry_time=utc_now(),
            quantity=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_id=signal.get('signal_id'),
            position_id=signal.get('position_id')
        )
        
        # Deduct from capital
        self.current_capital -= position_size * price
        
        # Store trade
        self.open_trades[symbol] = trade
        self.stats['total_trades'] += 1
        
        print(f"[Paper Trade] OPENED: {symbol} {action} | "
              f"Entry: ${price:.4f} | Qty: {position_size:.6f} | "
              f"Capital: ${self.current_capital + position_size * price:.2f} -> ${self.current_capital:.2f}")
        
        return trade
    
    def _update_paper_trades(self, portfolio_manager: PortfolioManager):
        """Update paper trades with current prices and check exits"""
        for symbol, trade in list(self.open_trades.items()):
            # Get current price from app's positions
            current_price = None
            
            # Try to get from portfolio manager's market data
            if hasattr(portfolio_manager, 'get_current_price'):
                current_price = portfolio_manager.get_current_price(symbol)
            else:
                # Fallback: check if position exists and get its current price
                for position in portfolio_manager.get_positions_model():
                    if position.get('symbol') == symbol and position.get('status') == 'OPEN':
                        current_price = position.get('current_price')
                        break
            
            if current_price is None:
                continue
            
            # Calculate current PnL
            if trade.action == "BUY":
                pnl = (current_price - trade.entry_price) * trade.quantity
            else:  # SELL
                pnl = (trade.entry_price - current_price) * trade.quantity
            
            pnl_percentage = (pnl / (trade.entry_price * trade.quantity)) * 100
            
            # Check exit conditions
            exit_reason = None
            
            if trade.action == "BUY":
                if current_price <= trade.stop_loss:
                    exit_reason = "STOP_LOSS"
                elif current_price >= trade.take_profit:
                    exit_reason = "TAKE_PROFIT"
            else:  # SELL
                if current_price >= trade.stop_loss:
                    exit_reason = "STOP_LOSS"
                elif current_price <= trade.take_profit:
                    exit_reason = "TAKE_PROFIT"
            
            # Check timeout (scalper max 4 hours)
            holding_hours = (utc_now() - trade.entry_time).total_seconds() / 3600
            if holding_hours >= 4 and not exit_reason:
                exit_reason = "TIMEOUT"
            
            # Close trade if exit condition met
            if exit_reason:
                self._close_paper_trade(symbol, current_price, exit_reason, pnl, pnl_percentage, holding_hours)
            else:
                # Update trade status (still open)
                trade.status = "OPEN"
    
    def _close_paper_trade(self, symbol: str, exit_price: float, reason: str, 
                          pnl: float, pnl_percentage: float, holding_hours: float):
        """Close a paper trade"""
        trade = self.open_trades.pop(symbol)
        
        # Update trade details
        trade.exit_price = exit_price
        trade.exit_time = utc_now()
        trade.pnl = pnl
        trade.pnl_percentage = pnl_percentage
        trade.holding_hours = holding_hours
        
        # Determine outcome
        if reason == "TAKE_PROFIT":
            trade.status = "CLOSED_WIN"
            self.stats['wins'] += 1
            outcome = "WIN"
        elif reason == "STOP_LOSS":
            trade.status = "CLOSED_LOSS"
            self.stats['losses'] += 1
            outcome = "LOSS"
        else:  # TIMEOUT
            if pnl > 0:
                trade.status = "CLOSED_WIN"
                self.stats['wins'] += 1
                outcome = "WIN"
            else:
                trade.status = "CLOSED_LOSS"
                self.stats['losses'] += 1
                outcome = "LOSS"
        
        # Update capital
        allocation = trade.entry_price * trade.quantity
        self.current_capital += allocation + pnl
        self.total_pnl += pnl
        
        # Update stats
        self.stats['total_pnl'] += pnl
        
        # Update symbol performance
        perf = self.stats['symbol_performance'][symbol]
        perf['trades'] += 1
        if outcome == "WIN":
            perf['wins'] += 1
        else:
            perf['losses'] += 1
        perf['pnl'] += pnl
        
        # Record trade
        self.closed_trades.append(trade)
        self.trade_history.append({
            'symbol': symbol,
            'action': trade.action,
            'entry_price': trade.entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'reason': reason,
            'holding_hours': holding_hours,
            'exit_time': trade.exit_time.isoformat()
        })
        
        # Calculate stats
        total_trades = self.stats['wins'] + self.stats['losses']
        self.stats['win_rate'] = self.stats['wins'] / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl for t in self.closed_trades if t.status == 'CLOSED_WIN']
        losses = [t.pnl for t in self.closed_trades if t.status == 'CLOSED_LOSS']
        self.stats['avg_win'] = sum(wins) / len(wins) if wins else 0
        self.stats['avg_loss'] = sum(losses) / len(losses) if losses else 0
        self.stats['max_win'] = max(wins) if wins else 0
        self.stats['max_loss'] = min(losses) if losses else 0
        
        # Find best/worst symbol
        if self.stats['symbol_performance']:
            best = max(self.stats['symbol_performance'].items(), key=lambda x: x[1]['pnl'])
            worst = min(self.stats['symbol_performance'].items(), key=lambda x: x[1]['pnl'])
            self.stats['best_symbol'] = best[0] if best[1]['trades'] > 0 else None
            self.stats['worst_symbol'] = worst[0] if worst[1]['trades'] > 0 else None
        
        print(f"[Paper Trade] CLOSED: {symbol} | {outcome} | "
              f"PnL: ${pnl:.4f} ({pnl_percentage:.2f}%) | {reason} | "
              f"Hold: {holding_hours:.1f}h | "
              f"Capital: ${self.current_capital - pnl:.2f} -> ${self.current_capital:.2f}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive paper trading report (ASCII-only for Windows compatibility)"""
        total_trades = len(self.closed_trades)
        open_count = len(self.open_trades)
        
        # Use ASCII-only characters for Windows compatibility
        report = f"""
======================================================================
                    PAPER TRADING REPORT
======================================================================
                                                                      
  CAPITAL:                                                          
     Initial: ${self.initial_capital:.4f}                                        
     Current: ${self.current_capital:.4f}                                        
     Total PnL: ${self.total_pnl:.4f} ({self.total_pnl/self.initial_capital*100:.2f}%)                      
                                                                      
  STATISTICS:                                                        
     Total Trades: {total_trades}                                              
     Open Trades: {open_count}                                               
     Wins: {self.stats['wins']} | Losses: {self.stats['losses']}                                  
     Win Rate: {self.stats['win_rate']:.1%}                                           
     Avg Win: ${self.stats['avg_win']:.4f} | Avg Loss: ${self.stats['avg_loss']:.4f}                    
     Max Win: ${self.stats['max_win']:.4f} | Max Loss: ${self.stats['max_loss']:.4f}                  
                                                                      
  BEST SYMBOL: {self.stats['best_symbol'] or 'N/A'} (${self.stats['symbol_performance'].get(self.stats['best_symbol'], {}).get('pnl', 0):.4f})             
  WORST SYMBOL: {self.stats['worst_symbol'] or 'N/A'} (${self.stats['symbol_performance'].get(self.stats['worst_symbol'], {}).get('pnl', 0):.4f})            
                                                                      
======================================================================
"""
        
        # Recent trades
        if self.trade_history:
            recent = self.trade_history[-5:]
            report += "\nRECENT TRADES:\n"
            for trade in recent:
                report += f"   {trade['symbol']} {trade['action']}: "
                report += f"${trade['pnl']:.4f} ({trade['pnl_percentage']:.2f}%) "
                report += f"- {trade['reason']} ({trade['holding_hours']:.1f}h)\n"
        
        # Open trades
        if self.open_trades:
            report += "\nOPEN TRADES:\n"
            for symbol, trade in self.open_trades.items():
                report += f"   {symbol} {trade.action}: Entry ${trade.entry_price:.4f} | "
                report += f"SL: ${trade.stop_loss:.4f} | TP: ${trade.take_profit:.4f}\n"
        
        # Symbol performance
        if self.stats['symbol_performance']:
            report += "\nSYMBOL PERFORMANCE:\n"
            for symbol, perf in sorted(
                self.stats['symbol_performance'].items(),
                key=lambda x: x[1]['pnl'],
                reverse=True
            ):
                if perf['trades'] > 0:
                    report += f"   {symbol}: {perf['trades']} trades, "
                    report += f"{perf['wins']}/{perf['losses']} W/L, "
                    report += f"${perf['pnl']:.4f}\n"
        
        report += f"\nReport Generated: {utc_now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        
        return report
    
    def save_report(self):
        """Save report to file"""
        report = self.generate_report()
        
        # Save text report with UTF-8 encoding
        report_path = Path("storage/paper_trading_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save JSON summary
        summary = {
            'timestamp': utc_now().isoformat(),
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_pnl': self.total_pnl,
            'total_return_pct': (self.total_pnl / self.initial_capital) * 100,
            'stats': {
                **self.stats,
                'symbol_performance': dict(self.stats['symbol_performance'])
            },
            'open_trades_count': len(self.open_trades),
            'closed_trades_count': len(self.closed_trades)
        }
        
        with open(self.paper_summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print report to console (using ASCII-only version)
        print(report)
        print(f"\nReport saved to: {report_path}")
        print(f"Summary saved to: {self.paper_summary_file}")
    
    def run_monitoring_loop(self, history_manager: HistoryManager, 
                           portfolio_manager: PortfolioManager,
                           interval_seconds: int = 60):
        """Run continuous monitoring loop"""
        print(f"\nStarting paper trading monitor (checking every {interval_seconds}s)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Sync with app
                self.sync_with_app(history_manager, portfolio_manager)
                
                # Generate and display report
                self.save_report()
                
                # Wait for next iteration
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            self._save_paper_trades()
            self.save_report()
        except Exception as e:
            print(f"Monitoring error: {e}")
            self._save_paper_trades()
            self.save_report()


def main():
    """Run the paper trading monitor"""
    print("=" * 70)
    print("SMARTCRYPTO PAPER TRADING MONITOR")
    print("=" * 70)
    
    settings = get_settings()
    history_manager = HistoryManager()
    portfolio_manager = PortfolioManager(
        initial_capital=settings.INITIAL_CAPITAL,
        history_manager=history_manager
    )
    
    monitor = PaperTradingMonitor(initial_capital=10.0)
    
    # Run monitoring loop
    monitor.run_monitoring_loop(history_manager, portfolio_manager, interval_seconds=60)


if __name__ == "__main__":
    main()