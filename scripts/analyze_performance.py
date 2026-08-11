#!/usr/bin/env python3
"""
SmartCrypto Performance Analysis Script
Analyzes all signals and positions to generate comprehensive performance reports
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_settings
from src.services.history_manager import HistoryManager
from src.services.portfolio_manager import PortfolioManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceAnalyzer:
    def __init__(self):
        self.settings = get_settings()
        self.history_manager = HistoryManager()
        self.portfolio_manager = PortfolioManager(
            initial_capital=self.settings.INITIAL_CAPITAL,
            history_manager=self.history_manager
        )
        self.results = {}
        
    def analyze_all(self) -> Dict:
        """Run complete analysis"""
        print("\n" + "=" * 70)
        print("📊 SMARTCRYPTO PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        # 1. Signal Analysis
        self.analyze_signals()
        
        # 2. Position/Trade Analysis
        self.analyze_positions()
        
        # 3. Pattern Analysis
        self.analyze_patterns()
        
        # 4. Timeframe Analysis
        self.analyze_timeframes()
        
        # 5. Symbol Performance
        self.analyze_symbols()
        
        # 6. Generate Summary
        self.generate_summary()
        
        return self.results
    
    def analyze_signals(self):
        """Analyze all signals from history"""
        print("\n" + "-" * 70)
        print("📡 SIGNAL ANALYSIS")
        print("-" * 70)
        
        signals = list(self.history_manager.signals_cache.values())
        
        if not signals:
            print("❌ No signals found in history")
            return
        
        total = len(signals)
        open_signals = len([s for s in signals if s.get('outcome') == 'OPEN'])
        closed_signals = len([s for s in signals if s.get('outcome') in ['WIN', 'LOSS']])
        wins = len([s for s in signals if s.get('outcome') == 'WIN'])
        losses = len([s for s in signals if s.get('outcome') == 'LOSS'])
        
        # Signal types
        signal_types = defaultdict(int)
        for s in signals:
            pattern = s.get('analysis', {}).get('detected_pattern', 'UNKNOWN')
            signal_types[pattern] += 1
        
        # Signal actions
        actions = defaultdict(int)
        for s in signals:
            actions[s.get('action', 'UNKNOWN')] += 1
        
        print(f"\n📈 SIGNAL STATISTICS:")
        print(f"   Total Signals: {total}")
        print(f"   Open Signals: {open_signals}")
        print(f"   Closed Signals: {closed_signals}")
        print(f"   Wins: {wins}")
        print(f"   Losses: {losses}")
        print(f"   Win Rate: {(wins / closed_signals * 100):.1f}%" if closed_signals > 0 else "   Win Rate: N/A")
        
        print(f"\n📊 SIGNAL TYPES:")
        for pattern, count in sorted(signal_types.items(), key=lambda x: -x[1])[:10]:
            print(f"   {pattern}: {count} ({count/total*100:.1f}%)")
        
        print(f"\n🎯 SIGNAL ACTIONS:")
        for action, count in actions.items():
            print(f"   {action}: {count} ({count/total*100:.1f}%)")
        
        self.results['signals'] = {
            'total': total,
            'open': open_signals,
            'closed': closed_signals,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / closed_signals if closed_signals > 0 else 0,
            'signal_types': dict(signal_types),
            'actions': dict(actions)
        }
    
    def analyze_positions(self):
        """Analyze all positions/trades"""
        print("\n" + "-" * 70)
        print("💰 POSITION / TRADE ANALYSIS")
        print("-" * 70)
        
        # Get positions from portfolio manager
        positions = self.portfolio_manager.get_positions_model()
        
        if not positions:
            print("❌ No positions found")
            return
        
        open_positions = [p for p in positions if p.get('status') == 'OPEN']
        closed_positions = [p for p in positions if p.get('status') == 'CLOSED']
        
        # Calculate PnL statistics
        pnls = [p.get('pnl', 0) for p in closed_positions if p.get('pnl') is not None]
        pnl_percentages = [p.get('pnl_percentage', 0) for p in closed_positions if p.get('pnl_percentage') is not None]
        
        winning_trades = [p for p in closed_positions if p.get('pnl', 0) > 0]
        losing_trades = [p for p in closed_positions if p.get('pnl', 0) < 0]
        
        total_pnl = sum(pnls) if pnls else 0
        avg_pnl = np.mean(pnls) if pnls else 0
        avg_pnl_pct = np.mean(pnl_percentages) if pnl_percentages else 0
        
        # Win/Loss ratio
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
        
        print(f"\n📈 POSITION STATISTICS:")
        print(f"   Total Positions: {len(positions)}")
        print(f"   Open Positions: {len(open_positions)}")
        print(f"   Closed Positions: {len(closed_positions)}")
        print(f"   Winning Trades: {win_count}")
        print(f"   Losing Trades: {loss_count}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Total PnL: ${total_pnl:.2f}")
        print(f"   Average PnL: ${avg_pnl:.2f}")
        print(f"   Average PnL %: {avg_pnl_pct:.2f}%")
        
        # Best and worst trades
        if pnls:
            best_trade = max(pnls)
            worst_trade = min(pnls)
            print(f"   Best Trade: ${best_trade:.2f}")
            print(f"   Worst Trade: ${worst_trade:.2f}")
        
        # By symbol
        symbol_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0})
        for p in closed_positions:
            symbol = p.get('symbol', 'UNKNOWN')
            symbol_stats[symbol]['trades'] += 1
            if p.get('pnl', 0) > 0:
                symbol_stats[symbol]['wins'] += 1
            else:
                symbol_stats[symbol]['losses'] += 1
            symbol_stats[symbol]['total_pnl'] += p.get('pnl', 0)
        
        print(f"\n📊 PERFORMANCE BY SYMBOL:")
        for symbol, stats in sorted(symbol_stats.items(), key=lambda x: -x[1]['total_pnl']):
            win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            print(f"   {symbol}: {stats['trades']} trades, {win_rate:.1%} win rate, ${stats['total_pnl']:.2f} PnL")
        
        self.results['positions'] = {
            'total': len(positions),
            'open': len(open_positions),
            'closed': len(closed_positions),
            'wins': win_count,
            'losses': loss_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_pnl_pct': avg_pnl_pct,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'symbol_stats': dict(symbol_stats)
        }
    
    def analyze_patterns(self):
        """Analyze pattern performance"""
        print("\n" + "-" * 70)
        print("🎨 PATTERN PERFORMANCE ANALYSIS")
        print("-" * 70)
        
        # Get pattern stats from history manager
        pattern_stats = self.history_manager.get_pattern_stats()
        
        if not pattern_stats or not pattern_stats.get('common_patterns'):
            print("❌ No pattern data available")
            return
        
        common_patterns = pattern_stats.get('common_patterns', {})
        
        print(f"\n📈 PATTERN STATISTICS:")
        print(f"   Total Patterns: {len(common_patterns)}")
        
        # Sort by win rate
        sorted_patterns = sorted(
            common_patterns.items(),
            key=lambda x: x[1].get('win_rate', 0),
            reverse=True
        )
        
        print(f"\n🏆 PATTERN WIN RATES:")
        for pattern, stats in sorted_patterns[:10]:
            win_rate = stats.get('win_rate', 0)
            total = stats.get('total', 0)
            print(f"   {pattern}: {win_rate:.1%} ({total} occurrences)")
        
        # Successful setups (>60% win rate)
        successful = pattern_stats.get('successful_setups', {})
        if successful:
            print(f"\n⭐ SUCCESSFUL PATTERNS (>60% Win Rate):")
            for pattern, stats in successful.items():
                print(f"   {pattern}: {stats.get('win_rate', 0):.1%} ({stats.get('total', 0)} trades)")
        
        self.results['patterns'] = {
            'total_patterns': len(common_patterns),
            'pattern_stats': common_patterns,
            'successful_patterns': successful
        }
    
    def analyze_timeframes(self):
        """Analyze timeframe accuracy"""
        print("\n" + "-" * 70)
        print("⏰ TIMEFRAME ACCURACY ANALYSIS")
        print("-" * 70)
        
        performance = self.history_manager.get_overall_performance()
        timeframe_accuracy = performance.get('timeframe_accuracy', {})
        
        if not timeframe_accuracy:
            print("❌ No timeframe accuracy data available")
            return
        
        print(f"\n📈 TIMEFRAME ACCURACY:")
        for tf, directions in timeframe_accuracy.items():
            print(f"\n   {tf} TIMEFRAME:")
            for direction, stats in directions.items():
                correct = stats.get('correct', 0)
                total = stats.get('total', 0)
                accuracy = correct / total if total > 0 else 0
                print(f"      {direction}: {accuracy:.1%} ({correct}/{total})")
        
        self.results['timeframes'] = timeframe_accuracy
    
    def analyze_symbols(self):
        """Analyze symbol performance"""
        print("\n" + "-" * 70)
        print("📊 SYMBOL PERFORMANCE ANALYSIS")
        print("-" * 70)
        
        performance = self.history_manager.get_overall_performance()
        symbol_performance = performance.get('symbol_performance', {})
        
        if not symbol_performance:
            print("❌ No symbol performance data available")
            return
        
        print(f"\n📈 SYMBOL PERFORMANCE:")
        for symbol, stats in sorted(
            symbol_performance.items(),
            key=lambda x: x[1].get('total_signals', 0),
            reverse=True
        ):
            total = stats.get('total_signals', 0)
            wins = stats.get('wins', 0)
            loss = stats.get('losses', 0)
            win_rate = wins / total if total > 0 else 0
            avg_pnl = stats.get('avg_pnl', 0)
            total_pnl = stats.get('total_pnl', 0)
            
            print(f"\n   {symbol}:")
            print(f"      Signals: {total} | Win Rate: {win_rate:.1%}")
            print(f"      Total PnL: ${total_pnl:.2f} | Avg PnL: {avg_pnl:.2f}%")
            print(f"      Wins: {wins} | Losses: {loss}")
        
        self.results['symbols'] = symbol_performance
    
    def generate_summary(self):
        """Generate comprehensive summary"""
        print("\n" + "=" * 70)
        print("📊 EXECUTIVE SUMMARY")
        print("=" * 70)
        
        signals = self.results.get('signals', {})
        positions = self.results.get('positions', {})
        patterns = self.results.get('patterns', {})
        
        total_signals = signals.get('total', 0)
        closed_signals = signals.get('closed', 0)
        signal_win_rate = signals.get('win_rate', 0)
        
        total_positions = positions.get('total', 0)
        closed_positions = positions.get('closed', 0)
        position_win_rate = positions.get('win_rate', 0)
        total_pnl = positions.get('total_pnl', 0)
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total Signals: {total_signals}")
        print(f"   Total Positions: {total_positions}")
        print(f"   Closed Trades: {closed_positions}")
        print(f"   Win Rate: {position_win_rate:.1%}")
        print(f"   Total PnL: ${total_pnl:.2f}")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if position_win_rate > 0.6:
            print("   ✅ EXCELLENT: Win rate above 60%")
        elif position_win_rate > 0.5:
            print("   ✅ GOOD: Win rate above 50%")
        elif position_win_rate > 0.4:
            print("   ⚠️ FAIR: Win rate above 40%")
        else:
            print("   ❌ NEEDS IMPROVEMENT: Win rate below 40%")
        
        if total_pnl > 0:
            print(f"   ✅ PROFITABLE: Total PnL is positive (${total_pnl:.2f})")
        else:
            print(f"   ❌ LOSING: Total PnL is negative (${total_pnl:.2f})")
        
        # Best pattern
        if patterns.get('pattern_stats'):
            best_pattern = max(
                patterns['pattern_stats'].items(),
                key=lambda x: x[1].get('win_rate', 0)
            )
            print(f"\n🏆 BEST PATTERN: {best_pattern[0]} ({best_pattern[1].get('win_rate', 0):.1%} win rate)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if position_win_rate < 0.5:
            print("   - Consider increasing confidence threshold")
        if total_pnl < 0:
            print("   - Reduce position sizes to limit losses")
        if len(positions.get('symbol_stats', {})) < 3:
            print("   - Consider diversifying across more symbols")
        
        self.results['summary'] = {
            'total_signals': total_signals,
            'total_positions': total_positions,
            'closed_trades': closed_positions,
            'win_rate': position_win_rate,
            'total_pnl': total_pnl,
            'status': 'PROFITABLE' if total_pnl > 0 else 'LOSING'
        }
    
    def export_report(self, filename: str = "performance_report.json"):
        """Export results to JSON file"""
        report_path = Path("storage") / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📁 Report saved to: {report_path}")
    
    def export_csv(self, filename: str = "performance_report.csv"):
        """Export signals and positions to CSV"""
        signals = list(self.history_manager.signals_cache.values())
        positions = self.portfolio_manager.get_positions_model()
        
        signals_df = pd.DataFrame(signals)
        positions_df = pd.DataFrame(positions)
        
        # Save to CSV
        csv_path = Path("storage") / filename
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(csv_path.parent / "performance_report.xlsx") as writer:
            if not signals_df.empty:
                signals_df.to_excel(writer, sheet_name='Signals', index=False)
            if not positions_df.empty:
                positions_df.to_excel(writer, sheet_name='Positions', index=False)
        
        print(f"📁 CSV report saved to: {csv_path.parent / 'performance_report.xlsx'}")


def main():
    """Run the analysis"""
    analyzer = PerformanceAnalyzer()
    results = analyzer.analyze_all()
    analyzer.export_report()
    analyzer.export_csv()
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()