#!/usr/bin/env python3
"""
SmartCrypto Analysis Dashboard
Analyzes signals.jsonl and trade_history.jsonl with beautiful visualizations
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Matplotlib or Seaborn not installed. Install with: pip install matplotlib seaborn")


class AnalysisDashboard:
    def __init__(self, signals_file: str = "signal_history/signals.jsonl",
                 trades_file: str = "positions/trade_history.jsonl"):
        self.signals_file = Path(signals_file)
        self.trades_file = Path(trades_file)
        self.signals = []
        self.trades = []
        self.df_signals = None
        self.df_trades = None
        
        # Load data
        self._load_data()
        
        # Process data
        if self.signals:
            self.df_signals = self._process_signals()
        if self.trades:
            self.df_trades = self._process_trades()
    
    def _load_data(self):
        """Load signals and trades from JSONL files"""
        # Load signals
        if self.signals_file.exists():
            with open(self.signals_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            self.signals.append(json.loads(line.strip()))
                        except:
                            pass
            print(f"📡 Loaded {len(self.signals)} signals")
        else:
            print(f"⚠️ Signals file not found: {self.signals_file}")
        
        # Load trades
        if self.trades_file.exists():
            with open(self.trades_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            self.trades.append(json.loads(line.strip()))
                        except:
                            pass
            print(f"💰 Loaded {len(self.trades)} trades")
        else:
            print(f"⚠️ Trades file not found: {self.trades_file}")
    
    def _parse_timestamp(self, value):
        """Safely parse timestamps with Z suffix"""
        if value is None:
            return pd.NaT
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            # Remove Z and parse
            value = value.replace('Z', '+00:00')
            try:
                return pd.to_datetime(value)
            except:
                try:
                    return pd.to_datetime(value[:-1])  # Try without Z
                except:
                    return pd.NaT
        return pd.NaT
    
    def _process_signals(self) -> pd.DataFrame:
        """Process signals into DataFrame"""
        df = pd.DataFrame(self.signals)
        
        # Convert timestamps safely
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].apply(self._parse_timestamp)
        
        if 'entry_time' in df.columns:
            df['entry_time'] = df['entry_time'].apply(self._parse_timestamp)
        
        if 'exit_time' in df.columns:
            df['exit_time'] = df['exit_time'].apply(self._parse_timestamp)
        
        # Extract pattern type
        if 'analysis' in df.columns:
            df['pattern_type'] = df['analysis'].apply(
                lambda x: x.get('detected_pattern', 'UNKNOWN') if isinstance(x, dict) else 'UNKNOWN'
            )
            df['signal_type'] = df['analysis'].apply(
                lambda x: x.get('signal_type', 'UNKNOWN') if isinstance(x, dict) else 'UNKNOWN'
            )
        
        # Extract timeframe consensus
        if 'analysis' in df.columns:
            df['timeframe_consensus'] = df['analysis'].apply(
                lambda x: x.get('timeframe_consensus', {}) if isinstance(x, dict) else {}
            )
        
        return df
    
    def _process_trades(self) -> pd.DataFrame:
        """Process trades into DataFrame"""
        df = pd.DataFrame(self.trades)
        
        # Convert timestamps safely
        if 'entry_time' in df.columns:
            df['entry_time'] = df['entry_time'].apply(self._parse_timestamp)
        
        if 'exit_time' in df.columns:
            df['exit_time'] = df['exit_time'].apply(self._parse_timestamp)
        
        # Calculate hold duration
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['hold_hours'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
        
        # Determine outcome if not present
        if 'status' in df.columns:
            df['outcome'] = df['status'].apply(
                lambda x: 'WIN' if x == 'CLOSED_WIN' else 'LOSS' if x == 'CLOSED_LOSS' else 'OPEN'
            )
        elif 'outcome' in df.columns:
            df['outcome'] = df['outcome'].fillna('OPEN')
        
        return df
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics"""
        summary = {
            'signals': {
                'total': len(self.signals),
                'open': 0,
                'closed': 0,
                'wins': 0,
                'losses': 0,
                'patterns': defaultdict(int),
                'actions': defaultdict(int)
            },
            'trades': {
                'total': len(self.trades),
                'open': 0,
                'closed': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'symbols': defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0})
            }
        }
        
        # Process signals
        for s in self.signals:
            outcome = s.get('outcome', 'OPEN')
            if outcome == 'OPEN':
                summary['signals']['open'] += 1
            elif outcome == 'WIN':
                summary['signals']['closed'] += 1
                summary['signals']['wins'] += 1
            elif outcome == 'LOSS':
                summary['signals']['closed'] += 1
                summary['signals']['losses'] += 1
            
            # Patterns
            if 'analysis' in s and isinstance(s['analysis'], dict):
                pattern = s['analysis'].get('detected_pattern', 'UNKNOWN')
                summary['signals']['patterns'][pattern] += 1
            
            # Actions
            action = s.get('action', 'UNKNOWN')
            summary['signals']['actions'][action] += 1
        
        # Process trades
        for t in self.trades:
            status = t.get('status', 'OPEN')
            pnl = t.get('pnl', 0)
            symbol = t.get('symbol', 'UNKNOWN')
            
            if status == 'OPEN':
                summary['trades']['open'] += 1
            else:
                summary['trades']['closed'] += 1
                if pnl > 0:
                    summary['trades']['wins'] += 1
                    outcome = 'WIN'
                else:
                    summary['trades']['losses'] += 1
                    outcome = 'LOSS'
                
                summary['trades']['total_pnl'] += pnl
                
                # Symbol stats
                sym_stats = summary['trades']['symbols'][symbol]
                sym_stats['trades'] += 1
                sym_stats['pnl'] += pnl
                if outcome == 'WIN':
                    sym_stats['wins'] += 1
                else:
                    sym_stats['losses'] += 1
        
        # Calculate rates
        total_closed = summary['trades']['closed']
        total_wins = summary['trades']['wins']
        if total_closed > 0:
            summary['trades']['win_rate'] = total_wins / total_closed
            summary['trades']['avg_pnl'] = summary['trades']['total_pnl'] / total_closed
        
        return summary
    
    def print_summary(self):
        """Print summary to console"""
        summary = self.generate_summary()
        
        print("\n" + "=" * 70)
        print("📊 SMARTCRYPTO ANALYSIS SUMMARY")
        print("=" * 70)
        
        print("\n📡 SIGNALS:")
        print(f"   Total Signals: {summary['signals']['total']}")
        print(f"   Open: {summary['signals']['open']} | Closed: {summary['signals']['closed']}")
        print(f"   Wins: {summary['signals']['wins']} | Losses: {summary['signals']['losses']}")
        
        if summary['signals']['patterns']:
            print(f"\n   Pattern Distribution:")
            for pattern, count in sorted(summary['signals']['patterns'].items(), key=lambda x: -x[1])[:10]:
                print(f"      {pattern}: {count}")
        
        if summary['signals']['actions']:
            print(f"\n   Action Distribution:")
            for action, count in summary['signals']['actions'].items():
                print(f"      {action}: {count}")
        
        print("\n💰 TRADES:")
        print(f"   Total Trades: {summary['trades']['total']}")
        print(f"   Open: {summary['trades']['open']} | Closed: {summary['trades']['closed']}")
        print(f"   Wins: {summary['trades']['wins']} | Losses: {summary['trades']['losses']}")
        print(f"   Win Rate: {summary['trades']['win_rate']:.1%}")
        print(f"   Total PnL: ${summary['trades']['total_pnl']:.4f}")
        print(f"   Average PnL: ${summary['trades']['avg_pnl']:.4f}")
        
        if summary['trades']['symbols']:
            print(f"\n   Symbol Performance:")
            for symbol, stats in sorted(
                summary['trades']['symbols'].items(),
                key=lambda x: x[1]['pnl'],
                reverse=True
            ):
                if stats['trades'] > 0:
                    win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
                    print(f"      {symbol}: {stats['trades']} trades, {win_rate:.1%} win rate, ${stats['pnl']:.4f}")
        
        print("=" * 70)
        
        return summary
    
    def create_visualizations(self, output_dir: str = "storage/analysis"):
        """Create analysis visualizations"""
        if not VISUALIZATION_AVAILABLE:
            print("❌ Visualization libraries not available")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('dark_background')
        sns.set_palette("viridis")
        
        # 1. Trade Performance Over Time
        self._plot_trade_performance(output_path)
        
        # 2. Win/Loss Distribution
        self._plot_win_loss_distribution(output_path)
        
        # 3. Symbol Performance
        self._plot_symbol_performance(output_path)
        
        # 4. Signal Pattern Distribution
        self._plot_pattern_distribution(output_path)
        
        # 5. Cumulative PnL
        self._plot_cumulative_pnl(output_path)
        
        # 6. Trade Duration Analysis
        self._plot_trade_duration(output_path)
        
        print(f"\n📊 Visualizations saved to: {output_path}")
    
    def _plot_trade_performance(self, output_path: Path):
        """Plot trade performance over time"""
        if self.df_trades is None or self.df_trades.empty:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Filter closed trades
        closed_trades = self.df_trades[self.df_trades['status'] != 'OPEN'].copy()
        
        if closed_trades.empty:
            axes[0].text(0.5, 0.5, 'No closed trades', ha='center', va='center', color='white')
            axes[1].text(0.5, 0.5, 'No closed trades', ha='center', va='center', color='white')
            plt.tight_layout()
            plt.savefig(output_path / 'trade_performance.png', dpi=150, bbox_inches='tight')
            plt.close()
            return
        
        # PnL by trade number
        closed_trades = closed_trades.sort_values('entry_time')
        closed_trades['trade_number'] = range(1, len(closed_trades) + 1)
        
        colors = ['#00ff88' if pnl > 0 else '#ff4444' for pnl in closed_trades['pnl']]
        
        axes[0].bar(closed_trades['trade_number'], closed_trades['pnl'], color=colors, alpha=0.7)
        axes[0].axhline(y=0, color='white', linestyle='-', alpha=0.3)
        axes[0].set_title('Trade PnL by Trade Number', color='white')
        axes[0].set_xlabel('Trade Number', color='white')
        axes[0].set_ylabel('PnL ($)', color='white')
        axes[0].tick_params(colors='white')
        axes[0].grid(True, alpha=0.2)
        
        # PnL by symbol
        symbol_pnl = closed_trades.groupby('symbol')['pnl'].sum().sort_values()
        colors = ['#00ff88' if v > 0 else '#ff4444' for v in symbol_pnl.values]
        axes[1].barh(symbol_pnl.index, symbol_pnl.values, color=colors, alpha=0.7)
        axes[1].axvline(x=0, color='white', linestyle='-', alpha=0.3)
        axes[1].set_title('Total PnL by Symbol', color='white')
        axes[1].set_xlabel('Total PnL ($)', color='white')
        axes[1].tick_params(colors='white')
        axes[1].grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(output_path / 'trade_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_win_loss_distribution(self, output_path: Path):
        """Plot win/loss distribution"""
        if self.df_trades is None or self.df_trades.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter closed trades
        closed_trades = self.df_trades[self.df_trades['status'] != 'OPEN'].copy()
        
        if closed_trades.empty:
            axes[0].text(0.5, 0.5, 'No closed trades', ha='center', va='center', color='white')
            axes[1].text(0.5, 0.5, 'No closed trades', ha='center', va='center', color='white')
            plt.tight_layout()
            plt.savefig(output_path / 'win_loss_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            return
        
        # Win/Loss pie chart
        outcomes = closed_trades['outcome'].value_counts()
        colors = ['#00ff88' if o == 'WIN' else '#ff4444' for o in outcomes.index]
        axes[0].pie(outcomes.values, labels=outcomes.index, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('Win/Loss Distribution', color='white')
        
        # PnL distribution histogram
        pnls = closed_trades['pnl']
        axes[1].hist(pnls, bins=20, color='#4488ff', alpha=0.7, edgecolor='white')
        axes[1].axvline(x=0, color='white', linestyle='-', alpha=0.5)
        if not pnls.empty:
            axes[1].axvline(x=pnls.mean(), color='#00ff88', linestyle='--', alpha=0.7, label=f'Mean: ${pnls.mean():.4f}')
        axes[1].set_title('PnL Distribution', color='white')
        axes[1].set_xlabel('PnL ($)', color='white')
        axes[1].set_ylabel('Frequency', color='white')
        axes[1].tick_params(colors='white')
        axes[1].grid(True, alpha=0.2)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'win_loss_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_symbol_performance(self, output_path: Path):
        """Plot symbol performance comparison"""
        if self.df_trades is None or self.df_trades.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter closed trades
        closed_trades = self.df_trades[self.df_trades['status'] != 'OPEN'].copy()
        
        if closed_trades.empty:
            axes[0].text(0.5, 0.5, 'No closed trades', ha='center', va='center', color='white')
            axes[1].text(0.5, 0.5, 'No closed trades', ha='center', va='center', color='white')
            plt.tight_layout()
            plt.savefig(output_path / 'symbol_performance.png', dpi=150, bbox_inches='tight')
            plt.close()
            return
        
        # PnL by symbol
        symbol_pnl = closed_trades.groupby('symbol').agg({
            'pnl': ['sum', 'mean', 'count'],
            'outcome': lambda x: (x == 'WIN').sum() / len(x) * 100 if len(x) > 0 else 0
        }).round(4)
        symbol_pnl.columns = ['total_pnl', 'avg_pnl', 'count', 'win_rate']
        symbol_pnl = symbol_pnl.sort_values('total_pnl', ascending=False)
        
        if not symbol_pnl.empty:
            # Total PnL bar chart
            colors = ['#00ff88' if v > 0 else '#ff4444' for v in symbol_pnl['total_pnl']]
            axes[0].bar(symbol_pnl.index, symbol_pnl['total_pnl'], color=colors, alpha=0.7)
            axes[0].axhline(y=0, color='white', linestyle='-', alpha=0.3)
            axes[0].set_title('Total PnL by Symbol', color='white')
            axes[0].set_xlabel('Symbol', color='white')
            axes[0].set_ylabel('Total PnL ($)', color='white')
            axes[0].tick_params(colors='white')
            axes[0].grid(True, alpha=0.2)
            
            # Win rate by symbol
            win_rates = symbol_pnl['win_rate']
            colors = ['#00ff88' if v > 50 else '#ff4444' for v in win_rates]
            axes[1].bar(win_rates.index, win_rates, color=colors, alpha=0.7)
            axes[1].axhline(y=50, color='white', linestyle='--', alpha=0.5, label='50% Baseline')
            axes[1].set_title('Win Rate by Symbol', color='white')
            axes[1].set_xlabel('Symbol', color='white')
            axes[1].set_ylabel('Win Rate (%)', color='white')
            axes[1].tick_params(colors='white')
            axes[1].grid(True, alpha=0.2)
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'symbol_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_pattern_distribution(self, output_path: Path):
        """Plot signal pattern distribution"""
        if self.df_signals is None or self.df_signals.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pattern distribution
        if 'pattern_type' in self.df_signals.columns:
            patterns = self.df_signals['pattern_type'].value_counts()
            if not patterns.empty:
                axes[0].barh(patterns.index[:10], patterns.values[:10], color='#4488ff', alpha=0.7)
                axes[0].set_title('Top 10 Signal Patterns', color='white')
                axes[0].set_xlabel('Count', color='white')
                axes[0].tick_params(colors='white')
                axes[0].grid(True, alpha=0.2)
        
        # Action distribution
        if 'action' in self.df_signals.columns:
            actions = self.df_signals['action'].value_counts()
            if not actions.empty:
                colors = {'BUY': '#00ff88', 'SELL': '#ff4444', 'HOLD': '#ffaa00'}
                bar_colors = [colors.get(a, '#888888') for a in actions.index]
                axes[1].bar(actions.index, actions.values, color=bar_colors, alpha=0.7)
                axes[1].set_title('Signal Actions', color='white')
                axes[1].set_xlabel('Action', color='white')
                axes[1].set_ylabel('Count', color='white')
                axes[1].tick_params(colors='white')
                axes[1].grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(output_path / 'pattern_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_cumulative_pnl(self, output_path: Path):
        """Plot cumulative PnL over time"""
        if self.df_trades is None or self.df_trades.empty:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Filter closed trades
        closed_trades = self.df_trades[self.df_trades['status'] != 'OPEN'].copy()
        
        if closed_trades.empty:
            ax.text(0.5, 0.5, 'No closed trades', ha='center', va='center', color='white')
            plt.tight_layout()
            plt.savefig(output_path / 'cumulative_pnl.png', dpi=150, bbox_inches='tight')
            plt.close()
            return
        
        closed_trades = closed_trades.sort_values('entry_time')
        closed_trades['cumulative_pnl'] = closed_trades['pnl'].cumsum()
        
        # Plot cumulative PnL
        ax.plot(closed_trades['entry_time'], closed_trades['cumulative_pnl'], 
                color='#00ff88', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        ax.fill_between(closed_trades['entry_time'], 0, closed_trades['cumulative_pnl'],
                        where=closed_trades['cumulative_pnl'] >= 0, color='#00ff88', alpha=0.2)
        ax.fill_between(closed_trades['entry_time'], 0, closed_trades['cumulative_pnl'],
                        where=closed_trades['cumulative_pnl'] < 0, color='#ff4444', alpha=0.2)
        ax.set_title('Cumulative PnL Over Time', color='white')
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Cumulative PnL ($)', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'cumulative_pnl.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_trade_duration(self, output_path: Path):
        """Plot trade duration analysis"""
        if self.df_trades is None or self.df_trades.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter closed trades with duration
        closed_trades = self.df_trades[self.df_trades['status'] != 'OPEN'].copy()
        
        if closed_trades.empty:
            axes[0].text(0.5, 0.5, 'No closed trades', ha='center', va='center', color='white')
            axes[1].text(0.5, 0.5, 'No closed trades', ha='center', va='center', color='white')
            plt.tight_layout()
            plt.savefig(output_path / 'trade_duration.png', dpi=150, bbox_inches='tight')
            plt.close()
            return
        
        if 'hold_hours' in closed_trades.columns:
            durations = closed_trades['hold_hours']
            # Remove NaN and infinite values
            durations = durations[durations.notna() & np.isfinite(durations)]
            
            if not durations.empty:
                # Duration histogram
                axes[0].hist(durations[durations < 24], bins=30, color='#4488ff', alpha=0.7, edgecolor='white')
                axes[0].axvline(x=durations.mean(), color='#00ff88', linestyle='--', alpha=0.7, 
                               label=f'Mean: {durations.mean():.1f}h')
                axes[0].axvline(x=durations.median(), color='#ffaa00', linestyle='--', alpha=0.7,
                               label=f'Median: {durations.median():.1f}h')
                axes[0].set_title('Trade Duration Distribution (<24h)', color='white')
                axes[0].set_xlabel('Duration (hours)', color='white')
                axes[0].set_ylabel('Frequency', color='white')
                axes[0].tick_params(colors='white')
                axes[0].grid(True, alpha=0.2)
                axes[0].legend()
                
                # Duration vs PnL scatter
                axes[1].scatter(durations, closed_trades['pnl'], 
                               c=['#00ff88' if p > 0 else '#ff4444' for p in closed_trades['pnl']],
                               alpha=0.6, s=50)
                axes[1].axhline(y=0, color='white', linestyle='-', alpha=0.3)
                axes[1].set_title('Duration vs PnL', color='white')
                axes[1].set_xlabel('Duration (hours)', color='white')
                axes[1].set_ylabel('PnL ($)', color='white')
                axes[1].tick_params(colors='white')
                axes[1].grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(output_path / 'trade_duration.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_html_report(self, output_dir: str = "storage/analysis"):
        """Generate an HTML report with all visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # First generate visualizations
        self.create_visualizations(output_dir)
        
        summary = self.generate_summary()
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartCrypto Analysis Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e1a;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(90deg, #00ff88, #4488ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header .badge {{
            background: rgba(255,255,255,0.05);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }}
        .card h3 {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: rgba(255,255,255,0.4);
            margin-bottom: 8px;
        }}
        .card .number {{
            font-size: 28px;
            font-weight: 700;
            color: #fff;
        }}
        .card .sub {{
            font-size: 13px;
            color: rgba(255,255,255,0.5);
            margin-top: 4px;
        }}
        .card .positive {{ color: #00ff88; }}
        .card .negative {{ color: #ff4444; }}
        .card .neutral {{ color: #ffaa00; }}
        .row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
        }}
        .chart-container h3 {{
            text-align: left;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: rgba(255,255,255,0.4);
            margin-bottom: 15px;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        @media (max-width: 768px) {{
            .row {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 SmartCrypto Analysis Dashboard</h1>
            <span class="badge">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📡 Signals</h3>
                <div class="number">{summary['signals']['total']}</div>
                <div class="sub">Open: {summary['signals']['open']} | Closed: {summary['signals']['closed']}</div>
                <div class="sub">Wins: <span class="positive">{summary['signals']['wins']}</span> | Losses: <span class="negative">{summary['signals']['losses']}</span></div>
            </div>
            <div class="card">
                <h3>💰 Trades</h3>
                <div class="number">{summary['trades']['total']}</div>
                <div class="sub">Open: {summary['trades']['open']} | Closed: {summary['trades']['closed']}</div>
                <div class="sub">Wins: <span class="positive">{summary['trades']['wins']}</span> | Losses: <span class="negative">{summary['trades']['losses']}</span></div>
            </div>
            <div class="card">
                <h3>📈 Win Rate</h3>
                <div class="number {'positive' if summary['trades']['win_rate'] > 0.5 else 'negative'}">{summary['trades']['win_rate']:.1%}</div>
                <div class="sub">{summary['trades']['wins']} wins out of {summary['trades']['closed']} closed trades</div>
            </div>
            <div class="card">
                <h3>💰 Total PnL</h3>
                <div class="number {'positive' if summary['trades']['total_pnl'] > 0 else 'negative'}">${summary['trades']['total_pnl']:.4f}</div>
                <div class="sub">Average: ${summary['trades']['avg_pnl']:.4f} per trade</div>
            </div>
        </div>
        
        <div class="row">
            <div class="chart-container">
                <h3>📈 Cumulative PnL Over Time</h3>
                <img src="cumulative_pnl.png" alt="Cumulative PnL" onerror="this.style.display='none'">
            </div>
            <div class="chart-container">
                <h3>📊 Win/Loss Distribution</h3>
                <img src="win_loss_distribution.png" alt="Win/Loss Distribution" onerror="this.style.display='none'">
            </div>
        </div>
        
        <div class="row">
            <div class="chart-container">
                <h3>💰 Trade Performance</h3>
                <img src="trade_performance.png" alt="Trade Performance" onerror="this.style.display='none'">
            </div>
            <div class="chart-container">
                <h3>📊 Symbol Performance</h3>
                <img src="symbol_performance.png" alt="Symbol Performance" onerror="this.style.display='none'">
            </div>
        </div>
        
        <div class="row">
            <div class="chart-container">
                <h3>🎨 Signal Patterns</h3>
                <img src="pattern_distribution.png" alt="Pattern Distribution" onerror="this.style.display='none'">
            </div>
            <div class="chart-container">
                <h3>⏰ Trade Duration</h3>
                <img src="trade_duration.png" alt="Trade Duration" onerror="this.style.display='none'">
            </div>
        </div>
        
        <div class="card" style="margin-top: 20px;">
            <h3>📋 Top Performing Symbols</h3>
            <table style="width:100%; border-collapse: collapse; margin-top: 10px;">
                <thead>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <th style="text-align:left; padding: 8px; color: rgba(255,255,255,0.5);">Symbol</th>
                        <th style="text-align:center; padding: 8px; color: rgba(255,255,255,0.5);">Trades</th>
                        <th style="text-align:center; padding: 8px; color: rgba(255,255,255,0.5);">Wins</th>
                        <th style="text-align:center; padding: 8px; color: rgba(255,255,255,0.5);">Losses</th>
                        <th style="text-align:center; padding: 8px; color: rgba(255,255,255,0.5);">Win Rate</th>
                        <th style="text-align:right; padding: 8px; color: rgba(255,255,255,0.5);">Total PnL</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add symbol rows
        for symbol, stats in sorted(
            summary['trades']['symbols'].items(),
            key=lambda x: x[1]['pnl'],
            reverse=True
        ):
            if stats['trades'] > 0:
                win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
                pnl_class = 'positive' if stats['pnl'] > 0 else 'negative'
                html += f"""
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                        <td style="padding: 8px;"><strong>{symbol}</strong></td>
                        <td style="text-align:center; padding: 8px;">{stats['trades']}</td>
                        <td style="text-align:center; padding: 8px; color: #00ff88;">{stats['wins']}</td>
                        <td style="text-align:center; padding: 8px; color: #ff4444;">{stats['losses']}</td>
                        <td style="text-align:center; padding: 8px;">{win_rate:.1%}</td>
                        <td style="text-align:right; padding: 8px;" class="{pnl_class}">${stats['pnl']:.4f}</td>
                    </tr>
                """
        
        html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        html_path = output_path / 'dashboard.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"\n🌐 HTML Dashboard saved to: {html_path}")
        return html_path


def main():
    """Run the analysis dashboard"""
    print("=" * 70)
    print("📊 SMARTCRYPTO ANALYSIS DASHBOARD")
    print("=" * 70)
    
    # Create analyzer
    analyzer = AnalysisDashboard()
    
    # Print summary
    analyzer.print_summary()
    
    # Generate visualizations and HTML report
    analyzer.generate_html_report()
    
    print("\n" + "=" * 70)
    print("✅ Analysis Complete!")
    print("📁 Open storage/analysis/dashboard.html in your browser")
    print("=" * 70)


if __name__ == "__main__":
    main()