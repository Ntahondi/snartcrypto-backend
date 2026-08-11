#!/usr/bin/env python3
"""
SmartCrypto Monitoring Script
Real-time monitoring of trading performance, system health, and alerts
"""

import os
import sys
import time
import json
import logging
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_settings
from src.services.history_manager import HistoryManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.settings = get_settings()
        self.history_manager = HistoryManager()
        self.alerts = []
        self.metrics = {}
        self.base_url = f"http://localhost:{self.settings.API_PORT if hasattr(self.settings, 'API_PORT') else 8000}"
        
    def check_system_health(self) -> Dict:
        """Check overall system health"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Check if API is running
        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=5)
            health['checks']['api'] = 'healthy' if response.status_code == 200 else 'unhealthy'
        except:
            health['checks']['api'] = 'unreachable'
            health['status'] = 'degraded'
        
        # Check storage
        storage_path = Path("storage")
        if storage_path.exists():
            health['checks']['storage'] = 'healthy'
        else:
            health['checks']['storage'] = 'missing'
            health['status'] = 'degraded'
        
        # Check if models exist
        model_path = Path(self.settings.MODEL_PATH)
        if model_path.exists():
            health['checks']['model'] = 'healthy'
        else:
            health['checks']['model'] = 'missing'
            health['status'] = 'critical'
        
        # Check memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            health['checks']['memory'] = {
                'status': 'healthy' if memory.percent < 80 else 'warning',
                'percent': memory.percent
            }
            if memory.percent > 80:
                health['status'] = 'degraded'
        except:
            pass
        
        return health
    
    def calculate_performance_metrics(self, hours: int = 24) -> Dict:
        """Calculate trading performance metrics"""
        signals = self.history_manager.get_recent_signals(hours=hours)
        
        if not signals:
            return {'status': 'no_data'}
        
        total_trades = len(signals)
        winning_trades = [s for s in signals if s.get('outcome') == 'WIN']
        losing_trades = [s for s in signals if s.get('outcome') == 'LOSS']
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(s.get('pnl_percentage', 0) for s in signals)
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl_percent': total_pnl,
            'avg_win': np.mean([s.get('pnl_percentage', 0) for s in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([s.get('pnl_percentage', 0) for s in losing_trades]) if losing_trades else 0,
            'profit_factor': abs(sum(s.get('pnl_percentage', 0) for s in winning_trades) / 
                                sum(s.get('pnl_percentage', 0) for s in losing_trades)) if losing_trades else float('inf'),
            'timeframe': f"Last {hours}h"
        }
        
        return metrics
    
    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check if any alerts need to be triggered"""
        alerts = []
        
        if metrics.get('win_rate', 0) < 0.4:
            alerts.append({
                'level': 'WARNING',
                'message': f"Low win rate: {metrics['win_rate']:.1%}",
                'timestamp': datetime.now().isoformat()
            })
        
        if metrics.get('total_pnl_percent', 0) < -5:
            alerts.append({
                'level': 'CRITICAL',
                'message': f"Large drawdown: {metrics['total_pnl_percent']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def generate_report(self) -> str:
        """Generate a performance report"""
        health = self.check_system_health()
        metrics = self.calculate_performance_metrics()
        alerts = self.check_alerts(metrics)
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    SMARTCRYPTO MONITORING REPORT             ║
╚══════════════════════════════════════════════════════════════╝

📊 SYSTEM HEALTH:
    Status: {health['status'].upper()}
    API: {health['checks'].get('api', 'unknown')}
    Model: {health['checks'].get('model', 'unknown')}
    Storage: {health['checks'].get('storage', 'unknown')}
"""
        
        if metrics and metrics.get('status') != 'no_data':
            report += f"""
📈 PERFORMANCE (Last 24h):
    Total Trades: {metrics.get('total_trades', 0)}
    Win Rate: {metrics.get('win_rate', 0):.1%}
    Total PnL: {metrics.get('total_pnl_percent', 0):.2f}%
    Avg Win: {metrics.get('avg_win', 0):.2f}%
    Avg Loss: {metrics.get('avg_loss', 0):.2f}%
    Profit Factor: {metrics.get('profit_factor', 0):.2f}
"""
        
        if alerts:
            report += """
⚠️ ACTIVE ALERTS:
"""
            for alert in alerts:
                report += f"    [{alert['level']}] {alert['message']}\n"
        else:
            report += """
✅ No active alerts - All systems normal
"""
        
        report += f"""
📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def send_alert(self, alert: Dict):
        """Send alert (console, email, Slack, etc.)"""
        if alert['level'] == 'CRITICAL':
            logger.critical(f"🔴 ALERT: {alert['message']}")
        elif alert['level'] == 'WARNING':
            logger.warning(f"🟡 WARNING: {alert['message']}")
        else:
            logger.info(f"🔵 INFO: {alert['message']}")

    def run_monitoring_loop(self):
        """Run continuous monitoring"""
        logger.info("🔄 Starting monitoring loop...")
        
        while True:
            try:
                # Check system health
                health = self.check_system_health()
                
                # Calculate performance
                metrics = self.calculate_performance_metrics()
                
                # Check alerts
                alerts = self.check_alerts(metrics)
                for alert in alerts:
                    self.send_alert(alert)
                
                # Generate and log report
                report = self.generate_report()
                logger.info("\n" + report)
                
                # Save metrics to file
                self.save_metrics(metrics)
                
                # Wait before next check
                time.sleep(300)  # 5 minutes
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring stopped")
                break
            except Exception as e:
                logger.error(f"❌ Monitor error: {e}")
                time.sleep(60)
    
    def save_metrics(self, metrics: Dict):
        """Save metrics to file for historical tracking"""
        metrics_file = Path("storage/metrics_history.json")
        
        existing = []
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                try:
                    existing = json.load(f)
                except:
                    pass
        
        existing.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        # Keep last 1000 entries
        if len(existing) > 1000:
            existing = existing[-1000:]
        
        with open(metrics_file, 'w') as f:
            json.dump(existing, f, indent=2)

def main():
    monitor = PerformanceMonitor()
    
    # Run one-time report
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        report = monitor.generate_report()
        print(report)
        return
    
    # Run continuous monitoring
    monitor.run_monitoring_loop()

if __name__ == "__main__":
    main()