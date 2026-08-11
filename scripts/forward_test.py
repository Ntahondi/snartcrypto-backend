#!/usr/bin/env python
"""
Forward Test - Simulate Live Trading with Real Data
"""
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ForwardTester:
    """
    Forward test: Simulate how the model would perform going forward
    Uses realistic execution and tracks performance in real-time
    """
    
    def __init__(
        self,
        model,
        scaler,
        transformer,
        feature_cols,
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005,
        confidence_threshold=0.5,
        position_size_pct=0.05,
    ):
        self.model = model
        self.scaler = scaler
        self.transformer = transformer
        self.feature_cols = feature_cols
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.confidence_threshold = confidence_threshold
        self.position_size_pct = position_size_pct
        
    def run_forward_test(self, df, start_date, end_date=None):
        """
        Run forward test from start_date to end_date
        Simulates live trading on a rolling window
        """
        if end_date is None:
            end_date = df['timestamp'].max()
            
        df_test = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
        logger.info(f"Forward test from {start_date} to {end_date}")
        logger.info(f"Test data: {len(df_test)} candles")
        
        # Process data
        engineer = FeatureEngineer()
        all_symbols = []
        for symbol in df_test["symbol"].unique():
            symbol_data = df_test[df_test["symbol"] == symbol].copy()
            symbol_data = engineer.add_technical_indicators(symbol_data)
            all_symbols.append(symbol_data)
        df_enhanced = pd.concat(all_symbols, ignore_index=True)
        
        # Ensure all features exist
        for col in self.feature_cols:
            if col not in df_enhanced.columns:
                df_enhanced[col] = 0.0
        
        symbols = df_enhanced['symbol'].unique()
        results = {}
        
        for symbol in symbols:
            logger.info(f"\n📊 Testing {symbol}")
            symbol_data = df_enhanced[df_enhanced['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
            result = self._run_single_forward_test(symbol_data, symbol)
            results[symbol] = result
            self._print_results(result, symbol)
        
        return results
    
    def _run_single_forward_test(self, df, symbol):
        """Run forward test for a single symbol with realistic execution"""
        
        portfolio = {
            "cash": self.initial_capital,
            "position_shares": 0,
            "entry_price": 0,
            "entry_bar": 0,
            "history": [],
            "transactions": [],
            "dates": [],
            "drawdowns": [],
        }
        
        # Track performance metrics
        returns = []
        
        # Prepare features
        X_mat = df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        
        try:
            X_scaled = self.scaler.transform(X_mat)
            X_trans = self.transformer.transform(X_scaled)
        except Exception as e:
            logger.error(f"❌ Feature transformation error: {e}")
            return self._empty_metrics()
        
        # Batch predictions
        logger.info(f"Running predictions on {len(df)} candles...")
        preds = self.model.predict(X_trans, batch_size=1024, verbose=0)
        
        dir_1h_probs = preds[0]
        confidence_scores = preds[3].flatten()
        
        # Track signal distribution
        actions = np.argmax(dir_1h_probs, axis=1)
        logger.info(f"📊 Signals: BUY={np.sum(actions==2)}, SELL={np.sum(actions==0)}, HOLD={np.sum(actions==1)}")
        
        # Simulate trading
        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            direction_probs = dir_1h_probs[i]
            confidence = float(confidence_scores[i])
            action = np.argmax(direction_probs)
            
            exec_price = next_row["open"]
            timestamp = next_row["timestamp"]
            
            # Force exit after 48 hours
            if portfolio["position_shares"] > 0:
                bars_held = i - portfolio["entry_bar"]
                if bars_held >= 48:
                    action = 0  # Force SELL
            
            # BUY
            if (
                action == 2
                and portfolio["position_shares"] == 0
                and confidence >= self.confidence_threshold
            ):
                fill_price = exec_price * (1 + self.slippage)
                cost = portfolio["cash"] * self.position_size_pct * (1 - self.commission)
                if cost > 0:
                    portfolio["position_shares"] = cost / fill_price
                    portfolio["cash"] -= cost
                    portfolio["entry_price"] = fill_price
                    portfolio["entry_bar"] = i
                    portfolio["transactions"].append({
                        "date": timestamp,
                        "action": "BUY",
                        "price": fill_price,
                        "shares": portfolio["position_shares"],
                        "confidence": confidence,
                    })
            
            # SELL
            elif (
                action == 0
                and portfolio["position_shares"] > 0
                and confidence >= self.confidence_threshold
            ):
                fill_price = exec_price * (1 - self.slippage)
                revenue = portfolio["position_shares"] * fill_price * (1 - self.commission)
                pnl = revenue - (portfolio["position_shares"] * portfolio["entry_price"])
                pnl_pct = pnl / (portfolio["position_shares"] * portfolio["entry_price"]) if portfolio["entry_price"] > 0 else 0
                
                portfolio["cash"] += revenue
                portfolio["transactions"].append({
                    "date": timestamp,
                    "action": "SELL",
                    "price": fill_price,
                    "shares": portfolio["position_shares"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "confidence": confidence,
                })
                portfolio["position_shares"] = 0
                portfolio["entry_price"] = 0
                portfolio["entry_bar"] = 0
            
            # Portfolio valuation
            current_value = portfolio["cash"] + (portfolio["position_shares"] * exec_price)
            portfolio["history"].append(current_value)
            portfolio["dates"].append(timestamp)
            
            # Track returns
            if i > 0 and portfolio["history"][-2] > 0:
                daily_return = (current_value - portfolio["history"][-2]) / portfolio["history"][-2]
                returns.append(daily_return)
        
        # Calculate metrics
        metrics = self._calculate_metrics(portfolio, returns)
        metrics["symbol"] = symbol
        metrics["transactions"] = portfolio["transactions"]
        metrics["history"] = portfolio["history"]
        metrics["dates"] = portfolio["dates"]
        
        return metrics
    
    def _calculate_metrics(self, portfolio, returns):
        values = portfolio["history"]
        transactions = portfolio["transactions"]
        
        if len(values) == 0:
            return self._empty_metrics()
        
        initial_value = self.initial_capital
        final_value = values[-1]
        total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0
        
        returns = np.array(returns)
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)
        else:
            sharpe = 0
        
        max_drawdown = 0
        peak = values[0] if values else 0
        for val in values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, dd)
        
        sell_transactions = [t for t in transactions if t["action"] == "SELL"]
        winning_trades = [t for t in sell_transactions if t.get("pnl", 0) > 0]
        win_rate = len(winning_trades) / len(sell_transactions) if sell_transactions else 0
        
        avg_pnl = np.mean([t.get("pnl", 0) for t in sell_transactions]) if sell_transactions else 0
        avg_pnl_pct = np.mean([t.get("pnl_pct", 0) for t in sell_transactions]) if sell_transactions else 0
        
        total_profit = sum([t.get("pnl", 0) for t in sell_transactions if t.get("pnl", 0) > 0])
        total_loss = abs(sum([t.get("pnl", 0) for t in sell_transactions if t.get("pnl", 0) < 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
        
        return {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "avg_pnl": avg_pnl,
            "avg_pnl_pct": avg_pnl_pct * 100,
            "profit_factor": profit_factor,
            "total_trades": len(transactions),
            "buy_trades": len([t for t in transactions if t["action"] == "BUY"]),
            "sell_trades": len(sell_transactions),
            "winning_trades": len(winning_trades),
            "losing_trades": len(sell_transactions) - len(winning_trades),
        }
    
    def _empty_metrics(self):
        return {
            "initial_value": self.initial_capital,
            "final_value": self.initial_capital,
            "total_return": 0,
            "total_return_pct": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "max_drawdown_pct": 0,
            "win_rate": 0,
            "win_rate_pct": 0,
            "avg_pnl": 0,
            "avg_pnl_pct": 0,
            "profit_factor": 0,
            "total_trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        }
    
    def _print_results(self, metrics, symbol):
        print(f"\n📊 {symbol} - FORWARD TEST RESULTS")
        print(f"{'='*50}")
        print(f"Initial Capital:      ${metrics['initial_value']:,.2f}")
        print(f"Final Value:          ${metrics['final_value']:,.2f}")
        print(f"Total Return:         {metrics['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:         {metrics['max_drawdown_pct']:.2f}%")
        print(f"Win Rate:             {metrics['win_rate_pct']:.1f}%")
        print(f"Profit Factor:        {metrics['profit_factor']:.2f}")
        print(f"Total Trades:         {metrics['total_trades']}")
        print(f"  Winning Trades:     {metrics['winning_trades']}")
        print(f"  Losing Trades:      {metrics['losing_trades']}")
        print(f"Avg PnL:              ${metrics['avg_pnl']:.2f} ({metrics['avg_pnl_pct']:.2f}%)")
    
    def plot_results(self, results):
        """Plot forward test results"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("AI Trading Strategy - Forward Test Results", fontsize=16)
        
        # Portfolio Value
        ax = axes[0, 0]
        for symbol, result in results.items():
            if result.get("history") and len(result["history"]) > 1:
                ax.plot(result["dates"], result["history"], label=symbol, linewidth=2)
        ax.legend()
        ax.set_title("Portfolio Value Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True, alpha=0.3)
        
        # Trade Returns
        ax = axes[0, 1]
        all_returns = []
        for symbol, result in results.items():
            if result.get("transactions"):
                pnls = [t.get("pnl_pct", 0) * 100 for t in result["transactions"] if t["action"] == "SELL"]
                all_returns.extend(pnls)
        if all_returns:
            ax.hist(all_returns, bins=30, alpha=0.7, edgecolor="black")
            ax.axvline(np.mean(all_returns), color="red", linestyle="--", label=f"Mean: {np.mean(all_returns):.2f}%")
            ax.legend()
        ax.set_title("Trade Returns Distribution")
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # Drawdown
        ax = axes[1, 0]
        for symbol, result in results.items():
            if result.get("history") and len(result["history"]) > 1:
                values = result["history"]
                drawdowns = []
                peak = values[0]
                for val in values:
                    if val > peak:
                        peak = val
                    dd = (peak - val) / peak if peak > 0 else 0
                    drawdowns.append(dd * 100)
                ax.plot(result["dates"], drawdowns, label=symbol, linewidth=2)
        ax.legend()
        ax.set_title("Portfolio Drawdown (%)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        
        # Cumulative Returns
        ax = axes[1, 1]
        for symbol, result in results.items():
            if result.get("history") and len(result["history"]) > 1:
                values = result["history"]
                cum_returns = [(v / result["initial_value"] - 1) * 100 for v in values if result["initial_value"] > 0]
                if cum_returns:
                    ax.plot(result["dates"][:len(cum_returns)], cum_returns, label=f"{symbol} (AI)", linewidth=2)
        ax.legend()
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.set_title("Cumulative Returns (%)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (%)")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    logger.info("🚀 Starting Forward Test")
    
    # Load model and artifacts
    model_path = config.MODELS_PATH / "smart_trader_ai_final.keras"
    scaler_path = config.MODELS_PATH / "robust_scaler.joblib"
    transformer_path = config.MODELS_PATH / "power_transformer.joblib"
    features_path = config.MODELS_PATH / "feature_columns.joblib"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    model = tf.keras.models.load_model(str(model_path))
    scaler = joblib.load(scaler_path)
    transformer = joblib.load(transformer_path)
    feature_cols = joblib.load(features_path)
    
    logger.info(f"✅ Loaded model expecting {len(feature_cols)} features")
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    
    # Run forward test on LAST 30 days of data (real forward test)
    end_date = df['timestamp'].max()
    start_date = end_date - timedelta(days=30)
    
    # Forward tester with realistic parameters
    forward_tester = ForwardTester(
        model=model,
        scaler=scaler,
        transformer=transformer,
        feature_cols=feature_cols,
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005,
        confidence_threshold=0.5,
        position_size_pct=0.05,
    )
    
    results = forward_tester.run_forward_test(df, start_date, end_date)
    
    # Plot results
    if any(r.get("total_trades", 0) > 0 for r in results.values()):
        forward_tester.plot_results(results)
    
    print("\n" + "="*60)
    print("📊 FORWARD TEST SUMMARY")
    print("="*60)
    
    summary_df = pd.DataFrame([{
        "Symbol": symbol,
        "Return": f"{r['total_return_pct']:.2f}%",
        "Sharpe": f"{r['sharpe_ratio']:.2f}",
        "Win Rate": f"{r['win_rate_pct']:.1f}%",
        "Max DD": f"{r['max_drawdown_pct']:.2f}%",
        "Trades": r["total_trades"],
        "Profit Factor": f"{r['profit_factor']:.2f}",
    } for symbol, r in results.items()])
    
    print(summary_df.to_string(index=False))
    logger.info("✅ Forward test completed!")


if __name__ == "__main__":
    main()