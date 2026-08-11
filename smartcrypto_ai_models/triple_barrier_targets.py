# smartcrypto_ai_models/triple_barrier_targets.py

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TripleBarrierTargetLabeler:
    """
    Simulates future price paths to eliminate fixed-hour timeout blindness.
    Labels trades based on whether Take Profit or Stop Loss barrier is touched first,
    and records exact hours required to reach Take Profit.
    """

    def __init__(self, tp_atr_mult: float = 2.0, sl_atr_mult: float = 2.0, max_holding_bars: int = 12):
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.max_holding_bars = max_holding_bars

    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label dataset with path-dependent direction and time-to-profit values"""
        df = df.copy()

        n_rows = len(df)
        target_directions = np.ones(n_rows, dtype=int)  # Default 1 = HOLD
        time_to_profit = np.full(n_rows, self.max_holding_bars, dtype=float)

        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        # Ensure atr14 exists
        if 'atr14' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr14'] = tr.rolling(14).mean().ffill().fillna(1e-4)

        atr_values = df['atr14'].values

        logger.info(f"🛣️ Scanning {n_rows:,} candle paths for Triple Barrier Labeling...")

        for i in range(n_rows - self.max_holding_bars):
            price_entry = close_prices[i]
            atr = atr_values[i]

            if price_entry <= 0 or np.isnan(atr) or atr <= 0:
                continue

            tp_barrier = price_entry + (self.tp_atr_mult * atr)
            sl_barrier = price_entry - (self.sl_atr_mult * atr)

            for h in range(1, self.max_holding_bars + 1):
                future_high = high_prices[i + h]
                future_low = low_prices[i + h]

                hit_tp = future_high >= tp_barrier
                hit_sl = future_low <= sl_barrier

                if hit_tp and not hit_sl:
                    target_directions[i] = 2  # BUY
                    time_to_profit[i] = float(h)
                    break
                elif hit_sl and not hit_tp:
                    target_directions[i] = 0  # SELL
                    time_to_profit[i] = float(h)
                    break
                elif hit_tp and hit_sl:
                    target_directions[i] = 1  # Volatile Wicks = HOLD
                    time_to_profit[i] = float(h)
                    break

        df['target_direction'] = target_directions
        df['time_to_profit_hours'] = time_to_profit

        buy_count = np.sum(target_directions == 2)
        hold_count = np.sum(target_directions == 1)
        sell_count = np.sum(target_directions == 0)

        logger.info(f"✅ Triple Barrier Path Scanning Complete:")
        logger.info(f"   BUY Wins: {buy_count:,} ({buy_count/n_rows:.1%}) | "
                    f"SELL Wins: {sell_count:,} ({sell_count/n_rows:.1%}) | "
                    f"HOLD/Stagnant: {hold_count:,} ({hold_count/n_rows:.1%})")

        return df

    # Method Aliases for backward compatibility
    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.label_dataset(df)

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.label_dataset(df)