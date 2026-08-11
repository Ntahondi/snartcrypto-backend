# smartcrypto_ai_models/regression_targets.py

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ContinuousRegressionLabeler:
    """
    Generates pure continuous percentage return vectors for future horizons.
    Clipped safely to prevent division-by-zero outlier spikes.
    """

    def __init__(self, horizons: dict = None):
        self.horizons = horizons or {'1h': 1, '4h': 4, '12h': 12}

    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes exact percentage returns for future time horizons safely"""
        df = df.copy()

        close_safe = df['close'].replace(0.0, np.nan).ffill().fillna(1.0)

        for label, shift in self.horizons.items():
            ret_col = f'target_ret_{label}'
            raw_ret = (close_safe.shift(-shift) - close_safe) / close_safe
            
            # Clip returns to realistic bounds [-50%, +50%]
            df[ret_col] = raw_ret.clip(-0.50, 0.50).fillna(0.0)

        logger.info("✅ Continuous Return Targets Generated:")
        for label in self.horizons.keys():
            col = f'target_ret_{label}'
            mean_ret = df[col].mean()
            std_ret = df[col].std()
            logger.info(f"   Horizon {label:<3}: Mean Return = {mean_ret:+.4%}, Std Dev = {std_ret:.4%}")

        return df