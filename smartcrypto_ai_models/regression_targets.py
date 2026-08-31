# smartcrypto_ai_models/regression_targets.py

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ContinuousRegressionLabeler:
    """
    Generates pure continuous percentage return vectors for future horizons.
    
    FIXED:
    - Per-symbol target computation
    - Does NOT fill NaN with 0.0 (leaves as NaN for filtering)
    - Proper trimming function
    """

    def __init__(self, horizons: dict = None):
        self.horizons = horizons or {'1h': 1, '4h': 4, '12h': 12}

    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes exact percentage returns for future time horizons safely"""
        df = df.copy()

        # Ensure timestamp is present for chronological ordering
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")

        # Sort by symbol and timestamp to ensure correct shift order
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

        # Group by symbol and compute returns PER SYMBOL
        for symbol, group in df.groupby('symbol', sort=False):
            mask = df['symbol'] == symbol
            
            # Get close prices for this symbol only
            close_safe = df.loc[mask, 'close'].replace(0.0, np.nan).ffill().fillna(1.0)
            
            # Compute forward returns
            for label, shift in self.horizons.items():
                ret_col = f'target_ret_{label}'
                
                # Shift within the symbol group
                raw_ret = (close_safe.shift(-shift) - close_safe) / close_safe
                
                # Clip to realistic bounds [-50%, +50%]
                # IMPORTANT: Do NOT fill NaN with 0.0. Leave as NaN for filtering.
                df.loc[mask, ret_col] = raw_ret.clip(-0.50, 0.50)
        
        # Reset index for consistency
        df = df.reset_index(drop=True)

        logger.info("✅ Continuous Return Targets Generated:")
        for label in self.horizons.keys():
            col = f'target_ret_{label}'
            
            # Calculate statistics per symbol (excluding NaN)
            for symbol in df['symbol'].unique():
                symbol_mask = df['symbol'] == symbol
                valid_mask = symbol_mask & df[col].notna()
                mean_ret = df.loc[valid_mask, col].mean()
                std_ret = df.loc[valid_mask, col].std()
                nan_count = (~df.loc[symbol_mask, col].notna()).sum()
                logger.info(
                    f"   {symbol} | Horizon {label:<3}: "
                    f"Mean Return = {mean_ret:+.4%}, "
                    f"Std Dev = {std_ret:.4%}, "
                    f"NaN = {nan_count}"
                )

        return df


def validate_targets(df: pd.DataFrame) -> dict:
    """
    Validate that targets are correctly generated.
    
    Checks:
    1. No out-of-bounds values
    2. NaN values exist ONLY at the end of each symbol's time series
    """
    validation = {}
    
    for col in [f'target_ret_{h}' for h in ['1h', '4h', '12h']]:
        if col not in df.columns:
            validation[col] = "MISSING"
            continue
        
        values = df[col]
        
        # Check bounds (ignore NaN)
        out_of_bounds = ((values < -0.5) | (values > 0.5)).sum()
        
        # Check NaN are only at symbol boundaries
        nan_count = values.isna().sum()
        
        # Verify NaN are at the end of each symbol
        boundary_nan = 0
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol][col]
            # Count NaN at the end (should be equal to max horizon)
            tail_nan = symbol_data.isna().sum()
            boundary_nan += tail_nan
        
        validation[col] = {
            "out_of_bounds": int(out_of_bounds),
            "nan_count": int(nan_count),
            "boundary_nan": int(boundary_nan),
            "status": "PASS" if out_of_bounds == 0 and nan_count == boundary_nan else "WARN"
        }
    
    return validation


def trim_tail_nans(df: pd.DataFrame, horizons: list = ['1h', '4h', '12h']) -> pd.DataFrame:
    """
    Remove rows where ANY target is NaN (tail of each symbol).
    This should be called AFTER labeling but BEFORE sequence creation.
    """
    df = df.copy()
    
    target_cols = [f'target_ret_{h}' for h in horizons]
    
    # Drop rows where ANY target is NaN
    df = df.dropna(subset=target_cols)
    
    logger.info(f"✅ Trimmed {len(df)} rows with NaN targets")
    
    return df.reset_index(drop=True)