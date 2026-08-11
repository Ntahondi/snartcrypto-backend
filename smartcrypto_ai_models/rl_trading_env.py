# smartcrypto_ai_models/rl_trading_env.py

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RLMarketEnvironment:
    """
    Custom Market Trading Environment for Deep Reinforcement Learning.
    State Space: 24-hour sequence matrix + account state (position, capital, PnL).
    Action Space: Continuous value in [-1.0, +1.0] (-1=100% Short, 0=Cash, +1=100% Long).
    Reward Function: Risk-adjusted PnL minus transaction fees and drawdown penalties.
    """

    def __init__(self, df_features: pd.DataFrame, feature_cols: list, initial_balance: float = 10000.0,
                 commission_rate: float = 0.001, window_size: int = 24):
        self.df = df_features.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.window_size = window_size

        self.n_steps = len(self.df) - self.window_size - 1
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0  # -1.0 to +1.0
        self.entry_price = 0.0
        self.peak_balance = self.initial_balance
        self.trade_history = []

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Returns observation state vector: 24h market matrix + account state"""
        window = self.df.iloc[self.current_step - self.window_size:self.current_step][self.feature_cols].values
        
        # Account state: [current_position, normalized_balance, drawdown_pct]
        drawdown = (self.peak_balance - self.balance) / (self.peak_balance + 1e-8)
        account_state = np.array([self.position, self.balance / self.initial_balance, drawdown], dtype=np.float32)
        
        # Flatten matrix + account state into single observation vector
        obs = np.concatenate([window.flatten(), account_state])
        return obs

    def step(self, action: float) -> tuple:
        """
        Executes one step in the market environment:
        action: float between -1.0 (Max Short) and +1.0 (Max Long)
        """
        action = float(np.clip(action, -1.0, 1.0))
        
        current_price = float(self.df.iloc[self.current_step]['close'])
        next_price = float(self.df.iloc[self.current_step + 1]['close'])
        price_return = (next_price - current_price) / current_price

        # 1. Calculate Position Change & Commission
        position_change = abs(action - self.position)
        commission = position_change * self.commission_rate * self.balance

        # 2. Calculate Trade PnL
        step_pnl = self.position * price_return * self.balance - commission
        self.balance += step_pnl

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # 3. Calculate Reward (Sharpe-style PnL - Drawdown Penalty)
        drawdown_penalty = max(0.0, (self.peak_balance - self.balance) / self.peak_balance) * 0.1
        reward = (step_pnl / self.initial_balance) - drawdown_penalty

        # Update position state
        self.position = action
        self.current_step += 1
        done = (self.current_step >= self.n_steps) or (self.balance <= self.initial_balance * 0.2)

        next_obs = self._get_observation() if not done else np.zeros_like(self._get_observation())
        
        info = {
            'balance': self.balance,
            'pnl': step_pnl,
            'position': self.position,
            'step': self.current_step
        }

        return next_obs, reward, done, info