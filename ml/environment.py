import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


class CryptoTradingEnv(gym.Env):
    """
    Custom Gymnasium RL environment for crypto trading.

    State  : technical indicators + portfolio state
    Actions: 0=Hold | 1=Buy25% | 2=Buy50% | 3=Buy100%
             4=Sell25% | 5=Sell50% | 6=Sell100%
    Reward : log-return minus drawdown penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, feature_cols: list, config):
        super().__init__()
        self.df = df.dropna().reset_index(drop=True)
        self.feature_cols = feature_cols
        self.config = config

        n = len(feature_cols) + 4   # features + 4 portfolio state vars
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)
        self.action_space = spaces.Discrete(7)
        self.reset()

    _SIM_BALANCE = 10_000.0   # fixed sim balance for RL training (live mode has INITIAL_BALANCE=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = self.config.LOOKBACK_WINDOW
        self.balance = self._SIM_BALANCE
        self.position = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.peak_value = self.balance
        return self._obs(), {}

    def _price(self):
        return float(self.df.iloc[self.step_idx]["close"])

    def _portfolio_value(self):
        return self.balance + self.position * self._price()

    def _obs(self) -> np.ndarray:
        row = self.df.iloc[self.step_idx]
        feats = row[self.feature_cols].values.astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)

        price = self._price()
        pv = self._portfolio_value()
        pos_pct = (self.position * price) / max(pv, 1e-8)
        upnl = (price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0.0
        dd = (pv - self.peak_value) / max(self.peak_value, 1e-8)

        state = np.array([pos_pct, upnl, dd, self.total_trades / 100.0], dtype=np.float32)
        return np.concatenate([feats, state])

    def step(self, action: int):
        price = self._price()
        fee = self.config.TRADE_FEE
        prev_pv = self._portfolio_value()

        if action == 1:
            self._buy(price, 0.25, fee)
        elif action == 2:
            self._buy(price, 0.50, fee)
        elif action == 3:
            self._buy(price, 1.00, fee)
        elif action == 4:
            self._sell(price, 0.25, fee)
        elif action == 5:
            self._sell(price, 0.50, fee)
        elif action == 6:
            self._sell(price, 1.00, fee)

        self.step_idx += 1
        curr_pv = self._portfolio_value()
        self.peak_value = max(self.peak_value, curr_pv)

        log_ret = np.log(curr_pv / max(prev_pv, 1e-8))
        dd_pen = min(0.0, (curr_pv - self.peak_value) / max(self.peak_value, 1e-8))
        reward = float(log_ret + 0.1 * dd_pen)

        done = (
            self.step_idx >= len(self.df) - 1
            or curr_pv < self._SIM_BALANCE * (1 - self.config.MAX_DRAWDOWN_PCT)
        )
        return self._obs(), reward, done, False, {"portfolio_value": curr_pv}

    def _buy(self, price: float, frac: float, fee: float):
        spend = self.balance * frac
        if spend < 10:
            return
        net = spend * (1 - fee)
        amount = net / price
        self.balance -= spend
        if self.position > 0:
            total = self.position + amount
            self.entry_price = (self.entry_price * self.position + price * amount) / total
        else:
            self.entry_price = price
        self.position += amount
        self.total_trades += 1

    def _sell(self, price: float, frac: float, fee: float):
        sell = self.position * frac
        if sell <= 0:
            return
        gross = sell * price
        self.balance += gross * (1 - fee)
        self.position -= sell
        if self.position < 1e-8:
            self.position = 0.0
            self.entry_price = 0.0
        self.total_trades += 1

    def render(self):
        logger.info(f"Step {self.step_idx}: value=${self._portfolio_value():.2f} pos={self.position:.6f}")
