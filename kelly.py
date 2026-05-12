"""
Kelly Criterion Position Sizer.

Mathematically optimal position sizing based on model edge.
Maximizes long-term growth while minimizing ruin risk.

Formula: f* = (bp - q) / b
  b = odds (reward/risk ratio)
  p = probability of winning
  q = 1 - p

We use a fractional Kelly (25%) to be more conservative.
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class KellyCriterion:
    def __init__(self, config, fraction: float = 0.25):
        """
        Args:
            config: bot config
            fraction: Kelly fraction (0.25 = quarter Kelly, safer)
        """
        self.config = config
        self.fraction = fraction  # 25% Kelly
        self._trade_history: list = []
        self._max_history = 50  # Last 50 trades

        # Running stats per symbol
        self._stats: dict = {}

    def add_trade_result(self, symbol: str, pnl: float, invested: float):
        """Add a completed trade result."""
        if invested > 0:
            ret = pnl / invested
            self._trade_history.append({
                "symbol": symbol,
                "pnl": pnl,
                "invested": invested,
                "return": ret,
                "win": pnl > 0,
            })
            self._trade_history = self._trade_history[-self._max_history:]
            self._update_stats(symbol)

    def _update_stats(self, symbol: str):
        """Update win/loss stats for a symbol."""
        symbol_trades = [t for t in self._trade_history if t["symbol"] == symbol]
        all_trades = self._trade_history

        if len(all_trades) < 5:
            return

        wins = [t for t in all_trades if t["win"]]
        losses = [t for t in all_trades if not t["win"]]

        p = len(wins) / len(all_trades) if all_trades else 0.5
        avg_win = np.mean([t["return"] for t in wins]) if wins else 0.02
        avg_loss = abs(np.mean([t["return"] for t in losses])) if losses else 0.02

        self._stats[symbol] = {
            "p": p,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "n_trades": len(all_trades),
        }

    def calculate_position_size(
        self,
        symbol: str,
        wallet_usdc: float,
        confidence: float,
        model_accuracy: float,
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Returns: position size in USDC
        """
        # Get stats
        stats = self._stats.get(symbol)

        if stats and stats["n_trades"] >= 10:
            # Use actual historical stats
            p = stats["p"]
            b = stats["avg_win"] / max(stats["avg_loss"], 0.001)
        else:
            # Use model accuracy + confidence as estimate
            p = model_accuracy * confidence  # Conservative estimate
            p = max(0.45, min(p, 0.75))  # Clamp 45-75%
            b = self.config.ATR_TP_MULTIPLIER / self.config.ATR_SL_MULTIPLIER  # TP/SL ratio

        q = 1 - p

        # Kelly formula
        if b <= 0 or p <= 0:
            kelly_f = 0
        else:
            kelly_f = (b * p - q) / b

        # Apply fraction (25% Kelly = safer)
        kelly_f = max(0, kelly_f * self.fraction)

        # Cap at 50% of wallet max
        kelly_f = min(kelly_f, 0.5)

        position_size = wallet_usdc * kelly_f

        # Minimum $5, maximum $50 (safety caps)
        position_size = max(5.0, min(position_size, 50.0))

        logger.info(
            f"Kelly [{symbol}]: p={p:.2f} b={b:.2f} f={kelly_f:.3f} "
            f"size=${position_size:.2f} (wallet=${wallet_usdc:.2f})"
        )

        return round(position_size, 2)

    def get_stats(self, symbol: str = None) -> dict:
        """Get current Kelly stats."""
        if symbol:
            return self._stats.get(symbol, {})
        return {
            "all_trades": len(self._trade_history),
            "win_rate": sum(1 for t in self._trade_history if t["win"]) / max(len(self._trade_history), 1),
            "per_symbol": self._stats,
        }
