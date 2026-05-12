"""
Adaptive Filters — auto-calibrate thresholds per symbol.

Instead of fixed MIN_ATR_PCT and MIN_VOLUME_RATIO for all symbols,
this module calculates thresholds based on each symbol's own history.

Example: SOL's 20th percentile ATR% might be 0.005 while BTC's is 0.002.
The bot adapts automatically.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AdaptiveFilters:
    def __init__(self, config):
        self.config = config
        # Per-symbol calibrated thresholds
        self._atr_thresholds: dict[str, float] = {}
        self._volume_thresholds: dict[str, float] = {}
        self._trend_thresholds: dict[str, float] = {}

        # Percentiles used for calibration
        self.atr_percentile: float = 15      # Skip bottom 20% volatility
        self.volume_percentile: float = 5   # Skip bottom 25% volume

    def calibrate(self, symbol: str, df) -> dict:
        """Calibrate filter thresholds from historical data for a symbol.
        Call this after add_indicators()."""
        result = {}

        try:
            # ATR% threshold: use 20th percentile as minimum
            if "atr_pct" in df.columns:
                atr_vals = df["atr_pct"].dropna()
                if len(atr_vals) > 50:
                    threshold = float(np.percentile(atr_vals, self.atr_percentile))
                    # Safety bounds
                    threshold = max(threshold, 0.001)   # never below 0.1%
                    threshold = min(threshold, 0.01)    # never above 1%
                    self._atr_thresholds[symbol] = threshold
                    result["atr_pct"] = threshold

            # Volume ratio threshold: use 25th percentile
            if "volume_ratio" in df.columns:
                vol_vals = df["volume_ratio"].dropna()
                if len(vol_vals) > 50:
                    threshold = float(np.percentile(vol_vals, self.volume_percentile))
                    threshold = max(threshold, 0.01)    # never below 0.3
                    threshold = min(threshold, 1.5)    # never above 1.5
                    self._volume_thresholds[symbol] = threshold
                    result["volume_ratio"] = threshold

            # Trend strength threshold: based on typical range
            if "trend_strength" in df.columns:
                trend_vals = df["trend_strength"].dropna().abs()
                if len(trend_vals) > 50:
                    # Use 75th percentile of absolute trend as "strong trend"
                    threshold = float(np.percentile(trend_vals, 75))
                    threshold = max(threshold, 0.005)
                    threshold = min(threshold, 0.05)
                    self._trend_thresholds[symbol] = threshold
                    result["trend_strength"] = threshold

            if result:
                logger.info(f"[{symbol}] Adaptive filters calibrated: {result}")

        except Exception as e:
            logger.warning(f"Calibration failed for {symbol}: {e}")

        return result

    def get_atr_threshold(self, symbol: str) -> float:
        """Get calibrated ATR% threshold for a symbol."""
        return self._atr_thresholds.get(symbol, self.config.MIN_ATR_PCT)

    def get_volume_threshold(self, symbol: str) -> float:
        """Get calibrated volume ratio threshold for a symbol."""
        return self._volume_thresholds.get(symbol, self.config.MIN_VOLUME_RATIO)

    def get_trend_threshold(self, symbol: str) -> float:
        """Get calibrated trend strength threshold for a symbol."""
        return self._trend_thresholds.get(symbol, 0.02)

    def check_volatility(self, symbol: str, current_atr_pct: float) -> bool:
        """Returns True if should SKIP (volatility too low)."""
        threshold = self.get_atr_threshold(symbol)
        if current_atr_pct < threshold:
            logger.info(f"[{symbol}] Adaptive volatility filter: ATR%={current_atr_pct:.4f} < {threshold:.4f}")
            return True
        return False

    def check_volume(self, symbol: str, current_vol_ratio: float) -> bool:
        """Disabled — volume check unreliable mid-candle."""
        return False

    def check_trend(self, symbol: str, trend_strength: float, signal: str) -> bool:
        """Returns True if signal goes against strong trend (should SKIP)."""
        threshold = self.get_trend_threshold(symbol)
        if signal == "SELL" and trend_strength > threshold:
            logger.info(f"[{symbol}] Adaptive trend filter: blocked SELL in uptrend ({trend_strength:.4f} > {threshold:.4f})")
            return True
        if signal == "BUY" and trend_strength < -threshold:
            logger.info(f"[{symbol}] Adaptive trend filter: blocked BUY in downtrend ({trend_strength:.4f} < -{threshold:.4f})")
            return True
        return False

    def get_all_thresholds(self) -> dict:
        """Return all calibrated thresholds (for dashboard)."""
        symbols = set(list(self._atr_thresholds.keys()) +
                      list(self._volume_thresholds.keys()) +
                      list(self._trend_thresholds.keys()))
        return {
            sym: {
                "atr_pct": self._atr_thresholds.get(sym, "default"),
                "volume_ratio": self._volume_thresholds.get(sym, "default"),
                "trend_strength": self._trend_thresholds.get(sym, "default"),
            }
            for sym in symbols
        }
