import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)

ALL_FEATURES = [
    # Trend
    "ema_8", "ema_21", "ema_50", "sma_20", "sma_50",
    "ema_cross", "price_above_sma50", "trend_strength",
    # MACD
    "macd", "macd_signal", "macd_diff", "macd_cross", "macd_hist_slope",
    # Momentum
    "rsi", "rsi_6", "stoch_k", "stoch_d", "williams_r", "cci", "roc",
    "rsi_divergence",
    # Volatility
    "bb_width", "bb_pct", "atr_pct", "volatility_regime",
    # Volume
    "volume_ratio", "mfi", "volume_trend",
    # Price action
    "returns", "log_returns", "high_low_ratio", "close_open_ratio",
    "returns_3", "returns_5", "returns_10",
    # Higher TF context
    "dist_from_high_20", "dist_from_low_20",
    # Lagged features (1h, 2h, 3h ago)
    "rsi_lag1", "rsi_lag2", "rsi_lag3",
    "returns_lag1", "returns_lag2", "returns_lag3",
    "macd_diff_lag1", "macd_diff_lag2",
    "volume_ratio_lag1", "volume_ratio_lag2",
    # Time
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    # Market regime (ADX-based)
    "market_regime", "regime_strength",
    # Feature interactions
    "rsi_x_volume", "momentum_x_vol", "macd_x_trend", "bb_x_rsi", "atr_x_regime",
    # Cross-asset (BTC context, injected at inference)
    "btc_returns_1h", "btc_returns_4h", "btc_rsi",
    # Market microstructure (injected at inference)
    "orderbook_ratio", "oi_change", "liquidation_score",
    # Sentiment
    "fear_greed", "funding_rate", "news_sentiment",
]


class FeatureProcessor:
    def __init__(self, config):
        self.config = config
        self._selected_features = None

    def add_indicators(self, df):
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        try:
            # Trend
            df["ema_8"] = ta.trend.ema_indicator(close, window=8)
            df["ema_21"] = ta.trend.ema_indicator(close, window=21)
            df["ema_50"] = ta.trend.ema_indicator(close, window=50)
            df["sma_20"] = ta.trend.sma_indicator(close, window=20)
            df["sma_50"] = ta.trend.sma_indicator(close, window=50)
            df["ema_cross"] = (df["ema_8"] > df["ema_21"]).astype(int)
            df["price_above_sma50"] = (close > df["sma_50"]).astype(int)
            df["trend_strength"] = (df["ema_8"] - df["ema_50"]) / close.replace(0, np.nan)

            # MACD
            macd_obj = ta.trend.MACD(close)
            df["macd"] = macd_obj.macd()
            df["macd_signal"] = macd_obj.macd_signal()
            df["macd_diff"] = macd_obj.macd_diff()
            df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)
            df["macd_hist_slope"] = df["macd_diff"].diff()

            # Momentum
            df["rsi"] = ta.momentum.rsi(close, window=14)
            df["rsi_6"] = ta.momentum.rsi(close, window=6)
            stoch = ta.momentum.StochasticOscillator(high, low, close)
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()
            df["williams_r"] = ta.momentum.williams_r(high, low, close)
            df["cci"] = ta.trend.cci(high, low, close)
            df["roc"] = ta.momentum.roc(close, window=12)
            price_chg_5 = close.diff(5)
            rsi_chg_5 = df["rsi"].diff(5)
            df["rsi_divergence"] = np.where(
                (price_chg_5 > 0) & (rsi_chg_5 < 0), -1,
                np.where((price_chg_5 < 0) & (rsi_chg_5 > 0), 1, 0)
            )

            # Volatility
            bb = ta.volatility.BollingerBands(close)
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_middle"] = bb.bollinger_mavg()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)
            df["bb_pct"] = bb.bollinger_pband()
            df["atr"] = ta.volatility.average_true_range(high, low, close)
            df["atr_pct"] = df["atr"] / close.replace(0, np.nan)
            atr_sma = df["atr"].rolling(50).mean()
            df["volatility_regime"] = df["atr"] / atr_sma.replace(0, np.nan)

            # Volume
            df["volume_sma"] = ta.trend.sma_indicator(volume, window=20)
            df["volume_ratio"] = volume / df["volume_sma"].replace(0, np.nan)
            df["mfi"] = ta.volume.money_flow_index(high, low, close, volume)
            vol_ema_5 = ta.trend.ema_indicator(volume, window=5)
            vol_ema_20 = ta.trend.ema_indicator(volume, window=20)
            df["volume_trend"] = vol_ema_5 / vol_ema_20.replace(0, np.nan)

            # Price action
            df["returns"] = close.pct_change()
            df["log_returns"] = np.log(close / close.shift(1))
            df["high_low_ratio"] = high / low.replace(0, np.nan)
            df["close_open_ratio"] = close / df["open"].replace(0, np.nan)
            df["returns_3"] = close.pct_change(3)
            df["returns_5"] = close.pct_change(5)
            df["returns_10"] = close.pct_change(10)

            # Higher TF context
            high_20 = high.rolling(20).max()
            low_20 = low.rolling(20).min()
            range_20 = (high_20 - low_20).replace(0, np.nan)
            df["dist_from_high_20"] = (high_20 - close) / range_20
            df["dist_from_low_20"] = (close - low_20) / range_20

            # ── Lagged features ──────────────────────────────────────
            df["rsi_lag1"] = df["rsi"].shift(1)
            df["rsi_lag2"] = df["rsi"].shift(2)
            df["rsi_lag3"] = df["rsi"].shift(3)
            df["returns_lag1"] = df["returns"].shift(1)
            df["returns_lag2"] = df["returns"].shift(2)
            df["returns_lag3"] = df["returns"].shift(3)
            df["macd_diff_lag1"] = df["macd_diff"].shift(1)
            df["macd_diff_lag2"] = df["macd_diff"].shift(2)
            df["volume_ratio_lag1"] = df["volume_ratio"].shift(1)
            df["volume_ratio_lag2"] = df["volume_ratio"].shift(2)

            # Time
            try:
                ts = pd.to_datetime(df.index)
                hour = ts.hour
                dow = ts.dayofweek
            except Exception:
                hour = pd.Series(12, index=df.index)
                dow = pd.Series(3, index=df.index)
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            df["day_sin"] = np.sin(2 * np.pi * dow / 7)
            df["day_cos"] = np.cos(2 * np.pi * dow / 7)

            # Market regime detection — 4 états
            # 0=Range, 1=Trend Up, 2=Trend Down, 3=High Volatility
            adx = ta.trend.adx(high, low, close, window=14)
            adx_pos = ta.trend.adx_pos(high, low, close, window=14)
            adx_neg = ta.trend.adx_neg(high, low, close, window=14)

            bb_width_val = (df.get("bb_upper", close) - df.get("bb_lower", close)) / close.replace(0, np.nan)
            bb_width_pct = bb_width_val.rolling(20).mean()
            high_vol = bb_width_val > (bb_width_pct * 1.5)

            regime = pd.Series(0, index=df.index)  # Default: Range
            trending = adx > 25
            regime[trending & (adx_pos > adx_neg)] = 1  # Trend Up
            regime[trending & (adx_neg > adx_pos)] = 2  # Trend Down
            regime[high_vol & (adx > 30)] = 3            # High Volatility

            df["market_regime"] = regime
            df["regime_strength"] = adx / 100.0  # Normalized 0-1
            df["adx"] = adx  # Garder ADX brut comme feature ML

            # Feature interactions
            df["rsi_x_volume"] = (df["rsi"] / 100) * df["volume_ratio"].fillna(1)
            df["momentum_x_vol"] = df["returns_5"].fillna(0) * df["volatility_regime"].fillna(1)
            df["macd_x_trend"] = df["macd_diff"].fillna(0) * df["trend_strength"].fillna(0)
            df["bb_x_rsi"] = df["bb_pct"].fillna(0.5) * (df["rsi"].fillna(50) / 100)
            atr_p = df["atr_pct"].fillna(0)
            regime = df.get("market_regime", pd.Series(0.5, index=df.index))
            df["atr_x_regime"] = atr_p * regime

            # Cross-asset placeholders
            df["btc_returns_1h"] = 0.0
            df["btc_returns_4h"] = 0.0
            df["btc_rsi"] = 50.0

            # Placeholders (injected at inference)
            df["orderbook_ratio"] = 1.0
            df["oi_change"] = 0.0
            df["liquidation_score"] = 0.0
            df["fear_greed"] = 50.0
            df["funding_rate"] = 0.0
            df["news_sentiment"] = 50.0

            # Label: 2-class, 3-candle lookahead
            future_ret = close.shift(-5) / close - 1
            # Label with threshold: ignore tiny moves (noise)
            df["label"] = None
            df.loc[future_ret > 0.003, "label"] = 1   # UP > 0.3%
            df.loc[future_ret < -0.003, "label"] = 0  # DOWN > 0.3%
            # Rows with |ret| < 0.3% become NaN → dropped during training

        except Exception as e:
            logger.error(f"add_indicators error: {e}")

        return df

    def get_feature_columns(self):
        if self._selected_features:
            return list(self._selected_features)
        return list(ALL_FEATURES)

    def set_selected_features(self, features):
        self._selected_features = features

    def get_all_feature_columns(self):
        return list(ALL_FEATURES)

    def get_current_features(self, df, fear_greed=50, funding_rate=0.0, btc_returns_1h=0.0, btc_returns_4h=0.0, btc_rsi=50.0,
                              news_sentiment=50.0, orderbook_ratio=1.0,
                              oi_change=0.0, liquidation_score=0.0):
        cols = [c for c in self.get_feature_columns() if c in df.columns]
        row = df[cols].iloc[-1].copy()

        # Inject live values
        injections = {
            "btc_returns_1h": float(btc_returns_1h),
            "btc_returns_4h": float(btc_returns_4h),
            "btc_rsi": float(btc_rsi),
            "fear_greed": float(fear_greed),
            "funding_rate": float(funding_rate),
            "news_sentiment": float(news_sentiment),
            "orderbook_ratio": float(orderbook_ratio),
            "oi_change": float(oi_change),
            "liquidation_score": float(liquidation_score),
        }
        for col, val in injections.items():
            if col in cols:
                row.iloc[cols.index(col)] = val

        values = row.values.astype(np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

    def get_current_atr(self, df):
        if "atr" in df.columns and len(df) > 0:
            v = df["atr"].iloc[-1]
            return float(v) if not np.isnan(v) else 0.0
        return 0.0

    def get_current_atr_pct(self, df):
        if "atr_pct" in df.columns and len(df) > 0:
            v = df["atr_pct"].iloc[-1]
            return float(v) if not np.isnan(v) else 0.0
        return 0.0

    def get_current_volume_ratio(self, df):
        if "volume_ratio" in df.columns and len(df) > 0:
            v = df["volume_ratio"].iloc[-1]
            return float(v) if not np.isnan(v) else 1.0
        return 1.0
