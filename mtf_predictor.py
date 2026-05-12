"""
Multi-Timeframe ML — trains separate models on 15min, 1h, 4h FOR EACH SYMBOL
and combines predictions via consensus voting.

Models are keyed by (symbol, timeframe) — training SOL does NOT overwrite BTC.
This runs alongside the main agent as an additional signal source.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


class MultiTimeframePredictorML:
    def __init__(self, config, fetcher, processor):
        self.config = config
        self.fetcher = fetcher
        self.processor = processor
        # Key: (symbol, timeframe) → model / scaler / feature_cols
        self._models: dict = {}
        self._scalers: dict = {}
        self._feat_cols: dict = {}
        self._last_trained = None
        self._retrain_hours = 4  # Retrain every 4h
        self._predictions: dict = {}

    def needs_retraining(self) -> bool:
        if not self._last_trained:
            return True
        elapsed = (datetime.now() - self._last_trained).total_seconds() / 3600
        return elapsed >= self._retrain_hours

    def train_all(self, symbol: str):
        """Train models on 15min, 1h, 4h timeframes FOR THIS SYMBOL.
        Each (symbol, tf) gets its own model; previous symbols' models are preserved."""
        from sklearn.preprocessing import StandardScaler

        timeframes = ["15m", "1h", "4h"]
        limits = {"15m": 3000, "1h": 2000, "4h": 1000}

        for tf in timeframes:
            try:
                df = self.fetcher.fetch_ohlcv(symbol, timeframe=tf, limit=limits[tf])
                if df.empty or len(df) < 200:
                    logger.warning(f"MTF [{symbol} {tf}]: Not enough data ({len(df)} rows)")
                    continue

                df = self.processor.add_indicators(df)
                if "label" in df.columns:
                    df["label"] = pd.to_numeric(df["label"], errors="coerce")
                feat_cols = [c for c in self.processor.get_all_feature_columns() if c in df.columns]
                clean = df[feat_cols + ["label"]].dropna()
                clean["label"] = clean["label"].astype(int)

                if len(clean) < 150:
                    logger.warning(f"MTF [{symbol} {tf}]: only {len(clean)} clean rows")
                    continue

                X = clean[feat_cols].values
                y = clean["label"].values

                split = int(len(X) * 0.8)
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[:split])
                y_tr = y[:split]

                if HAS_LGBM:
                    model = LGBMClassifier(
                        n_estimators=150, max_depth=4,
                        learning_rate=0.05, verbose=-1,
                        random_state=42, objective="binary",
                    )
                    model.fit(X_tr, y_tr)
                    key = (symbol, tf)
                    self._models[key] = model
                    self._scalers[key] = scaler
                    self._feat_cols[key] = feat_cols

                    # Test accuracy
                    X_te = scaler.transform(X[split:])
                    acc = (model.predict(X_te) == y[split:]).mean()
                    logger.info(f"MTF [{symbol} {tf}]: trained, acc={acc:.3f}")

            except Exception as e:
                logger.warning(f"MTF [{symbol} {tf}] training failed: {e}")

        self._last_trained = datetime.now()
        n_for_sym = sum(1 for k in self._models if k[0] == symbol)
        logger.info(f"MTF [{symbol}]: {n_for_sym}/{len(timeframes)} timeframes trained")

    def predict(self, symbol: str, features: np.ndarray) -> dict:
        """Get predictions from all trained timeframes for this symbol only."""
        results = {}

        # Filter to only models for this symbol
        symbol_models = {tf: (self._models[(symbol, tf)], self._scalers.get((symbol, tf)))
                         for (s, tf) in self._models.keys() if s == symbol}

        for tf, (model, scaler) in symbol_models.items():
            try:
                if scaler is None:
                    continue

                X = features.reshape(1, -1)
                expected = model.n_features_in_
                if X.shape[1] > expected:
                    X = X[:, :expected]
                elif X.shape[1] < expected:
                    pad = np.zeros((1, expected - X.shape[1]))
                    X = np.hstack([X, pad])

                X_s = scaler.transform(X)
                proba = model.predict_proba(X_s)[0]
                prob_up = float(proba[1]) if len(proba) > 1 else 0.5

                results[tf] = {
                    "prob_up": prob_up,
                    "signal": "BUY" if prob_up > 0.55 else "SELL" if prob_up < 0.45 else "HOLD",
                }

            except Exception as e:
                logger.debug(f"MTF [{symbol} {tf}] predict failed: {e}")

        self._predictions[symbol] = results

        # Consensus
        signals = [r["signal"] for r in results.values()]
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        total = len(signals)

        if total == 0:
            consensus = "NEUTRAL"
            strength = 0.0
        elif buy_count == total:
            consensus = "STRONG_BUY"
            strength = 1.0
        elif sell_count == total:
            consensus = "STRONG_SELL"
            strength = 1.0
        elif buy_count > sell_count:
            consensus = "BUY"
            strength = buy_count / total
        elif sell_count > buy_count:
            consensus = "SELL"
            strength = sell_count / total
        else:
            consensus = "NEUTRAL"
            strength = 0.0

        return {
            "timeframes": results,
            "consensus": consensus,
            "strength": strength,
        }

    def get_predictions(self, symbol: str = None) -> dict:
        if symbol:
            return self._predictions.get(symbol, {})
        return self._predictions
