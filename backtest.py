#!/usr/bin/env python3
"""
Backtest Module v2 — confidence-weighted results.
Trades are weighted by model confidence in PnL calculation.
"""
import argparse
import logging
import sys
import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config import config
from data.fetcher import DataFetcher
from data.processor import FeatureProcessor
from ml.agent import TradingAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class Backtester:
    def __init__(self, cfg):
        self.config = cfg
        self.fetcher = DataFetcher(cfg)
        self.processor = FeatureProcessor(cfg)
        self.agent = TradingAgent(cfg)

    def run(self, symbol="SOL/USDT", limit=1000):
        logger.info(f"Backtest: {symbol} — {limit} candles")

        df = self.fetcher.fetch_ohlcv(symbol, limit=limit)
        if df.empty:
            return {}

        df = self.processor.add_indicators(df)
        feat_cols = [c for c in self.processor.get_feature_columns() if c in df.columns]
        clean = df.dropna()

        if len(clean) < self.config.MIN_TRAIN_SAMPLES + 50:
            logger.error(f"Not enough data: {len(clean)}")
            return {}

        split = int(len(clean) * 0.8)
        train_df = clean.iloc[:split]
        test_df = clean.iloc[split:]

        logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")
        acc = self.agent.train(train_df, feat_cols)

        balance = 100.0
        position = 0.0
        entry_price = 0.0
        entry_conf = 0.0
        trades = []
        equity_curve = []

        for i in range(len(test_df)):
            row = test_df.iloc[i]
            price = float(row["close"])
            features = test_df[feat_cols].iloc[i].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

            prediction = self.agent.predict(features)
            signal = prediction["signal"]
            confidence = prediction["confidence"]

            portfolio_val = balance + (position * price)
            equity_curve.append({
                "idx": i, "price": price, "total": portfolio_val,
                "signal": signal, "confidence": confidence,
            })

            if confidence < self.config.MIN_CONFIDENCE:
                continue

            # Confidence-weighted position sizing
            # Higher confidence = larger position (0.5x to 1.5x)
            size_mult = 0.5 + (confidence - 0.5) * 2  # maps 0.5->0.5, 0.75->1.0, 1.0->1.5
            size_mult = max(0.5, min(size_mult, 1.5))

            if signal == "BUY" and position == 0 and balance > 10:
                spend = balance * 0.95 * size_mult
                spend = min(spend, balance * 0.95)
                position = spend / price
                balance -= spend
                entry_price = price
                entry_conf = confidence
                trades.append({
                    "idx": i, "side": "buy", "price": price,
                    "amount": position, "value": spend,
                    "confidence": confidence, "size_mult": size_mult,
                })

            elif signal == "SELL" and position > 0:
                value = position * price
                pnl = value - (position * entry_price)
                balance += value
                trades.append({
                    "idx": i, "side": "sell", "price": price,
                    "amount": position, "value": value,
                    "pnl": pnl, "confidence": confidence,
                    "entry_conf": entry_conf,
                })
                position = 0.0
                entry_price = 0.0

            # Stop loss
            if position > 0 and entry_price > 0:
                loss_pct = (price - entry_price) / entry_price
                if loss_pct <= -self.config.STOP_LOSS_PCT:
                    value = position * price
                    pnl = value - (position * entry_price)
                    balance += value
                    trades.append({
                        "idx": i, "side": "sell", "price": price,
                        "amount": position, "value": value,
                        "pnl": pnl, "reason": "stop_loss",
                        "confidence": 0, "entry_conf": entry_conf,
                    })
                    position = 0.0
                    entry_price = 0.0

        # Final close
        if position > 0:
            final_price = float(test_df.iloc[-1]["close"])
            value = position * final_price
            pnl = value - (position * entry_price)
            balance += value
            trades.append({
                "idx": len(test_df)-1, "side": "sell", "price": final_price,
                "amount": position, "value": value,
                "pnl": pnl, "reason": "backtest_end",
            })

        # Stats
        final_balance = balance
        total_return = (final_balance - 100) / 100 * 100
        sell_trades = [t for t in trades if t["side"] == "sell" and "pnl" in t]
        wins = [t for t in sell_trades if t["pnl"] > 0]
        losses = [t for t in sell_trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0

        # Confidence-weighted win rate
        if sell_trades:
            conf_wins = sum(t.get("entry_conf", 0.5) for t in wins)
            conf_total = sum(t.get("entry_conf", 0.5) for t in sell_trades)
            conf_win_rate = conf_wins / conf_total * 100 if conf_total > 0 else 0
        else:
            conf_win_rate = 0

        # Buy & hold
        start_price = float(test_df.iloc[0]["close"])
        end_price = float(test_df.iloc[-1]["close"])
        bnh_return = (end_price - start_price) / start_price * 100

        # Sharpe
        equity_vals = [e["total"] for e in equity_curve]
        if len(equity_vals) > 1:
            returns = pd.Series(equity_vals).pct_change().dropna()
            sharpe = float((returns.mean() / returns.std()) * (252**0.5)) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Confidence-weighted PnL
        conf_weighted_pnl = sum(
            t["pnl"] * t.get("entry_conf", 0.5)
            for t in sell_trades if "pnl" in t
        )

        results = {
            "symbol": symbol,
            "timeframe": self.config.TIMEFRAME,
            "model_accuracy": acc,
            "start_balance": 100.0,
            "final_balance": round(final_balance, 2),
            "total_return_pct": round(total_return, 2),
            "buy_hold_return_pct": round(bnh_return, 2),
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "conf_weighted_win_rate": round(conf_win_rate, 1),
            "conf_weighted_pnl": round(conf_weighted_pnl, 4),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(self._max_dd(equity_vals), 2),
            "avg_confidence": round(np.mean([t.get("confidence", 0.5) for t in trades]), 3) if trades else 0,
            "timestamp": datetime.now().isoformat(),
        }

        self._save_results(results, equity_curve)
        self._print_results(results)
        return results

    def _max_dd(self, equity):
        if len(equity) < 2:
            return 0.0
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _print_results(self, r):
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Symbol:              {r['symbol']}")
        logger.info(f"ML Accuracy:         {r['model_accuracy']:.1%}")
        logger.info(f"Final Balance:       ${r['final_balance']:.2f}")
        logger.info(f"Total Return:        {r['total_return_pct']:+.1f}%")
        logger.info(f"Buy & Hold:          {r['buy_hold_return_pct']:+.1f}%")
        logger.info(f"Alpha:               {r['total_return_pct'] - r['buy_hold_return_pct']:+.1f}%")
        logger.info(f"Win Rate:            {r['win_rate']:.1f}%")
        logger.info(f"Conf-Weighted WR:    {r['conf_weighted_win_rate']:.1f}%")
        logger.info(f"Conf-Weighted PnL:   ${r['conf_weighted_pnl']:.4f}")
        logger.info(f"Sharpe:              {r['sharpe']:.2f}")
        logger.info(f"Max Drawdown:        {r['max_drawdown']:.1f}%")
        logger.info(f"Avg Confidence:      {r['avg_confidence']:.3f}")
        logger.info("=" * 60)

    def _save_results(self, results, equity_curve):
        try:
            conn = sqlite3.connect(self.config.DB_PATH)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    timestamp TEXT, symbol TEXT, timeframe TEXT,
                    accuracy REAL, final_balance REAL, total_return REAL,
                    buy_hold_return REAL, total_trades INTEGER,
                    win_rate REAL, sharpe REAL, max_drawdown REAL,
                    data TEXT
                )
            """)
            c.execute(
                "INSERT INTO backtest_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (results["timestamp"], results["symbol"], results["timeframe"],
                 results["model_accuracy"], results["final_balance"],
                 results["total_return_pct"], results["buy_hold_return_pct"],
                 results["total_trades"], results["win_rate"],
                 results["sharpe"], results["max_drawdown"],
                 json.dumps(equity_curve[-100:]))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Save failed: {e}")



    def run_live_model(self, symbol="SOL/USDT", limit=500):
        """Backtest avec le modele de production (sans retraining) — mesure la vraie perf live."""
        logger.info(f"Backtest live model: {symbol} — {limit} candles")

        loaded = self.agent.load_saved_models()
        if not loaded:
            logger.error("Aucun modele sauvegarde trouve — fallback sur run() classique")
            return self.run(symbol, limit)

        df = self.fetcher.fetch_ohlcv(symbol, limit=limit)
        if df.empty:
            return {}

        df = self.processor.add_indicators(df)
        feat_cols = None  # resolved after clean
        clean = df.dropna()

        if len(clean) < 50:
            logger.error(f"Not enough data: {len(clean)}")
            return {}

        saved_cols = self.agent.feature_cols or self.agent.selected_features or []
        feat_cols = [c for c in saved_cols if c in clean.columns]
        if not feat_cols:
            feat_cols = [c for c in self.processor.get_feature_columns() if c in clean.columns]

        # Pas de split train/test — on teste sur tout avec le modele existant
        test_df = clean.copy()
        logger.info(f"Live model backtest: {len(test_df)} candles (no refit)")

        import numpy as np
        balance = 100.0
        position = 0.0
        entry_price = 0.0
        entry_conf = 0.0
        trades = []
        equity_curve = []

        for i in range(len(test_df)):
            row = test_df.iloc[i]
            price = float(row["close"])
            features = test_df[feat_cols].iloc[i].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

            prediction = self.agent.predict(features)
            signal = prediction["signal"]
            confidence = prediction["confidence"]

            portfolio_val = balance + (position * price)
            equity_curve.append({"idx": i, "price": price, "total": portfolio_val})

            if confidence < self.config.MIN_CONFIDENCE:
                continue

            if signal == "BUY" and position == 0 and balance > 10:
                spend = min(balance * 0.95, balance * 0.95)
                position = spend / price
                balance -= spend
                entry_price = price
                entry_conf = confidence
                trades.append({"idx": i, "side": "buy", "price": price, "value": spend, "confidence": confidence})

            elif signal == "SELL" and position > 0:
                value = position * price
                pnl = value - (position * entry_price)
                balance += value
                trades.append({"idx": i, "side": "sell", "price": price, "value": value, "pnl": pnl, "entry_conf": entry_conf})
                position = 0.0
                entry_price = 0.0

            if position > 0 and entry_price > 0:
                if (price - entry_price) / entry_price <= -self.config.STOP_LOSS_PCT:
                    value = position * price
                    pnl = value - (position * entry_price)
                    balance += value
                    trades.append({"idx": i, "side": "sell", "price": price, "value": value, "pnl": pnl, "reason": "stop_loss"})
                    position = 0.0
                    entry_price = 0.0

        if position > 0:
            final_price = float(test_df.iloc[-1]["close"])
            value = position * final_price
            pnl = value - (position * entry_price)
            balance += value
            trades.append({"idx": len(test_df)-1, "side": "sell", "price": final_price, "value": value, "pnl": pnl, "reason": "backtest_end"})

        sell_trades = [t for t in trades if t["side"] == "sell" and "pnl" in t]
        wins   = [t for t in sell_trades if t["pnl"] > 0]
        wr     = len(wins) / len(sell_trades) * 100 if sell_trades else 0
        ret    = (balance - 100) / 100 * 100
        bh     = (float(test_df.iloc[-1]["close"]) - float(test_df.iloc[0]["close"])) / float(test_df.iloc[0]["close"]) * 100

        import pandas as pd
        eq = pd.Series([e["total"] for e in equity_curve])
        roll_max = eq.cummax()
        dd = float(((eq - roll_max) / roll_max).min() * 100)
        returns = eq.pct_change().dropna()
        sharpe = float(returns.mean() / returns.std() * (252**0.5)) if returns.std() > 0 else 0

        logger.info(f"[LIVE MODEL] {symbol}: return={ret:+.1f}% | B&H={bh:+.1f}% | WR={wr:.1f}% | Sharpe={sharpe:.2f}")

        return {
            "symbol": symbol, "timeframe": "live_model",
            "model_accuracy": getattr(self.agent, "last_accuracy", 0),
            "start_balance": 100.0, "final_balance": balance,
            "total_return_pct": ret, "buy_hold_return_pct": bh,
            "total_trades": len(sell_trades), "wins": len(wins),
            "losses": len(sell_trades) - len(wins),
            "win_rate": wr, "sharpe": sharpe, "max_drawdown": dd,
            "avg_confidence": sum(t.get("confidence", 0) for t in trades) / max(len(trades), 1),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "mode": "live_model",
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    bt = Backtester(config)
    if args.all:
        for sym in config.SYMBOLS:
            bt.run(sym, args.limit)
    else:
        bt.run(args.symbol, args.limit)


if __name__ == "__main__":
    main()
