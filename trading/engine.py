import json
import os
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional

from config import config as _cfg
from data.fetcher import DataFetcher
from data.processor import FeatureProcessor
from data.storage import Database
from ml.agent import TradingAgent
from trading.portfolio import Portfolio
from trading.risk import RiskManager
from adaptive_filters import AdaptiveFilters
from news_sentiment import NewsSentiment
from claude_sentiment import ClaudeSentimentAnalyzer
from market_data import MarketData
from deep_learning import DeepLearningAgent
from whale_tracker import WhaleTracker
from kelly import KellyCriterion
from telegram_monitor import TelegramChannelMonitor
from mtf_predictor import MultiTimeframePredictorML

logger = logging.getLogger(__name__)

_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
               "1h": 3600, "4h": 14400, "1d": 86400}
_TP_STATE_FILE = "tp_state.json"
_BOT_STATE_FILE = "bot_state.json"


class TradingEngine:
    def __init__(self):
        self.config = _cfg
        self.fetcher = DataFetcher(_cfg)
        self.processor = FeatureProcessor(_cfg)
        self.db = Database(_cfg.DB_PATH)
        self.agent = TradingAgent(_cfg)
        self.portfolio = Portfolio(_cfg.INITIAL_BALANCE, _cfg)
        self.risk = RiskManager(_cfg)
        self.adaptive = AdaptiveFilters(_cfg)

        # Tracking for auto-compound and drawdown
        self._peak_wallet = 0.0
        self._btc_data = {}  # Cache BTC data for cross-asset
        self.news = NewsSentiment(_cfg)
        self.claude_sentiment = ClaudeSentimentAnalyzer(_cfg)
        self.market_data = MarketData(_cfg)
        self.deep_agent = DeepLearningAgent(_cfg)
        self.whale_tracker = WhaleTracker(_cfg)
        self.kelly = KellyCriterion(_cfg)

        # Multi-timeframe ML predictor
        self.mtf = MultiTimeframePredictorML(_cfg, self.fetcher, self.processor)

        # Telegram channel monitor (real-time news)
        self.tg_monitor = None
        try:
            self.tg_monitor = TelegramChannelMonitor(_cfg)
            # React to high-impact news
            self.tg_monitor.add_high_impact_callback(self._on_breaking_news)
            self.tg_monitor.start()
            logger.info("Telegram channel monitor started")
        except Exception as e:
            logger.warning(f"Telegram monitor init failed: {e}")

        # Trade tracking
        self._last_trade_time: dict[str, datetime] = {}
        self._daily_trade_count: int = 0
        self._daily_reset: datetime = datetime.now()
        self._status_lock = threading.RLock()  # Protege self.status en multi-thread
        self._trailing_stops: dict[str, dict] = {}
        self._scale_pending: dict[str, dict] = {}
        self._funding_rates: dict[str, float] = {}

        # Kelly tracking: map symbol → USDC invested on current open position
        # (used to feed Kelly.add_trade_result on close)
        self._pending_kelly_invested: dict[str, float] = {}

        # Persisted multi-TP level state (symbol → next TP level index)
        self._tp_levels: dict[str, int] = self._load_tp_state()
        self._load_bot_state()
        # Reconciliation differee apres init complete
        # Reload last trade times from disk (anti-flip-flop survives restarts)
        self._last_trade_time = self._load_last_trade_times()


        # Auto-pause tracking
        self._consecutive_losses: int = 0
        self._recent_trades: list = []  # last N trades for streak tracking
        self._auto_paused: bool = False
        self._auto_pause_until: datetime | None = None

        # Telegram
        self.telegram = None
        if _cfg.TELEGRAM_ENABLED:
            try:
                from telegram_alerts import TelegramAlerter
                self.telegram = TelegramAlerter(_cfg)
                logger.info("Telegram alerts enabled")
            except Exception as e:
                logger.warning(f"Telegram init failed: {e}")

        # Solana / Hyperliquid
        self.solana: object = None
        if _cfg.SOLANA_ENABLED:
            try:
                if _cfg.PERP_ENABLED:
                    from solana_wallet.hyperliquid_trader import HyperliquidTrader
                    self.solana = HyperliquidTrader(_cfg)
                    mode = "Hyperliquid PERP"
                else:
                    from solana_wallet.trader import SolanaTrader
                    self.solana = SolanaTrader(_cfg)
                    mode = "Jupiter SPOT"
                if self.solana.is_ready:
                    logger.info(f"Solana wallet active [{mode}]: {self.solana.public_key}")
                else:
                    logger.error("Solana wallet failed to initialise")
            except Exception as e:
                logger.error(f"Solana init error: {e}")

        self.running = False
        self.paused = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable] = []

        self.status = {
            "state": "stopped",
            "last_update": None,
            "current_prices": {},
            "last_signals": {},
            "cycle_count": 0,
            "errors": [],
            "filters": {},
            "positions": {},
            "funding_rates": {},
            "auto_pause": False,
            "consecutive_losses": 0,
        }

        self.agent.load_saved_models()

    # ------------------------------------------------------------------
    # Multi-TP level state (persisted across restarts)
    # ------------------------------------------------------------------

    def _load_last_trade_times(self) -> dict:
        """Reload _last_trade_time from disk so anti-flip-flop survives restarts."""
        import json, os
        try:
            if os.path.exists("trade_times.json"):
                with open("trade_times.json") as f:
                    data = json.load(f)
                result = {k: datetime.fromisoformat(v) for k, v in data.items()}
                logger.info(f"Loaded last trade times: {list(result.keys())}")
                return result
        except Exception as e:
            logger.debug(f"Failed to load trade times: {e}")
        return {}

    def _save_last_trade_times(self):
        import json
        try:
            data = {k: v.isoformat() for k, v in self._last_trade_time.items()}
            with open("trade_times.json", "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Failed to save trade times: {e}")

    def _reconcile_positions(self):
        """Reconcilier les positions ouvertes au demarrage:
        - Detecter positions existantes
        - Restaurer trailing stop si position en profit
        - Detecter si scale-out deja effectue (taille reduite)
        - Ne pas bloquer le cooldown (last_trade_time reste vide)
        """
        try:
            if not self.solana:
                return

            for symbol in self.config.SYMBOLS:
                sol_sym = symbol.split("/")[0]
                pos = self.solana.get_position(sol_sym)
                if not pos or pos.get("size", 0) <= 0:
                    continue

                side = pos.get("side", "")
                size = pos.get("size", 0)
                entry = pos.get("entry_price", 0)

                logger.info(f"Position existante detectee: {sol_sym} {side} size={size} entry={entry}")
                self.status["positions"][sol_sym] = pos

                # Detecter le scale-out: si position size << base size, scale deja fait
                try:
                    df = self.fetcher.fetch_ohlcv(symbol, limit=2)
                    if df is None or df.empty:
                        continue
                    current_price = float(df["close"].iloc[-1])

                    # Calculer le profit % actuel
                    if entry > 0:
                        if side == "long":
                            pnl_pct = (current_price - entry) / entry
                        else:
                            pnl_pct = (entry - current_price) / entry

                        # Si position en profit >= seuil scale-out, marquer comme deja scale (par prudence)
                        if pnl_pct >= self.config.SCALE_OUT_PROFIT_PCT * 0.7:
                            setattr(self, f"_scaled_{sol_sym}", True)
                            logger.info(f"  -> scale-out flag active (profit {pnl_pct:.1%})")

                        # Restaurer trailing stop si position en profit
                        # On utilise ATR rapide pour estimer la distance
                        if self.processor:
                            df_full = self.fetcher.fetch_ohlcv(symbol, limit=50)
                            if df_full is not None and not df_full.empty:
                                df_full = self.processor.add_indicators(df_full)
                                atr = self.processor.get_current_atr(df_full)
                                if atr > 0 and pnl_pct > 0.005:  # En profit > 0.5%
                                    self._update_trailing_stop(sol_sym, current_price, side, atr, entry=entry)
                                    ts = self._trailing_stops.get(sol_sym)
                                    if ts:
                                        logger.info(f"  -> trailing stop restaure @ ${ts['price']:.2f}")
                except Exception as e:
                    logger.debug(f"Reconcile detail {sol_sym}: {e}")

        except Exception as e:
            logger.warning(f"Reconciliation positions: {e}")

    def _load_bot_state(self):
        """Restaurer l'etat persistant du bot apres un restart."""
        try:
            import json, os
            path = os.path.join(os.path.dirname(__file__), _BOT_STATE_FILE)
            if not os.path.exists(path):
                return
            with open(path) as f:
                state = json.load(f)
            self._peak_value = state.get("peak_value", 0.0)
            self._lstm_hits = state.get("lstm_hits", {})
            self._current_regime = state.get("current_regime", -1)
            self._daily_start_value = state.get("daily_start_value", 0.0)
            self._bnh_start_prices = state.get("bnh_start_prices", {})
            self._recent_accuracies = state.get("recent_accuracies", [])
            self._daily_trade_count = state.get("daily_trade_count", 0)
            l2 = state.get("circuit_l2_time")
            self._circuit_l2_time = datetime.fromisoformat(l2) if l2 else None
            dt = state.get("circuit_daily_time")
            self._circuit_daily_time = datetime.fromisoformat(dt) if dt else None
            self._conf_adj = float(state.get("conf_adj", 0.0))
            self._regime_min_conf = float(state.get("regime_min_conf", 0.52))
            logger.info(
                f"Bot state restored: peak=${self._peak_value:.2f} "
                f"lstm_hits={len(self._lstm_hits)} regime={self._current_regime} "
                f"l2={'on' if self._circuit_l2_time else 'off'} adj={self._conf_adj:+.2f}"
            )
        except Exception as e:
            logger.warning(f"Failed to load bot state: {e}")

    def _save_bot_state(self):
        """Sauvegarder l'etat persistant du bot."""
        try:
            import json, os
            def _dt_str(v):
                return v.isoformat() if isinstance(v, datetime) else None
            state = {
                "peak_value": getattr(self, "_peak_value", 0.0),
                "lstm_hits": getattr(self, "_lstm_hits", {}),
                "current_regime": getattr(self, "_current_regime", -1),
                "daily_start_value": getattr(self, "_daily_start_value", 0.0),
                "bnh_start_prices": getattr(self, "_bnh_start_prices", {}),
                "recent_accuracies": getattr(self, "_recent_accuracies", []),
                "daily_trade_count": getattr(self, "_daily_trade_count", 0),
                "circuit_l2_time": _dt_str(getattr(self, "_circuit_l2_time", None)),
                "circuit_daily_time": _dt_str(getattr(self, "_circuit_daily_time", None)),
                "conf_adj": getattr(self, "_conf_adj", 0.0),
                "regime_min_conf": getattr(self, "_regime_min_conf", 0.52),
                "saved_at": datetime.now().isoformat(),
            }
            path = os.path.join(os.path.dirname(__file__), _BOT_STATE_FILE)
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save bot state: {e}")

    def _load_tp_state(self) -> dict:
        try:
            if os.path.exists(_TP_STATE_FILE):
                with open(_TP_STATE_FILE, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    logger.info(f"Loaded TP state: {data}")
                    return {k: int(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load TP state: {e}")
        return {}

    def _save_tp_state(self):
        try:
            with open(_TP_STATE_FILE, "w") as f:
                json.dump(self._tp_levels, f)
        except Exception as e:
            logger.debug(f"Failed to save TP state: {e}")

    def _get_tp_level(self, sym: str) -> int:
        return self._tp_levels.get(sym, 0)

    def _set_tp_level(self, sym: str, level: int):
        self._tp_levels[sym] = int(level)
        self._save_tp_state()

    def _reset_tp_level(self, sym: str):
        if sym in self._tp_levels:
            self._tp_levels.pop(sym, None)
            self._save_tp_state()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self):
        if self.running:
            return
        self.running = True
        self.paused = False
        self._auto_paused = False
        self.status["state"] = "running"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Engine started")
        self._alert(f"🟢 Bot started — ${self._get_wallet_value():.2f} USDC")

    def stop(self):
        self.running = False
        self.status["state"] = "stopped"
        logger.info("Engine stopped")
        self._alert(f"🔴 Bot stopped — ${self._get_wallet_value():.2f} USDC")

    def pause(self):
        self.paused = True
        self.status["state"] = "paused"

    def resume(self):
        self.paused = False
        self._auto_paused = False
        self._auto_pause_until = None
        self._consecutive_losses = 0
        self.status["state"] = "running"
        self.status["auto_pause"] = False
        logger.info("Engine resumed — loss counter reset")
        self._alert("▶️ Bot resumed manually — loss counter reset")

    def add_callback(self, cb: Callable):
        self._callbacks.append(cb)

    def _notify(self, event: str, data: dict = None):
        for cb in self._callbacks:
            try:
                cb(event, data or {})
            except Exception:
                pass

    def _alert(self, msg: str):
        if self.telegram:
            try:
                self.telegram.send(msg)
            except Exception:
                pass

    def _get_wallet_value(self) -> float:
        if self.solana:
            snap = self.status.get("solana_snapshot", {})
            return snap.get("total_value_usdc", 0.0)
        return self.portfolio.balance

    def get_status_dict(self) -> dict:
        """Return full status for API endpoint."""
        return {
            **self.status,
            "wallet_value": self._get_wallet_value(),
            "model_accuracy": self.agent.last_accuracy,
            "consecutive_losses": self._consecutive_losses,
            "auto_paused": self._auto_paused,
            "daily_trades": self._daily_trade_count,
            "feature_importances": self.agent.get_feature_importances(),
            "adaptive_thresholds": self.adaptive.get_all_thresholds(),
        }

    # ------------------------------------------------------------------
    # Auto-pause intelligence
    # ------------------------------------------------------------------

    def _check_auto_pause(self):
        """Auto-pause if: 3 consecutive losses OR accuracy < 35%."""
        # Check timed pause
        if self._auto_paused and self._auto_pause_until:
            if datetime.now() < self._auto_pause_until:
                return True  # Still paused
            else:
                logger.info("Auto-pause expired — resuming")
                self._auto_paused = False
                self._auto_pause_until = None
                self._consecutive_losses = 0
                self.status["auto_pause"] = False
                self._alert("▶️ Auto-pause expired — bot resuming")
                return False

        # Check consecutive losses
        max_losses = getattr(self.config, "AUTO_PAUSE_LOSSES", 3)
        if self._consecutive_losses >= max_losses:
            pause_hours = getattr(self.config, "AUTO_PAUSE_HOURS", 2)
            self._auto_paused = True
            self._auto_pause_until = datetime.now() + timedelta(hours=pause_hours)
            self.status["auto_pause"] = True
            logger.warning(f"⚠️ AUTO-PAUSE: {self._consecutive_losses} consecutive losses — pausing for {pause_hours}h")
            self._alert(f"⚠️ AUTO-PAUSE: {self._consecutive_losses} losses in a row — paused for {pause_hours}h until {self._auto_pause_until.strftime('%H:%M')}")
            return True

        # Check accuracy
        min_acc = getattr(self.config, "AUTO_PAUSE_MIN_ACCURACY", 0.35)
        if self.agent.last_accuracy > 0 and self.agent.last_accuracy < min_acc:
            self._auto_paused = True
            self._auto_pause_until = datetime.now() + timedelta(hours=1)
            self.status["auto_pause"] = True
            logger.warning(f"⚠️ AUTO-PAUSE: accuracy {self.agent.last_accuracy:.1%} < {min_acc:.0%} — pausing 1h")
            self._alert(f"⚠️ AUTO-PAUSE: accuracy {self.agent.last_accuracy:.1%} too low — paused 1h, will retrain")
            return True

        return False

    def _track_trade_result(self, trade: dict):
        """Track wins/losses for auto-pause. Only counts closed positions (not entries)."""
        if not trade:
            return
        # Only track actual position closures — entries have no realized PnL yet
        if trade.get("side") != "close":
            return
        pnl = trade.get("pnl", None)
        if pnl is None:
            return

        if pnl > 0:
            self._consecutive_losses = 0
            logger.info(f"Win: losses reset to 0")
        else:
            self._consecutive_losses += 1
            logger.info(f"Loss #{self._consecutive_losses} (pnl=${pnl:.2f})")

        # Sauvegarder contexte pour apprentissage
        try:
            symbol = trade.get("symbol", "")
            ctx = getattr(self, "_trade_contexts", {}).get(symbol, {})
            now = datetime.now()
            self.db.save_trade_context({
                "timestamp": now.isoformat(),
                "symbol": symbol,
                "side": trade.get("original_side", "?"),
                "pnl": float(pnl),
                "confidence": ctx.get("confidence", 0),
                "regime": ctx.get("regime", "Unknown"),
                "fear_greed": ctx.get("fear_greed", 0),
                "atr_pct": ctx.get("atr_pct", 0),
                "news_score": ctx.get("news_score", 0),
                "hour_utc": now.utcnow().hour,
                "day_of_week": now.weekday(),
                "lstm_agree": 1 if getattr(self, "_lstm_predictions", {}).get(symbol) else 0,
                "mtf_consensus": ctx.get("mtf_consensus", "NEUTRAL"),
                "ob_wall": ctx.get("ob_wall", 0),
            })
        except Exception as e:
            logger.debug(f"Save trade context: {e}")

        # Mettre a jour l'accuracy LSTM
        try:
            symbol = trade.get("symbol", "")
            if not hasattr(self, "_lstm_predictions"):
                self._lstm_predictions = {}
            if not hasattr(self, "_lstm_hits"):
                self._lstm_hits = {}
            lstm_agreed = self._lstm_predictions.get(symbol)
            if lstm_agreed is not None:
                if symbol not in self._lstm_hits:
                    self._lstm_hits[symbol] = {"correct": 0, "total": 0}
                self._lstm_hits[symbol]["total"] += 1
                if (pnl > 0 and lstm_agreed) or (pnl <= 0 and not lstm_agreed):
                    self._lstm_hits[symbol]["correct"] += 1
                acc = self._lstm_hits[symbol]["correct"] / self._lstm_hits[symbol]["total"]
                logger.info(f"[{symbol}] LSTM accuracy updated: {acc:.0%} ({self._lstm_hits[symbol]['correct']}/{self._lstm_hits[symbol]['total']})")
        except Exception as e:
            logger.debug(f"LSTM tracking error: {e}")

        self._recent_trades.append(trade)
        self._recent_trades = self._recent_trades[-20:]  # Keep last 20
        self.status["consecutive_losses"] = self._consecutive_losses

        # Feed Kelly with actual result (entry was recorded earlier via _remember_entry)
        try:
            invested = self._pending_kelly_invested.pop(trade["symbol"], None)
            if invested and invested > 0:
                self.kelly.add_trade_result(trade["symbol"], float(pnl), float(invested))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Funding rate
    # ------------------------------------------------------------------

    def _fetch_funding_rate(self, symbol: str) -> float:
        if not self.config.FUNDING_RATE_ENABLED:
            return 0.0
        try:
            if self.solana and hasattr(self.solana, "_info"):
                sym = symbol.split("/")[0]
                meta = self.solana._info.meta()
                for asset in meta.get("universe", []):
                    if asset.get("name") == sym:
                        fr = float(asset.get("funding", 0))
                        self._funding_rates[sym] = fr
                        return fr
        except Exception:
            pass
        return self._funding_rates.get(symbol.split("/")[0], 0.0)

    def _check_funding_rate(self, symbol: str, signal: str) -> bool:
        if not self.config.FUNDING_RATE_ENABLED:
            return False
        sym = symbol.split("/")[0]
        fr = self._funding_rates.get(sym, 0.0)
        t = self.config.FUNDING_RATE_THRESHOLD

        # Bloquer trades tres defavorables au funding
        if signal == "BUY" and fr > t * 2:
            logger.info(f"Funding fort: {sym} {fr:.4f} — BUY bloque (longs paient cher)")
            return True
        if signal == "SELL" and fr < -t * 2:
            logger.info(f"Funding fort: {sym} {fr:.4f} — SELL bloque (shorts paient cher)")
            return True

        # Signaler les opportunites favorables
        if signal == "SELL" and fr < -t:
            logger.info(f"Funding favorable: {sym} {fr:.4f} negatif + SELL = shorts recus paiement")
        elif signal == "BUY" and fr > t:
            logger.info(f"Funding defavorable: {sym} {fr:.4f} positif + BUY = longs paient")
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _send_daily_summary(self):
        """Send daily PnL summary via Telegram at midnight."""
        try:
            trades = self.db.get_trades_today() if hasattr(self.db, 'get_trades_today') else []
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
            losses = sum(1 for t in trades if t.get('pnl', 0) <= 0)
            wallet = getattr(self, '_wallet_usdc', 0) or (self.solana.get_balance() if self.solana else 0)
            acc = self.agent.last_accuracy if self.agent else 0

            # Buy & Hold benchmark temps reel
            bnh_lines = []
            try:
                for symbol in self.config.SYMBOLS:
                    df = self.fetcher.fetch_ohlcv(symbol, limit=96)  # ~24h de bougies 15min
                    if not df.empty and len(df) >= 2:
                        price_24h_ago = float(df["close"].iloc[0])
                        price_now = float(df["close"].iloc[-1])
                        bnh_pct = (price_now - price_24h_ago) / price_24h_ago * 100
                        bnh_lines.append(f"{symbol.split('/')[0]}: {bnh_pct:+.1f}%")
            except Exception:
                pass

            # Bot performance vs B&H
            start_val = getattr(self, "_daily_start_value", wallet)
            bot_pct = (wallet - start_val) / max(start_val, 1e-8) * 100

            msg = "DAILY SUMMARY\n"
            msg += "Wallet: $" + f"{wallet:.2f}" + "\n"
            msg += "Bot PnL: $" + f"{total_pnl:+.2f}" + " (" + f"{bot_pct:+.1f}" + "%)\n"
            msg += "Trades: " + str(len(trades)) + " (" + str(wins) + "W/" + str(losses) + "L)\n"
            msg += "Accuracy: " + f"{acc:.1%}" + "\n"
            msg += "Drawdown: " + f"{self.status.get('current_drawdown', 0):.1%}" + "\n"
            regime_name = self.status.get('market_regime', {}).get('name', '?')
            msg += "Regime: " + regime_name + "\n"
            if bnh_lines:
                msg += "B&H 24h: " + " | ".join(bnh_lines) + "\n"
            self._alert(msg)
            logger.info("Daily summary sent")
            prev = getattr(self, "_conf_adj", 0.0)
            if abs(prev) > 0.01:
                self._conf_adj = 0.0
                logger.info(f"Daily reset: conf_adj {prev:+.2f} -> 0.0")
        except Exception as e:
            logger.debug(f"Daily summary failed: {e}")


    def _apply_learned_adjustments(self):
        """Adapter les seuils selon les patterns appris (heure, regime, pertes recentes)."""
        try:
            now = datetime.now()
            hour = now.utcnow().hour
            regime = self.status.get("market_regime", {}).get("name", "Unknown")

            hour_stats = self.db.get_stats_by_hour(days=30)
            if hour in hour_stats:
                h = hour_stats[hour]
                if h["trades"] >= 3:
                    wr = h["win_rate"]
                    if wr < 0.35:
                        self._conf_adj = min(getattr(self, "_conf_adj", 0.0) + 0.05, 0.12)
                        logger.info("Time-adj: h" + str(hour) + " wr=" + str(round(wr*100)) + "% adj+5%")
                    elif wr > 0.70:
                        self._conf_adj = max(getattr(self, "_conf_adj", 0.0) - 0.02, -0.05)
                        logger.info("Time-adj: h" + str(hour) + " wr=" + str(round(wr*100)) + "% adj-2%")

            regime_stats = self.db.get_stats_by_regime(days=30)
            if regime in regime_stats:
                r = regime_stats[regime]
                if r["trades"] >= 5:
                    wr = r["win_rate"]
                    if wr < 0.40:
                        self._conf_adj = min(getattr(self, "_conf_adj", 0.0) + 0.04, 0.12)
                        logger.info("Regime-adj: " + regime + " wr=" + str(round(wr*100)) + "% adj+4%")
                    elif wr > 0.65:
                        self._conf_adj = max(getattr(self, "_conf_adj", 0.0) - 0.02, -0.05)
                        logger.info("Regime-adj: " + regime + " wr=" + str(round(wr*100)) + "% adj-2%")

            recent_losses = self.db.get_recent_losses_by_pattern(hours=48)
            if len(recent_losses) >= 3:
                regime_losses = [l for l in recent_losses if l.get("regime") == regime]
                if len(regime_losses) >= 3:
                    self._conf_adj = min(getattr(self, "_conf_adj", 0.0) + 0.03, 0.12)
                    logger.info("Pattern-adj: " + str(len(regime_losses)) + " pertes en " + regime + " adj+3%")

            self.config.MIN_CONFIDENCE = round(
                min(max(getattr(self, "_regime_min_conf", 0.52) + getattr(self, "_conf_adj", 0.0), 0.50), 0.75), 3
            )
            self.status["learned_min_conf"] = self.config.MIN_CONFIDENCE
            self.status["conf_adj"] = round(getattr(self, "_conf_adj", 0.0), 3)
        except Exception as e:
            logger.debug("_apply_learned_adjustments: " + str(e))


    def _auto_tune_confidence(self):
        try:
            now = datetime.now()
            last = getattr(self, '_last_autotune', None)
            if last and (now - last).total_seconds() < 86400:
                return
            self._last_autotune = now
            regime_stats = self.db.get_stats_by_regime(days=7)
            if not regime_stats:
                return
            total_trades = sum(s['trades'] for s in regime_stats.values())
            if total_trades < 10:
                return
            total_wins = sum(s['wins'] for s in regime_stats.values())
            global_wr = total_wins / max(total_trades, 1)
            total_pnl = sum(s['total_pnl'] for s in regime_stats.values())
            current = self.config.MIN_CONFIDENCE
            if global_wr > 0.65 and total_pnl > 0:
                new_conf = max(current - 0.01, 0.50)
                self.config.MIN_CONFIDENCE = new_conf
                logger.info('AutoTune: wr=' + str(round(global_wr*100)) + '% -> seuil ' + str(new_conf))
            elif global_wr < 0.45 or total_pnl < -0.5:
                new_conf = min(current + 0.02, 0.70)
                self.config.MIN_CONFIDENCE = new_conf
                logger.info('AutoTune: wr=' + str(round(global_wr*100)) + '% -> seuil ' + str(new_conf))
            self.status['autotune'] = {'win_rate': round(global_wr,3), 'total_pnl': round(total_pnl,2), 'trades': total_trades}
        except Exception as e:
            logger.debug('_auto_tune_confidence: ' + str(e))

    def _on_breaking_news(self, msg: dict):
        """React to high-impact breaking news from Telegram channels."""
        text = msg.get("text", "")[:200]
        channel = msg.get("channel", "?")
        sentiment = msg.get("sentiment", "neutral")
        self._alert(f"⚡ BREAKING [{channel}]: {text[:200]} | {sentiment}")
        logger.info(f"BREAKING NEWS [{channel}] {sentiment}: {text[:100]}")

    def _loop(self):
        while self.running:
            if not self.paused and not self._check_auto_pause():
                try:
                    self._cycle()
                except Exception as e:
                    logger.error(f"Cycle error: {e}")
                    self.status["errors"].append({"time": datetime.now().isoformat(), "error": str(e)})
                    self.status["errors"] = self.status["errors"][-10:]
            sleep_s = min(_TF_SECONDS.get(self.config.TIMEFRAME, 3600) // 10, 60)
            time.sleep(max(sleep_s, 10))

    def _cycle(self):
        self.status["cycle_count"] += 1
        if self.status["cycle_count"] == 1:
            self._reconcile_positions()
        self.status["last_update"] = datetime.now().isoformat()

        if datetime.now().date() > self._daily_reset.date():
            self._daily_trade_count = 0
            self._daily_reset = datetime.now()

        if self.solana:
            self.solana.refresh_prices()
            self.status["solana_snapshot"] = self.solana.get_snapshot()

        self._process_scale_ins()

        for sym in self.config.SYMBOLS:
            try:
                self._process_symbol(sym)
            except Exception as e:
                logger.error(f"Error on {sym}: {e}")

        if self.solana:
            sol_snap = self.status.get("solana_snapshot", {})
            self.db.save_portfolio_snapshot({
                "balance": sol_snap.get("balances", {}).get("USDC", 0.0),
                "total_value": sol_snap.get("total_value_usdc", 0.0),
                "positions": sol_snap.get("balances", {}),
            })

        self._check_drawdown()
        self._update_market_regime()

        cyc = self.status.get("cycle_count", 0)
        if cyc > 0 and cyc % 240 == 0:
            self._apply_learned_adjustments()
        if cyc > 0 and cyc % 5760 == 0:
            self._auto_tune_confidence()

        self.status["funding_rates"] = dict(self._funding_rates)
        self.status["news_sentiment"] = self.news.get_summary()

        # Daily summary at midnight
        now = datetime.now()
        if now.hour == 0 and now.minute < 2:
            if not hasattr(self, "_last_summary_date") or self._last_summary_date != now.date():
                self._send_daily_summary()
                self._last_summary_date = now.date()
        self.status["claude_sentiment"] = self.claude_sentiment.get_full_result()

        # Multi-timeframe training (every 4h)
        try:
            if self.mtf and self.mtf.needs_retraining():
                for sym in self.config.SYMBOLS:
                    self.mtf.train_all(sym)
        except Exception as e:
            logger.debug(f"MTF training: {e}")
        if self.tg_monitor:
            self.status["telegram_monitor"] = self.tg_monitor.get_stats()

        # Reload config overrides periodically
        if self.status.get("cycle_count", 0) % 50 == 0:
            try:
                from config import _apply_overrides
                _apply_overrides(self.config)
            except Exception:
                pass

        if self.agent.needs_retraining():
            self._retrain()
            # Drift detection: tracker les 3 dernieres accuracies
            try:
                if not hasattr(self, "_recent_accuracies"):
                    self._recent_accuracies = []
                acc = float(getattr(self.agent, "last_accuracy", 0.5))
                self._recent_accuracies.append(acc)
                self._recent_accuracies = self._recent_accuracies[-3:]
                if len(self._recent_accuracies) == 3 and all(a < 0.55 for a in self._recent_accuracies):
                    avg = sum(self._recent_accuracies) / 3
                    logger.warning(f"DRIFT detecte: 3 retrains consecutifs <55% (moyenne {avg:.1%})")

                    self._conf_adj = min(getattr(self, "_conf_adj", 0.0) + 0.06, 0.12)
                    self.config.MIN_CONFIDENCE = round(
                        min(getattr(self, "_regime_min_conf", 0.52) + self._conf_adj, 0.75), 3
                    )
                    logger.warning(f"DRIFT: conf_adj={self._conf_adj:+.2f} seuil={self.config.MIN_CONFIDENCE:.2f}")

                    # Action 2 — Reduire la taille des positions de 50%
                    self.config.POSITION_SIZE_BASE_USDC = max(
                        self.config.POSITION_SIZE_BASE_USDC * 0.5,
                        self.config.POSITION_SIZE_MIN_USDC
                    )
                    logger.warning(f"DRIFT: taille reduite -> ${self.config.POSITION_SIZE_BASE_USDC:.2f}")

                    self._alert("DRIFT: seuil " + str(round(self.config.MIN_CONFIDENCE*100)) + "%")

                    # Reset accuracies pour eviter alertes en boucle
                    self._recent_accuracies = []
            except Exception:
                pass

        if self.status.get('cycle_count', 0) % 10 == 0:
            self._reajust_positions()

        # Sauvegarder l'etat toutes les 40 cycles (~10 min)
        if self.status.get('cycle_count', 0) % 40 == 0:
            self._save_bot_state()

        self._notify("cycle", self.status)

    # ------------------------------------------------------------------
    # Scale in/out
    # ------------------------------------------------------------------

    def _process_scale_ins(self):
        if not self.config.SCALE_ENABLED:
            return
        now = datetime.now()
        done = []
        for sym, info in self._scale_pending.items():
            if now >= info["next_time"]:
                logger.info(f"Scale-in {info['step']}/{info['total_steps']} for {sym}")
                if info["side"] == "BUY":
                    trade = self.solana.buy(sym, info["step_size"], confidence=info["confidence"])
                else:
                    trade = self.solana.sell(sym, info["step_size"], confidence=info["confidence"])
                if trade:
                    trade["reason"] = f"scale_in_step{info['step']}"
                    self.db.save_trade(trade)
                    # Accumulate invested size for Kelly tracking
                    self._pending_kelly_invested[sym] = (
                        self._pending_kelly_invested.get(sym, 0.0) + info["step_size"]
                    )
                info["step"] += 1
                if info["step"] > info["total_steps"]:
                    done.append(sym)
                else:
                    info["next_time"] = now + timedelta(seconds=self.config.SCALE_IN_INTERVAL_SEC)
        for sym in done:
            del self._scale_pending[sym]

    def _start_scale_in(self, sym, side, total_size, confidence):
        steps = self.config.SCALE_IN_STEPS
        step_size = round(total_size / steps, 2)
        if step_size < self.config.POSITION_SIZE_MIN_USDC:
            return False
        if side == "BUY":
            trade = self.solana.buy(sym, step_size, confidence=confidence)
        else:
            trade = self.solana.sell(sym, step_size, confidence=confidence)
        if trade:
            trade["reason"] = "scale_in_step1"
            self.db.save_trade(trade)
            self._notify("trade", trade)
            # Harmonized cooldown key: always use the ccxt-style "BASE/USDT" symbol
            self._last_trade_time[f"{sym}/USDT"] = datetime.now()
            self._daily_trade_count += 1
            # Track initial investment for Kelly (step1 + future steps)
            self._pending_kelly_invested[sym] = (
                self._pending_kelly_invested.get(sym, 0.0) + step_size
            )
            if steps > 1:
                self._scale_pending[sym] = {
                    "side": side, "step_size": step_size, "step": 2,
                    "total_steps": steps, "confidence": confidence,
                    "next_time": datetime.now() + timedelta(seconds=self.config.SCALE_IN_INTERVAL_SEC),
                }
            emoji = "🟢" if side == "BUY" else "🔴"
            self._alert(f"{emoji} {side} {sym} ${step_size:.2f} (1/{steps}) | conf={confidence:.0%}")
            return True
        return False

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _check_cooldown(self, symbol):
        last = self._last_trade_time.get(symbol)
        if not last:
            return False
        # Check cooldown
        if (datetime.now() - last).total_seconds() < self.config.TRADE_COOLDOWN_SEC:
            return True
        # Also skip if same candle hour (1h timeframe = same hour)
        if last.hour == datetime.now().hour and last.date() == datetime.now().date():
            logger.debug(f"[{symbol}] Same candle hour — skipping")
            return True
        return False

    def _check_daily_limit(self):
        return self._daily_trade_count >= self.config.MAX_TRADES_PER_DAY

    def _check_volume_filter(self, symbol, df):
        v = self.processor.get_current_volume_ratio(df)
        return self.adaptive.check_volume(symbol, v)

    def _check_volatility_filter(self, symbol, df):
        a = self.processor.get_current_atr_pct(df)
        return self.adaptive.check_volatility(symbol, a)

    def _check_trend_filter(self, symbol, df, signal):
        if "trend_strength" not in df.columns:
            return False
        t = float(df["trend_strength"].iloc[-1])
        return self.adaptive.check_trend(symbol, t, signal)

    def _update_market_regime(self):
        """Détecter le régime de marché et adapter MIN_CONFIDENCE + taille position.
        Régimes: 0=Range, 1=Trend Up, 2=Trend Down, 3=High Volatility
        """
        try:
            REGIME_NAMES = {0: "Range", 1: "Trend↑", 2: "Trend↓", 3: "HighVol"}
            regime_counts = {}
            dominant_regime = 0

            for symbol in self.config.SYMBOLS:
                df = self.fetcher.fetch_ohlcv(symbol, limit=50)
                if df.empty:
                    continue
                df = self.processor.add_indicators(df)
                if "market_regime" not in df.columns:
                    continue
                regime = int(df["market_regime"].iloc[-1])
                strength = float(df["regime_strength"].iloc[-1]) if "regime_strength" in df.columns else 0.5
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                self.status[f"regime_{symbol.split('/')[0]}"] = {
                    "regime": regime,
                    "name": REGIME_NAMES.get(regime, "Unknown"),
                    "strength": round(strength, 3),
                }

            # Régime dominant sur tous les symboles
            if regime_counts:
                dominant_regime = max(regime_counts, key=regime_counts.get)

            prev_regime = getattr(self, "_current_regime", -1)
            self._current_regime = dominant_regime

            # Adapter les seuils selon le régime
            base_conf = 0.52
            base_size = self.config.POSITION_SIZE_BASE_USDC

            if dominant_regime == 0:
                self._regime_min_conf = base_conf + 0.05
                size_mult = 0.7
                regime_note = "Range: seuils élevés, taille réduite"
            elif dominant_regime == 1:
                self._regime_min_conf = base_conf
                size_mult = 1.2
                regime_note = "Trend↑: conditions optimales"
            elif dominant_regime == 2:
                self._regime_min_conf = base_conf + 0.03
                size_mult = 0.8
                regime_note = "Trend↓: prudence"
            else:
                self._regime_min_conf = base_conf + 0.08
                size_mult = 0.5
                regime_note = "HighVol: taille minimale"
            self.config.MIN_CONFIDENCE = round(
                min(max(self._regime_min_conf + getattr(self, "_conf_adj", 0.0), 0.50), 0.75), 3
            )

            # Appliquer la taille (respecter min/max)
            new_size = round(min(max(base_size * size_mult,
                                    self.config.POSITION_SIZE_MIN_USDC),
                                    self.config.POSITION_SIZE_MAX_USDC * 1.5), 2)
            self.config.POSITION_SIZE_BASE_USDC = new_size

            self.status["market_regime"] = {
                "dominant": dominant_regime,
                "name": REGIME_NAMES.get(dominant_regime, "Unknown"),
                "min_confidence": round(self.config.MIN_CONFIDENCE, 3),
                "position_size": new_size,
                "note": regime_note,
            }

            logger.info(f"Régime actuel: {REGIME_NAMES.get(dominant_regime,'?')} | conf={self.config.MIN_CONFIDENCE:.2f} | size=${new_size}")
            if dominant_regime != prev_regime:
                logger.info(f"Régime changé: {REGIME_NAMES.get(prev_regime,'?')} → {REGIME_NAMES.get(dominant_regime,'?')} | conf={self.config.MIN_CONFIDENCE:.2f} size=${new_size}")
                self._alert(f"📊 Régime: {REGIME_NAMES.get(dominant_regime,'?')} | conf≥{self.config.MIN_CONFIDENCE:.0%} | size=${new_size}")

        except Exception as e:
            logger.warning(f"_update_market_regime ERROR: {e}", exc_info=True)

    def _check_drawdown(self):
        """Circuit breaker multi-niveaux.
        L1: DD > 5%  → taille position réduite 50%
        L2: DD > 10% → pause 6h
        L3: DD > 15% → arrêt complet
        Daily: perte > 8% en 24h → pause 6h
        """
        if not self.solana:
            return
        snap = self.status.get("solana_snapshot", {})
        total = snap.get("total_value_usdc", 0.0)
        if total <= 0:
            return

        # ── Peak tracking ────────────────────────────────────────────────
        peak = getattr(self, "_peak_value", 0.0)
        if total > peak:
            self._peak_value = total
            peak = total

        if peak <= 0:
            return

        dd = (peak - total) / peak
        self.status["current_drawdown"] = dd
        self.status["peak_value"] = round(peak, 2)

        # ── Daily loss tracking ──────────────────────────────────────────
        now = datetime.now()
        if not hasattr(self, "_daily_start_value") or not hasattr(self, "_daily_start_date"):
            self._daily_start_value = total
            self._daily_start_date = now.date()
        elif self._daily_start_date != now.date():
            self._daily_start_value = total
            self._daily_start_date = now.date()

        daily_loss = (self._daily_start_value - total) / max(self._daily_start_value, 1e-8)
        self.status["daily_loss"] = round(daily_loss, 4)

        # ── Niveau 3 : Arrêt complet ─────────────────────────────────────
        l3 = getattr(self.config, "CIRCUIT_L3_DD", 0.15)
        if dd >= l3 and not self.paused:
            logger.warning(f"🚨 CIRCUIT L3: DD {dd:.1%} — arrêt complet!")
            self._alert(f"CIRCUIT BREAKER L3: DD {dd:.1%} - arret complet | Peak ${peak:.2f}")
            self.pause()
            return

        # ── Niveau 2 : Pause 6h ──────────────────────────────────────────
        l2 = getattr(self.config, "CIRCUIT_L2_DD", 0.10)
        last_l2 = getattr(self, "_circuit_l2_time", None)
        l2_cooldown = not last_l2 or (now - last_l2).total_seconds() > 21600  # 6h
        if dd >= l2 and not self.paused and l2_cooldown:
            logger.warning(f"⚠️ CIRCUIT L2: DD {dd:.1%} — pause 6h")
            self._alert(f"⚠️ CIRCUIT L2: DD {dd:.1%} | Pause 6h Peak ${peak:.2f} → ${total:.2f}")
            self._circuit_l2_time = now
            self.pause()
            return

        # ── Niveau 1 : Réduire taille 50% ───────────────────────────────
        l1 = getattr(self.config, "CIRCUIT_L1_DD", 0.05)
        if dd >= l1:
            base = 9.0  # Taille de base originale
            reduced = round(base * 0.5, 2)
            if self.config.POSITION_SIZE_BASE_USDC > reduced:
                self.config.POSITION_SIZE_BASE_USDC = reduced
                logger.info(f"⚡ CIRCUIT L1: DD {dd:.1%} — taille réduite ${reduced}")
                self.status["circuit_breaker"] = f"L1 actif: DD {dd:.1%}"
        else:
            self.status["circuit_breaker"] = "OK"

        # ── Perte journalière ────────────────────────────────────────────
        daily_limit = getattr(self.config, "CIRCUIT_DAILY_LOSS_PCT", 0.08)
        last_daily = getattr(self, "_circuit_daily_time", None)
        daily_cooldown = not last_daily or (now - last_daily).total_seconds() > 21600
        if daily_loss >= daily_limit and not self.paused and daily_cooldown:
            logger.warning(f"📉 CIRCUIT DAILY: perte {daily_loss:.1%} en 24h — pause 6h")
            self._alert(f"📉 PERTE JOURNALIÈRE {daily_loss:.1%} — Pause 6h ${self._daily_start_value:.2f} → ${total:.2f}")
            self._circuit_daily_time = now
            self.pause()

    def _calc_position_size(self, confidence, symbol=None):
        """Compute position size (USDC). Supports Kelly, accuracy multiplier, drawdown reduction."""
        # Base: confidence-scaled
        base = self.config.POSITION_SIZE_BASE_USDC
        size = base * (confidence / 0.55)

        # Kelly override (when enough trade history exists)
        if getattr(self.config, "KELLY_ENABLED", False) and symbol:
            try:
                wallet = self._get_wallet_value()
                if wallet > 0:
                    kelly_size = self.kelly.calculate_position_size(
                        symbol=symbol,
                        wallet_usdc=wallet,
                        confidence=confidence,
                        model_accuracy=self.agent.last_accuracy or 0.5,
                    )
                    if kelly_size > 0:
                        size = kelly_size
            except Exception as e:
                logger.debug(f"Kelly sizing failed: {e}")

        # Drawdown reduction: if we're in drawdown, cut size
        if getattr(self.config, "DRAWDOWN_PROTECTION", False):
            dd = self.status.get("current_drawdown", 0.0) or 0.0
            if dd >= getattr(self.config, "DD_REDUCE_THRESHOLD", 0.10):
                factor = getattr(self.config, "DD_REDUCE_FACTOR", 0.5)
                size *= factor
                logger.info(f"Drawdown {dd:.1%} → size reduced ×{factor}")

        # Bounds
        size = max(min(size, self.config.POSITION_SIZE_MAX_USDC), self.config.POSITION_SIZE_MIN_USDC)
        return round(size, 2)

    def _update_trailing_stop(self, sym, price, side, atr, entry=0):
        """Trailing stop dynamique: plus serre en gros profit pour lock les gains.

        - 0-2% profit:   ATR x 2.5  (large, laisser respirer)
        - 2-5% profit:   ATR x 2.0  (moyen)
        - 5-10% profit:  ATR x 1.5  (resserre)
        - 10%+ profit:   ATR x 1.0  (tres serre, lock gains)
        """
        # Calculer profit % si entry connu, sinon utiliser multiplier base
        if entry > 0:
            if side == "long":
                profit_pct = (price - entry) / entry
            else:
                profit_pct = (entry - price) / entry

            if profit_pct >= 0.10:
                mult = 1.0
            elif profit_pct >= 0.05:
                mult = 1.5
            elif profit_pct >= 0.02:
                mult = 2.0
            else:
                mult = self.config.TRAILING_STOP_ATR  # 2.5 par defaut
        else:
            mult = self.config.TRAILING_STOP_ATR

        dist = atr * mult
        ts = self._trailing_stops.get(sym)
        if side == "long":
            ns = price - dist
            if ts is None or ns > ts.get("price", 0):
                self._trailing_stops[sym] = {"side": "long", "price": ns, "mult": mult}
        elif side == "short":
            ns = price + dist
            if ts is None or ns < ts.get("price", float("inf")):
                self._trailing_stops[sym] = {"side": "short", "price": ns, "mult": mult}

    def _check_trailing_stop(self, sym, price):
        ts = self._trailing_stops.get(sym)
        if not ts:
            return False
        if ts["side"] == "long" and price <= ts["price"]:
            logger.info(f"TRAILING STOP {sym} LONG: ${price:.2f} <= ${ts['price']:.2f}")
            return True
        if ts["side"] == "short" and price >= ts["price"]:
            logger.info(f"TRAILING STOP {sym} SHORT: ${price:.2f} >= ${ts['price']:.2f}")
            return True
        return False

    # ------------------------------------------------------------------
    # Process symbol
    # ------------------------------------------------------------------


    def _reajust_positions(self):
        if not self.solana:
            return
        for symbol in self.config.SYMBOLS:
            sol_sym = symbol.split("/")[0]
            try:
                pos = self.solana.get_position(sol_sym)
                if pos["size"] == 0:
                    continue
                df = self.fetcher.fetch_ohlcv(symbol, limit=100)
                if df.empty:
                    continue
                df = self.processor.add_indicators(df)
                price = float(df["close"].iloc[-1])
                atr   = self.processor.get_current_atr(df)
                entry = pos["entry_price"]
                side  = pos["side"]
                upnl  = pos["unrealised_pnl"]
                age_h = 0
                last_t = self._last_trade_time.get(symbol)
                if last_t:
                    age_h = (datetime.now() - last_t).total_seconds() / 3600
                pct = (price - entry) / entry if side == "long" else (entry - price) / entry
                alerts = []
                if age_h > 12 and pct < -0.01:
                    alerts.append("Position bloquee " + str(round(age_h)) + "h PnL:" + str(round(pct*100,1)) + "%")
                cached_atr = getattr(self, "_entry_atr_" + sol_sym, None)
                if cached_atr is None:
                    setattr(self, "_entry_atr_" + sol_sym, atr)
                elif atr > cached_atr * 1.3:
                    alerts.append("ATR +" + str(round((atr/cached_atr-1)*100)) + "% volatilite accrue")
                    setattr(self, "_entry_atr_" + sol_sym, atr)
                if side == "long":
                    try:
                        news = self.news.get_sentiment_for_symbol(symbol) if hasattr(self.news, "get_sentiment_for_symbol") else self.news.get_sentiment()
                        if news.get("score", 0) < -0.5:
                            sl_price = entry - atr * 1.5
                            if price > sl_price:
                                alerts.append("Bearish fort SL suggere $" + str(round(sl_price, 2)))
                    except Exception:
                        pass
                if pct > 0.05 and age_h > 1:
                    trail = self._trailing_stops.get(sol_sym)
                    if trail and abs(price - trail) / price > 0.03:
                        alerts.append("Position +" + str(round(pct*100,1)) + "% trailing $" + str(round(trail,2)))
                if alerts:
                    msg = ("Position " + sol_sym + " " + side.upper() + " @$" + str(round(entry,2))
                           + " Prix:$" + str(round(price,2)) + " PnL:$" + str(round(upnl,2))
                           + " (" + str(round(pct*100,1)) + "%) Age:" + str(round(age_h,1)) + "h"
                           + "\n" + "\n".join(alerts))
                    logger.info("[" + symbol + "] Position review: " + " | ".join(alerts))

                    # Anti-spam: alerter max 1x/heure par symbole sauf urgence
                    if not hasattr(self, "_last_pos_alert"):
                        self._last_pos_alert = {}
                    last_alert_t = self._last_pos_alert.get(sol_sym)
                    is_urgent = any(k in msg for k in ["bloquee", "ATR", "trailing"])
                    cooldown_ok = not last_alert_t or (datetime.now() - last_alert_t).total_seconds() > 3600
                    if is_urgent or cooldown_ok:
                        self._alert(msg)
                        self._last_pos_alert[sol_sym] = datetime.now()
            except Exception as e:
                logger.debug("_reajust_positions " + symbol + ": " + str(e))
    def _process_symbol(self, symbol):
        df = self.fetcher.fetch_ohlcv(symbol, limit=self.config.HISTORY_LIMIT)
        if df.empty:
            return

        df = self.processor.add_indicators(df)
        price = float(df["close"].iloc[-1])
        atr = self.processor.get_current_atr(df)
        self.portfolio.update_price(symbol, price)
        self.status["current_prices"][symbol] = price
        sol_sym = symbol.split("/")[0]
        funding_rate = self._fetch_funding_rate(symbol)

        # Auto-calibrate filters for this symbol
        self.adaptive.calibrate(symbol, df)

        # Position management
        if self.solana and hasattr(self.solana, "get_position"):
            pos = self.solana.get_position(sol_sym)
            self.status["positions"][sol_sym] = pos

            if pos["size"] > 0:
                entry = pos["entry_price"]

                if self.config.USE_TRAILING_STOP and atr > 0:
                    # Delay trailing stop: wait 3h after entry before activating
                    pos_age_h = 0
                    try:
                        last_trade_t = self._last_trade_time.get(symbol)
                        if last_trade_t:
                            pos_age_h = (datetime.now() - last_trade_t).total_seconds() / 3600
                    except Exception:
                        pos_age_h = 99
                    if pos_age_h >= 1:  # Trailing apres 1h (etait 3h)
                        self._update_trailing_stop(sol_sym, price, pos["side"], atr, entry=entry)
                    else:
                        logger.debug(f"[{symbol}] Trailing stop delayed: position age {pos_age_h:.1f}h < 3h")

                if self._check_trailing_stop(sol_sym, price):
                    trade = self.solana.close_position(sol_sym, reason="trailing_stop")
                    if trade:
                        self.db.save_trade(trade)
                        self._track_trade_result(trade)
                        self._trailing_stops.pop(sol_sym, None)
                        self._reset_tp_level(sol_sym)
                        setattr(self, f"_scaled_{sol_sym}", False)
                        self._alert(f"📉 TRAILING STOP {sol_sym} @ ${price:.2f} | PnL: ${trade.get('pnl', 0):.2f}")
                    return

                # Scale out
                if self.config.SCALE_ENABLED and entry > 0:
                    pct = (price - entry) / entry if pos["side"] == "long" else (entry - price) / entry
                    if pct >= self.config.SCALE_OUT_PROFIT_PCT and not getattr(self, f"_scaled_{sol_sym}", False):
                        half = round(pos["size"] / 2, 1)
                        try:
                            # Use close_partial (reduce-only) for correctness
                            result = self.solana.close_partial(sol_sym, half, reason="scale_out_50pct")
                            if result:
                                setattr(self, f"_scaled_{sol_sym}", True)
                                self.db.save_trade(result)
                                self._alert(f"📊 Scale-out 50% {sol_sym} profit={pct:.1%}")
                        except Exception as e:
                            logger.error(f"Scale-out: {e}")

                # ATR SL/TP
                if entry > 0 and atr > 0:
                    sl_d = atr * self.config.ATR_SL_MULTIPLIER

                    if pos["side"] == "long":
                        sl_hit = price <= entry - sl_d
                    else:
                        sl_hit = price >= entry + sl_d

                    if sl_hit:
                        trade = self.solana.close_position(sol_sym, reason="atr_stop_loss")
                        if trade:
                            self.db.save_trade(trade)
                            self._track_trade_result(trade)
                            self._trailing_stops.pop(sol_sym, None)
                            self._reset_tp_level(sol_sym)
                            setattr(self, f"_scaled_{sol_sym}", False)
                            self._alert(f"🛑 SL {sol_sym} @ ${price:.2f} | PnL: ${trade.get('pnl', 0):.2f}")
                        return

                    # ── Multi-TP (each level checked independently) ────────────
                    if getattr(self.config, "MULTI_TP_ENABLED", False):
                        tp_levels = list(self.config.MULTI_TP_LEVELS)
                        tp_sizes = list(self.config.MULTI_TP_SIZES)
                        current_level = self._get_tp_level(sol_sym)

                        # Progress through as many levels as the current price allows
                        while current_level < len(tp_levels):
                            tp_mult = tp_levels[current_level]
                            tp_size_pct = tp_sizes[current_level]
                            if pos["side"] == "long":
                                tp_price = entry + atr * tp_mult
                                level_hit = price >= tp_price
                            else:
                                tp_price = entry - atr * tp_mult
                                level_hit = price <= tp_price
                            if not level_hit:
                                break

                            is_last = (current_level == len(tp_levels) - 1)
                            if is_last:
                                trade = self.solana.close_position(
                                    sol_sym, reason=f"tp{current_level+1}_final"
                                )
                                self._reset_tp_level(sol_sym)
                                self._trailing_stops.pop(sol_sym, None)
                                setattr(self, f"_scaled_{sol_sym}", False)
                            else:
                                partial_size = pos["size"] * tp_size_pct
                                trade = self.solana.close_partial(
                                    sol_sym, partial_size,
                                    reason=f"tp{current_level+1}_partial"
                                )
                                current_level += 1
                                self._set_tp_level(sol_sym, current_level)

                            if trade:
                                self.db.save_trade(trade)
                                self._track_trade_result(trade)
                                pnl = trade.get("pnl", 0)
                                self._alert(
                                    f"🎯 TP{current_level if is_last else current_level}"
                                    f"/{len(tp_levels)} {sol_sym} @ ${price:.2f} "
                                    f"| PnL: ${pnl:.2f} ({int(tp_size_pct*100)}% closed)"
                                )
                            if is_last:
                                return
                            # Refresh pos for next iteration in case of rapid successive fills
                            pos = self.solana.get_position(sol_sym)
                            if pos["size"] == 0:
                                return

                    else:
                        # Single TP fallback
                        tp_d = atr * self.config.ATR_TP_MULTIPLIER
                        if pos["side"] == "long":
                            tp_hit = price >= entry + tp_d
                        else:
                            tp_hit = price <= entry - tp_d
                        if tp_hit:
                            trade = self.solana.close_position(sol_sym, reason="atr_take_profit")
                            if trade:
                                self.db.save_trade(trade)
                                self._track_trade_result(trade)
                                self._trailing_stops.pop(sol_sym, None)
                                self._reset_tp_level(sol_sym)
                                setattr(self, f"_scaled_{sol_sym}", False)
                                self._alert(f"🎯 TP {sol_sym} @ ${price:.2f} | PnL: ${trade.get('pnl', 0):.2f}")
                            return

        if not self.agent.is_trained:
            return
        if self._check_cooldown(symbol) or self._check_daily_limit():
            return

        # ML prediction
        fear_greed = getattr(self.solana, "fear_greed", 50) if self.solana else 50

        # Dynamic position sizing: bet more when accuracy is high
        recent_acc = self.agent.last_accuracy if self.agent else 0.5
        if recent_acc >= 0.58:
            accuracy_multiplier = 1.5  # High accuracy → bigger position
        elif recent_acc >= 0.54:
            accuracy_multiplier = 1.2
        elif recent_acc >= 0.50:
            accuracy_multiplier = 1.0
        else:
            accuracy_multiplier = 0.7  # Low accuracy → smaller position
        # Use Claude AI for deep sentiment analysis
        news_summary = self.news.get_sentiment_for_symbol(symbol) if hasattr(self.news, 'get_sentiment_for_symbol') else self.news.get_sentiment()
        headlines = self.news.get_headlines()
        if self.claude_sentiment.is_enabled and headlines:
            claude_result = self.claude_sentiment.analyze_headlines(headlines)
            news_score = self.claude_sentiment.get_score_normalized()
            impact = self.claude_sentiment.get_market_impact()
            if impact == "critical":
                logger.warning(f"[{symbol}] ⚡ CRITICAL market impact detected!")
                self._alert(f"⚡ CRITICAL NEWS: {self.claude_sentiment.get_analysis()}")
        else:
            news_score = self.news.get_score_normalized()
        claude_label = self.claude_sentiment.get_full_result().get("label", "?") if self.claude_sentiment.is_enabled else "?"
        news_label = self.news.get_sentiment().get("label", "?")
        logger.info(f"[{symbol}] News: {news_label} | Claude: {claude_label} ({self.claude_sentiment.get_score():+.2f})")

        # ====== MOMENTUM OVERRIDE — detection pump/dump pour ignorer sentiment lent ======
        try:
            df_5m = self.fetcher.fetch_ohlcv(symbol, limit=10, timeframe="5m")
            if not df_5m.empty and len(df_5m) >= 7:
                price_now = float(df_5m["close"].iloc[-1])
                price_15m_ago = float(df_5m["close"].iloc[-4])
                price_30m_ago = float(df_5m["close"].iloc[-7])
                chg_15m = (price_now - price_15m_ago) / price_15m_ago
                chg_30m = (price_now - price_30m_ago) / price_30m_ago

                if chg_15m >= 0.05:
                    logger.warning(f"[{symbol}] 🚀 PUMP FORT +{chg_15m:.1%} en 15min — override news bearish")
                    self._alert(f"🚀 {symbol} PUMP +{chg_15m:.1%} en 15min")
                    news_score = max(news_score, 0.4)
                elif chg_15m >= 0.03 or chg_30m >= 0.05:
                    logger.info(f"[{symbol}] Momentum bullish 15m={chg_15m:+.1%} 30m={chg_30m:+.1%}")
                    news_score = max(news_score, 0.15)
                elif chg_15m <= -0.05:
                    logger.warning(f"[{symbol}] 📉 DUMP FORT {chg_15m:.1%} en 15min — override news bullish")
                    self._alert(f"📉 {symbol} DUMP {chg_15m:.1%} en 15min")
                    news_score = min(news_score, -0.4)
                elif chg_15m <= -0.03 or chg_30m <= -0.05:
                    logger.info(f"[{symbol}] Momentum bearish 15m={chg_15m:+.1%} 30m={chg_30m:+.1%}")
                    news_score = min(news_score, -0.15)
        except Exception as e:
            logger.debug(f"Momentum detection failed: {e}")
        # ==================================================================================
        # Cross-asset: get BTC data for altcoin prediction
        btc_ret_1h = 0.0
        btc_ret_4h = 0.0
        btc_rsi_val = 50.0
        if symbol != "BTC/USDT":
            try:
                if not self._btc_data or (datetime.now() - self._btc_data.get("ts", datetime.min)).seconds > 300:
                    btc_df = self.fetcher.fetch_ohlcv("BTC/USDT", limit=100)
                    if not btc_df.empty:
                        btc_df = self.processor.add_indicators(btc_df)
                        self._btc_data = {
                            "ret_1h": float(btc_df["returns"].iloc[-1]) if "returns" in btc_df else 0,
                            "ret_4h": float(btc_df["returns_5"].iloc[-1]) if "returns_5" in btc_df else 0,
                            "rsi": float(btc_df["rsi"].iloc[-1]) if "rsi" in btc_df else 50,
                            "ts": datetime.now(),
                        }
                btc_ret_1h = self._btc_data.get("ret_1h", 0)
                btc_ret_4h = self._btc_data.get("ret_4h", 0)
                btc_rsi_val = self._btc_data.get("rsi", 50)
            except Exception:
                pass

        # Fetch market microstructure data (orderbook L2 + OI + liquidations)
        mkt = self.market_data.get_all_features(sol_sym)
        ob_wall = mkt.get("ob_wall_signal", 0)
        ob_absorption = mkt.get("ob_absorption", 0.0)
        # Add Telegram channel sentiment
        tg_sentiment = 50.0
        if self.tg_monitor and self.tg_monitor.is_running:
            tg_sentiment = self.tg_monitor.get_sentiment_normalized()

        features = self.processor.get_current_features(
            df.dropna(), fear_greed=fear_greed, funding_rate=funding_rate,
            news_sentiment=news_score,
            btc_returns_1h=btc_ret_1h, btc_returns_4h=btc_ret_4h, btc_rsi=btc_rsi_val,
            orderbook_ratio=mkt.get("orderbook_ratio", 1.0),
            oi_change=mkt.get("oi_change", 0.0),
            liquidation_score=mkt.get("liquidation_score", 0.0),
        )
        prediction = self.agent.predict(features)
        self.status["last_signals"][symbol] = prediction
        signal = prediction["signal"]
        confidence = prediction["confidence"]

        # Stocker le contexte courant pour le tracking apprentissage
        if not hasattr(self, "_trade_contexts"):
            self._trade_contexts = {}
        self._trade_contexts[symbol] = {
            "confidence": float(confidence),
            "regime": self.status.get("market_regime", {}).get("name", "Unknown"),
            "fear_greed": int(fear_greed) if isinstance(fear_greed, (int, float)) else 0,
            "atr_pct": float(self.processor.get_current_atr_pct(df)),
            "news_score": float(news_score),
            "ob_wall": int(mkt.get("ob_wall_signal", 0)),
        }

        # Order book boost/penalty (calcule ici car signal est maintenant connu)
        ob_boost = 0.0
        if ob_wall == 1 and signal == "BUY":
            ob_boost = 0.03
        elif ob_wall == -1 and signal == "SELL":
            ob_boost = 0.03
        elif ob_wall == 1 and signal == "SELL":
            ob_boost = -0.02
        elif ob_wall == -1 and signal == "BUY":
            ob_boost = -0.02
        if ob_absorption == -1.0 and signal == "BUY":
            ob_boost += 0.02
        elif ob_absorption == 1.0 and signal == "SELL":
            ob_boost += 0.02
        if ob_boost != 0.0:
            confidence = round(min(max(confidence + ob_boost, 0.0), 0.99), 4)
            logger.info(f"[{symbol}] OB boost: {ob_boost:+.3f} -> conf={confidence:.2f} (wall={ob_wall} abs={ob_absorption})")

        self.status.setdefault("last_signals", {})[symbol] = {
            "signal": signal,
            "confidence": round(float(confidence), 3),
            "base_conf": round(float(prediction.get("confidence", 0)), 3),
            "threshold": round(float(self.config.MIN_CONFIDENCE), 3),
            "regime": self.status.get("market_regime", {}).get("name", "?"),
            "fear_greed": int(fear_greed) if isinstance(fear_greed, (int, float)) else 0,
            "timestamp": datetime.now().isoformat(),
        }
        logger.info(f"[{symbol}] Signal={signal}  conf={confidence:.2f}  F&G={fear_greed}  ATR%={self.processor.get_current_atr_pct(df):.4f}")

        # LSTM training (every 4h) - runs regardless of signal
        try:
            if self.deep_agent.needs_retraining():
                feat_cols = self.processor.get_feature_columns()
                clean = df.dropna()
                if len(clean) > 100:
                    logger.info(f"[{symbol}] Starting LSTM training...")
                    self.deep_agent.train(clean, feat_cols)
        except Exception as e:
            logger.warning(f"LSTM train error: {e}")

        # LSTM prediction
        try:
            if self.deep_agent.is_trained:
                feat_cols = self.processor.get_feature_columns()
                feat_matrix = df.dropna()[feat_cols].values if feat_cols else None
                if feat_matrix is not None and len(feat_matrix) >= 24:
                    lstm_result = self.deep_agent.predict(feat_matrix)
                    deep_signal = lstm_result.get("signal", "HOLD")
                    deep_conf = lstm_result.get("confidence", 0)
                    # Poids LSTM dynamique selon son historique de precision
                    if not hasattr(self, "_lstm_hits"):
                        self._lstm_hits = {}
                    stats = self._lstm_hits.get(symbol, {"correct": 0, "total": 0})
                    lstm_accuracy = stats["correct"] / stats["total"] if stats["total"] >= 10 else 0.5
                    lstm_weight = max(0.01, min(0.06, 0.03 * (lstm_accuracy / 0.5)))

                    # Stocker la prediction LSTM pour tracking ulterieur
                    if not hasattr(self, "_lstm_predictions"):
                        self._lstm_predictions = {}
                    lstm_agreed = deep_signal == signal and deep_conf > 0.3
                    self._lstm_predictions[symbol] = lstm_agreed

                    # Ignore LSTM tant qu'il n'a pas assez de données (< 10 trades)
                    if stats["total"] < 10:
                        logger.debug(f"[{symbol}] LSTM skipped (not enough data: {stats['total']}/10)")
                    elif lstm_agreed:
                        confidence = min(confidence + lstm_weight * deep_conf, 0.99)
                        logger.info(f"[{symbol}] LSTM agrees: +{lstm_weight*deep_conf:.3f} (acc={lstm_accuracy:.0%})")
                    elif deep_signal not in [signal, "HOLD"]:
                        penalty = lstm_weight * 0.7
                        confidence = max(confidence - penalty, 0)
                        logger.info(f"[{symbol}] LSTM disagrees: -{penalty:.3f} (acc={lstm_accuracy:.0%})")
        except Exception as e:
            logger.debug(f"LSTM: {e}")

        if signal == "HOLD":
            return
        if self._check_volume_filter(symbol, df) or self._check_volatility_filter(symbol, df):
            return
        if self._check_trend_filter(symbol, df, signal) or self._check_funding_rate(symbol, signal):
            return

        if self.config.MTF_ENABLED:
            mtf = self._get_mtf_signal(symbol)
            if mtf != signal and mtf != "HOLD":
                logger.info(f"MTF blocked {signal} ({mtf})")
                return

        # Skip if signal matches existing position direction
        if self.solana and hasattr(self.solana, 'get_position'):
            pos = self.solana.get_position(sol_sym)
            if pos['size'] > 0:
                if (signal == 'BUY' and pos['side'] == 'long') or (signal == 'SELL' and pos['side'] == 'short'):
                    logger.info(f'[{symbol}] Already has {pos["side"]} — same direction, skipping')
                    return

                # Anti-flip-flop: don't reverse too soon after opening
                min_reverse_sec = getattr(self.config, 'MIN_REVERSE_INTERVAL_SEC', 900)
                last_open = self._last_trade_time.get(symbol)
                if last_open:
                    age_sec = (datetime.now() - last_open).total_seconds()
                    if age_sec < min_reverse_sec:
                        logger.info(
                            f"[{symbol}] Reverse BLOCKED: position age {age_sec:.0f}s < "
                            f"{min_reverse_sec}s anti-flip-flop window"
                        )
                        return

                # Reverse requires higher confidence than normal entry
                min_reverse_conf = getattr(self.config, 'MIN_REVERSE_CONFIDENCE', 0.62)
                if confidence < min_reverse_conf:
                    logger.info(f'[{symbol}] Reverse BLOCKED: conf {confidence:.2%} < {min_reverse_conf:.2%} required')
                    return

                # Opposite signal = close current position (reverse will happen via buy/sell)
                logger.info(f'[{symbol}] Signal {signal} reverses {pos["side"]} — closing first')
                trade = self.solana.close_position(sol_sym, reason=f"reverse_to_{signal.lower()}")
                if trade:
                    self.db.save_trade(trade)
                    self._track_trade_result(trade)
                    self._trailing_stops.pop(sol_sym, None)
                    self._reset_tp_level(sol_sym)
                    self._alert(f"🔄 REVERSE {sol_sym}: closed {pos['side']} → {signal}")

        # Multi-timeframe confirmation
        mtf_boost = 0.0
        try:
            if self.mtf and self.mtf._models:
                mtf_result = self.mtf.predict(symbol, features)
                consensus = mtf_result.get("consensus", "NEUTRAL")
                strength = mtf_result.get("strength", 0)
                if ((signal == "BUY" and consensus in ["STRONG_BUY", "BUY"])
                        or (signal == "SELL" and consensus in ["STRONG_SELL", "SELL"])):
                    mtf_boost = 0.05 * strength  # Boost confidence up to 5%
                    logger.info(f"[{symbol}] MTF consensus: {consensus} (strength={strength:.1f}) → boost +{mtf_boost:.2f}")
                elif consensus.startswith("STRONG"):
                    # Strong disagreement → reduce confidence
                    mtf_boost = -0.03
                    logger.info(f"[{symbol}] MTF disagrees: {consensus} vs {signal} → penalty {mtf_boost:.2f}")
                confidence = min(confidence + mtf_boost, 0.99)
        except Exception as e:
            logger.debug(f"MTF predict: {e}")

        # Whale tracking
        try:
            whale = self.whale_tracker.get_whale_sentiment(symbol)
            whale_score = whale.get("score", 0)
            if (signal == "BUY" and whale_score > 0.3) or (signal == "SELL" and whale_score < -0.3):
                confidence = min(confidence + 0.02 * abs(whale_score), 0.99)
                logger.info(f"[{symbol}] Whale confirms: {whale.get('label')}")
            elif (signal == "BUY" and whale_score < -0.5) or (signal == "SELL" and whale_score > 0.5):
                confidence = max(confidence - 0.03, 0)
                logger.info(f"[{symbol}] Whale opposes: {whale.get('label')}")
        except Exception as e:
            logger.debug(f"Whale: {e}")

        # Support/Resistance filter
        try:
            high_20 = float(df["high"].rolling(20).max().iloc[-1])
            low_20 = float(df["low"].rolling(20).min().iloc[-1])
            price_range = high_20 - low_20
            if price_range > 0:
                near_resistance = (high_20 - price) / price_range < 0.05  # Within 5% of resistance
                near_support = (price - low_20) / price_range < 0.05     # Within 5% of support
                if signal == "BUY" and near_resistance:
                    logger.info(f"[{symbol}] Near resistance — reducing confidence")
                    confidence *= 0.9
                elif signal == "SELL" and near_support:
                    # Support casse = opportunite short — pas de penalite
                    logger.info(f"[{symbol}] SELL near support — no penalty (potential breakdown)")
        except Exception:
            pass

        # Final confidence gate after all adjustments (MTF / whale / S-R / LSTM)
        # SELL seuil legerement moins strict car marche baissier = signal plus fiable
        min_conf = self.config.MIN_CONFIDENCE - 0.02 if signal == "SELL" else self.config.MIN_CONFIDENCE
        if confidence < min_conf:
            logger.info(
                f"[{symbol}] Confidence {confidence:.2%} < {min_conf:.2%} "
                f"after adjustments — skip ({signal})"
            )
            return

        if signal == "BUY":
            self._execute_buy(symbol, sol_sym, price, confidence)
        elif signal == "SELL":
            self._execute_sell(symbol, sol_sym, price, confidence)

    def _count_open_positions(self, side=None):
        """Compter les positions ouvertes, optionnellement filtrees par side."""
        count = 0
        for sym in self.config.SYMBOLS:
            sol_sym = sym.split("/")[0]
            try:
                pos = self.solana.get_position(sol_sym) if self.solana else {}
                if pos.get("size", 0) > 0:
                    if side is None or pos.get("side") == side:
                        count += 1
            except Exception:
                pass
        return count

    def _execute_buy(self, symbol, sol_sym, price, confidence):
        if not self.solana:
            return

        # Correlation filter: bloquer si 2 longs deja ouverts (BTC/ETH/SOL correles)
        open_longs = self._count_open_positions(side="long")
        if open_longs >= 2:
            logger.info(f"[{symbol}] BUY bloque: {open_longs} longs deja ouverts (correlation risque)")
            return

        snap = self.status.get("solana_snapshot", {})
        usdc = snap.get("balances", {}).get(self.config.SOLANA_QUOTE_TOKEN, 0.0)
        size = self._calc_position_size(confidence, symbol=sol_sym)
        spend = min(size, usdc * 0.95)
        if spend >= self.config.POSITION_SIZE_MIN_USDC:
            if self.config.SCALE_ENABLED and spend >= self.config.POSITION_SIZE_MIN_USDC * 2:
                self._start_scale_in(sol_sym, "BUY", spend, confidence)
            else:
                trade = self.solana.buy(sol_sym, spend, confidence=confidence)
                if trade:
                    trade["reason"] = f"ml_buy_{confidence:.2f}"
                    self.db.save_trade(trade)
                    self._notify("trade", trade)
                    self._last_trade_time[symbol] = datetime.now()
                    self._save_last_trade_times()
                    self._daily_trade_count += 1
                    self._pending_kelly_invested[sol_sym] = spend
                    self._reset_tp_level(sol_sym)   # fresh position = fresh TP ladder
                    self._alert(f"🟢 BUY {sol_sym} ${spend:.2f} @ ${price:.2f} | conf={confidence:.0%}")

    def _execute_sell(self, symbol, sol_sym, price, confidence):
        if not self.solana:
            return

        # Correlation filter: bloquer si 2 shorts deja ouverts
        open_shorts = self._count_open_positions(side="short")
        if open_shorts >= 2:
            logger.info(f"[{symbol}] SELL bloque: {open_shorts} shorts deja ouverts (correlation risque)")
            return

        snap = self.status.get("solana_snapshot", {})
        usdc = snap.get("balances", {}).get(self.config.SOLANA_QUOTE_TOKEN, 0.0)
        size = self._calc_position_size(confidence, symbol=sol_sym)
        spend = min(size, usdc * 0.95)
        if spend >= self.config.POSITION_SIZE_MIN_USDC:
            if self.config.SCALE_ENABLED and spend >= self.config.POSITION_SIZE_MIN_USDC * 2:
                self._start_scale_in(sol_sym, "SELL", spend, confidence)
            else:
                trade = self.solana.sell(sol_sym, spend, confidence=confidence)
                if trade:
                    trade["reason"] = f"ml_sell_{confidence:.2f}"
                    self.db.save_trade(trade)
                    self._notify("trade", trade)
                    self._last_trade_time[symbol] = datetime.now()
                    self._save_last_trade_times()
                    self._daily_trade_count += 1
                    self._pending_kelly_invested[sol_sym] = spend
                    self._reset_tp_level(sol_sym)   # fresh position = fresh TP ladder
                    self._alert(f"🔴 SELL {sol_sym} ${spend:.2f} @ ${price:.2f} | conf={confidence:.0%}")

    def _get_mtf_signal(self, symbol):
        """Return the 4h signal using the dedicated MTF predictor model.
        Falls back to HOLD (no veto) if the (symbol, 4h) model is not yet trained."""
        tf = self.config.MTF_CONFIRM_TIMEFRAME
        try:
            if not (self.mtf and self.mtf._models):
                return "HOLD"
            key = (symbol, tf)
            model = self.mtf._models.get(key)
            scaler = self.mtf._scalers.get(key) if hasattr(self.mtf, "_scalers") else None
            if model is None or scaler is None:
                logger.debug(f"[{symbol}] MTF filter: no dedicated {tf} model yet")
                return "HOLD"
            df = self.fetcher.fetch_ohlcv(symbol, limit=100, timeframe=tf)
            if df.empty:
                return "HOLD"
            df = self.processor.add_indicators(df)
            fg = getattr(self.solana, "fear_greed", 50) if self.solana else 50
            fr = self._funding_rates.get(symbol.split("/")[0], 0.0)
            feat = self.processor.get_current_features(df.dropna(), fear_greed=fg, funding_rate=fr)
            import numpy as _np
            X = feat.reshape(1, -1)
            expected = model.n_features_in_
            if X.shape[1] > expected:
                X = X[:, :expected]
            elif X.shape[1] < expected:
                X = _np.hstack([X, _np.zeros((1, expected - X.shape[1]))])
            X_s = scaler.transform(X)
            proba = model.predict_proba(X_s)[0]
            prob_up = float(proba[1]) if len(proba) > 1 else 0.5
            sig = "BUY" if prob_up > 0.55 else "SELL" if prob_up < 0.45 else "HOLD"
            logger.info(f"[{symbol}] MTF filter ({tf} dedicated): prob_up={prob_up:.3f} -> {sig}")
            return sig
        except Exception as e:
            logger.debug(f"_get_mtf_signal error: {e}")
            return "HOLD"

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------

    def _build_combined_training_df(self, all_dfs: dict):
        """Concatenate indicator-enriched data from all symbols into a single
        training set. Each symbol contributes equally (by rows after cleaning)."""
        import pandas as _pd
        frames = []
        for sym, df in all_dfs.items():
            if df is None or df.empty:
                continue
            df2 = self.processor.add_indicators(df)
            if df2 is None or df2.empty:
                continue
            df2 = df2.copy()
            df2["_symbol"] = sym  # optional debug marker, not used as feature
            frames.append(df2)
        if not frames:
            return None
        combined = _pd.concat(frames, axis=0, ignore_index=False)
        combined.sort_index(inplace=True)  # chronological ordering across symbols
        return combined

    def _retrain(self):
        prev = self.status["state"]
        self.status["state"] = "retraining"
        logger.info("Retraining on all symbols…")
        try:
            all_dfs = self.fetcher.fetch_all_symbols()
            combined = self._build_combined_training_df(all_dfs)
            if combined is None or combined.empty:
                logger.warning("Retrain: no data fetched")
                self.agent.last_trained = datetime.now()
                return

            feat_cols = [c for c in self.processor.get_feature_columns() if c in combined.columns]
            clean = combined.dropna(subset=feat_cols + ["label"])
            if len(clean) < self.config.MIN_TRAIN_SAMPLES:
                logger.warning(f"Retrain: only {len(clean)} clean rows — skipping")
                self.agent.last_trained = datetime.now()
                return

            acc = self.agent.train(clean, feat_cols)
            self.db.save_model_metrics({
                "model_type": self.agent.models[0][0] if self.agent.models else "?",
                "accuracy": acc,
                "win_rate": self._win_rate(),
                "sharpe": self._sharpe(),
                "total_trades": self._daily_trade_count,
            })
            self._alert(f"🧠 Retrained on {len(clean)} rows — accuracy: {acc:.1%}")
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            self.agent.last_trained = datetime.now()
        finally:
            self.status["state"] = prev

    def initial_train(self):
        logger.info("Initial training on all symbols…")
        all_dfs = self.fetcher.fetch_all_symbols()
        combined = self._build_combined_training_df(all_dfs)
        if combined is None or combined.empty:
            logger.warning("initial_train: no data")
            return
        feat_cols = [c for c in self.processor.get_feature_columns() if c in combined.columns]
        clean = combined.dropna(subset=feat_cols + ["label"])
        if len(clean) >= self.config.MIN_TRAIN_SAMPLES:
            self.agent.train(clean, feat_cols)
        logger.info("Initial training done")

    def _win_rate(self):
        sells = [t for t in self.portfolio.trade_log if t.get("side") == "sell" and "pnl" in t]
        return sum(1 for t in sells if t["pnl"] > 0) / len(sells) if sells else 0.0

    def _sharpe(self):
        hist = self.db.get_portfolio_history(limit=100)
        if len(hist) < 2:
            return 0.0
        rets = hist["total_value"].pct_change().dropna()
        return float((rets.mean() / rets.std()) * (252 ** 0.5)) if rets.std() > 0 else 0.0
