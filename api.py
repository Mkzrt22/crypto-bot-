#!/usr/bin/env python3
"""
REST API for Crypto Trading Bot — control from iPhone or any client.

Endpoints:
  GET  /api/status      — bot status, wallet, signals, positions
  GET  /api/trades      — recent trades
  GET  /api/portfolio   — portfolio history
  GET  /api/metrics     — ML metrics history
  POST /api/start       — start bot
  POST /api/stop        — stop bot
  POST /api/pause       — pause bot
  POST /api/resume      — resume bot
  POST /api/retrain     — force retrain
  GET  /api/backtest    — backtest results

Runs on port 8502 alongside the Streamlit dashboard (8501).
"""
import logging
import sqlite3
import json
import os
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from flask import Flask, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    logger.warning("Flask not installed — API not available. pip install flask")

DB_PATH = "/root/crypto_trader/trading_data.db"
LOG_FILE = "/root/crypto_trader/bot.log"

# Global reference to engine (set by main.py)
_engine = None


def set_engine(engine):
    global _engine
    _engine = engine


def create_app():
    if not HAS_FLASK:
        return None

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    @app.route("/api/status")
    def status():
        if _engine is None:
            return jsonify({"error": "Engine not initialized"}), 503

        s = _engine.get_status_dict()
        return jsonify({
            "state": s.get("state", "unknown"),
            "wallet_usdc": s.get("wallet_value", 0),
            "cycle": s.get("cycle_count", 0),
            "prices": s.get("current_prices", {}),
            "signals": s.get("last_signals", {}),
            "positions": s.get("positions", {}),
            "funding_rates": s.get("funding_rates", {}),
            "model_accuracy": s.get("model_accuracy", 0),
            "consecutive_losses": s.get("consecutive_losses", 0),
            "auto_paused": s.get("auto_paused", False),
            "daily_trades": s.get("daily_trades", 0),
            "last_update": s.get("last_update"),
            "errors": s.get("errors", [])[-3:],
        })

    @app.route("/api/trades")
    def trades():
        limit = request.args.get("limit", 50, type=int)
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            return jsonify([dict(r) for r in rows])
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/portfolio")
    def portfolio():
        limit = request.args.get("limit", 200, type=int)
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            return jsonify([dict(r) for r in rows])
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/metrics")
    def metrics():
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM model_metrics ORDER BY timestamp DESC LIMIT 50"
            ).fetchall()
            conn.close()
            return jsonify([dict(r) for r in rows])
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/features")
    def features():
        if _engine is None:
            return jsonify({"error": "Engine not initialized"}), 503
        return jsonify(_engine.agent.get_feature_importances())

    @app.route("/api/backtest")
    def backtest():
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM backtest_results ORDER BY timestamp DESC LIMIT 10"
            ).fetchall()
            conn.close()
            results = []
            for r in rows:
                d = dict(r)
                d.pop("data", None)  # Remove large equity curve data
                results.append(d)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/logs")
    def logs():
        n = request.args.get("n", 50, type=int)
        filter_type = request.args.get("filter", "all")

        import subprocess
        if filter_type == "signals":
            cmd = f"grep 'Signal=' {LOG_FILE} | tail -n {n}"
        elif filter_type == "trades":
            cmd = f"grep -i 'BUY\\|SELL\\|STOP\\|PROFIT' {LOG_FILE} | tail -n {n}"
        elif filter_type == "errors":
            cmd = f"grep -i 'ERROR\\|WARN' {LOG_FILE} | tail -n {n}"
        else:
            cmd = f"tail -n {n} {LOG_FILE}"

        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            lines = r.stdout.strip().split("\n") if r.stdout.strip() else []
            return jsonify({"lines": lines, "count": len(lines)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── Control endpoints ─────────────────────────────────────────────

    @app.route("/api/start", methods=["POST"])
    def start():
        if _engine is None:
            return jsonify({"error": "Engine not initialized"}), 503
        _engine.start()
        return jsonify({"status": "started"})

    @app.route("/api/stop", methods=["POST"])
    def stop():
        if _engine is None:
            return jsonify({"error": "Engine not initialized"}), 503
        _engine.stop()
        return jsonify({"status": "stopped"})

    @app.route("/api/pause", methods=["POST"])
    def pause():
        if _engine is None:
            return jsonify({"error": "Engine not initialized"}), 503
        _engine.pause()
        return jsonify({"status": "paused"})

    @app.route("/api/resume", methods=["POST"])
    def resume():
        if _engine is None:
            return jsonify({"error": "Engine not initialized"}), 503
        _engine.resume()
        return jsonify({"status": "resumed", "consecutive_losses_reset": True})

    @app.route("/api/retrain", methods=["POST"])
    def retrain():
        if _engine is None:
            return jsonify({"error": "Engine not initialized"}), 503
        import threading
        threading.Thread(target=_engine._retrain, daemon=True).start()
        return jsonify({"status": "retraining_started"})

    @app.route("/api/close_all", methods=["POST"])
    def close_all():
        """Fermer TOUTES les positions immediatement (emergency)."""
        if _engine is None or not _engine.solana:
            return jsonify({"error": "Engine not initialized"}), 503
        closed = []
        errors = []
        for symbol in _engine.config.SYMBOLS:
            sol_sym = symbol.split("/")[0]
            try:
                pos = _engine.solana.get_position(sol_sym)
                if pos and pos.get("size", 0) > 0:
                    trade = _engine.solana.close_position(sol_sym, reason="emergency_close_all")
                    if trade:
                        _engine.db.save_trade(trade)
                        _engine._track_trade_result(trade)
                        _engine._trailing_stops.pop(sol_sym, None)
                        setattr(_engine, f"_scaled_{sol_sym}", False)
                        closed.append({
                            "symbol": sol_sym,
                            "pnl": trade.get("pnl", 0),
                            "price": trade.get("price", 0),
                        })
            except Exception as e:
                errors.append({"symbol": sol_sym, "error": str(e)})
        _engine._alert(f"🚨 EMERGENCY CLOSE ALL — {len(closed)} positions fermees")
        return jsonify({"status": "closed", "closed": closed, "errors": errors})

    @app.route("/api/reset_circuit", methods=["POST"])
    def reset_circuit():
        """Reset peak value + circuit breakers (sortir d'une pause infinie)."""
        if _engine is None:
            return jsonify({"error": "Engine not initialized"}), 503
        try:
            snap = _engine.status.get("solana_snapshot", {})
            current_total = snap.get("total_value_usdc", 0.0)
            old_peak = getattr(_engine, "_peak_value", 0.0)
            if current_total > 0:
                _engine._peak_value = current_total
                _engine._daily_start_value = current_total
            _engine._circuit_l2_time = None
            _engine._circuit_daily_time = None
            _engine._conf_adj = 0.0
            _engine.paused = False
            _engine._auto_paused = False
            _engine._auto_pause_until = None
            _engine._consecutive_losses = 0
            _engine.status["state"] = "running"
            _engine.status["auto_pause"] = False
            _engine._save_bot_state()
            _engine._alert(f"🔄 Circuit reset: peak ${old_peak:.2f} -> ${_engine._peak_value:.2f}")
            return jsonify({
                "status": "reset_ok",
                "old_peak": old_peak,
                "new_peak": _engine._peak_value,
                "current_total": current_total,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/signals", methods=["GET"])
    def signals():
        """Derniers signaux avec breakdown de confiance."""
        if _engine is None:
            return jsonify({"error": "Engine not initialized"}), 503
        return jsonify({
            "signals": _engine.status.get("last_signals", {}),
            "min_confidence": _engine.config.MIN_CONFIDENCE,
            "regime_base": getattr(_engine, "_regime_min_conf", 0.52),
            "conf_adj": getattr(_engine, "_conf_adj", 0.0),
        })

    @app.route("/api/perf_summary", methods=["GET"])
    def perf_summary():
        """Metriques avancees: Sharpe rolling, win rate, drawdown."""
        if _engine is None:
            return jsonify({"error": "Engine not initialized"}), 503
        try:
            import sqlite3, statistics
            from datetime import datetime, timedelta
            db = "/root/crypto_trader/trading_data.db"
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            rows = cur.execute(
                "SELECT pnl FROM trades WHERE timestamp >= ? AND pnl IS NOT NULL",
                (cutoff,)
            ).fetchall()
            conn.close()
            pnls = [float(r[0]) for r in rows if r[0] is not None]
            if not pnls:
                return jsonify({"trades_7d": 0, "sharpe_7d": 0, "win_rate_7d": 0, "total_pnl_7d": 0})
            wins = [p for p in pnls if p > 0]
            avg = sum(pnls) / len(pnls)
            std = statistics.stdev(pnls) if len(pnls) > 1 else 1
            sharpe = (avg / std) * (252 ** 0.5) if std > 0 else 0
            return jsonify({
                "trades_7d": len(pnls),
                "wins_7d": len(wins),
                "losses_7d": len(pnls) - len(wins),
                "win_rate_7d": round(len(wins) / len(pnls), 3),
                "total_pnl_7d": round(sum(pnls), 2),
                "avg_pnl_7d": round(avg, 3),
                "sharpe_7d": round(sharpe, 2),
                "best_trade_7d": round(max(pnls), 2),
                "worst_trade_7d": round(min(pnls), 2),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/positions", methods=["GET"])
    def positions():
        """Liste les positions ouvertes avec PnL en temps reel."""
        if _engine is None or not _engine.solana:
            return jsonify({"error": "Engine not initialized"}), 503
        result = []
        for symbol in _engine.config.SYMBOLS:
            sol_sym = symbol.split("/")[0]
            try:
                pos = _engine.solana.get_position(sol_sym)
                if pos and pos.get("size", 0) > 0:
                    result.append({
                        "symbol": sol_sym,
                        "side": pos.get("side"),
                        "size": pos.get("size"),
                        "entry": pos.get("entry_price"),
                        "unrealised_pnl": pos.get("unrealised_pnl", 0),
                        "leverage": pos.get("leverage", 1),
                        "trailing_stop": _engine._trailing_stops.get(sol_sym, {}).get("price"),
                    })
            except Exception as e:
                result.append({"symbol": sol_sym, "error": str(e)})
        return jsonify({"positions": result, "count": len(result)})

    return app


def run_api(engine, port=8502):
    """Start the API server in a thread."""
    set_engine(engine)
    app = create_app()
    if app is None:
        logger.warning("Flask not available — API disabled")
        return

    import threading
    def _run():
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    logger.info(f"REST API running on port {port}")
