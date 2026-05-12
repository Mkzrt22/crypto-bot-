#!/usr/bin/env python3
"""
AI Crypto Trading Bot v5
------------------------
Usage:
  python main.py              # start bot + dashboard + API
  python main.py dashboard    # dashboard only
  python main.py bot          # bot + API only (no dashboard)
  python main.py train        # train model then exit
  python main.py api          # API only (no trading)
"""
import argparse
import logging
import sys
import os
import time
import threading
import subprocess


def _setup_logging():
    from config import config
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE),
        ],
    )


def _run_dashboard():
    dashboard = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    from config import config
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard,
        "--server.port", str(config.DASHBOARD_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "none",
    ])


def _run_api(engine):
    try:
        from api import run_api
        run_api(engine, port=8502)
    except Exception as e:
        logging.getLogger("bot").warning(f"API startup failed: {e}")


def _run_bot(with_api=True):
    from config import config
    from trading.engine import TradingEngine

    _setup_logging()
    logger = logging.getLogger("bot")
    logger.info("=== AI Crypto Trading Bot v5 ===")
    logger.info(f"Symbols    : {config.SYMBOLS}")
    logger.info(f"Timeframe  : {config.TIMEFRAME}")
    logger.info(f"Mode       : LIVE TRADING — real funds")
    logger.info(f"Solana     : {config.SOLANA_ENABLED}")
    logger.info(f"Ensemble   : {config.ENSEMBLE_ENABLED}")

    engine = TradingEngine()

    # Start API
    if with_api:
        _run_api(engine)

    if not engine.agent.is_trained:
        logger.info("No saved model found — running initial training…")
        engine.initial_train()

    engine.start()

    try:
        while True:
            time.sleep(15)
            sol_snap = engine.status.get("solana_snapshot", {})
            total = sol_snap.get("total_value_usdc", 0.0)
            trades = len(engine.solana.trade_log) if engine.solana else 0
            auto = " [AUTO-PAUSED]" if engine._auto_paused else ""
            logger.info(
                f"Wallet ~${total:,.2f} USDC | "
                f"Trades {trades} | "
                f"Losses {engine._consecutive_losses} | "
                f"Cycle #{engine.status['cycle_count']}{auto}"
            )
    except KeyboardInterrupt:
        logger.info("Shutting down…")
        engine.stop()


def main():
    parser = argparse.ArgumentParser(description="AI Crypto Trading Bot v5")
    parser.add_argument(
        "command", nargs="?", default="start",
        choices=["start", "bot", "dashboard", "train", "api"],
    )
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--timeframe")
    parser.add_argument("--balance", type=float)

    args = parser.parse_args()

    if args.symbols or args.timeframe or args.balance:
        from config import config
        if args.symbols: config.SYMBOLS = args.symbols
        if args.timeframe: config.TIMEFRAME = args.timeframe
        if args.balance: config.INITIAL_BALANCE = args.balance

    if args.command == "dashboard":
        print("Starting dashboard at http://localhost:8501 …")
        _run_dashboard()

    elif args.command == "bot":
        _run_bot(with_api=True)

    elif args.command == "api":
        _setup_logging()
        from trading.engine import TradingEngine
        engine = TradingEngine()
        engine.agent.load_saved_models()
        _run_api(engine)
        print("API running on http://localhost:8502")
        while True:
            time.sleep(60)

    elif args.command == "train":
        _setup_logging()
        from trading.engine import TradingEngine
        e = TradingEngine()
        e.initial_train()
        print("Training complete.")

    else:  # "start"
        _setup_logging()
        print("Starting dashboard at http://localhost:8501 …")
        dash_t = threading.Thread(target=_run_dashboard, daemon=True)
        dash_t.start()
        time.sleep(2)
        _run_bot(with_api=True)


if __name__ == "__main__":
    main()
