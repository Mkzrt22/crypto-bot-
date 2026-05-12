#!/usr/bin/env python3
"""Watchdog: verifie que le bot tourne et que le log avance. Alerte Telegram si freeze."""
import os, sys, time, subprocess
sys.path.insert(0, "/root/crypto_trader")

LOG_PATH    = "/root/crypto_trader/bot.log"
MAX_SILENCE = 300  # secondes sans nouvelle ligne = freeze

def get_log_mtime():
    try:
        return os.path.getmtime(LOG_PATH)
    except Exception:
        return 0

def is_bot_running():
    result = subprocess.run(["systemctl", "is-active", "crypto-trader"], capture_output=True, text=True)
    return result.stdout.strip() == "active"

def send_telegram(msg):
    try:
        from telegram_alerts import TelegramAlerter
        from config import config
        TelegramAlerter(config).send(msg)
    except Exception as e:
        print("Telegram error:", e)

def main():
    now       = time.time()
    mtime     = get_log_mtime()
    silence   = now - mtime
    running   = is_bot_running()

    print(f"Bot running: {running} | Log silence: {silence:.0f}s")

    if not running:
        send_telegram("WATCHDOG: crypto-trader service DOWN - tentative de restart")
        subprocess.run(["systemctl", "restart", "crypto-trader"])
        return

    if silence > MAX_SILENCE:
        send_telegram(f"WATCHDOG: bot freeze detecte ({silence:.0f}s sans log) - restart")
        subprocess.run(["systemctl", "restart", "crypto-trader"])
        return

    print("OK - bot actif")

if __name__ == "__main__":
    main()
