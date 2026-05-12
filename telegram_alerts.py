"""
Telegram alerting for the crypto trading bot.
Sends trade notifications, SL/TP events, and drawdown warnings.

Setup:
  1. Create a bot via @BotFather on Telegram → get the token
  2. Send /start to your bot, then get your chat_id via:
     curl https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Add to .env:
     TELEGRAM_ENABLED=true
     TELEGRAM_BOT_TOKEN=123456:ABC-...
     TELEGRAM_CHAT_ID=123456789
"""
import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramAlerter:
    def __init__(self, config):
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        self._api = f"https://api.telegram.org/bot{self.token}"

        if not self.enabled:
            logger.warning("Telegram alerter: token or chat_id missing — disabled")

    def send(self, message: str, dedup_window: int = 60):
        """Send a message to the configured Telegram chat. Deduplicates identical messages within dedup_window seconds."""
        if not self.enabled:
            return
        import hashlib, time
        if not hasattr(self, '_sent_cache'):
            self._sent_cache = {}
        key = hashlib.md5(message[:200].encode()).hexdigest()
        now = time.time()
        if key in self._sent_cache and now - self._sent_cache[key] < dedup_window:
            logger.debug(f"Telegram dedup: message supprime (doublon < {dedup_window}s)")
            return
        self._sent_cache[key] = now
        # Nettoyage cache > 10 min
        self._sent_cache = {k: v for k, v in self._sent_cache.items() if now - v < 600}

        try:
            resp = requests.post(
                f"{self._api}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning(f"Telegram send failed: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Telegram send error: {e}")

    def send_trade(self, trade: dict):
        """Format and send a trade notification."""
        side = trade.get("side", "?").upper()
        symbol = trade.get("symbol", "?")
        price = trade.get("price", 0)
        value = trade.get("value", 0)
        confidence = trade.get("confidence", 0)
        reason = trade.get("reason", "")
        pnl = trade.get("pnl", None)

        emoji = "🟢" if side == "BUY" else "🔴" if side == "SELL" else "⚪"
        msg = (
            f"{emoji} <b>{side} {symbol}</b>\n"
            f"Price: ${price:.2f}\n"
            f"Size: ${value:.2f}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Reason: {reason}"
        )
        if pnl is not None:
            msg += f"\nPnL: ${pnl:.2f}"

        self.send(msg)

    def send_daily_summary(self, stats: dict):
        """Send end-of-day summary."""
        msg = (
            f"📊 <b>Daily Summary</b>\n"
            f"Wallet: ${stats.get('total_value', 0):.2f}\n"
            f"Trades: {stats.get('trades_today', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.0%}\n"
            f"PnL: ${stats.get('daily_pnl', 0):.2f}\n"
            f"F&G: {stats.get('fear_greed', 50)}"
        )
        self.send(msg)
