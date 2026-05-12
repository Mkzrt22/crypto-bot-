"""
Telegram Channel Monitor — real-time crypto news from Telegram channels.

Follows channels like WatcherGuru, WhaleAlert, etc. and analyzes
sentiment in real-time. Triggers immediate reactions for major news.

Requires: pip install telethon

Setup:
  1. Get api_id and api_hash from https://my.telegram.org
  2. Add to .env:
     TELEGRAM_API_ID=your_id
     TELEGRAM_API_HASH=your_hash
  3. First run will ask for phone number verification (one time only)
"""
import logging
import os
import re
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    from telethon import TelegramClient, events
    HAS_TELETHON = True
except ImportError:
    HAS_TELETHON = False
    logger.warning("telethon not installed — pip install telethon")

# Channels to follow (username or ID).
# Note: channel usernames must match EXACTLY what Telegram uses. Typos = silent skip.
DEFAULT_CHANNELS = [
    # Breaking news
    "WatcherGuru",           # Major breaking news
    "whale_alert_io",        # Whale movements
    "BitcoinNewsCom",        # Bitcoin news
    # Crypto news
    "CoinDesk",
    "Cointelegraph",
    "BitcoinMagazine",
    "cryptonews_channel",
    "dailyhodl",
    # Liquidations / Trading
    "CoinGlass_official",
    # Macro / Finance
    "financialjuice",
    "BBCBreaking",
    "Reuters",
]

# Impact keywords — high-priority news that should trigger immediate action
HIGH_IMPACT_KEYWORDS = [
    # Massive liquidations
    "liquidated", "liquidation", "billion liquidated", "million liquidated",
    # Major price moves
    "all-time high", "ath", "new high", "surpasses", "breaks above",
    "crashes", "flash crash", "plunges", "dumps",
    # Regulatory
    "sec approves", "etf approved", "ban crypto", "banned",
    "emergency", "executive order",
    # Geopolitical
    "war", "attack", "missile", "invasion", "ceasefire",
    "strait of hormuz", "sanctions",
    # Fed / Macro
    "rate cut", "rate hike", "fed announces", "fomc",
    "cpi", "inflation data", "jobs report",
    # Whale moves
    "whale", "transferred", "moved", "billion worth",
    # Exchange issues
    "halted", "suspended", "maintenance", "outage",
]

# Bullish signals
BULLISH_SIGNALS = [
    "surpasses", "breaks above", "all-time high", "ath", "new high",
    "etf approved", "rate cut", "ceasefire", "peace",
    "inflow", "accumulation", "buying", "rally",
    "bull", "moon", "pump", "green", "surge",
    "adoption", "partnership", "launch", "upgrade",
    "shorts liquidated", "short squeeze",
]

# Bearish signals
BEARISH_SIGNALS = [
    "crashes", "plunges", "dumps", "flash crash", "sell-off",
    "ban", "crackdown", "lawsuit", "charges", "hack",
    "rate hike", "war", "attack", "invasion", "sanctions",
    "outflow", "withdrawal", "bankrupt", "default",
    "longs liquidated", "long squeeze",
    "bear", "dump", "red", "correction",
]


class TelegramChannelMonitor:
    def __init__(self, config):
        self.config = config

        # Safe int() — tolerate empty strings and None
        raw_id = os.getenv("TELEGRAM_API_ID", "") or str(getattr(config, "TELEGRAM_API_ID", "") or "")
        try:
            self.api_id = int(raw_id) if raw_id.strip() else 0
        except (ValueError, TypeError):
            logger.warning(f"Invalid TELEGRAM_API_ID={raw_id!r} — monitor will stay disabled")
            self.api_id = 0
        self.api_hash = os.getenv("TELEGRAM_API_HASH", "") or getattr(config, "TELEGRAM_API_HASH", "")
        self.channels = getattr(config, "TELEGRAM_CHANNELS", DEFAULT_CHANNELS)

        self._client: Optional[object] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Callbacks
        self._on_news: list[Callable] = []
        self._on_high_impact: list[Callable] = []

        # Message cache
        self._recent_messages: list = []
        self._max_cache = 200
        self._sentiment_score: float = 0.0
        self._high_impact_count: int = 0
        self._last_high_impact: dict = {}

        # Stats
        self._messages_received: int = 0
        self._channels_connected: list = []

        # Pre-compiled word-boundary regex (avoid substring false positives)
        self._bull_re = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in BULLISH_SIGNALS) + r')\b',
            re.IGNORECASE,
        )
        self._bear_re = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in BEARISH_SIGNALS) + r')\b',
            re.IGNORECASE,
        )
        self._impact_re = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in HIGH_IMPACT_KEYWORDS) + r')\b',
            re.IGNORECASE,
        )

    def add_news_callback(self, cb: Callable):
        """Called for every news message."""
        self._on_news.append(cb)

    def add_high_impact_callback(self, cb: Callable):
        """Called only for high-impact breaking news."""
        self._on_high_impact.append(cb)

    def start(self):
        """Start monitoring in background thread."""
        if not HAS_TELETHON:
            logger.error("Cannot start: telethon not installed")
            return False

        if not self.api_id or not self.api_hash:
            logger.error("Cannot start: TELEGRAM_API_ID or TELEGRAM_API_HASH not set")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._run_async, daemon=True)
        self._thread.start()
        logger.info(f"Telegram monitor starting — following {len(self.channels)} channels")
        return True

    def stop(self):
        self._running = False
        if self._client and self._loop:
            asyncio.run_coroutine_threadsafe(self._client.disconnect(), self._loop)

    def _run_async(self):
        """Run async event loop in thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._monitor())

    async def _monitor(self):
        """Main async monitoring loop."""
        session_path = os.path.join(os.path.dirname(__file__), "telegram_monitor_session")
        self._client = TelegramClient(session_path, self.api_id, self.api_hash)

        await self._client.start()
        logger.info("Telegram client connected")

        # Join/follow channels
        entities = []
        for channel in self.channels:
            try:
                entity = await self._client.get_entity(channel)
                entities.append(entity)
                self._channels_connected.append(channel)
                logger.info(f"Following: {channel} (id={entity.id})")
            except Exception as e:
                logger.warning(f"Cannot follow {channel}: {e}")

        # Listen for new messages using entity objects
        @self._client.on(events.NewMessage(chats=entities))
        async def handler(event):
            await self._process_message(event)

        logger.info(f"Monitoring {len(self._channels_connected)} channels")

        # Keep running
        while self._running:
            await asyncio.sleep(1)

        await self._client.disconnect()

    async def _process_message(self, event):
        """Process incoming message from a channel."""
        try:
            text = event.message.text or ""
            if not text or len(text) < 10:
                return

            channel = ""
            try:
                chat = await event.get_chat()
                channel = getattr(chat, "username", "") or str(chat.id)
            except Exception:
                pass

            self._messages_received += 1

            # Analyze
            sentiment = self._analyze(text)
            is_high_impact = self._is_high_impact(text)

            msg_data = {
                "timestamp": datetime.now().isoformat(),
                "channel": channel,
                "text": text[:500],
                "sentiment": sentiment,
                "is_high_impact": is_high_impact,
            }

            # Cache
            self._recent_messages.append(msg_data)
            self._recent_messages = self._recent_messages[-self._max_cache:]

            # Update running sentiment
            self._update_sentiment()

            # Log
            impact_tag = " ⚡HIGH IMPACT" if is_high_impact else ""
            logger.info(f"[TG:{channel}] {sentiment.upper()}{impact_tag}: {text[:100]}")

            # Callbacks
            for cb in self._on_news:
                try:
                    cb(msg_data)
                except Exception:
                    pass

            if is_high_impact:
                self._high_impact_count += 1
                self._last_high_impact = msg_data
                for cb in self._on_high_impact:
                    try:
                        cb(msg_data)
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Message processing error: {e}")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze(self, text: str) -> str:
        """Analyze sentiment of a message (word-boundary matching)."""
        if not text:
            return "neutral"

        bull = len(self._bull_re.findall(text))
        bear = len(self._bear_re.findall(text))

        # Amounts like "$10 billion" / "$500M" amplify the dominant signal
        amounts = re.findall(r'\$[\d,.]+\s*(?:billion|million|B|M)\b', text, re.IGNORECASE)
        if amounts:
            has_large = any("billion" in a.lower() or re.search(r"\bB\b", a) for a in amounts)
            if has_large:
                if bull > bear:
                    bull += 2
                elif bear > bull:
                    bear += 2

        if bull > bear:
            return "bullish"
        elif bear > bull:
            return "bearish"
        return "neutral"

    def _is_high_impact(self, text: str) -> bool:
        """Check if this is high-impact breaking news (word-boundary matching)."""
        if not text:
            return False
        return bool(self._impact_re.search(text))

    def _update_sentiment(self):
        """Update running sentiment score from recent messages. Clamped to [-1, 1]."""
        recent = self._recent_messages[-30:]  # Last 30 messages
        if not recent:
            return

        scores = []
        for msg in recent:
            s = msg.get("sentiment", "neutral")
            weight = 3.0 if msg.get("is_high_impact") else 1.0
            if s == "bullish":
                scores.append(1.0 * weight)
            elif s == "bearish":
                scores.append(-1.0 * weight)
            else:
                scores.append(0.0)

        # Normalise by MAX possible weight (not count), so high-impact days don't explode the score
        max_weight_sum = sum(3.0 if m.get("is_high_impact") else 1.0 for m in recent)
        raw = sum(scores) / max(max_weight_sum, 1.0)
        # Clamp to [-1, 1]
        self._sentiment_score = max(-1.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sentiment_score(self) -> float:
        """Get current sentiment score (-1 to +1)."""
        return round(self._sentiment_score, 3)

    def get_sentiment_normalized(self) -> float:
        """Get sentiment 0-100 for ML features."""
        return (self._sentiment_score + 1) * 50

    def get_recent_messages(self, n: int = 20) -> list:
        """Get last N messages."""
        return self._recent_messages[-n:]

    def get_high_impact_messages(self, hours: int = 1) -> list:
        """Get high-impact messages from last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            m for m in self._recent_messages
            if m.get("is_high_impact") and
            datetime.fromisoformat(m["timestamp"]) > cutoff
        ]

    def get_stats(self) -> dict:
        """Get monitoring stats."""
        recent = self._recent_messages[-30:]
        bull = sum(1 for m in recent if m.get("sentiment") == "bullish")
        bear = sum(1 for m in recent if m.get("sentiment") == "bearish")
        return {
            "running": self._running,
            "channels": len(self._channels_connected),
            "channel_names": self._channels_connected,
            "messages_total": self._messages_received,
            "messages_cached": len(self._recent_messages),
            "sentiment_score": self.get_sentiment_score(),
            "recent_bullish": bull,
            "recent_bearish": bear,
            "high_impact_count": self._high_impact_count,
            "last_high_impact": self._last_high_impact,
        }

    @property
    def is_running(self):
        return self._running and self._thread and self._thread.is_alive()
