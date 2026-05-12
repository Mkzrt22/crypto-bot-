"""
Whale Tracker — monitors on-chain large transactions.

Sources:
  1. Blockchain.info / Blockchair (BTC)
  2. Etherscan (ETH)
  3. Solscan (SOL)
  4. WhaleAlert API (free tier)

Large exchange inflows → bearish (selling pressure)
Large exchange outflows → bullish (accumulation)
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Thresholds for "whale" transactions
WHALE_THRESHOLDS = {
    "BTC": 100,      # 100+ BTC = whale
    "ETH": 1000,     # 1000+ ETH
    "SOL": 100000,   # 100k+ SOL
}

# Exchange wallets (inflow = bearish, outflow = bullish)
KNOWN_EXCHANGE_WALLETS = {
    "BTC": [
        "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",  # Binance
        "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",  # Binance
    ],
    "ETH": [
        "0x28C6c06298d514Db089934071355E5743bf21d60",  # Binance
        "0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549",  # Binance
    ],
}


class WhaleTracker:
    def __init__(self, config):
        self.config = config
        self._cache: dict = {}
        self._last_fetch: dict = {}
        self._cache_minutes = 5  # Check every 5 min
        self._whale_events: list = []
        self._max_events = 100
        # Live price cache (populated from CoinGecko with 5-min TTL)
        self._price_cache: dict = {}
        self._price_cache_ts: datetime | None = None
        # NOTE: several endpoints below (Solscan public, Coinglass /public/v2/exchange_flows,
        # Blockchair threshold filter) are unreliable or no longer freely available.
        # The whale tracker will degrade gracefully to a neutral signal when they fail.

    def get_whale_sentiment(self, symbol: str) -> dict:
        """Get whale activity sentiment for a symbol."""
        sym = symbol.replace("/USDT", "")
        now = datetime.now()

        # Check cache
        last = self._last_fetch.get(sym)
        if last and (now - last).total_seconds() < self._cache_minutes * 60:
            return self._cache.get(sym, self._neutral())

        # Fetch
        result = self._fetch_whale_data(sym)
        self._cache[sym] = result
        self._last_fetch[sym] = now
        return result

    def _fetch_whale_data(self, symbol: str) -> dict:
        """Fetch whale transaction data."""
        events = []

        # Source 1: Blockchair (BTC + ETH) — free, no key needed
        if symbol in ["BTC", "ETH"]:
            events.extend(self._fetch_blockchair(symbol))

        # Source 2: Solscan (SOL)
        if symbol == "SOL":
            events.extend(self._fetch_solscan())

        # Source 3: Coinglass (exchange flows)
        events.extend(self._fetch_exchange_flows(symbol))

        if not events:
            return self._neutral()

        # Calculate sentiment
        inflow_usd = sum(e["usd_value"] for e in events if e.get("type") == "inflow")
        outflow_usd = sum(e["usd_value"] for e in events if e.get("type") == "outflow")
        transfer_usd = sum(e["usd_value"] for e in events if e.get("type") == "transfer")

        # Net flow: positive = outflow (bullish), negative = inflow (bearish)
        net_flow = outflow_usd - inflow_usd

        # Normalize to -1 / +1
        total = max(inflow_usd + outflow_usd, 1)
        score = net_flow / total
        score = max(-1.0, min(1.0, score))

        result = {
            "score": round(score, 3),
            "label": "Bullish" if score > 0.2 else "Bearish" if score < -0.2 else "Neutral",
            "inflow_usd": inflow_usd,
            "outflow_usd": outflow_usd,
            "event_count": len(events),
            "events": events[:5],  # Top 5 events
        }

        if events:
            logger.info(
                f"Whale {symbol}: score={score:+.2f} "
                f"inflow=${inflow_usd/1e6:.1f}M outflow=${outflow_usd/1e6:.1f}M"
            )

        # Save to history
        for e in events:
            self._whale_events.append({**e, "symbol": symbol, "timestamp": datetime.now().isoformat()})
        self._whale_events = self._whale_events[-self._max_events:]

        return result

    def _fetch_blockchair(self, symbol: str) -> list:
        """Fetch large transactions from Blockchair."""
        try:
            chain = "bitcoin" if symbol == "BTC" else "ethereum"
            resp = requests.get(
                f"https://api.blockchair.com/{chain}/mempool/transactions",
                params={
                    "s": "value(desc)",
                    "limit": 10,
                    "threshold": WHALE_THRESHOLDS.get(symbol, 100),
                },
                timeout=8,
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            events = []
            for tx in data.get("data", [])[:5]:
                value = float(tx.get("value", 0)) / 1e8
                usd = value * self._get_price_estimate(symbol)
                if usd > 1_000_000:  # Only $1M+ transactions
                    events.append({
                        "type": "transfer",
                        "amount": value,
                        "usd_value": usd,
                        "source": "blockchair",
                    })
            return events

        except Exception as e:
            logger.debug(f"Blockchair {symbol}: {e}")
            return []

    def _fetch_solscan(self) -> list:
        """Fetch large SOL transactions."""
        try:
            resp = requests.get(
                "https://public-api.solscan.io/transaction/last",
                params={"limit": 20},
                headers={"Accept": "application/json"},
                timeout=8,
            )
            if resp.status_code != 200:
                return []

            txs = resp.json()
            events = []
            for tx in txs:
                fee = float(tx.get("fee", 0)) / 1e9
                lamports = float(tx.get("lamport", 0)) / 1e9
                if lamports > WHALE_THRESHOLDS["SOL"]:
                    usd = lamports * self._get_price_estimate("SOL")
                    events.append({
                        "type": "transfer",
                        "amount": lamports,
                        "usd_value": usd,
                        "source": "solscan",
                    })
            return events

        except Exception as e:
            logger.debug(f"Solscan: {e}")
            return []

    def _fetch_exchange_flows(self, symbol: str) -> list:
        """Fetch exchange inflow/outflow data from Coinglass."""
        try:
            resp = requests.get(
                "https://open-api.coinglass.com/public/v2/exchange_flows",
                params={"symbol": symbol},
                timeout=8,
            )
            if resp.status_code != 200:
                return []

            data = resp.json().get("data", {})
            events = []

            inflow = float(data.get("inflow", 0))
            outflow = float(data.get("outflow", 0))
            price = self._get_price_estimate(symbol)

            if inflow > WHALE_THRESHOLDS.get(symbol, 100):
                events.append({
                    "type": "inflow",
                    "amount": inflow,
                    "usd_value": inflow * price,
                    "source": "coinglass",
                })

            if outflow > WHALE_THRESHOLDS.get(symbol, 100):
                events.append({
                    "type": "outflow",
                    "amount": outflow,
                    "usd_value": outflow * price,
                    "source": "coinglass",
                })

            return events

        except Exception as e:
            logger.debug(f"Coinglass flows {symbol}: {e}")
            return []

    def _get_price_estimate(self, symbol: str) -> float:
        """Fetch a rough USD price for USD conversion of whale events.
        Uses CoinGecko with a 5-minute cache. Falls back to last known price."""
        now = datetime.now()
        # Refresh cache every 5 minutes at most
        if (not self._price_cache_ts
                or (now - self._price_cache_ts).total_seconds() > 300):
            try:
                resp = requests.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={
                        "ids": "bitcoin,ethereum,solana",
                        "vs_currencies": "usd",
                    },
                    timeout=8,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self._price_cache = {
                        "BTC": float(data.get("bitcoin", {}).get("usd", 0)),
                        "ETH": float(data.get("ethereum", {}).get("usd", 0)),
                        "SOL": float(data.get("solana", {}).get("usd", 0)),
                    }
                    self._price_cache_ts = now
            except Exception as e:
                logger.debug(f"Whale price fetch failed: {e}")

        price = self._price_cache.get(symbol, 0.0)
        if price > 0:
            return price
        # Last-resort fallback (used only if CoinGecko has never succeeded)
        fallback = {"BTC": 60000, "ETH": 3000, "SOL": 140}
        return fallback.get(symbol, 1.0)

    def _neutral(self) -> dict:
        return {"score": 0.0, "label": "Neutral", "inflow_usd": 0, "outflow_usd": 0, "event_count": 0, "events": []}

    def get_score_normalized(self, symbol: str) -> float:
        """Get score as 0-100 for ML features."""
        result = self.get_whale_sentiment(symbol)
        return (result["score"] + 1) * 50

    def get_recent_events(self, hours: int = 1) -> list:
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            e for e in self._whale_events
            if datetime.fromisoformat(e["timestamp"]) > cutoff
        ]
