"""
Market Microstructure Data — orderbook, open interest, liquidations.
Fetches from Hyperliquid L1 API.
"""
import logging
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

HL_INFO_API = "https://api.hyperliquid.xyz/info"
_META_CACHE_SEC = 30  # Cache metaAndAssetCtxs response for 30s to save API calls


class MarketData:
    def __init__(self, config):
        self.config = config
        self._cache = {}
        self._meta_cache: dict = {"data": None, "ts": None}

    # ------------------------------------------------------------------
    # Shared meta fetch (cached)
    # ------------------------------------------------------------------

    def _get_meta(self):
        """Fetch metaAndAssetCtxs with 30s cache to avoid hammering."""
        now = datetime.now()
        ts = self._meta_cache.get("ts")
        if ts and (now - ts).total_seconds() < _META_CACHE_SEC and self._meta_cache.get("data"):
            return self._meta_cache["data"]
        try:
            resp = requests.post(
                HL_INFO_API, json={"type": "metaAndAssetCtxs"}, timeout=8
            )
            if resp.status_code != 200:
                return self._meta_cache.get("data")
            data = resp.json()
            self._meta_cache = {"data": data, "ts": now}
            return data
        except Exception as e:
            logger.debug(f"_get_meta failed: {e}")
            return self._meta_cache.get("data")

    # ------------------------------------------------------------------
    # Orderbook
    # ------------------------------------------------------------------

    def get_orderbook_features(self, symbol: str) -> dict:
        """Analyse complete du orderbook L2.
        Retourne:
        - ratio: bid/ask volume ratio (>1 = pression achat)
        - bid_wall: niveau du plus gros mur achat (prix)
        - ask_wall: niveau du plus gros mur vente (prix)
        - wall_signal: 1=wall achat proche, -1=wall vente proche, 0=neutre
        - absorption: 1 si gros ordre absorbe (liquidite disparait vite)
        """
        default = {
            "ratio": self._cache.get(f"{symbol}_ob_ratio", 1.0),
            "bid_wall": 0.0, "ask_wall": 0.0,
            "wall_signal": 0, "absorption": 0.0,
        }
        try:
            resp = requests.post(
                HL_INFO_API, json={"type": "l2Book", "coin": symbol}, timeout=8
            )
            if resp.status_code != 200:
                return default
            data = resp.json()
            levels = data.get("levels") or [[], []]
            if len(levels) < 2:
                return default

            bids = levels[0][:20]
            asks = levels[1][:20]

            # Volume total bid/ask
            bid_vol = sum(float(b.get("sz", 0)) for b in bids)
            ask_vol = sum(float(a.get("sz", 0)) for a in asks)
            ratio = bid_vol / max(ask_vol, 1e-8)

            # Wall detection - niveau avec volume > 3x la moyenne
            def find_wall(levels):
                if not levels:
                    return 0.0, 0.0
                vols = [float(l.get("sz", 0)) for l in levels]
                avg = sum(vols) / max(len(vols), 1)
                walls = [(float(l.get("px", 0)), v)
                         for l, v in zip(levels, vols) if v > avg * 3]
                if not walls:
                    return 0.0, 0.0
                biggest = max(walls, key=lambda x: x[1])
                return biggest[0], biggest[1]

            bid_wall_px, bid_wall_sz = find_wall(bids)
            ask_wall_px, ask_wall_sz = find_wall(asks)

            # Signal wall: si mur achat > 5x mur vente -> bullish
            wall_signal = 0
            if bid_wall_sz > 0 and ask_wall_sz > 0:
                if bid_wall_sz > ask_wall_sz * 2:
                    wall_signal = 1   # Gros mur achat = support fort
                elif ask_wall_sz > bid_wall_sz * 2:
                    wall_signal = -1  # Gros mur vente = resistance forte

            # Absorption: comparer avec snapshot precedent
            prev_bid = self._cache.get(f"{symbol}_prev_bid_vol", bid_vol)
            prev_ask = self._cache.get(f"{symbol}_prev_ask_vol", ask_vol)
            absorption = 0.0
            if prev_bid > 0 and bid_vol < prev_bid * 0.7:
                absorption = 1.0   # Bids absorbes rapidement -> pression vente
            elif prev_ask > 0 and ask_vol < prev_ask * 0.7:
                absorption = -1.0  # Asks absorbes rapidement -> pression achat

            # Mettre a jour cache
            self._cache[f"{symbol}_ob_ratio"] = ratio
            self._cache[f"{symbol}_prev_bid_vol"] = bid_vol
            self._cache[f"{symbol}_prev_ask_vol"] = ask_vol

            result = {
                "ratio": round(ratio, 4),
                "bid_wall": bid_wall_px,
                "ask_wall": ask_wall_px,
                "wall_signal": wall_signal,
                "absorption": absorption,
            }

            if wall_signal != 0 or absorption != 0:
                logger.info(f"[{symbol}] OB: ratio={ratio:.2f} wall={'BUY' if wall_signal>0 else 'SELL' if wall_signal<0 else '-'} absorption={absorption}")

            return result

        except Exception as e:
            logger.debug(f"Orderbook {symbol}: {e}")
            return default

    def get_orderbook_ratio(self, symbol: str) -> float:
        """Backward compat - retourne juste le ratio."""
        return self.get_orderbook_features(symbol)["ratio"]

    # ------------------------------------------------------------------
    # Open interest & funding (from shared meta)
    # ------------------------------------------------------------------

    def _read_ctx(self, symbol: str):
        """Return the ctx dict for `symbol` from cached meta, or None."""
        data = self._get_meta()
        if not data or len(data) < 2:
            return None
        meta, ctxs = data[0], data[1]
        for i, asset in enumerate(meta.get("universe", [])):
            if asset.get("name") == symbol and i < len(ctxs):
                return ctxs[i]
        return None

    def get_open_interest(self, symbol: str) -> float:
        """Get open interest in USD for a symbol."""
        ctx = self._read_ctx(symbol)
        if not ctx:
            return self._cache.get(f"{symbol}_oi", 0.0)
        try:
            oi = float(ctx.get("openInterest", 0))
            price = float(ctx.get("markPx", 0))
            oi_usd = oi * price
            self._cache[f"{symbol}_oi"] = oi_usd
            return oi_usd
        except Exception:
            return self._cache.get(f"{symbol}_oi", 0.0)

    def get_funding_rate(self, symbol: str) -> float:
        ctx = self._read_ctx(symbol)
        if not ctx:
            return 0.0
        try:
            return float(ctx.get("funding", 0))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Combined features (fixed: no more cache-flush bug)
    # ------------------------------------------------------------------

    def get_all_features(self, symbol: str) -> dict:
        """Get all market microstructure features for ML in ONE consistent snapshot."""
        # Compute current OI once, then compute change vs previous snapshot
        current_oi = self.get_open_interest(symbol)
        prev_key = f"{symbol}_oi_prev"
        prev_oi = self._cache.get(prev_key, current_oi)
        if prev_oi > 0:
            oi_change = round((current_oi - prev_oi) / prev_oi, 4)
        else:
            oi_change = 0.0
        # Update prev AFTER computing change
        self._cache[prev_key] = current_oi

        # Liquidation score derived from the SAME oi_change (no second fetch)
        liq_score = 0.0
        if oi_change < -0.02:
            liq_score = min(abs(oi_change) * 10, 1.0)
            logger.info(f"[{symbol}] Liquidation signal: OI dropped {oi_change:.2%}")

        ob = self.get_orderbook_features(symbol)
        liq = self.get_liquidation_levels(symbol)
        return {
            "orderbook_ratio": ob["ratio"],
            "ob_wall_signal": ob["wall_signal"],
            "ob_absorption": ob["absorption"],
            "ob_bid_wall": ob["bid_wall"],
            "ob_ask_wall": ob["ask_wall"],
            "oi_change": oi_change,
            "liquidation_score": liq_score,
            "liq_pressure": liq.get("liq_pressure", 0.0),
            "nearest_liq_long": liq.get("nearest_liq_long", 0.0),
            "nearest_liq_short": liq.get("nearest_liq_short", 0.0),
        }

    # Backward-compat (not used internally anymore)
    def get_oi_change_pct(self, symbol: str) -> float:
        return self.get_all_features(symbol)["oi_change"]

    def get_liquidation_levels(self, symbol: str) -> dict:
        """Recuperer les niveaux de liquidation concentres sur Hyperliquid.
        Le prix est attire vers ces zones — signal tres puissant en perp.
        Retourne: nearest_liq_long, nearest_liq_short, liq_pressure
        """
        default = {"nearest_liq_long": 0.0, "nearest_liq_short": 0.0, "liq_pressure": 0.0}
        try:
            # Recuperer les positions ouvertes aggregees
            resp = requests.post(
                HL_INFO_API,
                json={"type": "clearinghouseState", "user": "0x0000000000000000000000000000000000000000"},
                timeout=8
            )
            # Utiliser metaAndAssetCtxs pour estimer les niveaux de liquidation
            data = self._get_meta()
            if not data or len(data) < 2:
                return default

            meta, ctxs = data[0], data[1]
            sym = symbol.split("/")[0]

            for i, asset in enumerate(meta.get("universe", [])):
                if asset.get("name") == sym and i < len(ctxs):
                    ctx = ctxs[i]
                    mark_px = float(ctx.get("markPx", 0))
                    oi = float(ctx.get("openInterest", 0))
                    funding = float(ctx.get("funding", 0))

                    if mark_px <= 0:
                        return default

                    # Estimer les zones de liquidation basees sur:
                    # - funding negatif = beaucoup de longs = liquidations longs en bas
                    # - funding positif = beaucoup de shorts = liquidations shorts en haut
                    liq_pressure = 0.0
                    nearest_liq_long = 0.0
                    nearest_liq_short = 0.0

                    if funding < -0.0001:  # Beaucoup de longs, risque liquidation en bas
                        liq_distance = abs(funding) * 1000  # Distance estimee en %
                        nearest_liq_long = mark_px * (1 - min(liq_distance, 0.15))
                        liq_pressure = -min(abs(funding) * 5000, 1.0)  # Pression baissiere
                    elif funding > 0.0001:  # Beaucoup de shorts, risque liquidation en haut
                        liq_distance = abs(funding) * 1000
                        nearest_liq_short = mark_px * (1 + min(liq_distance, 0.15))
                        liq_pressure = min(abs(funding) * 5000, 1.0)  # Pression haussiere

                    result = {
                        "nearest_liq_long": round(nearest_liq_long, 2),
                        "nearest_liq_short": round(nearest_liq_short, 2),
                        "liq_pressure": round(liq_pressure, 4),
                        "funding": funding,
                        "mark_px": mark_px,
                    }

                    self._cache[f"{symbol}_liq"] = result
                    if abs(liq_pressure) > 0.3:
                        logger.info(f"[{sym}] Liq pressure={liq_pressure:.2f} funding={funding:.4f}")
                    return result

            return default
        except Exception as e:
            logger.debug(f"get_liquidation_levels {symbol}: {e}")
            return self._cache.get(f"{symbol}_liq", default)

    def get_recent_liquidations(self, symbol: str) -> dict:
        features = self.get_all_features(symbol)
        return {
            "score": features["liquidation_score"],
            "oi_change_pct": features["oi_change"],
        }
