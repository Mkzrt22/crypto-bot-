"""
HyperliquidTrader — perpetual futures on Hyperliquid L1.
v2 — Fixes:
  1. Null API response protection (no more 'NoneType' crashes)
  2. close_position() method to properly close longs/shorts
  3. Dynamic leverage: 2x default, 3x when confidence > 75%
  4. Take profit orders placed directly on Hyperliquid after entry
  5. Fear & greed index fetched and exposed for ML features
  6. Minimum order value check ($10) before sending to API
"""
import logging
import requests
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
FEAR_GREED_URL      = "https://api.alternative.me/fng/?limit=1"

_COINGECKO_IDS = {
    "SOL":  "solana",
    "BTC":  "bitcoin",
    "ETH":  "ethereum",
    "USDC": "usd-coin",
}

HL_API = "https://api.hyperliquid.xyz"

# Hyperliquid minimum order value in USD
HL_MIN_ORDER_USD = 10.0

# Hyperliquid size decimals per asset
HL_SIZE_DECIMALS = {
    "BTC": 5,    # 0.00001 BTC increments
    "ETH": 4,    # 0.0001 ETH increments
    "SOL": 1,    # 0.1 SOL increments
}

def hl_round_size(sym: str, size: float) -> float:
    """Round size to Hyperliquid's allowed decimals per asset."""
    decimals = HL_SIZE_DECIMALS.get(sym, 4)
    return round(size, decimals)


class HyperliquidTrader:
    def __init__(self, config):
        self.config = config
        self._ready = False
        self._exchange = None
        self._info = None
        self._prices: dict = {}
        self.trade_log: list = []
        self._wallet_address = ""
        self.fear_greed: int = 50   # 0=extreme fear, 100=extreme greed

        try:
            self._init_client()
        except Exception as e:
            logger.error(f"HyperliquidTrader init failed: {e}")

    def _init_client(self):
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        import eth_account

        pk = self.config.HYPERLIQUID_PRIVATE_KEY
        if not pk:
            logger.error("HYPERLIQUID_PRIVATE_KEY not set in .env")
            return

        account = eth_account.Account.from_key(pk)
        self._wallet_address = account.address
        self._info = Info(HL_API, skip_ws=True)
        self._exchange = Exchange(account, HL_API)
        self._ready = True
        logger.info(f"Hyperliquid client ready — address: {self._wallet_address}")

    # ------------------------------------------------------------------
    # Dynamic leverage based on confidence
    # ------------------------------------------------------------------

    def _get_leverage(self, confidence: float) -> float:
        if confidence >= self.config.PERP_HIGH_CONF_THRESHOLD:
            lev = min(self.config.PERP_LEVERAGE * 1.5, self.config.PERP_MAX_LEVERAGE)
            logger.info(f"High confidence {confidence:.0%} → leverage {lev}x")
            return lev
        return self.config.PERP_LEVERAGE

    # ------------------------------------------------------------------
    # Fear & Greed index
    # ------------------------------------------------------------------

    def refresh_fear_greed(self) -> int:
        try:
            resp = requests.get(FEAR_GREED_URL, timeout=8)
            resp.raise_for_status()
            value = int(resp.json()["data"][0]["value"])
            self.fear_greed = value
            logger.info(f"Fear & Greed index: {value}")
        except Exception as e:
            logger.warning(f"Fear & greed fetch failed: {e}")
        return self.fear_greed

    # ------------------------------------------------------------------
    # Price feeds
    # ------------------------------------------------------------------

    def refresh_prices(self, symbols: Optional[list] = None) -> dict:
        syms = symbols or ["SOL", "BTC", "ETH"]
        try:
            cg_ids = [_COINGECKO_IDS[s] for s in syms if s in _COINGECKO_IDS]
            resp = requests.get(
                COINGECKO_PRICE_URL,
                params={"ids": ",".join(cg_ids), "vs_currencies": "usd"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            id_to_sym = {v: k for k, v in _COINGECKO_IDS.items()}
            self._prices = {
                id_to_sym[cg_id]: float(info["usd"])
                for cg_id, info in data.items()
                if cg_id in id_to_sym
            }
        except Exception as e:
            logger.error(f"refresh_prices: {e}")
        # Also refresh fear & greed every cycle
        self.refresh_fear_greed()
        return self._prices

    def get_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    # ------------------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------------------

    def get_position(self, sym: str) -> dict:
        """Return full position info: size, side, entry price, unrealised PnL."""
        try:
            state = self._info.user_state(self._wallet_address)
            for pos in state.get("assetPositions", []):
                p = pos.get("position", {})
                if p.get("coin") == sym:
                    szi = float(p.get("szi", 0))
                    return {
                        "size": abs(szi),
                        "signed_size": szi,
                        "side": "long" if szi > 0 else "short" if szi < 0 else "none",
                        "entry_price": float(p.get("entryPx", 0)),
                        "unrealised_pnl": float(p.get("unrealizedPnl", 0)),
                        "leverage": float(p.get("leverage", {}).get("value", 1)),
                    }
        except Exception as e:
            logger.warning(f"get_position: {e}")
        return {"size": 0, "signed_size": 0, "side": "none", "entry_price": 0,
                "unrealised_pnl": 0, "leverage": 1}

    def get_open_position_size(self, sym: str) -> float:
        """Return absolute size of current open position (0 if none)."""
        return self.get_position(sym)["size"]

    def get_balances(self) -> dict:
        if not self._ready:
            return {}
        try:
            state = self._info.user_state(self._wallet_address)
            usdc = float(state.get("marginSummary", {}).get("accountValue", 0))
            return {"USDC": usdc}
        except Exception as e:
            logger.error(f"get_balances: {e}")
            return {}

    def get_snapshot(self) -> dict:
        balances = self.get_balances()
        usdc = balances.get("USDC", 0.0)
        unrealised = self._get_unrealised_pnl()
        return {
            "timestamp": datetime.now().isoformat(),
            "wallet": self._wallet_address,
            "balances": balances,
            "prices": dict(self._prices),
            "total_value_usdc": usdc + unrealised,
            "unrealised_pnl": unrealised,
            "fear_greed": self.fear_greed,
        }

    def _get_unrealised_pnl(self) -> float:
        try:
            state = self._info.user_state(self._wallet_address)
            return float(state.get("marginSummary", {}).get("totalUnrealizedPnl", 0))
        except Exception:
            return 0.0

    def get_total_value_usdc(self) -> float:
        return self.get_snapshot()["total_value_usdc"]

    # ------------------------------------------------------------------
    # Leverage management (force Hyperliquid to use the configured leverage)
    # ------------------------------------------------------------------

    _leverage_set: dict = {}

    def _ensure_leverage(self, sym: str, leverage):
        """Update Hyperliquid per-asset leverage. Cached to avoid API spam."""
        try:
            lev_int = int(round(float(leverage)))
            if self._leverage_set.get(sym) == lev_int:
                return True
            if hasattr(self._exchange, "update_leverage"):
                result = self._exchange.update_leverage(lev_int, sym)
                logger.info(f"Hyperliquid leverage set: {sym}={lev_int}x (result={result})")
                self._leverage_set[sym] = lev_int
                return True
        except Exception as e:
            logger.warning(f"update_leverage({sym}, {leverage}) failed: {e}")
        return False

    # ------------------------------------------------------------------
    # Safe API call wrapper
    # ------------------------------------------------------------------

    def _safe_market_open(self, sym: str, is_buy: bool, size: float) -> dict:
        """Wrapper around market_open with null/error protection."""
        try:
            result = self._exchange.market_open(sym, is_buy, size)
            if result is None:
                logger.error(f"market_open returned None for {sym} is_buy={is_buy} size={size}")
                return {"status": "error", "msg": "API returned None"}

            logger.info(f"market_open {'LONG' if is_buy else 'SHORT'} result: {result}")

            # Check for order errors in response
            if result.get("status") == "ok":
                statuses = (result.get("response", {}) or {}).get("data", {}).get("statuses", [])
                if statuses and isinstance(statuses[0], dict) and "error" in statuses[0]:
                    error_msg = statuses[0]["error"]
                    logger.error(f"Order rejected by Hyperliquid: {error_msg}")
                    return {"status": "error", "msg": error_msg}

            return result
        except Exception as e:
            logger.error(f"market_open exception: {e}")
            return {"status": "error", "msg": str(e)}

    def _safe_market_close(self, sym: str) -> dict:
        """Wrapper around market_close with null/error protection."""
        try:
            result = self._exchange.market_close(sym)
            if result is None:
                logger.error(f"market_close returned None for {sym}")
                return {"status": "error", "msg": "API returned None"}
            logger.info(f"market_close {sym} result: {result}")
            return result
        except Exception as e:
            logger.error(f"market_close exception: {e}")
            return {"status": "error", "msg": str(e)}

    # ------------------------------------------------------------------
    # Minimum order validation
    # ------------------------------------------------------------------

    def _validate_order_size(self, sym: str, size: float, price: float) -> bool:
        """Check that notional value meets Hyperliquid's $10 minimum."""
        notional = size * price
        if notional < HL_MIN_ORDER_USD:
            logger.warning(
                f"Order too small: {sym} size={size:.4f} × ${price:.2f} = "
                f"${notional:.2f} < ${HL_MIN_ORDER_USD} minimum"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Take profit helper
    # ------------------------------------------------------------------

    def _place_take_profit(self, sym: str, is_long: bool, entry_price: float, size: float):
        """Place a limit take-profit order on Hyperliquid (survives bot restarts)."""
        try:
            tp_pct = self.config.TAKE_PROFIT_PCT
            if is_long:
                tp_price = round(entry_price * (1 + tp_pct), 2)
                result = self._exchange.order(
                    sym, False, size, tp_price,
                    {"limit": {"tif": "Gtc"}},
                    reduce_only=True,
                )
            else:
                tp_price = round(entry_price * (1 - tp_pct), 2)
                result = self._exchange.order(
                    sym, True, size, tp_price,
                    {"limit": {"tif": "Gtc"}},
                    reduce_only=True,
                )
            if result and result.get("status") == "ok":
                logger.info(f"Take profit set @ ${tp_price:.2f} for {sym}")
            else:
                logger.warning(f"Take profit order failed: {result}")
        except Exception as e:
            logger.error(f"_place_take_profit: {e}")

    # ------------------------------------------------------------------
    # Close position (NEW)
    # ------------------------------------------------------------------

    def close_position(self, sym: str, reason: str = "close") -> dict:
        """Fully close any open position on sym using market_close."""
        if not self._ready:
            return {}

        pos = self.get_position(sym)
        if pos["size"] == 0:
            logger.info(f"No open position on {sym} to close")
            return {}

        price = self.get_price(sym)
        logger.info(f"Closing {pos['side']} {sym}: size={pos['size']:.4f} entry=${pos['entry_price']:.2f} pnl=${pos['unrealised_pnl']:.2f}")

        # Save PnL before closing (pos data lost after close)
        saved_pnl = pos["unrealised_pnl"]
        saved_entry = pos["entry_price"]
        saved_size = pos["size"]
        saved_side = pos["side"]

        result = self._safe_market_close(sym)
        if result.get("status") == "ok":
            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": sym,
                "side": "close",
                "type": f"close_{saved_side}",
                "price": price,
                "amount": saved_size,
                "value": saved_size * price,
                "pnl": saved_pnl,
                "reason": reason,
                "fear_greed": self.fear_greed,
            }
            self.trade_log.append(trade)
            logger.info(f"CLOSED {saved_side.upper()} {sym} | PnL: ${saved_pnl:.2f} | reason: {reason}")
            return trade
        else:
            logger.error(f"Failed to close {sym}: {result}")
            return {}

    def close_partial(self, sym: str, size: float, reason: str = "partial_close") -> dict:
        """Partially close an open position by placing a reduce-only market order.

        Args:
            sym:   asset name (e.g. "SOL")
            size:  size to close in coin units (not USDC). Will be rounded to the
                   exchange's size precision.
            reason: string for logging / the trade record.
        """
        if not self._ready:
            return {}

        pos = self.get_position(sym)
        if pos["size"] == 0:
            logger.info(f"No open position on {sym} to partially close")
            return {}

        close_size = hl_round_size(sym, min(size, pos["size"]))
        if close_size <= 0:
            logger.warning(f"close_partial {sym}: rounded size <= 0")
            return {}

        price = self.get_price(sym)

        # If the partial close would leave a dust position below min notional,
        # just close the whole thing.
        remaining_notional = (pos["size"] - close_size) * price
        if remaining_notional < HL_MIN_ORDER_USD:
            logger.info(
                f"close_partial {sym}: residual ${remaining_notional:.2f} < "
                f"${HL_MIN_ORDER_USD} min — closing full position instead"
            )
            return self.close_position(sym, reason=f"{reason}_full")

        # is_buy is the opposite of the position side (we want to reduce, not add)
        is_buy = pos["side"] == "short"

        try:
            result = self._exchange.market_open(
                sym, is_buy, close_size, None, 0.01,
            )
        except TypeError:
            # Fallback for SDK versions that don't accept extra kwargs; try positional-only
            try:
                result = self._exchange.market_open(sym, is_buy, close_size)
            except Exception as e:
                logger.error(f"close_partial market_open exception: {e}")
                return {}
        except Exception as e:
            logger.error(f"close_partial market_open exception: {e}")
            return {}

        if result is None:
            logger.error(f"close_partial returned None for {sym}")
            return {}

        # Estimate PnL of the partial slice
        if pos["entry_price"] > 0:
            if pos["side"] == "long":
                slice_pnl = close_size * (price - pos["entry_price"])
            else:
                slice_pnl = close_size * (pos["entry_price"] - price)
        else:
            slice_pnl = 0.0

        if result.get("status") == "ok":
            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": sym,
                "side": "close",
                "type": f"partial_close_{pos['side']}",
                "price": price,
                "amount": close_size,
                "value": close_size * price,
                "pnl": round(slice_pnl, 4),
                "reason": reason,
                "fear_greed": self.fear_greed,
            }
            self.trade_log.append(trade)
            logger.info(
                f"PARTIAL CLOSE {pos['side'].upper()} {sym} size={close_size} "
                f"| est PnL: ${slice_pnl:.2f} | reason: {reason}"
            )
            return trade
        else:
            logger.error(f"Failed partial close {sym}: {result}")
            return {}

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def buy(self, symbol: str, usdc_amount: float, confidence: float = 0.5) -> dict:
        """Open LONG position with dynamic leverage + take profit.
        If a SHORT is open, close it first."""
        if not self._ready:
            return {}
        sym = symbol.split("/")[0]
        price = self.get_price(sym)
        if price <= 0:
            logger.error(f"No price for {sym}")
            return {}

        # If we have an opposing position, close it first
        pos = self.get_position(sym)
        if pos["side"] == "short":
            logger.info(f"Closing existing SHORT before opening LONG on {sym}")
            self.close_position(sym, reason="reverse_to_long")

        leverage = self._get_leverage(confidence)
        # usdc_amount is MARGIN. Notional = margin * leverage.
        size = hl_round_size(sym, (usdc_amount * leverage) / price)
        if size <= 0:
            return {}

        # Validate minimum order (notional)
        if not self._validate_order_size(sym, size, price):
            return {}

        # Force Hyperliquid to use the configured leverage
        self._ensure_leverage(sym, leverage)

        result = self._safe_market_open(sym, True, size)
        if result.get("status") == "ok":
            self._place_take_profit(sym, True, price, size)
            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": sym,
                "side": "buy",
                "type": "perp_long",
                "price": price,
                "amount": size,
                "value": usdc_amount,
                "leverage": leverage,
                "confidence": confidence,
                "fear_greed": self.fear_greed,
                "reason": "ml_signal",
            }
            self.trade_log.append(trade)
            logger.info(f"LONG {sym}  size={size}  price=${price:.2f}  lev={leverage}x  conf={confidence:.0%}  F&G={self.fear_greed}")
            return trade
        else:
            logger.warning(f"Buy {sym} failed: {result}")
            return {}

    def sell(self, symbol: str, usdc_amount: float, confidence: float = 0.5) -> dict:
        """Open SHORT position with dynamic leverage + take profit.
        If a LONG is open, close it first.
        Note: `usdc_amount` is the USDC notional (pre-leverage), not the coin size."""
        if not self._ready:
            return {}
        sym = symbol.split("/")[0]
        price = self.get_price(sym)
        if price <= 0:
            logger.warning(f"No price for {sym}")
            return {}

        # If we have an opposing position, close it first
        pos = self.get_position(sym)
        if pos["side"] == "long":
            logger.info(f"Closing existing LONG before opening SHORT on {sym}")
            self.close_position(sym, reason="reverse_to_short")

        leverage = self._get_leverage(confidence)
        # usdc_amount is MARGIN. Notional = margin * leverage.
        size = hl_round_size(sym, (usdc_amount * leverage) / price)
        if size <= 0:
            logger.warning("sell: calculated size=0")
            return {}

        # Validate minimum order (notional)
        if not self._validate_order_size(sym, size, price):
            return {}

        # Force Hyperliquid to use the configured leverage
        self._ensure_leverage(sym, leverage)

        result = self._safe_market_open(sym, False, size)
        if result.get("status") == "ok":
            self._place_take_profit(sym, False, price, size)
            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": sym,
                "side": "sell",
                "type": "perp_short",
                "price": price,
                "amount": size,
                "value": usdc_amount,
                "leverage": leverage,
                "confidence": confidence,
                "fear_greed": self.fear_greed,
                "reason": "ml_signal",
            }
            self.trade_log.append(trade)
            logger.info(f"SHORT {sym}  size={size}  price=${price:.2f}  lev={leverage}x  conf={confidence:.0%}  F&G={self.fear_greed}")
            return trade
        else:
            logger.warning(f"Sell {sym} failed: {result}")
            return {}

    def quote(self, input_sym: str, output_sym: str, amount: float) -> dict:
        price = self.get_price(input_sym)
        size = round((amount * self.config.PERP_LEVERAGE) / max(price, 1e-8), 4)
        return {"success": True, "out_amount": size, "price": price,
                "leverage": self.config.PERP_LEVERAGE,
                "fear_greed": self.fear_greed}

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def public_key(self) -> str:
        return self._wallet_address
