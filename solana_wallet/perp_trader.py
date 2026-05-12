"""
DriftPerpTrader — perpetual futures trading on Drift Protocol (Solana).
Uses driftpy SDK to open/close long & short positions with USDC collateral.

BUY signal  → open LONG  (or close SHORT if one exists)
SELL signal → open SHORT (or close LONG  if one exists)
"""
import logging
import asyncio
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Drift market indices for perps
_MARKET_INDEX = {
    "SOL": 0,
    "BTC": 1,
    "ETH": 2,
}

# CoinGecko fallback for prices (same as dex.py)
_COINGECKO_IDS = {
    "SOL":  "solana",
    "BTC":  "bitcoin",
    "ETH":  "ethereum",
    "USDC": "usd-coin",
}

COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"


class DriftPerpTrader:
    """
    Wraps driftpy to execute perp trades on Drift Protocol.
    Exposes the same buy/sell/get_snapshot interface as SolanaTrader
    so TradingEngine can use either backend without changes.
    """

    def __init__(self, config):
        self.config = config
        self._ready = False
        self._drift_client = None
        self._prices: dict = {}
        self.trade_log: list = []
        self._position: dict = {}   # {symbol: {"side": "long"/"short", "size": float, "entry": float}}

        try:
            from solana_wallet.wallet import SolanaWallet
            self._wallet = SolanaWallet(config)
            if not self._wallet.is_ready:
                logger.error("DriftPerpTrader: wallet not ready")
                return
            self._init_drift()
        except Exception as e:
            logger.error(f"DriftPerpTrader init failed: {e}")

    def _init_drift(self):
        try:
            from driftpy.drift_client import DriftClient
            from driftpy.account_subscription_config import AccountSubscriptionConfig
            from solders.keypair import Keypair
            from solana.rpc.async_api import AsyncClient

            kp = self._wallet.keypair
            rpc = self.config.SOLANA_RPC_URL

            async def _build():
                conn = AsyncClient(rpc)
                client = DriftClient(
                    conn,
                    kp,
                    "mainnet",
                    account_subscription=AccountSubscriptionConfig("cached"),
                )
                await client.subscribe()
                return client

            self._drift_client = asyncio.get_event_loop().run_until_complete(_build())
            self._ready = True
            logger.info(f"Drift perp client ready — wallet: {self._wallet.public_key}")
        except Exception as e:
            logger.error(f"Drift client init error: {e}")

    # ------------------------------------------------------------------
    # Price feeds
    # ------------------------------------------------------------------

    def refresh_prices(self, symbols: Optional[list] = None) -> dict:
        syms = symbols or ["SOL", "BTC", "ETH", "USDC"]
        try:
            import requests
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
        return self._prices

    def get_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    # ------------------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------------------

    def get_balances(self) -> dict:
        if not self._wallet.is_ready:
            return {}
        return self._wallet.get_all_balances()

    def get_snapshot(self) -> dict:
        balances = self.get_balances()
        usdc = balances.get("USDC", 0.0)
        sol_price = self._prices.get("SOL", 0.0)
        # Add unrealised PnL from open positions
        unrealised = 0.0
        for sym, pos in self._position.items():
            curr = self._prices.get(sym, pos["entry"])
            if pos["side"] == "long":
                unrealised += pos["size"] * (curr - pos["entry"])
            else:
                unrealised += pos["size"] * (pos["entry"] - curr)
        total = usdc + unrealised + balances.get("SOL", 0.0) * sol_price
        return {
            "timestamp": datetime.now().isoformat(),
            "wallet": self._wallet.public_key,
            "balances": balances,
            "prices": dict(self._prices),
            "total_value_usdc": total,
            "positions": dict(self._position),
            "unrealised_pnl": unrealised,
        }

    def get_total_value_usdc(self) -> float:
        return self.get_snapshot()["total_value_usdc"]

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def buy(self, symbol: str, usdc_amount: float) -> dict:
        """Open LONG or close SHORT."""
        if not self._ready:
            return {}
        sym = symbol.split("/")[0]  # "SOL/USDT" → "SOL"
        market_idx = _MARKET_INDEX.get(sym)
        if market_idx is None:
            logger.error(f"No Drift market for {sym}")
            return {}

        price = self.get_price(sym)
        if price <= 0:
            logger.error(f"No price for {sym}")
            return {}

        size = (usdc_amount * self.config.PERP_LEVERAGE) / price

        # Close existing short first
        existing = self._position.get(sym, {})
        if existing.get("side") == "short":
            self._close_position(sym, market_idx, existing)

        sig = self._open_long(sym, market_idx, size, price)
        if not sig:
            return {}

        self._position[sym] = {"side": "long", "size": size, "entry": price}
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": sym,
            "side": "buy",
            "type": "perp_long",
            "price": price,
            "amount": size,
            "value": usdc_amount,
            "leverage": self.config.PERP_LEVERAGE,
            "signature": sig,
            "reason": "ml_signal",
        }
        self.trade_log.append(trade)
        logger.info(f"LONG {sym}  size={size:.4f}  price=${price:.2f}  leverage={self.config.PERP_LEVERAGE}x  tx={sig[:20]}")
        return trade

    def sell(self, symbol: str, amount: float) -> dict:
        """Close LONG or open SHORT."""
        if not self._ready:
            return {}
        sym = symbol.split("/")[0]
        market_idx = _MARKET_INDEX.get(sym)
        if market_idx is None:
            logger.error(f"No Drift market for {sym}")
            return {}

        price = self.get_price(sym)
        if price <= 0:
            return {}

        usdc_amount = amount * price / self.config.PERP_LEVERAGE
        size = (usdc_amount * self.config.PERP_LEVERAGE) / price

        # Close existing long first
        existing = self._position.get(sym, {})
        if existing.get("side") == "long":
            self._close_position(sym, market_idx, existing)

        sig = self._open_short(sym, market_idx, size, price)
        if not sig:
            return {}

        self._position[sym] = {"side": "short", "size": size, "entry": price}
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": sym,
            "side": "sell",
            "type": "perp_short",
            "price": price,
            "amount": size,
            "value": usdc_amount,
            "leverage": self.config.PERP_LEVERAGE,
            "signature": sig,
            "reason": "ml_signal",
        }
        self.trade_log.append(trade)
        logger.info(f"SHORT {sym}  size={size:.4f}  price=${price:.2f}  leverage={self.config.PERP_LEVERAGE}x  tx={sig[:20]}")
        return trade

    def quote(self, input_sym: str, output_sym: str, amount: float) -> dict:
        price = self.get_price(input_sym)
        size = (amount * self.config.PERP_LEVERAGE) / max(price, 1e-8)
        return {"success": True, "out_amount": size, "price": price,
                "leverage": self.config.PERP_LEVERAGE}

    # ------------------------------------------------------------------
    # Internal Drift calls
    # ------------------------------------------------------------------

    def _open_long(self, symbol: str, market_idx: int, size: float, price: float) -> str:
        try:
            from driftpy.types import PositionDirection, OrderType, OrderParams, PostOnlyParams
            from driftpy.constants.numeric_constants import BASE_PRECISION, PRICE_PRECISION

            base_amt = int(size * BASE_PRECISION)
            params = OrderParams(
                order_type=OrderType.Market(),
                market_index=market_idx,
                direction=PositionDirection.Long(),
                base_asset_amount=base_amt,
            )
            sig = asyncio.get_event_loop().run_until_complete(
                self._drift_client.place_perp_order(params)
            )
            return str(sig)
        except Exception as e:
            logger.error(f"_open_long {symbol}: {e}")
            return ""

    def _open_short(self, symbol: str, market_idx: int, size: float, price: float) -> str:
        try:
            from driftpy.types import PositionDirection, OrderType, OrderParams
            from driftpy.constants.numeric_constants import BASE_PRECISION

            base_amt = int(size * BASE_PRECISION)
            params = OrderParams(
                order_type=OrderType.Market(),
                market_index=market_idx,
                direction=PositionDirection.Short(),
                base_asset_amount=base_amt,
            )
            sig = asyncio.get_event_loop().run_until_complete(
                self._drift_client.place_perp_order(params)
            )
            return str(sig)
        except Exception as e:
            logger.error(f"_open_short {symbol}: {e}")
            return ""

    def _close_position(self, symbol: str, market_idx: int, pos: dict):
        try:
            sig = asyncio.get_event_loop().run_until_complete(
                self._drift_client.close_position(market_idx)
            )
            logger.info(f"Closed {pos['side']} {symbol}  tx={str(sig)[:20]}")
            self._position.pop(symbol, None)
        except Exception as e:
            logger.error(f"_close_position {symbol}: {e}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def public_key(self) -> str:
        return self._wallet.public_key if hasattr(self, "_wallet") else ""
