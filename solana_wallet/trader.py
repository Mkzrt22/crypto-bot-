"""
SolanaTrader — bridges the ML engine signals to on-chain Jupiter swaps.
Plugs into TradingEngine as an optional execution backend.
"""
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class SolanaTrader:
    """
    Wraps wallet + JupiterDEX and exposes the same buy/sell interface
    used by the paper-trading Portfolio, so TradingEngine can call either
    without knowing which backend is active.
    """

    def __init__(self, config):
        self.config = config
        self._ready = False

        try:
            from solana_wallet.wallet import SolanaWallet
            from solana_wallet.dex import JupiterDEX

            self.wallet = SolanaWallet(config)
            self.dex = JupiterDEX(self.wallet, config)
            self._ready = self.wallet.is_ready
        except Exception as e:
            logger.error(f"SolanaTrader init failed: {e}")

        self.trade_log: list = []
        self._prices: dict = {}

    # ------------------------------------------------------------------
    # Price feeds (called by engine each cycle)
    # ------------------------------------------------------------------

    def refresh_prices(self, symbols: Optional[list] = None) -> dict:
        if not self._ready:
            return {}
        syms = symbols or list(self.config.SOLANA_TOKENS.keys())
        self._prices = self.dex.get_prices(syms)
        return self._prices

    def get_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    # ------------------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------------------

    def get_balances(self) -> dict:
        if not self._ready:
            return {}
        return self.wallet.get_all_balances()

    def get_total_value_usdc(self) -> float:
        return self.wallet.get_portfolio_value_usdc(self._prices)

    def get_snapshot(self) -> dict:
        balances = self.get_balances()
        total = self.get_total_value_usdc()
        return {
            "timestamp": datetime.now().isoformat(),
            "wallet": self.wallet.public_key,
            "balances": balances,
            "prices": dict(self._prices),
            "total_value_usdc": total,
        }

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def buy(self, symbol: str, usdc_amount: float) -> dict:
        """Swap USDC → token."""
        if not self._ready:
            return {}
        result = self.dex.execute_swap(
            input_symbol=self.config.SOLANA_QUOTE_TOKEN,
            output_symbol=symbol,
            amount_ui=usdc_amount,
            dry_run=False,
        )
        if result["success"]:
            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "side": "buy",
                "price": self.get_price(symbol),
                "amount": result["out_amount"],
                "value": usdc_amount,
                "fee": 0,
                "signature": result["signature"],
                "reason": "ml_signal",
            }
            self.trade_log.append(trade)
            return trade
        logger.warning(f"Buy {symbol} failed: {result['error']}")
        return {}

    def sell(self, symbol: str, amount: float) -> dict:
        """Swap token → USDC."""
        if not self._ready:
            return {}
        result = self.dex.execute_swap(
            input_symbol=symbol,
            output_symbol=self.config.SOLANA_QUOTE_TOKEN,
            amount_ui=amount,
            dry_run=False,
        )
        if result["success"]:
            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "side": "sell",
                "price": self.get_price(symbol),
                "amount": amount,
                "value": result["out_amount"],
                "fee": 0,
                "signature": result["signature"],
                "reason": "ml_signal",
            }
            self.trade_log.append(trade)
            return trade
        logger.warning(f"Sell {symbol} failed: {result['error']}")
        return {}

    def quote(self, input_sym: str, output_sym: str, amount: float) -> dict:
        """Dry-run quote — no transaction sent."""
        return self.dex.execute_swap(input_sym, output_sym, amount, dry_run=True)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def public_key(self) -> str:
        return self.wallet.public_key if self._ready else ""
