"""
Jupiter DEX integration (v6 API).
Handles price quotes, swap execution, and price feeds for Solana tokens.
"""
import base64
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

JUPITER_QUOTE_URL  = "https://lite-api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL   = "https://lite-api.jup.ag/swap/v1/swap"
COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"

# CoinGecko token IDs (used instead of Jupiter price API which is DNS-blocked on some VPS)
_COINGECKO_IDS = {
    "SOL":  "solana",
    "USDC": "usd-coin",
    "USDT": "tether",
    "JUP":  "jupiter-exchange-solana",
    "BONK": "bonk",
    "WIF":  "dogwifcoin",
}


class JupiterDEX:
    def __init__(self, wallet, config):
        self.wallet = wallet
        self.config = config
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Origin": "https://jup.ag",
            "Referer": "https://jup.ag/",
        })

    # ------------------------------------------------------------------
    # Price feeds
    # ------------------------------------------------------------------

    def get_price(self, symbol: str, vs: str = "USDC") -> float:
        """Get current USD price via CoinGecko."""
        prices = self.get_prices([symbol])
        return prices.get(symbol, 0.0)

    def get_prices(self, symbols: list) -> dict:
        """Returns {symbol: usd_price} using CoinGecko free API."""
        cg_ids = [_COINGECKO_IDS[s] for s in symbols if s in _COINGECKO_IDS]
        if not cg_ids:
            return {}
        try:
            resp = self._session.get(
                COINGECKO_PRICE_URL,
                params={"ids": ",".join(cg_ids), "vs_currencies": "usd"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            id_to_sym = {v: k for k, v in _COINGECKO_IDS.items()}
            return {
                id_to_sym[cg_id]: float(info["usd"])
                for cg_id, info in data.items()
                if cg_id in id_to_sym
            }
        except Exception as e:
            logger.error(f"get_prices (CoinGecko): {e}")
            return {}

    # ------------------------------------------------------------------
    # Quote
    # ------------------------------------------------------------------

    def get_quote(
        self,
        input_symbol: str,
        output_symbol: str,
        amount_ui: float,          # human-readable (e.g. 1.5 SOL)
        slippage_bps: int = None,
    ) -> Optional[dict]:
        """
        Fetch a swap quote from Jupiter.
        amount_ui is in the input token's natural units (e.g. SOL, USDC).
        Returns the raw quote dict or None on failure.
        """
        input_mint  = self.config.SOLANA_TOKENS.get(input_symbol)
        output_mint = self.config.SOLANA_TOKENS.get(output_symbol)
        if not input_mint or not output_mint:
            logger.error(f"Unknown token: {input_symbol} or {output_symbol}")
            return None

        decimals = _token_decimals(input_symbol)
        amount_raw = int(amount_ui * (10 ** decimals))
        slippage = slippage_bps if slippage_bps is not None else self.config.SOLANA_SLIPPAGE_BPS

        try:
            resp = self._session.get(
                JUPITER_QUOTE_URL,
                params={
                    "inputMint":   input_mint,
                    "outputMint":  output_mint,
                    "amount":      amount_raw,
                    "slippageBps": slippage,
                },
                timeout=10,
            )
            resp.raise_for_status()
            quote = resp.json()
            out_dec  = _token_decimals(output_symbol)
            out_ui   = int(quote.get("outAmount", 0)) / (10 ** out_dec)
            logger.info(
                f"Quote {amount_ui} {input_symbol} → {out_ui:.4f} {output_symbol} "
                f"(price impact {quote.get('priceImpactPct', '?')}%)"
            )
            return quote
        except Exception as e:
            logger.error(f"get_quote({input_symbol}→{output_symbol}): {e}")
            return None

    # ------------------------------------------------------------------
    # Swap execution
    # ------------------------------------------------------------------

    def execute_swap(
        self,
        input_symbol: str,
        output_symbol: str,
        amount_ui: float,
        dry_run: bool = False,
    ) -> dict:
        """
        Execute a swap via Jupiter.
        dry_run=True quotes only (no transaction sent).
        Returns a result dict with keys: success, signature, in_amount,
        out_amount, input_symbol, output_symbol, error.
        """
        result = {
            "success": False,
            "signature": "",
            "in_amount": amount_ui,
            "out_amount": 0.0,
            "input_symbol": input_symbol,
            "output_symbol": output_symbol,
            "error": "",
            "dry_run": dry_run,
        }

        quote = self.get_quote(input_symbol, output_symbol, amount_ui)
        if not quote:
            result["error"] = "Failed to get quote"
            return result

        out_dec = _token_decimals(output_symbol)
        result["out_amount"] = int(quote.get("outAmount", 0)) / (10 ** out_dec)

        if dry_run:
            result["success"] = True
            return result

        if not self.wallet.is_ready:
            result["error"] = "Wallet not ready"
            return result

        try:
            swap_resp = self._session.post(
                JUPITER_SWAP_URL,
                json={
                    "quoteResponse":  quote,
                    "userPublicKey":  self.wallet.public_key,
                    "wrapAndUnwrapSol": True,
                    "dynamicComputeUnitLimit": True,
                    "prioritizationFeeLamports": "auto",
                },
                timeout=20,
            )
            swap_resp.raise_for_status()
            swap_data = swap_resp.json()

            tx_bytes = base64.b64decode(swap_data["swapTransaction"])
            sig = self.wallet.sign_and_send(tx_bytes)

            result["success"] = True
            result["signature"] = sig
            logger.info(
                f"Swap {amount_ui} {input_symbol} → {result['out_amount']:.4f} {output_symbol}  tx={sig}"
            )
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"execute_swap: {e}")

        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_DECIMALS = {
    "SOL": 9, "WSOL": 9,
    "USDC": 6, "USDT": 6,
    "JUP": 6,
    "BONK": 5,
    "WIF": 6,
}

def _token_decimals(symbol: str) -> int:
    return _DECIMALS.get(symbol.upper(), 9)
