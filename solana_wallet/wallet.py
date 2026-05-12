"""
Solana wallet — load keypair, query balances (SOL + SPL tokens).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports so the rest of the bot still works without solana packages
try:
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solana.rpc.api import Client
    from solana.rpc.types import TokenAccountOpts
    _SOLANA_AVAILABLE = True
except ImportError:
    _SOLANA_AVAILABLE = False
    logger.warning("solana/solders packages not installed — Solana wallet disabled")


class SolanaWallet:
    def __init__(self, config):
        self.config = config
        self.keypair: Optional[object] = None
        self.client: Optional[object] = None
        self._ready = False

        if not _SOLANA_AVAILABLE:
            logger.error("Install solana + solders:  pip install solana solders base58")
            return

        if not config.SOLANA_PRIVATE_KEY:
            logger.error("SOLANA_PRIVATE_KEY not set in .env")
            return

        self._load_keypair()
        self._connect()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_keypair(self):
        try:
            self.keypair = Keypair.from_base58_string(self.config.SOLANA_PRIVATE_KEY)
            self._ready = True
            logger.info(f"Solana wallet loaded: {self.public_key}")
        except Exception as e:
            logger.error(f"Failed to load keypair: {e}")

    def _connect(self):
        try:
            self.client = Client(self.config.SOLANA_RPC_URL)
            # Quick connectivity check
            slot = self.client.get_slot()
            logger.info(f"Connected to Solana RPC — slot {slot.value}")
        except Exception as e:
            logger.error(f"RPC connection failed: {e}")
            self._ready = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def public_key(self) -> str:
        if self.keypair:
            return str(self.keypair.pubkey())
        return ""

    @property
    def is_ready(self) -> bool:
        return self._ready and self.keypair is not None and self.client is not None

    # ------------------------------------------------------------------
    # Balance queries
    # ------------------------------------------------------------------

    def get_sol_balance(self) -> float:
        """Returns SOL balance (lamports → SOL)."""
        if not self.is_ready:
            return 0.0
        try:
            resp = self.client.get_balance(self.keypair.pubkey())
            lamports = resp.value
            return lamports / 1_000_000_000
        except Exception as e:
            logger.error(f"get_sol_balance: {e}")
            return 0.0

    def get_token_balance(self, mint_address: str) -> float:
        """Returns SPL token balance for the given mint."""
        if not self.is_ready:
            return 0.0
        try:
            from solders.pubkey import Pubkey as _Pubkey
            mint_pubkey = _Pubkey.from_string(mint_address)
            opts = TokenAccountOpts(mint=mint_pubkey)
            resp = self.client.get_token_accounts_by_owner(self.keypair.pubkey(), opts)
            accounts = resp.value
            if not accounts:
                return 0.0
            # Sum across all associated token accounts (usually one)
            total = 0.0
            for acc in accounts:
                info = self.client.get_token_account_balance(acc.pubkey)
                ui_amount = info.value.ui_amount
                if ui_amount:
                    total += float(ui_amount)
            return total
        except Exception as e:
            logger.error(f"get_token_balance({mint_address}): {e}")
            return 0.0

    def get_all_balances(self) -> dict:
        """Returns {symbol: balance} for SOL + all configured tokens."""
        if not self.is_ready:
            return {}

        balances: dict = {"SOL": self.get_sol_balance()}
        for symbol, mint in self.config.SOLANA_TOKENS.items():
            if symbol == "SOL":
                continue
            bal = self.get_token_balance(mint)
            if bal > 0:
                balances[symbol] = bal
        return balances

    def get_portfolio_value_usdc(self, prices: dict) -> float:
        """
        Estimates total wallet value in USDC.
        prices: {symbol: usd_price}
        """
        bals = self.get_all_balances()
        total = 0.0
        for sym, amt in bals.items():
            price = prices.get(sym, 1.0 if sym in ("USDC", "USDT") else 0.0)
            total += amt * price
        return total

    def sign_and_send(self, transaction_bytes: bytes) -> str:
        """
        Sign a serialised (base64/bytes) transaction returned by Jupiter
        and broadcast it.  Returns the transaction signature string.
        """
        if not self.is_ready:
            raise RuntimeError("Wallet not ready")
        try:
            from solders.transaction import VersionedTransaction
            from solders.message import to_bytes_versioned

            # Deserialise the unsigned transaction from Jupiter
            tx = VersionedTransaction.from_bytes(transaction_bytes)
            # Re-sign: create a new signed VersionedTransaction
            signed_tx = VersionedTransaction(tx.message, [self.keypair])
            # Send
            resp = self.client.send_raw_transaction(bytes(signed_tx))
            sig = str(resp.value)
            logger.info(f"Transaction sent: {sig}")
            return sig
        except Exception as e:
            logger.error(f"sign_and_send failed: {e}")
            raise
