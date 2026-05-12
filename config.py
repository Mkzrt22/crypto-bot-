import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TradingConfig:
    EXCHANGE: str = "binance"
    API_KEY: str = os.getenv("API_KEY", "")
    API_SECRET: str = os.getenv("API_SECRET", "")
    SANDBOX: bool = os.getenv("SANDBOX", "false").lower() == "true"

    SYMBOLS: List[str] = field(default_factory=lambda: ["SOL/USDT", "BTC/USDT", "ETH/USDT"])
    TIMEFRAME: str = "1h"

    PAPER_TRADING: bool = False
    INITIAL_BALANCE: float = 0.0

    # ── Risk ──────────────────────────────────────────────────────────
    MAX_POSITION_PCT: float = 0.25
    STOP_LOSS_PCT: float = 0.03
    TAKE_PROFIT_PCT: float = 0.06
    # MAX_DRAWDOWN_PCT: fraction (0.0-1.0). 0.15 = pause trading when drawdown >= 15%.
    MAX_DRAWDOWN_PCT: float = 0.15

    # ── Circuit Breaker multi-niveaux ─────────────────────────────────────
    # Niveau 1 : DD > 5% → réduire taille 50%
    CIRCUIT_L1_DD: float = 0.05
    # Niveau 2 : DD > 10% → pause 6h
    CIRCUIT_L2_DD: float = 0.10
    # Niveau 3 : DD > 15% → arrêt complet (= MAX_DRAWDOWN_PCT existant)
    CIRCUIT_L3_DD: float = 0.15
    # Perte journalière max avant pause
    CIRCUIT_DAILY_LOSS_PCT: float = 0.08   # -8% du wallet en 24h → pause
    TRADE_FEE: float = 0.001

    # ── Dynamic SL/TP ─────────────────────────────────────────────────
    ATR_SL_MULTIPLIER: float = 2.5
    ATR_TP_MULTIPLIER: float = 4.0  # Final TP (fallback when MULTI_TP disabled)
    # Multi-TP levels (checked independently; must be sorted ascending)
    MULTI_TP_ENABLED: bool = True
    MULTI_TP_LEVELS: tuple = (1.5, 2.5, 4.0)    # ATR multipliers
    MULTI_TP_SIZES: tuple = (0.33, 0.33, 0.34)   # % of position to close at each level
    USE_TRAILING_STOP: bool = True
    TRAILING_STOP_ATR: float = 2.5  # ATR x2 au lieu de x4

    # ── Filters ───────────────────────────────────────────────────────
    MIN_VOLUME_RATIO: float = 0.8
    MIN_ATR_PCT: float = 0.003
    TRADE_COOLDOWN_SEC: int = 14400
    MAX_TRADES_PER_DAY: int = 6

    # ── Position sizing ───────────────────────────────────────────────
    POSITION_SIZE_MIN_USDC: float = 9.0
    POSITION_SIZE_MAX_USDC: float = 9.0
    POSITION_SIZE_BASE_USDC: float = 9.0
    # Anti-flip-flop: minimum time between reverse trades on the same symbol (seconds)
    MIN_REVERSE_INTERVAL_SEC: int = 900   # 15 minutes
    # Kelly-based sizing (set True to let Kelly override the confidence-based formula)
    KELLY_ENABLED: bool = False
    KELLY_FRACTION: float = 0.25  # Quarter Kelly for safety

    # ── Scale in/out ──────────────────────────────────────────────────
    SCALE_ENABLED: bool = True
    SCALE_IN_STEPS: int = 2
    SCALE_IN_INTERVAL_SEC: int = 300
    SCALE_OUT_PROFIT_PCT: float = 0.03

    # ── ML ────────────────────────────────────────────────────────────
    LOOKBACK_WINDOW: int = 60
    TRAIN_EPISODES: int = 50
    MIN_TRAIN_SAMPLES: int = 100
    MIN_CONFIDENCE: float = 0.52
    MIN_REVERSE_CONFIDENCE: float = 0.62
    HIGH_CONFIDENCE: float = 0.65
    ENSEMBLE_ENABLED: bool = True

    HISTORY_LIMIT: int = 10000
    DB_PATH: str = "trading_data.db"

    DASHBOARD_PORT: int = 8501
    REFRESH_INTERVAL: int = 30

    # ── Solana ────────────────────────────────────────────────────────
    SOLANA_ENABLED: bool = os.getenv("SOLANA_ENABLED", "false").lower() == "true"
    SOLANA_PRIVATE_KEY: str = os.getenv("SOLANA_PRIVATE_KEY", "")
    SOLANA_RPC_URL: str = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    SOLANA_TOKENS: dict = None
    SOLANA_QUOTE_TOKEN: str = "USDC"
    SOLANA_SLIPPAGE_BPS: int = 50
    SOLANA_MIN_SOL_RESERVE: float = 0.05

    # ── Perp ──────────────────────────────────────────────────────────
    PERP_ENABLED: bool = os.getenv("PERP_ENABLED", "true").lower() == "true"
    PERP_LEVERAGE: float = float(os.getenv("PERP_LEVERAGE", "20.0"))
    PERP_MAX_LEVERAGE: float = float(os.getenv("PERP_MAX_LEVERAGE", "20.0"))
    PERP_HIGH_CONF_THRESHOLD: float = 0.75
    PERP_TRADE_SIZE_USDC: float = float(os.getenv("PERP_TRADE_SIZE_USDC", "10.0"))
    HYPERLIQUID_PRIVATE_KEY: str = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")

    # ── Funding rate ──────────────────────────────────────────────────
    FUNDING_RATE_ENABLED: bool = True
    FUNDING_RATE_THRESHOLD: float = 0.01

    # ── Multi-timeframe ───────────────────────────────────────────────
    MTF_ENABLED: bool = True
    MTF_CONFIRM_TIMEFRAME: str = "4h"

    RETRAIN_EVERY_HOURS: int = 2

    # ── Auto-pause ────────────────────────────────────────────────────
    AUTO_PAUSE_LOSSES: int = 3             # Pause after N consecutive losses
    AUTO_PAUSE_HOURS: float = 2.0          # Pause duration in hours
    AUTO_PAUSE_MIN_ACCURACY: float = 0.35  # Pause if accuracy drops below

    # ── Drawdown protection (used to reduce size, not to pause) ───────
    DRAWDOWN_PROTECTION: bool = True
    DD_REDUCE_THRESHOLD: float = 0.10  # Reduce size if DD > 10%
    DD_REDUCE_FACTOR: float = 0.5      # Cut position size in half

    # ── Telegram ──────────────────────────────────────────────────────
    TELEGRAM_ENABLED: bool = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # ── API ───────────────────────────────────────────────────────────
    API_PORT: int = 8502

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "bot.log"

    def __post_init__(self):
        if self.SOLANA_TOKENS is None:
            self.SOLANA_TOKENS = {
                "SOL":  "So11111111111111111111111111111111111111112",
                "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                "JUP":  "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
                "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
                "WIF":  "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
            }

        # Validate MULTI_TP consistency
        if len(self.MULTI_TP_LEVELS) != len(self.MULTI_TP_SIZES):
            raise ValueError("MULTI_TP_LEVELS and MULTI_TP_SIZES must have the same length")
        if abs(sum(self.MULTI_TP_SIZES) - 1.0) > 0.01:
            raise ValueError(f"MULTI_TP_SIZES must sum to 1.0, got {sum(self.MULTI_TP_SIZES)}")


def _apply_overrides(cfg):
    import json, os
    path = os.path.join(os.path.dirname(__file__), "config_overrides.json")
    if os.path.exists(path):
        try:
            overrides = json.load(open(path))
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except Exception as e:
            print(f"Override load error: {e}")

config = TradingConfig()
_apply_overrides(config)
