import ccxt
import pandas as pd
import time
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, config):
        self.config = config
        self.exchange = self._init_exchange()

    def _init_exchange(self):
        exchange_class = getattr(ccxt, self.config.EXCHANGE)
        params = {"enableRateLimit": True}
        if self.config.API_KEY:
            params["apiKey"] = self.config.API_KEY
            params["secret"] = self.config.API_SECRET
        ex = exchange_class(params)
        if self.config.SANDBOX:
            try:
                ex.set_sandbox_mode(True)
            except Exception:
                pass
        return ex

    def fetch_ohlcv(self, symbol: str, timeframe: str = None, limit: int = None) -> pd.DataFrame:
        tf = timeframe or self.config.TIMEFRAME
        lim = limit or self.config.HISTORY_LIMIT
        for attempt in range(3):
            try:
                raw = self.exchange.fetch_ohlcv(symbol, tf, limit=lim)
                if not raw:
                    return pd.DataFrame()
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return df.astype(float)
            except ccxt.RateLimitExceeded:
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"fetch_ohlcv {symbol}: {e}")
                if attempt == 2:
                    return pd.DataFrame()
                time.sleep(1)
        return pd.DataFrame()

    def fetch_ticker(self, symbol: str) -> dict:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"fetch_ticker {symbol}: {e}")
            return {}

    def get_current_price(self, symbol: str) -> float:
        ticker = self.fetch_ticker(symbol)
        return float(ticker.get("last", 0.0))

    def fetch_all_symbols(self) -> dict:
        results = {}
        for sym in self.config.SYMBOLS:
            df = self.fetch_ohlcv(sym)
            if not df.empty:
                results[sym] = df
            time.sleep(0.2)
        return results
