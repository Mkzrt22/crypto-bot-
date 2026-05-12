# CHANGELOG — Bug fixes & improvements (2026-04-19)

This patch reviews and fixes the crypto_trader project. **Test on paper/testnet
before resuming live trading.**

---

## ⚠️ Security — action required on your side

- **`.env` has been removed from this bundle.** The original backup zip shipped a
  `.env` file containing the Hyperliquid private key, Anthropic API key, and
  Telegram tokens. **If you have shared that backup with anyone, rotate all
  those credentials immediately.**
- Restore your own `.env` from your VPS (don't re-upload it from the backup).
- Never include `.env` in a backup/zip/git push. Add `.env` to `.gitignore` and
  the backup script's exclude list.

---

## 🔴 Critical bugs fixed

### 1. `MAX_DRAWDOWN_PCT` was declared twice in `config.py`
The second declaration (`10.0`) silently overrode the first (`0.20`). Drawdown
is always a fraction `[0,1]` so `dd >= 10.0` was impossible — **the drawdown
circuit-breaker was permanently disabled.** Now a single `0.15` value (= 15%).
Drawdown-based *size reduction* lives in separate `DD_REDUCE_THRESHOLD` (0.10)
and `DD_REDUCE_FACTOR` (0.5) variables.

### 2. Multi-TP was gated by the final TP (`engine.py`)
The old code required `price >= entry + 4*ATR` (= the *final* TP) before it
even looked at the partial TP ladder. That meant TP1 (1.5×) and TP2 (2.5×) could
**never fire independently.** Rewritten as a proper ladder: each level is
checked at its own ATR multiple, with a `while` loop so rapid moves can cascade
through multiple levels in one cycle.

### 3. "Partial" TPs were closing 100% of the position
The old code called `close_position(sym)` (full market_close) with a
`partial_size` variable that was never used. Added a **new `close_partial()`
method** to `HyperliquidTrader` that opens a reduce-only market order in the
opposite direction for the exact requested size, with a dust-check: if the
residual notional would be below Hyperliquid's $10 minimum, it closes the
whole position instead.

### 4. `HyperliquidTrader.sell()` ignored its `amount` argument
`buy()` correctly used `usdc_amount`, but `sell()` always used
`PERP_TRADE_SIZE_USDC` (hardcoded). All SHORT positions were the same fixed
size, regardless of confidence/Kelly/sizing logic. Fixed.

### 5. Auto-pause triggered on fake losses
`_track_trade_result` counted any trade with `pnl <= 0` as a loss, but entry
trades (`buy`/`sell`) have no realized PnL yet — they always got `pnl=0` and
were counted as losses. Three entries in a row → auto-pause. Now only trades
with `side == "close"` AND a non-None `pnl` contribute.

### 6. MTF models were overwriting each other between symbols
`_models` was keyed by timeframe only, so the loop
`for sym in SYMBOLS: mtf.train_all(sym)` left only the *last* symbol's models
(ETH by default). Rewritten to key by `(symbol, timeframe)`. `predict(symbol,…)`
now filters to only that symbol's models.

### 7. `MarketData.get_all_features` double-fetch cache flush
Calling `get_oi_change_pct` updated the cache to the current value, so when
`get_recent_liquidations` internally called it again, it saw `prev == current`
and always returned 0. **Liquidation score was permanently 0.**
Rewrote as a single consistent snapshot, plus a 30-second cache on the shared
`metaAndAssetCtxs` fetch.

### 8. Confidence could dip below `MIN_CONFIDENCE` after adjustments
MTF / whale / S-R adjustments could push confidence below the ML gate, but no
re-check was done before executing. Added a final
`if confidence < MIN_CONFIDENCE: return` after all adjustments.

---

## 🟠 Serious bugs fixed

### 9. `requirements.txt` was missing ~10 actually-imported packages
Added: `lightgbm`, `xgboost`, `catboost`, `optuna`, `torch`,
`hyperliquid-python-sdk`, `eth-account`, `telethon`, `flask`. Reorganized
by logical section.

### 10. Retrain stopped after the first symbol
`for sym in all_dfs: ... break` trained only on SOL (first in SYMBOLS).
BTC and ETH data was fetched but never used. New `_build_combined_training_df`
concatenates all symbols chronologically into one dataset.

### 11. Engineered features were computed but never used by the model
`market_regime`, `regime_strength`, `rsi_x_volume`, `momentum_x_vol`,
`macd_x_trend`, `bb_x_rsi`, `atr_x_regime`, `btc_returns_1h/4h`, `btc_rsi`
were added to each dataframe row but not listed in `ALL_FEATURES`, so they
were stripped during feature selection. Added to the list.

### 12. News sentiment cache: 3 min instead of the commented "30 min"
APIs were hit at 20× the intended rate. Fixed to 30 minutes.
Risk of 429s from CryptoPanic / NewsData reduced.

### 13. Kelly Criterion instantiated but never called
Kelly was initialized in `__init__` and then completely dead code.
Now integrated: `_calc_position_size` optionally uses Kelly via
`KELLY_ENABLED` flag (default False, opt-in). Trade results are fed back to
Kelly in `_track_trade_result` via `_pending_kelly_invested`.

### 14. Multi-TP state was not persisted across restarts
Used `getattr(self, f"_tp_level_{sym}", 0)` — lost on restart. If a position
was still open, TP1 could re-fire. Now persisted in `tp_state.json` via
`_load_tp_state` / `_save_tp_state`. Reset automatically on full close
(SL, trailing stop, final TP, reverse).

### 15. LSTM normalization stats were never persisted
`_feature_mean` / `_feature_std` were computed in-memory only. After a
restart, the saved `.pt` weights were loaded but normalization parameters
were missing → prediction garbage until next retrain. Now saved to
`models/lstm_stats.npz` and auto-loaded on `__init__`.

### 16. `torch.load` without `weights_only=True`
Triggers a warning on PyTorch ≥ 2.4 and becomes an error on ≥ 2.6.
Fixed with a `try/except TypeError` for older-PyTorch compatibility.

### 17. Keyword matching used substring (`in`) instead of word boundaries
`"war" in "award"`, `"ban" in "urban"`, `"ath" in "path"` → massive false
positives in both `news_sentiment.py` and `telegram_monitor.py`.
Both now use pre-compiled `re.compile(r'\b(...)\b', re.IGNORECASE)`.

### 18. Telegram sentiment could exceed [-1, 1]
With `weight=3.0` on high-impact messages, `_sentiment_score` was
`sum(scores)/count` and could reach ±3.0. `get_sentiment_normalized()` then
returned up to 200/100. Now divided by `max_weight_sum` and clamped to [-1,1].

### 19. Dead RSS feed: Reuters (shut down ~2020)
Removed from the feed list.

### 20. Hardcoded stale prices in `whale_tracker.py`
`BTC=75000, ETH=2300, SOL=85` were outdated. Replaced with live CoinGecko
fetch (5-min cache), with updated fallback values.

### 21. `get_portfolio_history` returned the OLDEST 500 rows, not newest
`ORDER BY timestamp LIMIT 500` without `DESC`. Sharpe ratio was being
computed on ancient history. Fixed.

### 22. `get_trades_today` was referenced but not implemented
Engine falls back to `[]` via `hasattr()`, so daily summary was always empty.
Now implemented using `WHERE timestamp >= start_of_day`.

### 23. `int(os.getenv("TELEGRAM_API_ID", "0"))` crashed on empty string
`int("")` → `ValueError`, aborting the monitor at init. Now tolerates empty
strings and logs a warning instead.

### 24. Telegram default channels had corrupted usernames
`theaboreal`, `binaboreal`, `defaboreal`, `zaboreal` were clearly typos /
rot-like corruption. They silently fail channel resolution.
Removed; kept only verified usernames.

### 25. Cooldown key inconsistency
`_start_scale_in` used `sym + "/USDT"`, other paths used `symbol`.
Harmonized to `f"{base}/USDT"` everywhere.

### 26. SQLite contention on concurrent bot + dashboard
Added WAL mode, 15-second busy timeout, `synchronous=NORMAL`, indexes on
`timestamp` and `symbol`. Reduces risk of `database is locked` under load.

---

## 🟡 Cleanups

- Removed unused `import time as _time` in `engine.py`.
- Fixed weird whitespace continuations around the MTF confirmation block.
- Removed leftover editor artifacts (`app.py.bak`, `app.py.backup.*`,
  `engine.pyoDqy4b9MuXPy`) from the zip.
- Removed training residuals that shouldn't be versioned: `bot.log`,
  `trading_data.db`, `telegram_monitor_session.session`, `catboost_info/`,
  `*.pt`, `*.pkl`, all `__pycache__` dirs.
- Updated `CLAUDE_MODEL` to `claude-sonnet-4-5-20250929`. Haiku 4.5 is listed
  as a cheaper alternative in the comment.

---

## 🆕 New files / functions

- `HyperliquidTrader.close_partial(sym, size, reason)` — reduce-only partial close.
- `TradingEngine._build_combined_training_df(all_dfs)` — multi-symbol training.
- `TradingEngine._load_tp_state / _save_tp_state / _get_tp_level / _set_tp_level / _reset_tp_level` — TP ladder persistence.
- `Database.get_trades_today()` — daily trade list.
- `MarketData._get_meta()` — shared cached Hyperliquid meta fetch.
- `DeepLearningAgent._save_stats / _try_load_stats` — persist normalization.

---

## 🔧 Recommended next steps (not done in this patch)

1. **Test on Hyperliquid testnet.** Especially: multi-TP ladder, `close_partial`,
   reverse-then-open flows. Hyperliquid SDK has changed its signature before —
   `close_partial` falls back if `market_open` rejects extra kwargs.
2. **Add a heartbeat** to Telegram every 6h ("bot alive, cycle #X, wallet $Y").
3. **Rate-limit CoinGecko calls** (now hit from two places: `hyperliquid_trader.refresh_prices`
   and `whale_tracker._get_price_estimate`). Could share a singleton.
4. **Coinglass / Solscan endpoints in `whale_tracker.py` are likely broken**
   (Coinglass moved to paid API, Solscan public API requires key). The module
   degrades gracefully to neutral, but if whale tracking matters, budget for
   paid API access or a different source.
5. **Schema migration**: the new `Database.__init__` adds indexes via
   `CREATE INDEX IF NOT EXISTS` so existing DBs upgrade automatically on first
   call — no manual migration needed.
6. **Consider a `run_backtest.sh`** that trains on 2024 data and validates PnL
   on 2025 out-of-sample, to check whether the multi-TP fix actually improves
   returns before going live.
