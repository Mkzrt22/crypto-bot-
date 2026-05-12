"""
Crypto Trader Bot — Dashboard v3
Complete UI with all bot features visible.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

import streamlit.components.v1 as components
components.html(
    '<script src="https://telegram.org/js/telegram-web-app.js"></script>'
    '<script>window.Telegram.WebApp.expand();</script>',
    height=0
)

st.set_page_config(
    page_title="Crypto Trader Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark theme
st.markdown("""
<style>
    .stApp { background: #0d1117; }
    .stMetric { background: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .big-number { font-size: 2.5em; font-weight: bold; }
    .green { color: #3fb950; }
    .red { color: #f85149; }
    .yellow { color: #d29922; }
    div[data-testid="stMetricValue"] { font-size: 1.8em; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

BOT_DIR = Path("/root/crypto_trader")
DB_PATH = BOT_DIR / "trading_data.db"
LOG_PATH = BOT_DIR / "bot.log"

# ── Helpers ─────────────────────────────────────────────────────────────────
def grep_log(pattern: str, n: int = 20) -> str:
    """Get last N matching log lines."""
    try:
        result = subprocess.run(
            ["grep", "-E", pattern, str(LOG_PATH)],
            capture_output=True, text=True, timeout=3
        )
        lines = result.stdout.strip().split("\n")
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def tail_log(n: int = 50) -> str:
    try:
        result = subprocess.run(
            ["tail", "-n", str(n), str(LOG_PATH)],
            capture_output=True, text=True, timeout=3
        )
        return result.stdout
    except Exception:
        return ""


def get_db(query: str, params=()) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_wallet() -> float:
    """Extract wallet from latest log line."""
    try:
        result = subprocess.run(
            ["grep", "-oE", r"Wallet ~\$[0-9.]+", str(LOG_PATH)],
            capture_output=True, text=True, timeout=3
        )
        lines = result.stdout.strip().split("\n")
        if lines:
            match = re.search(r"\$([0-9.]+)", lines[-1])
            if match:
                return float(match.group(1))
    except Exception:
        pass
    return 0.0


def get_latest_value(pattern: str) -> str | None:
    """Get latest matched group from log."""
    try:
        result = subprocess.run(
            ["grep", "-oE", pattern, str(LOG_PATH)],
            capture_output=True, text=True, timeout=3
        )
        lines = result.stdout.strip().split("\n")
        return lines[-1] if lines else None
    except Exception:
        return None


def bot_running() -> bool:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "crypto-trader"],
            capture_output=True, text=True, timeout=2
        )
        return result.stdout.strip() == "active"
    except Exception:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "main.py"],
                capture_output=True, text=True, timeout=2
            )
            return bool(result.stdout.strip())
        except Exception:
            return False


# ── Header ──────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# 🤖 Crypto Trader Bot")
    st.caption("AI-powered multi-symbol perp trading with Claude sentiment analysis")

with col_status:
    if bot_running():
        st.success("🟢 Bot Running")
    else:
        st.error("🔴 Bot Stopped")
    st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")

st.divider()

# ── Top KPIs ────────────────────────────────────────────────────────────────
wallet = get_wallet()

# Get accuracy
acc_line = get_latest_value(r"accuracy: [0-9.]+")
accuracy = float(acc_line.split(": ")[1]) if acc_line else 0.0

# Get today's trades
today = datetime.now().strftime("%Y-%m-%d")
trades_df = get_db(
    "SELECT * FROM trades WHERE date(timestamp) = date('now') AND reason LIKE '%stop%' OR reason LIKE '%profit%' OR reason LIKE '%close%' ORDER BY timestamp DESC"
)
today_pnl = trades_df["pnl"].sum() if not trades_df.empty and "pnl" in trades_df else 0
today_trades = len(trades_df)

# Get positions directly from DB/logs
positions = set()
try:
    import sys
    sys.path.insert(0, str(BOT_DIR))
    from dotenv import load_dotenv
    load_dotenv(BOT_DIR / ".env")
    from config import config
    from solana_wallet.hyperliquid_trader import HyperliquidTrader
    hl = HyperliquidTrader(config)
    for sym in ["SOL", "BTC", "ETH"]:
        pos = hl.get_position(sym)
        if pos["size"] > 0:
            positions.add(f"{sym} {pos['side'].upper()}")
except Exception:
    positions_raw = grep_log(r"Already has", 30)
    for line in positions_raw.split("\n"):
        match = re.search(r"\[(\w+)/USDT\] Already has (\w+)", line)
        if match:
            positions.add(f"{match.group(1)} {match.group(2).upper()}")

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("💰 Wallet", f"${wallet:.2f}", f"{today_pnl:+.2f} today")
with k2:
    # Get LSTM accuracy too
    lstm_acc_line = get_latest_value(r"LSTM.*acc=[0-9.]+")
    lstm_acc_raw = float(re.search(r"acc=([0-9.]+)", lstm_acc_line).group(1)) if lstm_acc_line and re.search(r"acc=([0-9.]+)", lstm_acc_line) else None
    lstm_acc = lstm_acc_raw / 100 if lstm_acc_raw and lstm_acc_raw > 1 else lstm_acc_raw

    color = "🟢" if accuracy >= 0.55 else "🟡" if accuracy >= 0.5 else "🔴"
    lstm_label = f"LSTM: {lstm_acc:.1%}" if lstm_acc else "LSTM: N/A"
    st.metric(f"{color} Accuracy", f"{accuracy:.1%}", lstm_label)
with k3:
    st.metric("📊 Trades Today", today_trades)
with k4:
    st.metric("🎯 Open Positions", len(positions), ", ".join(positions) if positions else "—")
with k5:
    fg_line = get_latest_value(r"F&G=[0-9]+")
    fg = fg_line.split("=")[1] if fg_line else "?"
    emoji = "😱" if int(fg) < 25 else "😐" if int(fg) < 50 else "😊" if int(fg) < 75 else "🤑"
    st.metric(f"{emoji} Fear & Greed", fg)

st.divider()

# ── Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Overview",
    "💼 Trades",
    "🧠 ML & AI",
    "📰 News",
    "🛡️ Risk & Filters",
    "🔴 Live Logs",
    "⚙️ System",
])

# ── TAB 1: Overview ─────────────────────────────────────────────────────────
with tabs[0]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Wallet Evolution")
        wallet_hist = get_db(
            "SELECT timestamp, wallet_usdc FROM bot_status ORDER BY timestamp DESC LIMIT 200"
        )
        if not wallet_hist.empty:
            wallet_hist["ts"] = pd.to_datetime(wallet_hist["timestamp"])
            wallet_hist = wallet_hist.sort_values("ts")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=wallet_hist["ts"], y=wallet_hist["wallet_usdc"],
                fill="tozeroy", line=dict(color="#3fb950", width=2),
                fillcolor="rgba(63,185,80,0.1)", name="Wallet"
            ))
            fig.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="USDC",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Wallet history will appear after a few cycles.")

    with col2:
        st.markdown("### 🎯 Daily PnL")
        daily_pnl = get_db("""
            SELECT date(timestamp) as day, SUM(pnl) as pnl, COUNT(*) as trades
            FROM trades
            WHERE pnl IS NOT NULL
            GROUP BY date(timestamp)
            ORDER BY day DESC LIMIT 14
        """)
        if not daily_pnl.empty:
            daily_pnl = daily_pnl.sort_values("day")
            colors = ["#3fb950" if v > 0 else "#f85149" for v in daily_pnl["pnl"]]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily_pnl["day"], y=daily_pnl["pnl"],
                marker_color=colors, name="Daily PnL",
            ))
            fig.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="USD",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No closed trades yet.")

    # Recent activity timeline
    st.markdown("### 📜 Recent Activity")
    recent = grep_log(r"BUY|SELL|TRAILING|SL|TP|CLOSED|REVERSE", 15)
    if recent:
        for line in recent.split("\n")[::-1]:
            # Extract timestamp
            ts_match = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2})", line)
            time_str = ts_match.group(1) if ts_match else ""
            msg = line.split("] ", 2)[-1] if "] " in line else line

            if "BUY" in msg:
                st.markdown(f"🟢 **{time_str}** — {msg}")
            elif "SELL" in msg:
                st.markdown(f"🔴 **{time_str}** — {msg}")
            elif "TRAILING" in msg or "TP" in msg:
                st.markdown(f"🎯 **{time_str}** — {msg}")
            elif "SL" in msg or "stop_loss" in msg:
                st.markdown(f"⛔ **{time_str}** — {msg}")
            elif "REVERSE" in msg:
                st.markdown(f"🔄 **{time_str}** — {msg}")
            elif "CLOSED" in msg:
                st.markdown(f"✅ **{time_str}** — {msg}")


# ── TAB 2: Trades ───────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 📋 Trade History")

    trades = get_db(
        "SELECT timestamp, symbol, side, price, amount, pnl, reason FROM trades WHERE reason LIKE '%stop%' OR reason LIKE '%profit%' OR reason LIKE '%close%' OR pnl != 0 ORDER BY timestamp DESC LIMIT 100"
    )
    if not trades.empty:
        # Stats
        closed = trades[trades["pnl"].notna()]
        wins = closed[closed["pnl"] > 0]
        losses = closed[closed["pnl"] <= 0]
        win_rate = len(wins) / len(closed) * 100 if len(closed) > 0 else 0
        total_pnl = closed["pnl"].sum()
        avg_win = wins["pnl"].mean() if not wins.empty else 0
        avg_loss = losses["pnl"].mean() if not losses.empty else 0

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Trades", len(trades))
        s2.metric("Win Rate", f"{win_rate:.1f}%")
        s3.metric("Total PnL", f"${total_pnl:+.2f}")
        s4.metric("Avg Win", f"${avg_win:+.2f}" if avg_win else "—")
        s5.metric("Avg Loss", f"${avg_loss:+.2f}" if avg_loss else "—")

        st.divider()

        # Format for display
        display = trades.copy()
        display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        display["side"] = display["side"].str.upper()

        def color_pnl(val):
            if pd.isna(val):
                return "color: #888"
            return "color: #3fb950" if val > 0 else "color: #f85149"

        st.dataframe(
            display.style.map(color_pnl, subset=["pnl"]).format({
                "price": "${:.2f}", "amount": "{:.4f}", "pnl": "${:+.2f}"
            }),
            use_container_width=True, height=500,
        )
    else:
        st.info("No trades yet.")


# ── TAB 3: ML & AI ──────────────────────────────────────────────────────────
with tabs[2]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 ML Model Status")

        # LSTM Status
        st.markdown("#### 🧬 LSTM + Transformer")
        lstm_lines = grep_log(r"LSTM.*trained|LSTM.*acc=", 5)
        if lstm_lines:
            for line in lstm_lines.strip().split("\n")[-3:]:
                match = re.search(r"acc=([0-9.]+)", line)
                tf_match = re.search(r"LSTM (\w+):", line)
                if match:
                    acc = float(match.group(1))
                    acc = acc / 100 if acc > 1 else acc
                    tf = tf_match.group(1) if tf_match else "main"
                    color = "🟢" if acc >= 0.60 else "🟡" if acc >= 0.55 else "🔴"
                    st.markdown(f"{color} **{tf}**: {acc:.1%}")
        else:
            st.caption("LSTM training in progress...")

        # LSTM signals
        lstm_sig = grep_log(r"LSTM agrees|LSTM disagrees", 10)
        if lstm_sig:
            for line in lstm_sig.strip().split("\n")[-3:]:
                msg = line.split("] ")[-1]
                if "agrees" in msg:
                    st.success(f"✅ {msg}")
                else:
                    st.warning(f"⚠️ {msg}")

        st.divider()

        # Accuracy history
        acc_lines = grep_log(r"Training complete — accuracy:", 30)
        if acc_lines:
            accs = []
            for line in acc_lines.split("\n"):
                match = re.search(r"accuracy: ([0-9.]+)", line)
                ts_match = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2})", line)
                if match and ts_match:
                    accs.append({
                        "time": pd.to_datetime(ts_match.group(1)),
                        "accuracy": float(match.group(1)) * 100,
                    })

            if accs:
                acc_df = pd.DataFrame(accs).drop_duplicates("time").sort_values("time")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=acc_df["time"], y=acc_df["accuracy"],
                    mode="lines+markers", line=dict(color="#58a6ff", width=2),
                    marker=dict(size=6), name="Accuracy"
                ))
                fig.add_hline(y=50, line_dash="dash", line_color="#888", annotation_text="Random")
                fig.add_hline(y=55, line_dash="dash", line_color="#d29922", annotation_text="Profitable")
                fig.add_hline(y=60, line_dash="dash", line_color="#3fb950", annotation_text="Excellent")
                fig.update_layout(
                    template="plotly_dark", height=300,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title="Accuracy %", yaxis_range=[40, 70],
                )
                st.plotly_chart(fig, use_container_width=True)

        # Model weights (adaptive ensemble)
        st.markdown("#### ⚖️ Ensemble Weights")
        weight_lines = grep_log(r"lgbm:|xgb:|catboost:", 5)
        if weight_lines:
            cols = st.columns(3)
            models = ["lgbm", "xgb", "catboost"]
            colors = ["#58a6ff", "#f85149", "#3fb950"]
            weights = []
            for m in models:
                match = re.search(f"{m}.*avg_acc=([0-9.]+)", weight_lines)
                weights.append(float(match.group(1)) if match else 0.33)
            for i, (model, weight, color) in enumerate(zip(models, weights, colors)):
                cols[i].metric(model.upper(), f"{weight:.1%}")
        else:
            st.caption("Weights update after retraining")

        st.divider()

        # Model info
        st.markdown("#### ⚙️ Model Config")
        info = {
            "Models": "LightGBM + XGBoost + CatBoost",
            "Meta-learner": "Logistic Regression (Stacking)",
            "Features": "35 (selected from 57)",
            "Lookahead": "5 candles (5h)",
            "Label threshold": "0.3% movement",
            "Training": "Walk-forward (purged)",
            "Optuna trials": "50",
            "Retraining": "Every 2h",
        }
        st.json(info)

    with col2:
        st.markdown("### 🤖 Claude AI Sentiment")

        claude_lines = grep_log(r"Claude sentiment:", 20)
        if claude_lines:
            last = claude_lines.strip().split("\n")[-1]
            score_match = re.search(r"Claude sentiment: ([+-]?[0-9.]+)", last)
            label_match = re.search(r"\(([^)]+)\)", last)
            impact_match = re.search(r"impact=(\w+)", last)
            analysis_match = re.search(r"— (.+)$", last)

            if score_match:
                score = float(score_match.group(1))
                label = label_match.group(1) if label_match else "?"
                impact = impact_match.group(1) if impact_match else "?"
                analysis = analysis_match.group(1) if analysis_match else ""

                c1, c2, c3 = st.columns(3)
                emoji = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "🟡"
                c1.metric(f"{emoji} Sentiment", label, f"Score: {score:+.2f}")

                impact_colors = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                c2.metric("Impact", f"{impact_colors.get(impact, '⚪')} {impact.title()}")

                c3.metric("Analysis", "By Claude AI")

                if analysis:
                    st.info(f"💭 **Claude's Analysis:** {analysis[:400]}")

        # Sentiment history
        sentiment_hist = get_db(
            "SELECT * FROM news_sentiment_history ORDER BY timestamp DESC LIMIT 100"
        )
        if not sentiment_hist.empty:
            sentiment_hist["ts"] = pd.to_datetime(sentiment_hist["timestamp"])
            sentiment_hist = sentiment_hist.sort_values("ts")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sentiment_hist["ts"], y=sentiment_hist["score"],
                fill="tozeroy", line=dict(color="#8be9fd", width=2),
                fillcolor="rgba(139,233,253,0.1)", name="Sentiment"
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#888")
            fig.update_layout(
                template="plotly_dark", height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Sentiment Score", yaxis_range=[-1, 1],
            )
            st.plotly_chart(fig, use_container_width=True)


# ── TAB 4: News ─────────────────────────────────────────────────────────────
with tabs[3]:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📡 Sources")
        st.markdown("""
        **RSS Feeds (95+)**
        - Crypto (28): CoinDesk, CoinTelegraph, The Block, Decrypt...
        - Finance (23): Reuters, Bloomberg, WSJ, Barron's...
        - Central Banks (3): Fed, ECB, BoE
        - Geopolitics (12): BBC, NYT, Al Jazeera, Guardian...
        - Asia (6): SCMP, Nikkei, Korea Herald...

        **Telegram (20+)**
        - WatcherGuru (breaking)
        - WhaleAlert (whales)
        - CoinDesk, CoinTelegraph
        - CoinGlass (liquidations)
        - Reuters, BBC Breaking

        **AI Analysis**
        - Claude Sonnet 4
        - Every 5 minutes
        - Context-aware
        """)

    with col2:
        st.markdown("### 📋 Recent Headlines")
        headlines = get_db(
            "SELECT timestamp, title, source, sentiment FROM news_headlines ORDER BY timestamp DESC LIMIT 50"
        )
        if not headlines.empty:
            headlines["time"] = pd.to_datetime(headlines["timestamp"]).dt.strftime("%H:%M")

            def color_sent(val):
                return {
                    "bullish": "color: #3fb950",
                    "bearish": "color: #f85149",
                    "neutral": "color: #888",
                }.get(val, "")

            st.dataframe(
                headlines[["time", "title", "source", "sentiment"]].style.map(
                    color_sent, subset=["sentiment"]
                ),
                use_container_width=True, height=500,
            )

        # Telegram breaking news
        st.markdown("### ⚡ Telegram Breaking News")
        tg_lines = grep_log(r"TG:|BREAKING", 10)
        if tg_lines:
            for line in tg_lines.split("\n")[::-1]:
                msg = line.split("] ", 2)[-1] if "] " in line else line
                if "BREAKING" in msg or "HIGH IMPACT" in msg:
                    st.warning(msg[:300])
                elif "TG:" in msg:
                    st.caption(msg[:300])
        else:
            st.caption("No Telegram messages yet — waiting for channel updates.")


# ── TAB 5: Risk & Filters ───────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 🛡️ Risk Management & Filters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⚙️ Risk Config")
        st.json({
            "Stop Loss": "2.5× ATR",
            "Take Profit": "4.0× ATR",
            "Trailing Stop": "4.0× ATR (3h delay)",
            "Max trades/day": 6,
            "Cooldown": "2 hours",
            "Min confidence": "55%",
            "Max leverage": "3.0x",
            "Auto-compound": "33% of wallet",
            "Drawdown protection": "Cut 50% if DD > 10%",
            "Auto-pause": "After 3 losses (2h)",
        })

    with col2:
        st.markdown("#### 🎯 Adaptive Filters (per symbol)")
        filter_lines = grep_log(r"Adaptive filters calibrated", 3)
        if filter_lines:
            for line in filter_lines.split("\n"):
                sym_match = re.search(r"\[(\w+/\w+)\]", line)
                values_match = re.search(r"\{([^}]+)\}", line)
                if sym_match and values_match:
                    sym = sym_match.group(1)
                    try:
                        # Parse values
                        import ast
                        vals = ast.literal_eval("{" + values_match.group(1) + "}")
                        st.markdown(f"**{sym}**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("ATR%", f"{vals.get('atr_pct', 0)*100:.3f}%")
                        c2.metric("Volume ratio", f"{vals.get('volume_ratio', 0):.2f}")
                        c3.metric("Trend", f"{vals.get('trend_strength', 0)*100:.2f}%")
                    except Exception:
                        st.code(line[:200])

    st.divider()

    # Blocked signals
    st.markdown("#### 🚫 Recently Blocked Signals")
    blocked = grep_log(r"filter|blocked|skipping", 15)
    if blocked:
        for line in blocked.split("\n")[-10:][::-1]:
            msg = line.split("] ", 2)[-1] if "] " in line else line
            if "Already has" in msg:
                st.caption(f"⏸️ {msg}")
            elif "blocked" in msg or "filter" in msg.lower():
                st.caption(f"🛡️ {msg}")


# ── TAB 6: Live Logs ────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### 🔴 Live Log Stream")

    filter_type = st.selectbox(
        "Filter",
        ["All", "Trades", "ML Training", "News", "Errors", "Filters"]
    )

    patterns = {
        "All": ".",
        "Trades": r"BUY|SELL|LONG|SHORT|CLOSE|STOP|TP|PROFIT|scale|REVERSE",
        "ML Training": r"accuracy|Optuna|Walk-forward|Feature selection|Stacker",
        "News": r"News|sentiment|TG:|Claude",
        "Errors": r"ERROR|WARN|error|fail",
        "Filters": r"filter|blocked|cooldown|skipped|Funding",
    }

    logs = grep_log(patterns[filter_type], 100)
    if logs:
        st.code(logs, language=None)
    else:
        st.info("No matching logs.")


# ── TAB 7: System ───────────────────────────────────────────────────────────
with tabs[6]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖥️ System Health")
        try:
            # CPU & Memory
            import psutil
            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            m1, m2, m3 = st.columns(3)
            m1.metric("CPU", f"{cpu:.0f}%")
            m2.metric("RAM", f"{mem.percent:.0f}%", f"{mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB")
            m3.metric("Disk", f"{disk.percent:.0f}%", f"{disk.used/1e9:.0f}/{disk.total/1e9:.0f} GB")

        except ImportError:
            st.caption("Install psutil for system stats")

        # Bot service status
        try:
            result = subprocess.run(
                ["systemctl", "status", "crypto-trader", "--no-pager"],
                capture_output=True, text=True, timeout=3
            )
            st.code(result.stdout[:500], language=None)
        except Exception:
            pass

    with col2:
        st.markdown("### 🔌 Services")

        services = {
            "Bot Engine": "crypto-trader" in subprocess.run(["systemctl", "is-active", "crypto-trader"], capture_output=True, text=True).stdout,
            "Telegram Monitor": bool(grep_log(r"Telegram monitor", 1)),
            "Claude Sentiment": bool(grep_log(r"Claude sentiment analyzer enabled", 1)),
            "News RSS": bool(grep_log(r"News sentiment:", 1)),
        }

        for name, active in services.items():
            icon = "🟢" if active else "🔴"
            st.markdown(f"{icon} **{name}** — {'Active' if active else 'Inactive'}")

        st.divider()

        st.markdown("### 💾 Database")
        try:
            conn = sqlite3.connect(DB_PATH)
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )
            for table in tables["name"]:
                count = pd.read_sql(f"SELECT COUNT(*) as c FROM {table}", conn)["c"].iloc[0]
                st.caption(f"📊 **{table}**: {count:,} rows")
            conn.close()
        except Exception:
            st.caption("DB not accessible")

        st.divider()

        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Restart Bot"):
            subprocess.run(["systemctl", "restart", "crypto-trader"])
            st.success("Restarting... refresh in 30s")

        if st.button("🧹 Clear old logs"):
            subprocess.run(["truncate", "-s", "0", str(LOG_PATH)])
            st.success("Logs cleared")


# ── Auto-refresh ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"🔄 Auto-refresh every 30s • Last update: {datetime.now().strftime('%H:%M:%S')}")

# Use st.rerun every 30s (approximation)
import time
time.sleep(30)
st.rerun()
