import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init()

    def _conn(self):
        # 15s timeout to survive brief lock contention (bot + dashboard + threads)
        conn = sqlite3.connect(self.db_path, timeout=15, isolation_level=None)
        # WAL mode → readers don't block writers
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=15000")
        except Exception:
            pass
        return conn

    def _init(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol    TEXT NOT NULL,
                    side      TEXT NOT NULL,
                    price     REAL NOT NULL,
                    amount    REAL NOT NULL,
                    value     REAL NOT NULL,
                    fee       REAL DEFAULT 0,
                    pnl       REAL DEFAULT 0,
                    reason    TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);

                CREATE TABLE IF NOT EXISTS trade_context (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    symbol      TEXT NOT NULL,
                    side        TEXT NOT NULL,
                    pnl         REAL,
                    confidence  REAL,
                    regime      TEXT,
                    fear_greed  INTEGER,
                    atr_pct     REAL,
                    news_score  REAL,
                    hour_utc    INTEGER,
                    day_of_week INTEGER,
                    lstm_agree  INTEGER,
                    mtf_consensus TEXT,
                    ob_wall     INTEGER,
                    won         INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_ctx_symbol ON trade_context(symbol);
                CREATE INDEX IF NOT EXISTS idx_ctx_hour ON trade_context(hour_utc);
                CREATE INDEX IF NOT EXISTS idx_ctx_regime ON trade_context(regime);

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT NOT NULL,
                    balance      REAL NOT NULL,
                    total_value  REAL NOT NULL,
                    positions    TEXT,
                    daily_return REAL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_snap_timestamp ON portfolio_snapshots(timestamp);

                CREATE TABLE IF NOT EXISTS model_metrics (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT NOT NULL,
                    model_type   TEXT,
                    accuracy     REAL,
                    win_rate     REAL,
                    sharpe       REAL,
                    total_trades INTEGER,
                    extra        TEXT
                );
            """)
        logger.info(f"Database ready (WAL): {self.db_path}")

    def save_trade(self, trade: dict):
        if not trade:
            return
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO trades (timestamp,symbol,side,price,amount,value,fee,pnl,reason) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        trade.get("timestamp", datetime.now().isoformat()),
                        trade.get("symbol", "?"), trade.get("side", "?"),
                        trade.get("price", 0), trade.get("amount", 0),
                        trade.get("value", 0),
                        trade.get("fee", 0), trade.get("pnl", 0),
                        trade.get("reason", ""),
                    ),
                )
        except Exception as e:
            logger.warning(f"save_trade failed: {e}")

    def save_portfolio_snapshot(self, snap: dict):
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO portfolio_snapshots (timestamp,balance,total_value,positions,daily_return) "
                    "VALUES (?,?,?,?,?)",
                    (
                        snap.get("timestamp", datetime.now().isoformat()),
                        snap.get("balance", 0), snap.get("total_value", 0),
                        json.dumps(snap.get("positions", {})),
                        snap.get("daily_return", 0),
                    ),
                )
        except Exception as e:
            logger.warning(f"save_portfolio_snapshot failed: {e}")

    def save_model_metrics(self, m: dict):
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO model_metrics (timestamp,model_type,accuracy,win_rate,sharpe,total_trades,extra) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        m.get("timestamp", datetime.now().isoformat()),
                        m.get("model_type", "supervised"),
                        m.get("accuracy", 0), m.get("win_rate", 0),
                        m.get("sharpe", 0), m.get("total_trades", 0),
                        json.dumps(m.get("extra", {})),
                    ),
                )
        except Exception as e:
            logger.warning(f"save_model_metrics failed: {e}")

    def save_trade_context(self, ctx: dict):
        """Sauvegarder le contexte complet d'un trade ferme pour apprentissage."""
        try:
            with self._conn() as c:
                c.execute(
                    """INSERT INTO trade_context
                    (timestamp,symbol,side,pnl,confidence,regime,fear_greed,
                     atr_pct,news_score,hour_utc,day_of_week,lstm_agree,
                     mtf_consensus,ob_wall,won)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (ctx.get("timestamp"), ctx.get("symbol"), ctx.get("side"),
                     ctx.get("pnl", 0), ctx.get("confidence", 0),
                     ctx.get("regime"), ctx.get("fear_greed", 0),
                     ctx.get("atr_pct", 0), ctx.get("news_score", 0),
                     ctx.get("hour_utc", 0), ctx.get("day_of_week", 0),
                     ctx.get("lstm_agree", -1), ctx.get("mtf_consensus", "NEUTRAL"),
                     ctx.get("ob_wall", 0), 1 if ctx.get("pnl", 0) > 0 else 0)
                )
        except Exception as e:
            logger.warning(f"save_trade_context failed: {e}")

    def get_stats_by_hour(self, days: int = 30) -> dict:
        """Win rate et PnL par tranche horaire UTC sur N derniers jours."""
        try:
            with self._conn() as c:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                rows = c.execute(
                    """SELECT hour_utc, COUNT(*), SUM(won), SUM(pnl)
                       FROM trade_context WHERE timestamp >= ?
                       GROUP BY hour_utc""",
                    (cutoff,)
                ).fetchall()
            return {
                int(r[0]): {
                    "trades": int(r[1]),
                    "wins": int(r[2] or 0),
                    "win_rate": (r[2] or 0) / max(r[1], 1),
                    "total_pnl": float(r[3] or 0),
                }
                for r in rows
            }
        except Exception as e:
            logger.warning(f"get_stats_by_hour failed: {e}")
            return {}

    def get_stats_by_regime(self, days: int = 30) -> dict:
        """Win rate et PnL par regime de marche."""
        try:
            with self._conn() as c:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                rows = c.execute(
                    """SELECT regime, COUNT(*), SUM(won), SUM(pnl)
                       FROM trade_context WHERE timestamp >= ?
                       GROUP BY regime""",
                    (cutoff,)
                ).fetchall()
            return {
                str(r[0] or "Unknown"): {
                    "trades": int(r[1]),
                    "wins": int(r[2] or 0),
                    "win_rate": (r[2] or 0) / max(r[1], 1),
                    "total_pnl": float(r[3] or 0),
                }
                for r in rows
            }
        except Exception as e:
            logger.warning(f"get_stats_by_regime failed: {e}")
            return {}

    def get_recent_losses_by_pattern(self, hours: int = 48) -> list:
        """Patterns recents ayant donne des pertes."""
        try:
            with self._conn() as c:
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                rows = c.execute(
                    """SELECT regime, hour_utc, side, pnl FROM trade_context
                       WHERE timestamp >= ? AND won = 0
                       ORDER BY timestamp DESC LIMIT 20""",
                    (cutoff,)
                ).fetchall()
            return [{"regime": r[0], "hour": r[1], "side": r[2], "pnl": r[3]} for r in rows]
        except Exception as e:
            logger.warning(f"get_recent_losses failed: {e}")
            return []

    def get_trades(self, symbol: str = None, limit: int = 100) -> pd.DataFrame:
        q = "SELECT * FROM trades"
        params = []
        if symbol:
            q += " WHERE symbol=?"
            params.append(symbol)
        q += f" ORDER BY timestamp DESC LIMIT {int(limit)}"
        try:
            with self._conn() as c:
                return pd.read_sql_query(q, c, params=params)
        except Exception as e:
            logger.warning(f"get_trades failed: {e}")
            return pd.DataFrame()

    def get_trades_today(self) -> list:
        """Return all trades (as dicts) from the current calendar day."""
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        try:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT timestamp,symbol,side,price,amount,value,fee,pnl,reason "
                    "FROM trades WHERE timestamp >= ? ORDER BY timestamp ASC",
                    (start,),
                ).fetchall()
            cols = ["timestamp", "symbol", "side", "price", "amount", "value", "fee", "pnl", "reason"]
            return [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            logger.warning(f"get_trades_today failed: {e}")
            return []

    def get_portfolio_history(self, limit: int = 500) -> pd.DataFrame:
        """Return the N MOST RECENT portfolio snapshots in chronological order."""
        try:
            with self._conn() as c:
                df = pd.read_sql_query(
                    f"SELECT * FROM portfolio_snapshots "
                    f"ORDER BY timestamp DESC LIMIT {int(limit)}",
                    c,
                )
            # Return in ascending chronological order for plotting / returns calc
            return df.iloc[::-1].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"get_portfolio_history failed: {e}")
            return pd.DataFrame()

    def get_latest_metrics(self) -> dict:
        try:
            with self._conn() as c:
                row = c.execute(
                    "SELECT * FROM model_metrics ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if row:
                    return {
                        "timestamp": row[1], "model_type": row[2],
                        "accuracy": row[3], "win_rate": row[4],
                        "sharpe": row[5], "total_trades": row[6],
                    }
        except Exception as e:
            logger.warning(f"get_latest_metrics failed: {e}")
        return {}
