#!/usr/bin/env python3
import os, sys, sqlite3, logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import config
from backtest import Backtester
from telegram_alerts import TelegramAlerter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FLIP_THRESHOLD  = getattr(config, "MIN_REVERSE_CONFIDENCE", 0.62)
ENTRY_THRESHOLD = getattr(config, "MIN_CONFIDENCE", 0.52)

def get_last_result(conn, symbol):
    try:
        row = conn.execute(
            "SELECT * FROM backtest_results WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
            (symbol,)
        ).fetchone()
        if not row:
            return None
        raw = dict(row)
        raw["total_return_pct"]    = raw.pop("total_return", 0)
        raw["buy_hold_return_pct"] = raw.pop("buy_hold_return", 0)
        raw["model_accuracy"]      = raw.pop("accuracy", 0)
        return raw
    except Exception:
        return None

def arrow(cur, prev, key):
    if prev is None:
        return ""
    d = cur[key] - prev[key]
    return f" ↑{d:+.1f}" if d > 0 else (f" ↓{d:+.1f}" if d < 0 else " →")

def format_report(results_map, previous_map):
    lines = [f"📊 *Backtest — {datetime.now().strftime('%d/%m %H:%M')}*\n"]
    overall_ok = True
    for sym, r in results_map.items():
        prev = previous_map.get(sym)
        ret  = r["total_return_pct"]
        bh   = r["buy_hold_return_pct"]
        wr   = r["win_rate"]
        sh   = r["sharpe"]
        dd   = r["max_drawdown"]
        acc  = r["model_accuracy"]
        n    = r["total_trades"]
        alpha = ret - bh
        if ret > bh and wr >= 50 and sh > 0.5:
            v = "✅"
        elif ret > 0 and wr >= 45:
            v = "⚠️"
        else:
            v = "❌"
            overall_ok = False
        lines.append(
            f"{v} *{sym}*\n"
            f"  Retour: `{ret:+.1f}%`{arrow(r,prev,'total_return_pct')} | B&H: `{bh:+.1f}%` | Alpha: `{alpha:+.1f}%`\n"
            f"  Win rate: `{wr:.1f}%`{arrow(r,prev,'win_rate')} | Sharpe: `{sh:.2f}`{arrow(r,prev,'sharpe')}\n"
            f"  Drawdown: `{dd:.1f}%` | Trades: `{n}` | Accuracy: `{acc:.1%}`\n"
        )
    lines.append("─────────────────────")
    if overall_ok:
        lines.append("🟢 Modèles OK — aucune action requise")
    else:
        lines.append(
            f"🔴 Performance dégradée\n"
            f"→ Retraining manuel conseillé\n"
            f"→ Seuils: entry={ENTRY_THRESHOLD:.0%} | flip={FLIP_THRESHOLD:.0%}"
        )
    return "\n".join(lines)

def main():
    logger.info("=== Auto-backtest démarré ===")
    bt   = Backtester(config)
    tg   = TelegramAlerter(config)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    results_map, previous_map = {}, {}
    for sym in config.SYMBOLS:
        logger.info(f"Backtest {sym}...")
        previous_map[sym] = get_last_result(conn, sym)
        r = bt.run_live_model(sym, limit=500)
        if not r:
            r = bt.run(sym, limit=1000)
        if r:
            results_map[sym] = r
    conn.close()
    if not results_map:
        tg.send("⚠️ Auto-backtest: aucun résultat généré")
        return
    report = format_report(results_map, previous_map)
    logger.info(f"Rapport:\n{report}")
    new_conf, new_flip, reason = adjust_thresholds(results_map)
    report += "\nSeuils MAJ: entry=" + str(round(new_conf*100)) + "% flip=" + str(round(new_flip*100)) + "% - " + reason
    tg.send(report)
    logger.info("=== Auto-backtest terminé ===")


def adjust_thresholds(results_map):
    overrides_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_overrides.json")
    scores = []
    for r in results_map.values():
        scores.append({
            "alpha_ok":  r["total_return_pct"] > r["buy_hold_return_pct"],
            "wr_ok":     r["win_rate"] >= 50,
            "sharpe_ok": r["sharpe"] > 0.5,
            "losing":    r["total_return_pct"] < 0 or r["win_rate"] < 45,
        })
    n      = len(scores)
    good   = sum(1 for s in scores if s["alpha_ok"] and s["wr_ok"] and s["sharpe_ok"])
    losing = sum(1 for s in scores if s["losing"])
    current = getattr(config, "MIN_CONFIDENCE", 0.52)
    if good == n:
        new_conf = max(0.50, round(current - 0.01, 2))
        reason   = "perf optimale, seuil abaisse"
    elif losing >= max(1, n // 2):
        new_conf = min(0.65, round(current + 0.02, 2))
        reason   = "perf degradee, seuil releve"
    else:
        new_conf = current
        reason   = "perf mixte, inchange"
    new_flip = round(new_conf + 0.10, 2)
    try:
        existing = {}
        if os.path.exists(overrides_path):
            existing = json.load(open(overrides_path))
        existing.update({"MIN_CONFIDENCE": new_conf, "MIN_REVERSE_CONFIDENCE": new_flip})
        json.dump(existing, open(overrides_path, "w"), indent=2)
        logger.info("Overrides: entry=" + str(new_conf) + " flip=" + str(new_flip) + " (" + reason + ")")
    except Exception as e:
        logger.error("Override write error: " + str(e))
    return new_conf, new_flip, reason


def main():
    logger.info("=== Auto-backtest demarre ===")
    bt   = Backtester(config)
    tg   = TelegramAlerter(config)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    results_map, previous_map = {}, {}
    for sym in config.SYMBOLS:
        logger.info("Backtest " + sym + "...")
        previous_map[sym] = get_last_result(conn, sym)
        r = bt.run_live_model(sym, limit=500)
        if not r:
            r = bt.run(sym, limit=1000)
        if r:
            results_map[sym] = r
    conn.close()
    if not results_map:
        tg.send("Auto-backtest: aucun resultat")
        return
    report = format_report(results_map, previous_map)
    new_conf, new_flip, reason = adjust_thresholds(results_map)
    report += "\nSeuils MAJ: entry=" + str(round(new_conf*100)) + "% flip=" + str(round(new_flip*100)) + "% - " + reason
    logger.info("Rapport:\n" + report)
    tg.send(report)
    logger.info("=== Auto-backtest termine ===")


if __name__ == "__main__":
    main()
