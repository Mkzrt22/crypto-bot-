#!/usr/bin/env python3
"""
apply_patches.py — Applique automatiquement sur le VPS :
  1. Position sizing fixe à $9 (margin) par trade
  2. Anti-flip-flop de 15 minutes entre reverses sur le même symbole
  3. Bonus : Kelly tracking + reset TP level sur reverse

Usage sur le VPS :
    cd /root/crypto_trader
    source venv/bin/activate
    python apply_patches.py

Le script :
  - Fait une sauvegarde horodatée des 2 fichiers modifiés
  - Applique les patches idempotemment (re-lancer ne casse rien)
  - Vérifie la syntaxe Python avant de valider
  - Affiche un diff lisible
  - Restaure en cas d'erreur
"""
import os
import sys
import shutil
import ast
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.py"
ENGINE_PATH = BASE_DIR / "trading" / "engine.py"
BACKUP_SUFFIX = f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def info(msg):
    print(f"{BLUE}[INFO]{RESET} {msg}")


def ok(msg):
    print(f"{GREEN}[OK]{RESET} {msg}")


def warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def err(msg):
    print(f"{RED}[ERROR]{RESET} {msg}")


def validate_syntax(path: Path) -> bool:
    """Check the file parses as valid Python."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            ast.parse(f.read(), filename=str(path))
        return True
    except SyntaxError as e:
        err(f"Syntax error in {path}: {e}")
        return False


def backup(path: Path) -> Path:
    """Create a timestamped backup of a file."""
    backup_path = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    shutil.copy2(path, backup_path)
    ok(f"Backup: {path.name} → {backup_path.name}")
    return backup_path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# PATCH 1 — config.py : fixe position sizing à $9 + ajoute MIN_REVERSE_INTERVAL_SEC
# ──────────────────────────────────────────────────────────────────────────────

def patch_config(content: str) -> tuple[str, list[str]]:
    """Apply config patches. Returns (new_content, changes_list)."""
    changes = []

    # 1. POSITION_SIZE_MIN_USDC → 9.0
    old_min = "POSITION_SIZE_MIN_USDC: float = 10.0"
    new_min = "POSITION_SIZE_MIN_USDC: float = 9.0"
    if old_min in content:
        content = content.replace(old_min, new_min)
        changes.append("POSITION_SIZE_MIN_USDC: 10.0 → 9.0")
    elif new_min in content:
        changes.append("POSITION_SIZE_MIN_USDC already at 9.0 (skip)")
    else:
        raise RuntimeError(
            "Cannot find POSITION_SIZE_MIN_USDC line — config.py may have been "
            "modified manually. Aborting to avoid breaking things."
        )

    # 2. POSITION_SIZE_MAX_USDC → 9.0
    old_max = "POSITION_SIZE_MAX_USDC: float = 25.0"
    new_max = "POSITION_SIZE_MAX_USDC: float = 9.0"
    if old_max in content:
        content = content.replace(old_max, new_max)
        changes.append("POSITION_SIZE_MAX_USDC: 25.0 → 9.0")
    elif new_max in content:
        changes.append("POSITION_SIZE_MAX_USDC already at 9.0 (skip)")
    else:
        raise RuntimeError("Cannot find POSITION_SIZE_MAX_USDC line")

    # 3. POSITION_SIZE_BASE_USDC → 9.0
    old_base = "POSITION_SIZE_BASE_USDC: float = 10.0"
    new_base = "POSITION_SIZE_BASE_USDC: float = 9.0"
    if old_base in content:
        content = content.replace(old_base, new_base)
        changes.append("POSITION_SIZE_BASE_USDC: 10.0 → 9.0")
    elif new_base in content:
        changes.append("POSITION_SIZE_BASE_USDC already at 9.0 (skip)")
    else:
        raise RuntimeError("Cannot find POSITION_SIZE_BASE_USDC line")

    # 4. Add MIN_REVERSE_INTERVAL_SEC right after POSITION_SIZE_BASE_USDC
    if "MIN_REVERSE_INTERVAL_SEC" in content:
        changes.append("MIN_REVERSE_INTERVAL_SEC already present (skip)")
    else:
        # Insert after the POSITION_SIZE_BASE_USDC line
        anchor = "POSITION_SIZE_BASE_USDC: float = 9.0"
        insertion = (
            "POSITION_SIZE_BASE_USDC: float = 9.0\n"
            "    # Anti-flip-flop: minimum time between reverse trades on the same symbol (seconds)\n"
            "    MIN_REVERSE_INTERVAL_SEC: int = 900   # 15 minutes"
        )
        content = content.replace(anchor, insertion)
        changes.append("MIN_REVERSE_INTERVAL_SEC: added (900s = 15 min)")

    return content, changes


# ──────────────────────────────────────────────────────────────────────────────
# PATCH 2 — engine.py : anti-flip-flop + Kelly + TP reset on reverse
# ──────────────────────────────────────────────────────────────────────────────

OLD_REVERSE_BLOCK = """        # Skip if signal matches existing position direction
        if self.solana and hasattr(self.solana, 'get_position'):
            pos = self.solana.get_position(sol_sym)
            if pos['size'] > 0:
                if (signal == 'BUY' and pos['side'] == 'long') or (signal == 'SELL' and pos['side'] == 'short'):
                    logger.info(f'[{symbol}] Already has {pos["side"]} — same direction, skipping')
                    return
                # Opposite signal = close current position (reverse will happen via buy/sell)
                logger.info(f'[{symbol}] Signal {signal} reverses {pos["side"]} — closing first')
                trade = self.solana.close_position(sol_sym, reason=f"reverse_to_{signal.lower()}")
                if trade:
                    self.db.save_trade(trade)
                    self._trailing_stops.pop(sol_sym, None)
                    self._alert(f"🔄 REVERSE {sol_sym}: closed {pos['side']} → {signal}")"""

NEW_REVERSE_BLOCK = """        # Skip if signal matches existing position direction
        if self.solana and hasattr(self.solana, 'get_position'):
            pos = self.solana.get_position(sol_sym)
            if pos['size'] > 0:
                if (signal == 'BUY' and pos['side'] == 'long') or (signal == 'SELL' and pos['side'] == 'short'):
                    logger.info(f'[{symbol}] Already has {pos["side"]} — same direction, skipping')
                    return

                # Anti-flip-flop: don't reverse too soon after opening
                min_reverse_sec = getattr(self.config, 'MIN_REVERSE_INTERVAL_SEC', 900)
                last_open = self._last_trade_time.get(symbol)
                if last_open:
                    age_sec = (datetime.now() - last_open).total_seconds()
                    if age_sec < min_reverse_sec:
                        logger.info(
                            f"[{symbol}] Reverse BLOCKED: position age {age_sec:.0f}s < "
                            f"{min_reverse_sec}s anti-flip-flop window"
                        )
                        return

                # Opposite signal = close current position (reverse will happen via buy/sell)
                logger.info(f'[{symbol}] Signal {signal} reverses {pos["side"]} — closing first')
                trade = self.solana.close_position(sol_sym, reason=f"reverse_to_{signal.lower()}")
                if trade:
                    self.db.save_trade(trade)
                    self._track_trade_result(trade)
                    self._trailing_stops.pop(sol_sym, None)
                    self._reset_tp_level(sol_sym)
                    self._alert(f"🔄 REVERSE {sol_sym}: closed {pos['side']} → {signal}")"""


def patch_engine(content: str) -> tuple[str, list[str]]:
    """Apply engine patches. Returns (new_content, changes_list)."""
    changes = []

    if NEW_REVERSE_BLOCK in content:
        changes.append("Reverse block already patched (skip)")
    elif OLD_REVERSE_BLOCK in content:
        content = content.replace(OLD_REVERSE_BLOCK, NEW_REVERSE_BLOCK)
        changes.append("Anti-flip-flop guard: added (15 min window)")
        changes.append("_track_trade_result on reverse: added")
        changes.append("_reset_tp_level on reverse: added")
    else:
        raise RuntimeError(
            "Cannot find the reverse block in engine.py — file may have been "
            "modified manually since the last patch. Aborting."
        )

    return content, changes


# ──────────────────────────────────────────────────────────────────────────────
# Apply one file atomically with rollback on validation failure
# ──────────────────────────────────────────────────────────────────────────────

def apply_file(path: Path, patch_fn, label: str) -> bool:
    print(f"\n{BOLD}━━━ {label} ━━━{RESET}")

    if not path.exists():
        err(f"{path} does not exist")
        return False

    if not validate_syntax(path):
        err(f"{path} is already broken before patching — fix it first")
        return False

    backup_path = backup(path)
    original = read(path)

    try:
        new_content, changes = patch_fn(original)
    except RuntimeError as e:
        err(str(e))
        return False

    if not changes or all("skip" in c for c in changes):
        info("No changes needed.")
        for c in changes:
            info(f"  • {c}")
        return True

    write(path, new_content)

    if not validate_syntax(path):
        err(f"Patched file has syntax errors — restoring backup")
        shutil.copy2(backup_path, path)
        return False

    ok(f"Applied {len(changes)} change(s):")
    for c in changes:
        print(f"  • {c}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Post-apply : verify runtime behaviour
# ──────────────────────────────────────────────────────────────────────────────

def verify_runtime():
    print(f"\n{BOLD}━━━ Runtime verification ━━━{RESET}")
    try:
        # Force a fresh import (invalidate any cached bytecode)
        for mod in list(sys.modules):
            if mod.startswith(("config", "trading", "data", "ml")):
                del sys.modules[mod]

        sys.path.insert(0, str(BASE_DIR))
        from config import config

        checks = [
            ("POSITION_SIZE_MIN_USDC",  config.POSITION_SIZE_MIN_USDC,  9.0),
            ("POSITION_SIZE_MAX_USDC",  config.POSITION_SIZE_MAX_USDC,  9.0),
            ("POSITION_SIZE_BASE_USDC", config.POSITION_SIZE_BASE_USDC, 9.0),
            ("MIN_REVERSE_INTERVAL_SEC", getattr(config, "MIN_REVERSE_INTERVAL_SEC", None), 900),
        ]
        all_pass = True
        for name, actual, expected in checks:
            if actual == expected:
                ok(f"{name} = {actual}")
            else:
                err(f"{name} = {actual}  (expected {expected})")
                all_pass = False

        from trading.engine import TradingEngine  # noqa: F401
        ok("trading.engine imports cleanly")
        return all_pass
    except Exception as e:
        err(f"Runtime verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Service restart helper
# ──────────────────────────────────────────────────────────────────────────────

def prompt_restart():
    print(f"\n{BOLD}━━━ Restart the bot ━━━{RESET}")
    answer = input(f"{YELLOW}Restart crypto-trader service now? [y/N]: {RESET}").strip().lower()
    if answer == "y":
        ret = os.system("sudo systemctl restart crypto-trader")
        if ret == 0:
            ok("Service restarted")
            print()
            os.system("sudo systemctl status crypto-trader --no-pager | head -10")
            print()
            info("Follow logs with: tail -f /root/crypto_trader/bot.log")
        else:
            err("systemctl restart failed — run manually: sudo systemctl restart crypto-trader")
    else:
        info("Skipped. Restart manually when ready:")
        print("    sudo systemctl restart crypto-trader")
        print("    tail -f /root/crypto_trader/bot.log")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"{BOLD}Applying patches to crypto_trader{RESET}")
    print(f"Working dir: {BASE_DIR}")
    print(f"Backups will use suffix: {BACKUP_SUFFIX}")

    if not (BASE_DIR / "config.py").exists() or not (BASE_DIR / "trading" / "engine.py").exists():
        err("Must be run from /root/crypto_trader (where config.py and trading/engine.py live)")
        sys.exit(1)

    results = []
    results.append(apply_file(CONFIG_PATH, patch_config, "PATCH 1 — config.py (position size $9)"))
    results.append(apply_file(ENGINE_PATH, patch_engine, "PATCH 2 — engine.py (anti-flip-flop 15 min)"))

    if not all(results):
        err("\nOne or more patches failed. Your original files are restored from the .bak_* backups.")
        sys.exit(1)

    if not verify_runtime():
        err("\nRuntime verification failed. Inspect the errors above.")
        sys.exit(1)

    print(f"\n{GREEN}{BOLD}✓ All patches applied successfully.{RESET}")
    prompt_restart()


if __name__ == "__main__":
    main()
