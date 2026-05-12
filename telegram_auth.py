#!/usr/bin/env python3
"""
First-time Telegram authentication.
Run this ONCE to verify your phone number.
After that, the bot can connect automatically.

Usage:
  cd /root/crypto_trader
  source venv/bin/activate
  python3 telegram_auth.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH", "")

if not API_ID or not API_HASH:
    print("ERROR: Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env")
    sys.exit(1)

async def main():
    from telethon import TelegramClient

    session_path = os.path.join(os.path.dirname(__file__), "telegram_monitor_session")
    client = TelegramClient(session_path, API_ID, API_HASH)

    print("=" * 50)
    print("  Telegram Authentication")
    print("=" * 50)
    print()
    print("This will ask for your phone number and a code.")
    print("You only need to do this ONCE.")
    print()

    await client.start()

    me = await client.get_me()
    print(f"\n✓ Authenticated as: {me.first_name} ({me.phone})")
    print(f"✓ Session saved to: {session_path}")

    # Test: try to access WatcherGuru
    print("\nTesting channel access...")
    channels = ["WatcherGuru", "whale_alert_io", "CoinDesk"]
    for ch in channels:
        try:
            entity = await client.get_entity(ch)
            name = getattr(entity, "title", ch)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {ch}: {e}")

    await client.disconnect()
    print("\n✓ Done! You can now start the bot normally.")
    print("  The bot will automatically connect to Telegram.")


asyncio.run(main())
