import asyncio
import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "live_test.db")
os.environ["BOT_VERSION"] = "test"

from dotenv import load_dotenv
load_dotenv()

import analyze


async def main() -> None:
    print("=" * 30, "CASE 1: MANUAL", "=" * 30, flush=True)
    manual = await analyze.analyze_symbol("BTCUSDT", "short", user_id=111111, chat_id=222222)
    print("MANUAL RESULT:", flush=True)
    print(manual["text"], flush=True)

    print("=" * 30, "CASE 2: AUTO SCAN", "=" * 30, flush=True)
    auto = await analyze.auto_scan_symbol_for_user(
        "BTCUSDT", "short", user_id=333333, chat_id=444444, scan_slot="live-test"
    )
    print("AUTO RESULT send=", auto.get("send"), "stage=", auto.get("stage"), flush=True)
    print(auto.get("text") or auto.get("reason"), flush=True)


asyncio.run(main())
