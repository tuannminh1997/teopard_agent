import sqlite3
import os
import asyncio
import traceback
import tempfile
import uuid
from pathlib import Path

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

DB_PATH = os.getenv("DB_PATH", "bot.db")
ANALYZE_SHORT_CALLBACK_PREFIX = "analyze_short"
ANALYZE_LONG_CALLBACK_PREFIX  = "analyze_long"


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().lstrip("/").upper()


def init_symbol_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS allowed_symbols (
                symbol TEXT PRIMARY KEY
            )
        """)
        conn.commit()


def add_allowed_symbol(symbol: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO allowed_symbols (symbol) VALUES (?)",
            (normalize_symbol(symbol),),
        )
        conn.commit()


def remove_allowed_symbol(symbol: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM allowed_symbols WHERE symbol = ?",
            (normalize_symbol(symbol),),
        )
        conn.commit()


def is_allowed_symbol(symbol: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM allowed_symbols WHERE symbol = ?",
            (normalize_symbol(symbol),),
        ).fetchone()
    return row is not None


def get_allowed_symbols() -> list[str]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT symbol FROM allowed_symbols ORDER BY symbol"
        ).fetchall()
    return [r[0] for r in rows]


def symbol_analysis_keyboard(symbol: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Scalp (15m/1H/4H)", callback_data=f"{ANALYZE_SHORT_CALLBACK_PREFIX}:{symbol}"),
        InlineKeyboardButton("Swing (4H/1D/1W)",  callback_data=f"{ANALYZE_LONG_CALLBACK_PREFIX}:{symbol}"),
    ]])




def split_telegram_message(text: str, limit: int = 3900) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) <= limit:
            current += line
            continue
        if current:
            chunks.append(current.strip())
            current = ""
        while len(line) > limit:
            chunks.append(line[:limit].strip())
            line = line[limit:]
        current = line
    if current:
        chunks.append(current.strip())
    return chunks


# ─── Command handlers ─────────────────────────────────────────────────────────

async def add_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from auth import is_admin
    admin = update.effective_user
    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return
    if not context.args:
        await update.effective_message.reply_text("Cú pháp đúng: /addsymbol BTC")
        return
    symbol = normalize_symbol(context.args[0])
    add_allowed_symbol(symbol)
    await update.effective_message.reply_text(f"Đã thêm symbol {symbol}.")


async def remove_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from auth import is_admin
    admin = update.effective_user
    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return
    if not context.args:
        await update.effective_message.reply_text("Cú pháp đúng: /removesymbol BTC")
        return
    symbol = normalize_symbol(context.args[0])
    remove_allowed_symbol(symbol)
    await update.effective_message.reply_text(f"Đã xóa symbol {symbol}.")


async def list_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbols = get_allowed_symbols()
    if not symbols:
        await update.effective_message.reply_text("Danh sách symbol hiện đang trống.")
        return
    await update.effective_message.reply_text(
        "Danh sách symbol được phép:\n" + "\n".join(f"• {s}" for s in symbols)
    )


# ─── Message handler: user types a coin symbol ──────────────────────────────

async def handle_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    from auth import is_account_activated, show_start_menu, verified_users

    user = update.effective_user
    message = update.effective_message
    if not user or not message or not message.text:
        return False

    symbol = normalize_symbol(message.text)
    if not is_allowed_symbol(symbol):
        return False

    if not is_account_activated(user.id):
        verified_users.discard(user.id)
        await show_start_menu(update)
        return True

    await message.reply_text(
        f"Bạn muốn phân tích {symbol}/USDT theo kiểu nào?",
        reply_markup=symbol_analysis_keyboard(symbol),
    )
    return True


async def symbol_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    handled = await handle_symbol(update, context)
    if handled:
        raise ApplicationHandlerStop


# ─── Callback: user chooses Scalp/Swing ─────────────────────────────────────

async def analyze_symbol_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import analyze_symbol
    from auth import get_user_usage, increment_user_usage, is_account_activated, show_start_menu, verified_users

    query = update.callback_query
    user  = update.effective_user
    if not query or not user or not query.data:
        return

    await query.answer()

    if not is_account_activated(user.id):
        verified_users.discard(user.id)
        await show_start_menu(update)
        return

    action, symbol = query.data.split(":", 1)
    mode = "short" if action == ANALYZE_SHORT_CALLBACK_PREFIX else "long"
    mode_label = "Scalp (15m/1H/4H)" if mode == "short" else "Swing (4H/1D/1W)"

    daily_limit, used_today = get_user_usage(user.id)
    remaining = daily_limit - used_today

    if remaining <= 0:
        await query.message.reply_text(
            f"Bạn đã hết {daily_limit} lượt hôm nay. "
            "Vui lòng chờ sang ngày mới hoặc liên hệ admin."
        )
        return

    await query.message.reply_text(
        f"Đang phân tích {symbol}/USDT — {mode_label}. "
        f"Vui lòng chờ... (còn {remaining - 1} lượt hôm nay)"
    )

    try:
        result_payload = await analyze_symbol(symbol, mode, user_id=user.id, chat_id=query.message.chat_id)
    except Exception as exc:
        error_text = str(exc)
        print(
            f"[MANUAL_ERROR] symbol={symbol} mode={mode} user_id={user.id} "
            f"error_type={type(exc).__name__} error={error_text}",
            flush=True,
        )
        traceback.print_exc()
        if "timed out" in error_text.lower() or "timeout" in error_text.lower():
            await query.message.reply_text(
                "Phân tích thất bại: AI cuối không trả lời kịp sau lần thử chính và một lần retry. "
                "Lượt sử dụng không bị trừ; bạn có thể chạy lại sau ít phút."
            )
        else:
            await query.message.reply_text(f"Phân tích thất bại: {error_text}")
        return

    increment_user_usage(user.id)

    if isinstance(result_payload, dict):
        result_text = result_payload.get("text", "")
    else:
        result_text = str(result_payload)

    chunks = split_telegram_message(result_text)
    for chunk in chunks:
        await query.message.reply_text(chunk)



# ─── Background job: auto-check WIN/LOSS ────────────────────────────────────

async def job_check_predictions(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Runs periodically, auto-checks due predictions, and only updates the DB without sending automatic messages."""
    from datetime import datetime
    from analyze import auto_check_pending_predictions
    from evaluation_store import cleanup_evaluation_data, update_evaluation_tracking

    print(f"[AUTO_CHECK] Job chạy lúc {datetime.now().isoformat()}", flush=True)
    payload = await auto_check_pending_predictions()
    await asyncio.to_thread(cleanup_evaluation_data)
    tracking = await asyncio.to_thread(update_evaluation_tracking)

    if isinstance(payload, dict):
        print(
            "[AUTO_CHECK] Done: "
            f"due={payload.get('due_count', 0)}, "
            f"entry_filled={payload.get('entry_filled_count', 0)}, "
            f"closed={payload.get('closed_count', 0)}, "
            f"rescheduled={payload.get('rescheduled_count', 0)}, eval_updated={tracking.get('updated', 0)}",
            flush=True,
        )

    # No automatic notification is sent to the user/admin.
    # Users who want to see results should use /history, /stats, or /dashboard.


def command_scope_user_id(update: Update) -> int | None:
    user = update.effective_user
    if not user:
        return None
    # By default everyone, including admin, sees their own data.
    # Admin uses dedicated commands to see the whole system: /statsall, /historyall, /dashboardall.
    return user.id


def is_current_user_admin(update: Update) -> bool:
    from auth import is_admin

    user = update.effective_user
    return bool(user and is_admin(user.id))


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_stats
    symbol = context.args[0] if context.args else None
    await update.effective_message.reply_text(format_stats(symbol, user_id=command_scope_user_id(update)))


async def statsall_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_stats

    if not is_current_user_admin(update):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return
    symbol = context.args[0] if context.args else None
    await update.effective_message.reply_text(format_stats(symbol, user_id=None))


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_history
    symbol = context.args[0] if context.args else None
    await update.effective_message.reply_text(format_history(symbol, user_id=command_scope_user_id(update)))


async def historyall_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_history

    if not is_current_user_admin(update):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return
    symbol = context.args[0] if context.args else None
    await update.effective_message.reply_text(format_history(symbol, user_id=None))


async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_stats
    await update.effective_message.reply_text(format_stats(user_id=command_scope_user_id(update)))


async def dashboardall_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_stats

    if not is_current_user_admin(update):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return
    await update.effective_message.reply_text(format_stats(user_id=None))


async def clearhistory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from auth import is_admin
    from analyze import clear_prediction_history

    admin = update.effective_user
    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return

    if not context.args or context.args[0].upper() != "CONFIRM":
        await update.effective_message.reply_text(
            "Lệnh này sẽ xóa toàn bộ lịch sử phân tích/prediction nhưng vẫn giữ whitelist và danh sách symbol.\n"
            "Gõ: /clearhistory CONFIRM"
        )
        return

    payload = clear_prediction_history()
    if isinstance(payload, dict):
        await update.effective_message.reply_text(
            "Đã xóa lịch sử theo dõi. Whitelist và danh sách symbol vẫn được giữ.\n"
            f"Lệnh đã trade/đang theo dõi: {payload.get('visible_count', 0)}\n"
            f"Tổng dòng predictions cũ đã xóa: {payload.get('total_prediction_count', 0)}\n"
            f"Lệnh nháp chưa xác nhận đã xóa: {payload.get('draft_count', 0)}"
        )
    else:
        await update.effective_message.reply_text(
            f"Đã xóa {payload} prediction khỏi lịch sử. Whitelist và danh sách symbol vẫn được giữ."
        )


async def cleardrafts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Deletes only draft/candidate trades, leaving the traded history untouched."""
    from auth import is_admin
    from analyze import clear_trade_candidates

    user = update.effective_user
    if not user:
        return

    args_upper = [arg.upper() for arg in (context.args or [])]
    clear_all = "ALL" in args_upper

    if clear_all and not is_admin(user.id):
        await update.effective_message.reply_text("Bạn không có quyền xóa lệnh nháp của toàn hệ thống.")
        return

    if "CONFIRM" not in args_upper:
        if clear_all:
            await update.effective_message.reply_text(
                "Lệnh này chỉ xóa lệnh nháp/candidate của toàn hệ thống, KHÔNG xóa /history.\n"
                "Gõ: /cleardrafts ALL CONFIRM"
            )
        else:
            await update.effective_message.reply_text(
                "Lệnh này chỉ xóa lệnh nháp/candidate của bạn, KHÔNG xóa /history.\n"
                "Gõ: /cleardrafts CONFIRM\n"
                "Admin muốn xóa toàn hệ thống: /cleardrafts ALL CONFIRM"
            )
        return

    payload = await asyncio.to_thread(clear_trade_candidates, None if clear_all else user.id)
    scope_text = "toàn hệ thống" if clear_all else "của bạn"
    reset_text = "Có" if payload.get("sequence_reset") else "Không, vì vẫn còn candidate của user khác"
    await update.effective_message.reply_text(
        f"Đã xóa lệnh nháp {scope_text}. History đã trade vẫn được giữ nguyên.\n"
        f"Tổng candidate đã xóa: {payload.get('deleted_count', 0)}\n"
        f"- Nháp còn chờ: {payload.get('draft_count', 0)}\n"
        f"- Đã hết hạn: {payload.get('expired_count', 0)}\n"
        f"- Đã bỏ qua: {payload.get('discarded_count', 0)}\n"
        f"- Đang xác nhận: {payload.get('confirming_count', 0)}\n"
        f"- Đã xác nhận/copy sang history: {payload.get('confirmed_count', 0)}\n"
        f"Reset ID lệnh nháp: {reset_text}"
    )


async def checknow_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from auth import is_admin
    from analyze import auto_check_pending_predictions
    from evaluation_store import cleanup_evaluation_data, update_evaluation_tracking

    admin = update.effective_user
    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return

    await update.effective_message.reply_text("Đang ép kiểm tra toàn bộ prediction đang mở ngay bây giờ...")
    payload = await auto_check_pending_predictions(force=True)

    if not isinstance(payload, dict):
        await update.effective_message.reply_text("Đã kiểm tra xong.")
        return

    closed_count = int(payload.get("closed_count", 0))
    entry_filled_count = int(payload.get("entry_filled_count", 0))
    rescheduled_count = int(payload.get("rescheduled_count", 0))
    due_count = int(payload.get("due_count", 0))

    if due_count == 0:
        await update.effective_message.reply_text("Không có prediction đang mở để kiểm tra.")
        return

    await update.effective_message.reply_text(
        "Đã kiểm tra xong và cập nhật DB.\n"
        f"Prediction đang mở đã kiểm tra: {due_count}\n"
        f"Mới khớp Entry: {entry_filled_count}\n"
        f"Có kết quả cuối: {closed_count}\n"
        f"Tiếp tục chờ: {rescheduled_count}\n\n"
        "Bot không gửi thông báo tự động cho user/admin nữa. "
        "Cần xem chi tiết thì dùng /history, /stats, /dashboard hoặc /historyall."
    )


async def autoscanon_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from auth import is_account_activated
    from analyze import (
        set_auto_scan_enabled, _normalize_auto_scan_modes, AUTO_SCAN_INTERVAL_SECONDS,
        AUTO_SCAN_MIN_PREFILTER_CONFIDENCE, AUTO_SCAN_PREFILTER_MIN_DIRECTION_GAP,
        AUTO_SCAN_MIN_FINAL_CONFIDENCE, AUTO_SCAN_MAX_GLM_CALLS_PER_DAY, normalize_auto_scan_symbol
    )

    user = update.effective_user
    message = update.effective_message
    if not user or not message:
        return
    if not is_account_activated(user.id):
        from auth import show_start_menu, verified_users
        verified_users.discard(user.id)
        await show_start_menu(update)
        return

    if not context.args:
        await message.reply_text(
            "Cú pháp: /autoscanon BTC\n"
            "Ví dụ: /autoscanon btc\n"
            "Auto Scan chỉ chạy 1 symbol tại một thời điểm cho mỗi tài khoản."
        )
        return

    raw_symbols = context.args
    symbols = []
    seen = set()
    for raw in raw_symbols:
        for part in str(raw).replace(",", " ").split():
            sym = normalize_auto_scan_symbol(part)
            if sym and sym not in seen:
                symbols.append(sym)
                seen.add(sym)

    if not symbols:
        await message.reply_text("Không đọc được symbol. Ví dụ đúng: /autoscanon BTC")
        return
    if len(symbols) > 1:
        await message.reply_text(
            "Auto Scan chỉ cho quét 1 symbol tại một thời điểm để tiết kiệm tài nguyên.\n"
            "Ví dụ đúng: /autoscanon BTC\n"
            "Muốn đổi symbol thì gõ lại /autoscanon <symbol_mới>."
        )
        return

    # Only allow enabling Auto Scan for a symbol that's already on the allowed list.
    not_allowed = []
    for sym in symbols:
        base = sym[:-4] if sym.endswith("USDT") else sym
        if not is_allowed_symbol(base) and not is_allowed_symbol(sym):
            not_allowed.append(base)
    if not_allowed:
        await message.reply_text(
            "Symbol chưa có trong danh sách được phép: " + ", ".join(not_allowed) +
            "\nAdmin cần thêm bằng /addsymbol <symbol> trước."
        )
        return

    enable_result = await asyncio.to_thread(set_auto_scan_enabled, user.id, message.chat_id, True, symbols)
    if enable_result.get("quota_blocked"):
        await message.reply_text(
            f"Auto Scan đã dùng đủ {AUTO_SCAN_MAX_GLM_CALLS_PER_DAY} lượt gọi AI cuối trong ngày. "
            "Bot sẽ tự bật lại và reset quota lúc 07:00 sáng mai theo giờ Việt Nam."
        )
        return
    modes = ", ".join("SCALP" if m == "short" else "SWING" for m in _normalize_auto_scan_modes())
    await message.reply_text(
        "Đã bật Auto Scan cho tài khoản của bạn.\n"
        f"Symbol đang quét: {symbols[0]}.\n"
        f"Chu kỳ quét: mỗi {int(AUTO_SCAN_INTERVAL_SECONDS // 60)} phút.\n"
        f"Mode đang quét: {modes}.\n"
        f"DeepSeek mini-rubric tối thiểu: {AUTO_SCAN_MIN_PREFILTER_CONFIDENCE}/100.\n"
        f"Chênh lệch LONG/SHORT tối thiểu: {AUTO_SCAN_PREFILTER_MIN_DIRECTION_GAP} điểm.\n"
        f"Điểm tín hiệu AI cuối tối thiểu: {AUTO_SCAN_MIN_FINAL_CONFIDENCE}/100.\n"
        f"Giới hạn gọi AI cuối: {AUTO_SCAN_MAX_GLM_CALLS_PER_DAY} lần/ngày Auto Scan.\n"
        "Đủ quota thì Auto Scan tự dừng; 07:00 sáng hôm sau tự bật và reset quota.\n"
        "Giờ nghỉ tự động: 00:00-07:00 theo giờ Việt Nam; sáng bot tự bật lại nếu trước đó đang bật.\n"
        "Khi có tín hiệu đủ tốt, bot sẽ tự gửi và tự lưu theo dõi, không cần bấm xác nhận."
    )

async def autoscanoff_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import set_auto_scan_enabled

    user = update.effective_user
    message = update.effective_message
    if not user or not message:
        return
    await asyncio.to_thread(set_auto_scan_enabled, user.id, message.chat_id, False)
    await message.reply_text("Đã tắt Auto Scan cho tài khoản của bạn.")




def _display_scan_direction(value) -> str:
    raw = str(value or "-").strip().upper().replace("_", " ")
    if raw in {"", "-"}:
        return "-"
    if raw in {"NO TRADE", "NO  TRADE", "NOTRADE"}:
        return "NO TRADE"
    return raw




def _display_scan_stage(stage, status=None) -> str:
    stage_raw = str(stage or "-").lower()
    status_raw = str(status or "-").lower()
    stage_map = {
        "deepseek": "DeepSeek",
        "planner": "Planner Pro",
        "reviewer": "Flash reviewer",
        "trigger": "Trigger",
        "confirmation": "Xác nhận bias",
        "binance": "Binance",
        "cooldown": "Cooldown",
        "quota": "Quota AI cuối",
        "sent": "Đã gửi",
        "error": "Lỗi",
    }
    status_map = {
        "rejected": "bỏ qua",
        "skipped": "bỏ qua",
        "error": "lỗi",
        "sent": "đã gửi",
        "ok": "đã gửi",
        "waiting": "đang chờ",
    }
    return f"{stage_map.get(stage_raw, stage_raw.upper() if stage_raw != '-' else '-')} → {status_map.get(status_raw, status_raw)}"


def _display_scan_reason(reason) -> str:
    text = str(reason or "-").strip()
    if not text or text == "-":
        return "-"
    lower = text.lower()
    replacements = {
        "prefilter rejected: no actionable long/short signal": "DeepSeek bỏ qua vì tín hiệu chưa đạt ngưỡng.",
        "prefilter rejected: no actionable long/short structure": "DeepSeek bỏ qua vì tín hiệu chưa đạt ngưỡng.",
        "deepseek không chọn được hướng long/short để gửi glm.": "DeepSeek bỏ qua vì tín hiệu chưa đạt ngưỡng.",
        "glm returned no trade": "AI cuối chọn NO TRADE sau phân tích đầy đủ.",
        "cooldown": "Đang trong thời gian chờ, chưa gửi lại tín hiệu cùng symbol/mode.",
        "direction cooldown": "Đang trong thời gian chờ, chưa gửi lại tín hiệu cùng hướng.",
        "no binance data": "Không lấy được dữ liệu Binance.",
    }
    if lower in replacements:
        return replacements[lower]
    # Clean older English prefixes if any remain.
    text = text.replace("prefilter rejected:", "DeepSeek bỏ qua:").replace("final rejected:", "AI cuối bỏ qua:")
    text = text.replace("signal score", "điểm tín hiệu").replace("below", "dưới ngưỡng")
    return text

def _display_scan_score(direction, confidence, *, source: str) -> str:
    label = _display_scan_direction(direction)
    if label == "-":
        return "Chưa gọi" if source == "planner" else "-"
    if label == "NO TRADE":
        return "NO TRADE"
    if confidence is None:
        return f"{label} -/100"
    if source == "deepseek":
        return f"{label} {confidence}/100"
    return f"{label} {confidence}/100"


def _int_or_none(value):
    try:
        if value is None or value == "":
            return None
        return int(round(float(value)))
    except Exception:
        return None


def _extract_prefilter_pair(item: dict | None) -> tuple[int | None, int | None, int | None]:
    """Return (long_score, short_score, gap) for DeepSeek prefilter display.

    New logs store these fields directly. Old logs can still be readable because the
    reason usually contains: LONG 42/100, SHORT 43/100; gap 1 point.
    """
    item = item or {}
    long_score = _int_or_none(item.get("pre_long_score"))
    short_score = _int_or_none(item.get("pre_short_score"))
    gap = _int_or_none(item.get("pre_gap"))
    if long_score is not None and short_score is not None:
        if gap is None:
            gap = abs(long_score - short_score)
        return long_score, short_score, gap

    import re
    reason = str(item.get("reason") or "")
    m = re.search(r"LONG\s+(\d+)\s*/\s*100\s*,\s*SHORT\s+(\d+)\s*/\s*100", reason, flags=re.IGNORECASE)
    if m:
        long_score = int(m.group(1))
        short_score = int(m.group(2))
        gap_m = re.search(r"chênh\s+(\d+)", reason, flags=re.IGNORECASE)
        gap = int(gap_m.group(1)) if gap_m else abs(long_score - short_score)
        return long_score, short_score, gap
    return long_score, short_score, gap


def _display_prefilter_score(item: dict | None) -> str:
    """DeepSeek prefilter user-facing display.

    Avoid showing fake `LONG 0 / SHORT 0` when the Flash response was not parseable.
    0/100 should only be shown when it was a real parsed score.
    """
    item = item or {}
    reason = str(item.get("reason") or "")
    if "Không parse được mini-rubric" in reason or "Khong parse duoc mini-rubric" in reason:
        return "Không parse được mini-rubric"

    direction = _display_scan_direction(item.get("pre_direction"))
    long_score, short_score, gap = _extract_prefilter_pair(item)

    if long_score is not None and short_score is not None:
        if gap is None:
            gap = abs(long_score - short_score)
        if direction == "NEUTRAL":
            return f"LONG {long_score}/100 | SHORT {short_score}/100 (gần cân bằng, chênh {gap})"
        if direction in {"LONG", "SHORT"}:
            other_label = "SHORT" if direction == "LONG" else "LONG"
            other_score = short_score if direction == "LONG" else long_score
            main_score = long_score if direction == "LONG" else short_score
            return f"{direction} {main_score}/100 | {other_label} {other_score}/100 (chênh {gap})"

    # Fallback for very old logs.
    return _display_scan_score(item.get("pre_direction"), item.get("pre_confidence"), source="deepseek")

async def autoscanstatus_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import (
        get_auto_scan_runtime_status, _parse_auto_scan_symbols_text, _auto_scan_symbols_from_env_or_db,
        _normalize_auto_scan_modes, AUTO_SCAN_INTERVAL_SECONDS, AUTO_SCAN_MIN_PREFILTER_CONFIDENCE,
        AUTO_SCAN_PREFILTER_MIN_DIRECTION_GAP, AUTO_SCAN_MIN_FINAL_CONFIDENCE,
        AUTO_SCAN_SIGNAL_COOLDOWN_MINUTES, AUTO_SCAN_MAX_GLM_CALLS_PER_DAY, DEEPSEEK_MODEL,
        DEEPSEEK_REVIEW_MODEL, FINAL_REVIEW_MIN_SIGNAL_SCORE,
        get_ai_model_name, get_ai_provider_label, _auto_scan_format_dt,
    )

    user = update.effective_user
    message = update.effective_message
    if not user or not message:
        return
    status = await asyncio.to_thread(get_auto_scan_runtime_status, user.id)
    symbols = _parse_auto_scan_symbols_text(status.get("symbols")) or await asyncio.to_thread(_auto_scan_symbols_from_env_or_db)
    modes = ", ".join("SCALP" if m == "short" else "SWING" for m in _normalize_auto_scan_modes())
    last_log = status.get("last_log") or {}
    last_line = "Chưa có log scan."
    if last_log:
        pre = _display_prefilter_score(last_log)
        planner_direction = _display_scan_direction(last_log.get('final_direction'))
        reviewer_score = last_log.get('final_confidence')
        reviewer_verdict = last_log.get('reviewer_verdict') or (
            "REJECT" if str(last_log.get('stage') or '').lower() == "reviewer" and str(last_log.get('status') or '').lower() == "rejected" else "-"
        )
        reviewer_text = (
            f"{reviewer_score}/100 — {reviewer_verdict}" if reviewer_score is not None
            else ("PARSE ERROR — REJECT" if reviewer_verdict == "REJECT" else "Chưa gọi")
        )
        last_line = (
            f"{_auto_scan_format_dt(last_log.get('scanned_at'))} | "
            f"{last_log.get('symbol')} {'SCALP' if last_log.get('mode') == 'short' else 'SWING'} | "
            f"{_display_scan_stage(last_log.get('stage'), last_log.get('status'))} | "
            f"Prefilter Flash: {pre} | Planner Pro: {planner_direction} | "
            f"Flash reviewer: {reviewer_text} | {_display_scan_reason(last_log.get('reason'))}"
        )
    if status.get("quota_resume"):
        state_text = "⏸ ĐÃ ĐỦ QUOTA AI CUỐI — sẽ tự bật lại lúc 07:00"
    elif status.get("in_sleep_window") and status.get("night_resume"):
        state_text = "🌙 ĐANG NGHỈ ĐÊM — sẽ tự bật lại lúc 07:00"
    else:
        state_text = "🟢 ĐANG BẬT" if status.get("enabled") else "🔴 ĐANG TẮT"

    await message.reply_text(
        "🤖 Auto Scan status:\n"
        f"Trạng thái: {state_text}\n"
        f"Giờ hoạt động tự động: 07:00-24:00 theo giờ Việt Nam\n"
        f"Symbol: {', '.join(symbols) if symbols else 'chưa chọn'}\n"
        f"Chu kỳ nến: {int(AUTO_SCAN_INTERVAL_SECONDS // 60)} phút, quét theo nến đóng\n"
        f"Mode: {modes}\n"
        "Giới hạn: 1 symbol/tài khoản\n"
        f"DeepSeek prefilter: {DEEPSEEK_MODEL}\n"
        f"Planner Pro: {get_ai_model_name()} ({get_ai_provider_label()})\n"
        f"Flash reviewer: {DEEPSEEK_REVIEW_MODEL}\n"
        f"Ngưỡng mini-rubric DeepSeek: {AUTO_SCAN_MIN_PREFILTER_CONFIDENCE}/100\n"
        f"Chênh lệch hướng tối thiểu: {AUTO_SCAN_PREFILTER_MIN_DIRECTION_GAP} điểm\n"
        f"Ngưỡng reviewer chung/Manual: {FINAL_REVIEW_MIN_SIGNAL_SCORE}/100\n"
        f"Ngưỡng gửi Auto Scan: {AUTO_SCAN_MIN_FINAL_CONFIDENCE}/100\n"
        f"Quota gọi AI cuối hôm nay: {status.get('glm_calls_today', 0)}/{AUTO_SCAN_MAX_GLM_CALLS_PER_DAY} "
        f"(còn {status.get('glm_calls_remaining', AUTO_SCAN_MAX_GLM_CALLS_PER_DAY)} lượt)\n"
        f"Cooldown cùng symbol/mode: {AUTO_SCAN_SIGNAL_COOLDOWN_MINUTES} phút\n"
        f"Lần quét gần nhất: {_auto_scan_format_dt(status.get('last_scan_at'))}\n"
        f"Lần quét kế tiếp: {_auto_scan_format_dt(status.get('next_scan_at'))}\n"
        f"Log gần nhất: {last_line}"
    )


async def autoscanlog_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import get_auto_scan_logs, _auto_scan_format_dt

    user = update.effective_user
    message = update.effective_message
    if not user or not message:
        return
    logs = await asyncio.to_thread(get_auto_scan_logs, user.id, 5)
    if not logs:
        await message.reply_text("Chưa có log Auto Scan nào. Bot sẽ có log sau lần quét đầu tiên theo nến đóng.")
        return
    lines = ["🧾 Auto Scan log gần nhất:"]
    for item in reversed(logs):
        mode_label = "SCALP" if item.get("mode") == "short" else "SWING"
        pre = _display_prefilter_score(item)
        planner_direction = _display_scan_direction(item.get('final_direction'))
        reviewer_score = item.get('final_confidence')
        reviewer_verdict = item.get('reviewer_verdict') or (
            "REJECT" if str(item.get('stage') or '').lower() == "reviewer" and str(item.get('status') or '').lower() == "rejected" else "-"
        )
        reviewer_text = (
            f"{reviewer_score}/100 — {reviewer_verdict}" if reviewer_score is not None
            else ("PARSE ERROR — REJECT" if reviewer_verdict == "REJECT" else "Chưa gọi")
        )
        pid = f" | prediction #{item.get('prediction_id')}" if item.get("prediction_id") else ""
        lines.append(
            f"\n{_auto_scan_format_dt(item.get('scanned_at'))}\n"
            f"{item.get('symbol')} {mode_label}\n"
            f"Kết quả: {_display_scan_stage(item.get('stage'), item.get('status'))}\n"
            f"Prefilter Flash: {pre}\n"
            f"Planner Pro: {planner_direction}\n"
            f"Flash reviewer: {reviewer_text}\n"
            f"Ghi chú: {_display_scan_reason(item.get('reason'))}{pid}"
        )
    # Still split the message safely, since a single log entry can contain a long note even though only 5 items are kept.
    log_text = "\n".join(lines)
    chunks = split_telegram_message(log_text, limit=3800)
    total_chunks = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        if index > 1:
            chunk = f"🧾 Auto Scan log (tiếp {index}/{total_chunks}):\n{chunk}"
        await message.reply_text(chunk)


async def job_auto_scan(context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import run_auto_scan_once

    try:
        payload = await run_auto_scan_once(bot=context.bot)
        if payload.get("skipped"):
            from analyze import AUTO_SCAN_DEBUG
            if AUTO_SCAN_DEBUG:
                print(f"[AUTO_SCAN] skipped: {payload.get('reason')} next={payload.get('next_scan_at')}", flush=True)
        else:
            print(
                "[AUTO_SCAN] Done: "
                f"users={payload.get('users', 0)}, symbols={payload.get('symbols', 0)}, "
                f"checked={payload.get('checked', 0)}, sent={payload.get('sent', 0)}, errors={payload.get('errors', 0)}, "
                f"next={payload.get('next_scan_at')}",
                flush=True,
            )
    except Exception as exc:
        print(f"[AUTO_SCAN] Job failed: {exc}", flush=True)


async def exportdb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin-only: create a consistent SQLite snapshot and send it via Telegram."""
    from auth import is_admin
    from evaluation_store import export_database_snapshot

    user = update.effective_user
    message = update.effective_message
    if not user or not message:
        return
    if not is_admin(user.id):
        await message.reply_text("Lệnh này chỉ dành cho admin.")
        return

    tmp_dir = Path(tempfile.gettempdir())
    export_path = tmp_dir / f"teopard_bot_export_{user.id}_{uuid.uuid4().hex}.db"
    try:
        await asyncio.to_thread(export_database_snapshot, str(export_path))
        size_mb = export_path.stat().st_size / (1024 * 1024)
        with export_path.open("rb") as fh:
            await message.reply_document(
                document=fh,
                filename="bot_export.db",
                caption=f"SQLite snapshot nhất quán — {size_mb:.2f} MB. Gửi kèm ZIP source đang deploy để audit.",
            )
    except Exception as exc:
        await message.reply_text(f"Không thể export database: {exc}")
    finally:
        try:
            export_path.unlink(missing_ok=True)
        except Exception:
            pass


# ─── Register ─────────────────────────────────────────────────────────────

def register_symbol_handlers(app: Application) -> None:
    init_symbol_db()

    app.add_handler(CommandHandler("addsymbol",    add_symbol))
    app.add_handler(CommandHandler("removesymbol", remove_symbol))
    app.add_handler(CommandHandler("listsymbols",  list_symbols))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("statsall", statsall_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("historyall", historyall_command))
    app.add_handler(CommandHandler("dashboard", dashboard_command))
    app.add_handler(CommandHandler("dashboardall", dashboardall_command))
    app.add_handler(CommandHandler("clearhistory", clearhistory_command))
    app.add_handler(CommandHandler("cleardrafts", cleardrafts_command))
    app.add_handler(CommandHandler("checknow", checknow_command))
    app.add_handler(CommandHandler("autoscanon", autoscanon_command))
    app.add_handler(CommandHandler("autoscanoff", autoscanoff_command))
    app.add_handler(CommandHandler("autoscanstatus", autoscanstatus_command))
    app.add_handler(CommandHandler("autoscanlog", autoscanlog_command))
    app.add_handler(CommandHandler("exportdb", exportdb_command))
    app.add_handler(CallbackQueryHandler(
        analyze_symbol_callback,
        pattern=f"^({ANALYZE_SHORT_CALLBACK_PREFIX}|{ANALYZE_LONG_CALLBACK_PREFIX}):",
    ))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, symbol_message_handler), group=1)
    app.add_handler(MessageHandler(filters.COMMAND, symbol_message_handler), group=2)

    # Background job: check pending predictions every 60 minutes
    if app.job_queue is None:
        print("JobQueue is not available. Install python-telegram-bot[job-queue].")
    else:
        app.job_queue.run_repeating(job_check_predictions, interval=3600, first=300)
        try:
            # This job only wakes up to check whether a candle-close slot is due; it doesn't call Binance/LLM if the slot was already scanned.
            from analyze import AUTO_SCAN_SCHEDULER_TICK_SECONDS
            app.job_queue.run_repeating(
                job_auto_scan,
                interval=AUTO_SCAN_SCHEDULER_TICK_SECONDS,
                first=10,
                job_kwargs={"misfire_grace_time": 60},
            )
        except Exception as exc:
            print(f"Auto Scan job was not started: {exc}", flush=True)


def symbol_control_commands() -> list[BotCommand]:
    """Minimal user menu; maintenance/rarely-used commands can still be typed manually."""
    return [
        BotCommand("listsymbols", "Danh sách coin hỗ trợ"),
        BotCommand("history", "5 lệnh gần nhất"),
        BotCommand("stats", "Thống kê kết quả"),
        BotCommand("autoscanon", "Bật Auto Scan"),
        BotCommand("autoscanoff", "Tắt Auto Scan"),
        BotCommand("autoscanstatus", "Trạng thái Auto Scan"),
        BotCommand("autoscanlog", "5 log Auto Scan gần nhất"),
    ]
