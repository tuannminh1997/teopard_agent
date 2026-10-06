import sqlite3
import os
import asyncio
import math
import time
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
ANALYZE_FUTURES_CALLBACK_PREFIX = "analyze_futures"
ANALYZE_SPOT_CALLBACK_PREFIX  = "analyze_spot"

# Blocks a user from firing a second manual analysis (double-tap) while one is still running.
_analyzing_users: set[int] = set()


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
        InlineKeyboardButton("Futures (4H/1H/15m)", callback_data=f"{ANALYZE_FUTURES_CALLBACK_PREFIX}:{symbol}"),
        InlineKeyboardButton("Spot (1W/1D/4H)",  callback_data=f"{ANALYZE_SPOT_CALLBACK_PREFIX}:{symbol}"),
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
    from analyze import resolve_binance_symbol, get_current_price_raw

    admin = update.effective_user
    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return
    if not context.args:
        await update.effective_message.reply_text("Cú pháp đúng: /addsymbol BTC")
        return
    symbol = normalize_symbol(context.args[0])
    futures_symbol = resolve_binance_symbol(symbol, "futures")
    spot_symbol = resolve_binance_symbol(symbol, "spot")
    futures_price, spot_price = await asyncio.gather(
        asyncio.to_thread(get_current_price_raw, futures_symbol, "futures"),
        asyncio.to_thread(get_current_price_raw, spot_symbol, "spot"),
    )
    if futures_price is None and spot_price is None:
        await update.effective_message.reply_text(
            f"Không thêm được {symbol}: không tìm thấy cặp USDT tương ứng trên Binance Futures hoặc Spot."
        )
        return
    await asyncio.to_thread(add_allowed_symbol, symbol)
    markets = ", ".join(name for name, price in (("Futures", futures_price), ("Spot", spot_price)) if price is not None)
    await update.effective_message.reply_text(f"Đã thêm symbol {symbol}. Thị trường khả dụng: {markets}.")


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
    await asyncio.to_thread(remove_allowed_symbol, symbol)
    await update.effective_message.reply_text(f"Đã xóa symbol {symbol}.")


async def list_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbols = await asyncio.to_thread(get_allowed_symbols)
    if not symbols:
        await update.effective_message.reply_text("Danh sách symbol hiện đang trống.")
        return
    await update.effective_message.reply_text(
        "Danh sách symbol được phép:\n" + "\n".join(f"• {s}" for s in symbols)
    )


# ─── Message handler: user types a coin symbol ──────────────────────────────

async def handle_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    from auth import is_account_activated, show_start_menu
    from analyze import BINANCE_QUOTE_ASSET

    user = update.effective_user
    message = update.effective_message
    if not user or not message or not message.text:
        return False

    symbol = normalize_symbol(message.text)
    if not await asyncio.to_thread(is_allowed_symbol, symbol):
        return False

    # Tin non-command đã được bot xử lý → xóa; tin bắt đầu bằng "/" thì handler lệnh giữ nguyên.
    try:
        await message.delete()
    except Exception:
        pass

    if not await asyncio.to_thread(is_account_activated, user.id):
        await show_start_menu(update)
        return True

    await message.reply_text(
        f"✅ Đã nhận {symbol}/{BINANCE_QUOTE_ASSET}\n"
        f"Bạn muốn phân tích {symbol}/{BINANCE_QUOTE_ASSET} theo kiểu nào?",
        reply_markup=symbol_analysis_keyboard(symbol),
    )
    return True


async def symbol_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    handled = await handle_symbol(update, context)
    if handled:
        raise ApplicationHandlerStop


# ─── Callback: user chooses Futures/Spot ─────────────────────────────────────

async def analyze_symbol_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import analyze_symbol, BINANCE_QUOTE_ASSET
    from auth import decrement_user_usage, get_user_usage, increment_user_usage, is_account_activated, show_start_menu

    query = update.callback_query
    user  = update.effective_user
    if not query or not user or not query.data:
        return

    await query.answer()

    if not await asyncio.to_thread(is_account_activated, user.id):
        await show_start_menu(update)
        return

    if user.id in _analyzing_users:
        await query.message.reply_text(
            "Bạn đang có 1 phân tích khác đang chạy, vui lòng chờ kết quả trước khi bấm tiếp."
        )
        return

    action, symbol = query.data.split(":", 1)
    mode = "futures" if action == ANALYZE_FUTURES_CALLBACK_PREFIX else "spot"
    mode_label = "Futures (4H/1H/15m)" if mode == "futures" else "Spot (1W/1D/4H)"

    daily_limit, used_today = await asyncio.to_thread(get_user_usage, user.id)
    remaining = daily_limit - used_today

    if remaining <= 0:
        await query.message.reply_text(
            f"Bạn đã hết {daily_limit} lượt hôm nay. "
            "Vui lòng chờ sang ngày mới hoặc liên hệ admin."
        )
        return

    _analyzing_users.add(user.id)
    status_target = None  # message đang giữ thông báo xác nhận ✅ Đã chọn ...

    async def keep_confirmation() -> None:
        """Cắt bỏ phần 'Đang phân tích — vui lòng chờ', chỉ giữ dòng xác nhận đã chọn mode.
        Output sẽ gửi thành tin mới nên user luôn nhớ mình đang chạy phân tích gì."""
        if status_target is None:
            return
        try:
            await status_target.edit_text(
                text=f"✅ Đã chọn {mode_label} cho {symbol}/{BINANCE_QUOTE_ASSET}.",
                reply_markup=None,
            )
        except Exception:
            pass

    async def update_analysis_message(text: str) -> None:
        # Không đè lên thông báo xác nhận — giữ dòng ✅, gửi nội dung (lỗi) thành tin mới.
        await keep_confirmation()
        await query.message.reply_text(text)

    try:
        # Replace the chooser itself so the old question is not left above a second status message.
        status_text = (
            f"✅ Đã chọn {mode_label} cho {symbol}/{BINANCE_QUOTE_ASSET}.\n"
            f"Đang phân tích bằng dữ liệu {mode_label} — vui lòng chờ... "
            f"(còn {remaining - 1} lượt hôm nay)"
        )
        try:
            await query.edit_message_text(text=status_text, reply_markup=None)
            status_target = query.message
        except Exception:
            # If Telegram won't let us edit the chooser, at least remove its buttons and send
            # a clear status message so a tap is not mistaken for the final plan.
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass
            status_target = await query.message.reply_text(status_text)

        await asyncio.to_thread(increment_user_usage, user.id)
        remaining -= 1

        try:
            result_payload = await analyze_symbol(symbol, mode, user_id=user.id, chat_id=query.message.chat_id)
        except Exception as exc:
            await asyncio.to_thread(decrement_user_usage, user.id)
            error_text = str(exc)
            print(
                f"[MANUAL_ERROR] symbol={symbol} mode={mode} user_id={user.id} "
                f"error_type={type(exc).__name__} error={error_text}",
                flush=True,
            )
            traceback.print_exc()
            error_lower = error_text.lower()
            if "timed out" in error_lower or "timeout" in error_lower:
                await update_analysis_message(
                    "Phân tích thất bại: AI cuối không trả lời kịp sau lần thử chính và một lần retry. "
                    "Lượt sử dụng không bị trừ; bạn có thể chạy lại sau ít phút."
                )
            elif "could not fetch binance data" in error_lower:
                if mode == "futures":
                    fetch_error = (
                        f"Không lấy được dữ liệu Futures cho {symbol}/{BINANCE_QUOTE_ASSET} — có thể coin này chưa có hợp đồng "
                        "perpetual, tên hợp đồng khác tên Spot (một số token bị rebase), hoặc lỗi mạng tạm thời. "
                        "Lượt sử dụng không bị trừ; vui lòng thử lại sau hoặc báo admin nếu lặp lại."
                    )
                else:
                    fetch_error = (
                        f"Không lấy được dữ liệu Spot cho {symbol}/{BINANCE_QUOTE_ASSET} — có thể cặp Spot chưa được niêm yết "
                        "hoặc lỗi mạng tạm thời. Lượt sử dụng không bị trừ; vui lòng thử lại sau hoặc báo admin nếu lặp lại."
                    )
                await update_analysis_message(fetch_error)
            elif "thiếu dữ liệu binance cho khung quan trọng" in error_lower:
                await update_analysis_message(
                    f"Không đủ dữ liệu Binance {mode_label} cho {symbol}/{BINANCE_QUOTE_ASSET} ở các khung quyết định hướng/Entry/SL/TP — "
                    "có thể lỗi mạng tạm thời khi lấy nến. Lượt sử dụng không bị trừ; vui lòng thử lại sau."
                )
            else:
                await update_analysis_message(f"Phân tích thất bại: {error_text}")
            return
    finally:
        _analyzing_users.discard(user.id)

    if isinstance(result_payload, dict):
        result_text = result_payload.get("text", "")
    else:
        result_text = str(result_payload)

    chunks = split_telegram_message(result_text)
    try:
        # Giữ nguyên thông báo xác nhận (đã chọn mode nào), output gửi thành tin mới bên dưới.
        await keep_confirmation()
        for chunk in chunks:
            await query.message.reply_text(chunk)
    except Exception as exc:
        # The result was computed and usage already charged, but the user never actually saw it
        # (network blip, user blocked the bot, etc.) — refund so a hung-looking interaction doesn't
        # also cost a quota slot. The plan itself is still saved in /history if it was trackable.
        await asyncio.to_thread(decrement_user_usage, user.id)
        print(
            f"[MANUAL_SEND_FAILED] symbol={symbol} mode={mode} user_id={user.id} error={exc}",
            flush=True,
        )
        traceback.print_exc()



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
    # Users who want to see results should use /history.


def command_scope_user_id(update: Update) -> int | None:
    user = update.effective_user
    if not user:
        return None
    # Everyone, including admin, sees their own data only.
    return user.id


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_history
    text = await asyncio.to_thread(format_history, user_id=command_scope_user_id(update))
    await update.effective_message.reply_text(text)


async def clearhistory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from auth import is_admin
    from analyze import clear_prediction_history

    admin = update.effective_user
    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return

    if not context.args or context.args[0].upper() != "CONFIRM":
        await update.effective_message.reply_text(
            "Lệnh này sẽ xóa toàn bộ lịch sử/theo dõi (predictions, evaluation data, log và tín hiệu Auto Scan) "
            "nhưng vẫn giữ whitelist, danh sách symbol, và trạng thái bật/tắt + symbol đang chọn của Auto Scan.\n"
            "Gõ: /clearhistory CONFIRM"
        )
        return

    payload = await asyncio.to_thread(clear_prediction_history)
    if isinstance(payload, dict):
        await update.effective_message.reply_text(
            "Đã xóa lịch sử theo dõi. Whitelist, danh sách symbol và cấu hình Auto Scan vẫn được giữ.\n"
            f"Lệnh đã trade/đang theo dõi: {payload.get('visible_count', 0)}\n"
            f"Tổng dòng predictions cũ đã xóa: {payload.get('total_prediction_count', 0)}\n"
            f"Tổng dòng evaluation data cũ đã xóa: {payload.get('evaluation_count', 0)}"
        )
    else:
        await update.effective_message.reply_text(
            f"Đã xóa {payload} prediction khỏi lịch sử. Whitelist và danh sách symbol vẫn được giữ."
        )


# ─── Auto Scan phiên theo market: bật/tắt/log riêng cho futures & spot ───────
# Pending state cho luồng nhập liệu: user_id -> {stage, market, symbol, api_key, qty, ts}
_AUTO_PENDING: dict[int, dict] = {}
# Nhập key/qty dở dang mà bỏ đó rồi thì tin nhắn SAU (giờ, ngày…) cũng bị coi là API key
# và lưu đè key đang dùng → mọi lệnh đặt lệnh abort. State hết hạn sau 10 phút.
_PENDING_TTL_SECONDS = 600


def _new_pending(user_id: int, **state) -> dict:
    payload = dict(state)
    payload["ts"] = time.time()
    _AUTO_PENDING[user_id] = payload
    return payload


def _normalize_decimal(raw: str) -> str:
    """Chuẩn hóa số user gõ về dạng Python đọc được.

    Quy tắc: dấu xuất hiện CUỐI CÙNG là dấu thập phân.
    - "0,97"      → 0.97   (chỉ có phẩy = thập phân, kiểu VN)
    - "1.000,50"  → 1000.50 (chấm nghìn, phẩy thập phân, kiểu châu Âu)
    - "1,000.50"  → 1000.50 (phẩy nghìn, chấm thập phân, kiểu US)
    Thay vì replace(",","")blind từng làm "0,97" → 97 (sai ~100 lần).
    """
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            return raw.replace(".", "").replace(",", ".")
        return raw.replace(",", "")
    if "," in raw:
        return raw.replace(",", ".")
    return raw


def _parse_qty(text: str) -> float | None:
    """Đọc khối lượng user gõ. Dấu phẩy là DẤU THẬP PHÂN (0,97 = 0.97) — trước đây replace(",")
    thẳng làm "0,97" thành 97 (sai ~100 lần)."""
    raw = _normalize_decimal(text.strip())
    if not raw:
        return None
    try:
        qty = float(raw)
    except Exception:
        return None
    if not math.isfinite(qty) or qty <= 0:
        return None
    return qty


def _parse_leverage(text: str) -> int | None:
    try:
        lev = int(float(text.strip()))
        return lev if 1 <= lev <= 125 else None
    except Exception:
        return None


def _fmt_qty(qty: float) -> str:
    """Lưu qty dạng số thuần ('0.97') — chuỗi gốc user gõ (có thể '0,97') mà bị replace(',')
    sẽ thành '097' và các lần đặt lệnh sau đọc ra 97."""
    return repr(float(qty))


async def _enable_session(
    update: Update, market: str, symbol: str, qty: str = "", leverage: int = 0,
    automation: bool = False,
) -> None:
    from analyze import (
        set_auto_scan_market_enabled, AUTOSCAN_INTERVAL_SECONDS,
        AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY,
    )

    message = update.effective_message
    user = update.effective_user
    if not user or not message:
        return
    result = await asyncio.to_thread(
        set_auto_scan_market_enabled, user.id, message.chat_id, market, True, symbol, qty, leverage,
    )
    if result.get("quota_blocked"):
        await message.reply_text(
            f"Auto Scan đã dùng đủ {AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY} lượt gọi AI cuối trong ngày. "
            "Bot sẽ tự bật lại và reset quota lúc 07:00 sáng mai theo giờ Việt Nam."
        )
        return
    label = "FUTURES" if market == "futures" else "SPOT"
    lines = [
        f"Đã bật Auto Scan {label} cho {symbol}.",
        f"Chu kỳ quét: mỗi {int(AUTOSCAN_INTERVAL_SECONDS // 60)} phút, theo nến đóng.",
        "Planner tự quyết LONG, SHORT (SPOT: BUY) hoặc NO TRADE; NO TRADE thì không gửi.",
    ]
    if automation:
        lev_note = f" | đòn bẩy x{leverage}" if market == "futures" else ""
        lines.append(f"🤖 Đặt lệnh tự động: BẬT — khối lượng {qty}{lev_note} (API key đã mã hóa trong DB).")
        lines.append("Có tín hiệu LONG/SHORT/BUY là bot đặt lệnh + gửi thông báo kèm ID lệnh.")
    else:
        lines.append("🤖 Đặt lệnh tự động: TẮT — chỉ gửi tín hiệu.")
    lines.append(f"Giới hạn {AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY} lần gọi Planner/ngày; nghỉ 00:00-07:00 VN.")
    lines.append(f"Xem lệnh phiên: /autoscanlog{market[:4] if market == 'futures' else 'spot'}")
    await message.reply_text("\n".join(lines))


async def _autoscan_on_command(update: Update, context: ContextTypes.DEFAULT_TYPE, market: str) -> None:
    from auth import is_account_activated
    from analyze import normalize_auto_scan_symbol, BINANCE_QUOTE_ASSET

    user = update.effective_user
    message = update.effective_message
    if not user or not message:
        return
    if not await asyncio.to_thread(is_account_activated, user.id):
        from auth import show_start_menu
        await show_start_menu(update)
        return

    if not context.args:
        label = "/onfutu" if market == "futures" else "/onspot"
        await message.reply_text(
            f"Cú pháp: {label} ETH\n"
            f"Ví dụ: {label} eth\n"
            "Mỗi tài khoản 1 symbol cho mỗi market (futures và spot chạy độc lập)."
        )
        return

    seen: set = set()
    symbols = []
    for raw in context.args:
        for part in str(raw).replace(",", " ").split():
            sym = normalize_auto_scan_symbol(part)
            if sym and sym not in seen:
                symbols.append(sym)
                seen.add(sym)
    if not symbols:
        await message.reply_text("Không đọc được symbol. Ví dụ: /onfutu eth")
        return
    if len(symbols) > 1:
        await message.reply_text(
            "Mỗi phiên chỉ quét 1 symbol. Ví dụ: /onfutu eth\n"
            "Muốn đổi symbol thì gõ lại lệnh với symbol mới."
        )
        return

    symbol = symbols[0]
    base = symbol[:-len(BINANCE_QUOTE_ASSET)] if symbol.endswith(BINANCE_QUOTE_ASSET) else symbol
    if not await asyncio.to_thread(is_allowed_symbol, base) and not await asyncio.to_thread(is_allowed_symbol, symbol):
        await message.reply_text(
            f"Symbol {base} chưa có trong danh sách được phép. Admin cần thêm bằng /addsymbol {base} trước."
        )
        return

    label = "FUTURES" if market == "futures" else "SPOT"
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(
            "Có, cần thêm API key",
            callback_data=f"asauto:yes:{market}:{symbol}",
        ),
        InlineKeyboardButton("Không", callback_data=f"asauto:no:{market}:{symbol}"),
    ]])
    await message.reply_text(
        f"Bạn có muốn tự động hóa việc đặt lệnh không?\n"
        f"(phiên {label} cho {symbol})",
        reply_markup=keyboard,
    )


async def onfutu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _autoscan_on_command(update, context, "futures")


async def onspot_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _autoscan_on_command(update, context, "spot")


async def autoscan_auto_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """asauto:{yes|no}:{market}:{symbol} — trả lời câu hỏi tự động hóa đặt lệnh."""
    from key_store import has_api_keys

    query = update.callback_query
    user = update.effective_user
    if not query or not user or not query.data:
        return
    await query.answer()
    try:
        _, answer, market, symbol = query.data.split(":", 3)
    except ValueError:
        return

    if answer == "no":
        await _enable_session(update, market, symbol, automation=False)
        return

    # answer == "yes"
    has_key = await asyncio.to_thread(has_api_keys, user.id, market)
    if has_key:
        _new_pending(user.id, stage="qty", market=market, symbol=symbol)
        base = symbol[:-4] if symbol.endswith("USDT") else symbol
        await query.message.reply_text(
            f"Đã có API key {market} trong DB.\n"
            f"Bước 3 - Nhập số lượng {base} cần đặt mỗi lệnh (ví dụ 0.97):"
        )
    else:
        _new_pending(user.id, stage="api_key", market=market, symbol=symbol)
        await query.message.reply_text(
            f"Nhập cho tôi lần lượt nhé (market: {market.upper()} — key futures và key spot là KHÁC nhau).\n\n"
            "Bước 1 - GỬI API KEY\n"
            "Chuỗi ký tự dài hiển thị đầu tiên trong trang API Management của Binance. "
            "Chưa tới bước Secret Key, đừng gửi nhầm Secret Key vào đây."
        )


async def apikey_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """apikey:{add|change}:{market} — nút Thêm / Đổi-Gỡ API key từ /autoscanstatus."""
    query = update.callback_query
    user = update.effective_user
    if not query or not user or not query.data:
        return
    await query.answer()
    try:
        _, action, market = query.data.split(":", 2)
    except ValueError:
        return

    if action == "add":
        _new_pending(user.id, stage="api_key", market=market, symbol="", intent="keyonly")
        await query.message.reply_text(
            f"Thêm API key {market.upper()} — nhập cho tôi lần lượt nhé.\n\n"
            "Bước 1 - GỬI API KEY\n"
            "Chuỗi ký tự dài hiển thị đầu tiên trong trang API Management của Binance."
        )
    else:
        _new_pending(user.id, stage="rekey", market=market, symbol="", intent="keyonly")
        await query.message.reply_text(
            f"Đổi/Gỡ API key {market.upper()}.\n\n"
            "Bước 1 - GỬI API KEY MỚI\n"
            "Để Gỡ Key: gửi tin trống (hoặc gõ \"xóa\")."
        )


async def autoscan_pending_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Bắt tin nhắn text trong lúc chờ nhập key/secret/số lượng/đòn bẩy (handler group 0)."""
    from key_store import KeyError_, delete_api_keys, save_api_keys

    user = update.effective_user
    message = update.effective_message
    if not user or not message or not message.text:
        return
    state = _AUTO_PENDING.get(user.id)
    if not state:
        return
    if time.time() - float(state.get("ts") or 0) > _PENDING_TTL_SECONDS:
        # Quá hạn nhập liệu → quên state để tin nhắn thường của user không bị coi là API key.
        _AUTO_PENDING.pop(user.id, None)
        await message.reply_text(
            "⏳ Phiên nhập liệu đã hết hạn (10 phút không có phản hồi) — đã hủy.\n"
            "Nếu đang nhập API key/secret thì gửi lại từ đầu nhé."
        )
        return
    text = message.text.strip()
    stage = state["stage"]
    market = state["market"]
    symbol = state["symbol"]
    # Người dùng đang tích cực trả lời → gia hạn TTL cho cả luồng nhiều bước.
    state["ts"] = time.time()

    # Quy tắc chung: mọi tin non-command trong luồng nhập liệu này xử lý xong là xóa;
    # chỉ tin bắt đầu bằng "/" (lệnh) mới được giữ nguyên.
    delete_failed = False
    try:
        deleted = await message.delete()
        delete_failed = deleted is False
    except Exception:
        delete_failed = True
    if delete_failed and stage in ("api_key", "rekey", "rekey_secret"):
        # Bot KHÔNG xóa được tin của user trong private chat → secret nằm nguyên trong lịch sử.
        await message.reply_text(
            "⚠️ Bot không tự xóa được tin nhắn của bạn — hãy tự xóa/ thu hồi 2 tin vừa gửi "
            "(API key + Secret) trong vòng 48 giờ."
        )

    if stage == "rekey":
        if not text or text.lower() in {"xóa", "xoa", "xoa key", "delete"}:
            from analyze import (
                cancel_pending_plan_orders_for, delete_session_signals,
                get_auto_scan_market_settings, set_auto_scan_market_enabled,
            )

            # Hủy lệnh TREO TRƯỚC, trong khi key CÒN. Nếu xóa key trước thì bot không còn
            # cách nào ký lệnh hủy → lệnh treo thành mồ côi trên sàn cho tới khi tự tay hủy.
            cancel_res = await asyncio.to_thread(cancel_pending_plan_orders_for, user.id, market)
            n_cancel = len(cancel_res.get("cancelled") or [])
            cancel_failed = bool(cancel_res.get("no_keys"))

            removed = await asyncio.to_thread(delete_api_keys, user.id, market)
            _AUTO_PENDING.pop(user.id, None)
            if not removed:
                await message.reply_text(f"Không có API key {market.upper()} nào đang lưu.")
                return
            # Gỡ key mà phiên đang bật TỰ ĐỘNG ĐẶT LỆNH (có qty) → không còn key để đặt lệnh,
            # tắt luôn phiên. Dòng ledger đã đặt lệnh mà chưa hủy được thì PHẢI giữ lại.
            cfg = await asyncio.to_thread(get_auto_scan_market_settings, user.id, market)
            automation_on = bool(
                cfg and cfg.get("enabled") and str(cfg.get("qty") or "").strip()
            )
            if automation_on:
                await asyncio.to_thread(
                    set_auto_scan_market_enabled, user.id, message.chat_id, market, False, cfg.get("symbol") or "",
                )
                deleted = await asyncio.to_thread(
                    delete_session_signals, user.id, market, cancel_failed,
                )
                label = "FUTURES" if market == "futures" else "SPOT"
                cancel_note = (
                    f"✅ Đã hủy {n_cancel} lệnh TREO trước khi gỡ key.\n"
                    if n_cancel and not cancel_failed else
                    "⚠️ Không hủy được lệnh treo — bạn vào GUI hủy tay giúp "
                    "(lệnh đã khớp thì để nguyên).\n"
                    if cancel_failed else
                    "Không có lệnh treo nào cần hủy.\n"
                )
                await message.reply_text(
                    f"✅ Đã gỡ API key {market.upper()}.\n"
                    f"Phiên {label} đang bật tự động đặt lệnh nên không còn key — "
                    f"đã tắt Auto Scan và xóa {deleted} lệnh trong phiên.\n"
                    + cancel_note
                    + ("Dòng lệnh chưa hủy được được giữ lại trong /autoscanlog để bạn đối chiếu.\n"
                       if cancel_failed else "")
                )
            else:
                cancel_note = (
                    f"✅ Đã hủy {n_cancel} lệnh TREO trước khi gỡ key.\n" if n_cancel else ""
                )
                await message.reply_text(f"✅ Đã gỡ API key {market.upper()}.\n" + cancel_note)
            return
        if len(text) < 20:
            await message.reply_text("API key quá ngắn — gửi lại, hoặc gửi tin trống/gõ \"xóa\" để gỡ key.")
            return
        state["api_key"] = text
        state["stage"] = "rekey_secret"
        await message.reply_text(
            "✅ Đã nhận API KEY MỚI\n"
            "Bước 2 - GỬI SECRET KEY MỚI\n"
            "Chuỗi chỉ hiển thị 1 lần lúc tạo key trên Binance (mất thì phải tạo lại key mới)."
        )
        return

    if stage == "rekey_secret":
        if len(text) < 20:
            await message.reply_text("Secret quá ngắn — kiểm tra lại và gửi lại.")
            return
        try:
            await asyncio.to_thread(save_api_keys, user.id, market, state["api_key"], text)
        except KeyError_ as exc:
            _AUTO_PENDING.pop(user.id, None)
            await message.reply_text(f"❌ Không lưu được API key: {exc}")
            return
        _AUTO_PENDING.pop(user.id, None)
        await message.reply_text(
            f"✅ Đã thay API key {market.upper()} bằng KEY mới."
        )
        return

    if stage == "api_key":
        if len(text) < 20:
            await message.reply_text("API key quá ngắn — kiểm tra lại và gửi lại.")
            return
        state["api_key"] = text
        state["stage"] = "secret"
        await message.reply_text(
            "✅ Đã nhận API KEY\n"
            "Bước 2 - GỬI SECRET KEY\n"
            "Chuỗi chỉ hiển thị 1 lần lúc tạo key trên Binance (mất thì phải tạo lại key mới)."
        )
    elif stage == "secret":
        if len(text) < 20:
            await message.reply_text("Secret quá ngắn — kiểm tra lại và gửi lại.")
            return
        try:
            await asyncio.to_thread(save_api_keys, user.id, market, state["api_key"], text)
        except KeyError_ as exc:
            _AUTO_PENDING.pop(user.id, None)
            await message.reply_text(f"❌ Không lưu được API key: {exc}")
            return
        if state.get("intent") == "keyonly":
            # Nhập key từ /autoscanstatus (Thêm key) — không hỏi qty/đòn bẩy, phiên đã cấu hình sẵn.
            _AUTO_PENDING.pop(user.id, None)
            await message.reply_text(
                "✅ Đã lưu KEY và SECRET KEY."
            )
            return
        state["stage"] = "qty"
        base = symbol[:-4] if symbol.endswith("USDT") else symbol
        await message.reply_text(
            "✅ Đã lưu KEY và SECRET KEY.\n\n"
            f"Bước 3 - Nhập số lượng {base} cần đặt mỗi lệnh (ví dụ 0.97):"
        )
    elif stage == "qty":
        qty = _parse_qty(text)
        if qty is None:
            await message.reply_text("Số lượng không hợp lệ. Nhập dạng số > 0, ví dụ 0.008")
            return
        state["qty"] = _fmt_qty(qty)
        if market == "spot":
            # Spot không dùng đòn bẩy — bỏ qua bước leverage.
            _AUTO_PENDING.pop(user.id, None)
            await message.reply_text(f"✅ Đã lưu khối lượng: {state['qty']}")
            await _enable_session(update, market, symbol, qty=state["qty"], leverage=1, automation=True)
            return
        state["stage"] = "leverage"
        await message.reply_text(
            f"✅ Đã lưu khối lượng: {state['qty']}\n\n"
            "Bước 4 - Nhập đòn bẩy (ví dụ 20, từ 1 đến 125):"
        )
    elif stage == "leverage":
        leverage = _parse_leverage(text)
        if leverage is None:
            await message.reply_text("Đòn bẩy không hợp lệ. Nhập số nguyên từ 1 đến 125, ví dụ 20")
            return
        _AUTO_PENDING.pop(user.id, None)
        await message.reply_text(f"✅ Đã lưu đòn bẩy: {leverage}")
        await _enable_session(update, market, symbol, qty=state.get("qty", ""), leverage=leverage, automation=True)

    # Nuốt tin nhắn này khỏi các handler khác (symbol/fallback).
    raise ApplicationHandlerStop

async def _autoscan_off_command(update: Update, context: ContextTypes.DEFAULT_TYPE, market: str) -> None:
    from analyze import (cancel_pending_plan_orders_for, delete_session_signals,
                         normalize_auto_scan_symbol, set_auto_scan_market_enabled)

    user = update.effective_user
    message = update.effective_message
    if not user or not message:
        return
    label = "FUTURES" if market == "futures" else "SPOT"
    if not context.args:
        cmd = "/offfutu" if market == "futures" else "/offspot"
        await message.reply_text(f"Cú pháp: {cmd} eth\n(Ví dụ: {cmd} eth)")
        return
    symbol = normalize_auto_scan_symbol(context.args[0])
    # Tắt phiên → hủy MỌI lệnh TREO chưa khớp (lệnh đã khớp giữ nguyên) TRƯỚC khi xóa ledger.
    res = await asyncio.to_thread(cancel_pending_plan_orders_for, user.id, market)
    n_cancel = len(res.get("cancelled") or [])
    no_keys = res.get("no_keys") or []
    await asyncio.to_thread(
        set_auto_scan_market_enabled, user.id, message.chat_id, market, False, symbol,
    )
    # Thiếu key → KHÔNG xóa các dòng đã đặt lệnh: đó là bản ghi duy nhất về orderId trên sàn.
    deleted = await asyncio.to_thread(delete_session_signals, user.id, market, bool(no_keys))
    if no_keys:
        pending_note = (
            f"⚠️ KHÔNG hủy được lệnh treo vì thiếu API key ({market}) — "
            "lệnh vẫn còn trên sàn.\n"
            "Bạn tự hủy tay trên GUI, hoặc thêm lại API key rồi gõ lại lệnh tắt.\n"
        )
    elif n_cancel:
        pending_note = f"✅ Đã hủy {n_cancel} lệnh TREO chưa khớp (lệnh đã khớp giữ nguyên).\n"
    else:
        pending_note = "Không có lệnh treo nào cần hủy (lệnh đã khớp giữ nguyên).\n"
    await message.reply_text(
        f"✅ Đã tắt Auto Scan {label} cho {symbol}.\n"
        f"✅ Đã xóa {deleted} lệnh trong phiên (log phiên sẽ trống).\n"
        + pending_note
    )


async def offfutu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _autoscan_off_command(update, context, "futures")


async def offspot_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _autoscan_off_command(update, context, "spot")




async def autoscanstatus_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import (
        get_auto_scan_runtime_status,
        AUTOSCAN_INTERVAL_SECONDS,
        AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY,
        _auto_scan_format_dt,
    )
    from key_store import api_key_status

    user = update.effective_user
    message = update.effective_message
    if not user or not message:
        return
    status = await asyncio.to_thread(get_auto_scan_runtime_status, user.id)
    markets = status.get("markets") or []

    # API key theo từng market (futures và spot là 2 key riêng) + nút quản lý key.
    api_lines: list[str] = []
    auto_lines: list[str] = []
    buttons = []
    multi = len(markets) > 1
    for m in markets:
        key_state = await asyncio.to_thread(api_key_status, user.id, m["market"])
        has_key = key_state != "missing"
        tag = f" {m['market'].upper()}" if multi else ""
        if key_state == "ok":
            api_lines.append(f"API key{tag}: Đã Thêm")
        elif key_state == "broken":
            # Dòng key tồn tại nhưng không giải mã được → mọi lần đặt lệnh sẽ abort.
            api_lines.append(
                f"API key{tag}: ⚠️ KHÔNG đọc được (đã đổi DATA_ENCRYPTION_KEY?) — gửi lại key mới"
            )
        else:
            api_lines.append(f"API key{tag}: Chưa thêm")
        label = ("Đổi/Gỡ API Key" if has_key else "Thêm API Key") + tag
        buttons.append(
            InlineKeyboardButton(label, callback_data=f"apikey:{'change' if has_key else 'add'}:{m['market']}")
        )
        # Trạng thái TỰ ĐỘNG ĐẶT LỆNH tách riêng khỏi API key: có key nhưng chọn "Không"
        # thì vẫn là TẮT (chỉ nhận tín hiệu). Nguồn sự thật = qty trong cấu hình phiên.
        qty = str(m.get("qty") or "").strip()
        if qty:
            lev = f" | đòn bẩy x{m['leverage']}" if m["market"] == "futures" and m.get("leverage") else ""
            auto_lines.append(f"Tự động đặt lệnh{tag}: BẬT — khối lượng {qty}{lev}")
        else:
            auto_lines.append(f"Tự động đặt lệnh{tag}: TẮT (chỉ nhận tín hiệu)")

    if not markets:
        market_lines = ["(chưa bật phiên nào — dùng /onfutu hoặc /onspot)"]
    else:
        market_lines = []
        for m in markets:
            label = "FUTURES" if m["market"] == "futures" else "SPOT"
            if status.get("quota_resume"):
                state = "⏸ ĐỦ QUOTA — tự bật lại 07:00"
            elif status.get("in_sleep_window") and m["night_resume"]:
                state = "🌙 NGHỈ ĐÊM — tự bật lại 07:00"
            else:
                state = "🟢 ĐANG BẬT" if m["enabled"] else "🔴 ĐANG TẮT"
            market_lines.append(f"{label} {m.get('symbol') or 'chưa chọn'}: {state}")

    lines = ["Auto Scan status:"]
    lines.extend(api_lines)
    lines.extend(auto_lines)
    lines.extend(market_lines)
    lines += [
        "Giờ hoạt động tự động: 07:00-24:00 theo giờ Việt Nam",
        f"Chu kỳ nến: {int(AUTOSCAN_INTERVAL_SECONDS // 60)} phút, quét theo nến đóng",
        "Cơ chế: Planner tự quyết LONG, SHORT hoặc NO TRADE",
        f"Quota gọi Planner hôm nay: {status.get('glm_calls_today', 0)}/{AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY} "
        f"(còn {status.get('glm_calls_remaining', AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY)} lượt)",
        f"Lần quét gần nhất: {_auto_scan_format_dt(status.get('last_scan_at'))}",
        f"Lần quét kế tiếp: {_auto_scan_format_dt(status.get('next_scan_at'))}",
    ]
    await message.reply_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup([buttons]) if buttons else None,
    )


def _order_status_text(row: dict) -> str:
    status = str(row.get("order_status") or "pending")
    ids = []
    if row.get("entry_order_id"):
        ids.append(f"entry {row['entry_order_id']}")
    if row.get("tp_algo_id"):
        ids.append(f"tp {row['tp_algo_id']}")
    if row.get("sl_algo_id"):
        ids.append(f"sl {row['sl_algo_id']}")
    id_note = f" ({', '.join(ids)})" if ids else ""
    if status == "placed":
        return f"đã đặt ✓{id_note}"
    if status == "entry_failed":
        return f"lỗi đặt lệnh{id_note}"
    if status == "no_auto":
        return "chưa bật tự động"
    return "đang chờ"


async def _autoscan_log_command(update: Update, context: ContextTypes.DEFAULT_TYPE, market: str) -> None:
    from analyze import list_session_signals, _auto_scan_format_dt, fmt

    user = update.effective_user
    message = update.effective_message
    if not user or not message:
        return
    rows = await asyncio.to_thread(list_session_signals, user.id, market)
    label = "FUTURES" if market == "futures" else "SPOT"
    if not rows:
        cmd = "/onfutu" if market == "futures" else "/onspot"
        await message.reply_text(
            f"Phiên {label} chưa có lệnh nào. Lệnh chỉ xuất hiện khi Planner trả LONG/SHORT/BUY "
            f"(NO TRADE không lưu). Bật phiên: {cmd} <coin>"
        )
        return
    lines = [f"🧾 Lệnh trong phiên {label} ({len(rows)} lệnh):"]
    for idx, row in enumerate(rows, 1):
        qty = f" | qty {row.get('qty')}" if row.get("qty") else ""
        lev = f" x{row.get('leverage')}" if market == "futures" and row.get("leverage") else ""
        entry = row.get("entry_low")
        entry_high = row.get("entry_high")
        entry_text = f"{fmt(entry)}–{fmt(entry_high)}" if entry is not None else "n/a"
        lines.append(
            f"\n{idx}. {row.get('plan_id') or '(chưa có id)'} | "
            f"{_auto_scan_format_dt(row.get('sent_at'))} | {row.get('direction')}{qty}{lev}\n"
            f"   Entry {entry_text} | SL {fmt(row.get('sl'))} | TP {fmt(row.get('tp1'))}\n"
            f"   Lệnh: {_order_status_text(row)}"
        )
    log_text = "\n".join(lines)
    chunks = split_telegram_message(log_text, limit=3800)
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        if index > 1:
            chunk = f"🧾 Lệnh phiên {label} (tiếp {index}/{total}):\n{chunk}"
        await message.reply_text(chunk)


async def autoscanlogfutu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _autoscan_log_command(update, context, "futures")


async def autoscanlogspot_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _autoscan_log_command(update, context, "spot")


async def job_auto_scan(context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import run_auto_scan_once

    try:
        payload = await run_auto_scan_once(bot=context.bot)
        if payload.get("skipped"):
            from analyze import AUTOSCAN_DEBUG
            if AUTOSCAN_DEBUG:
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
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("clearhistory", clearhistory_command))
    app.add_handler(CommandHandler("onfutu", onfutu_command))
    app.add_handler(CommandHandler("onspot", onspot_command))
    app.add_handler(CommandHandler("offfutu", offfutu_command))
    app.add_handler(CommandHandler("offspot", offspot_command))
    app.add_handler(CommandHandler("autoscanstatus", autoscanstatus_command))
    app.add_handler(CommandHandler("autoscanlogfutu", autoscanlogfutu_command))
    app.add_handler(CommandHandler("autoscanlogspot", autoscanlogspot_command))
    app.add_handler(CommandHandler("exportdb", exportdb_command))
    app.add_handler(CallbackQueryHandler(
        analyze_symbol_callback,
        pattern=f"^({ANALYZE_FUTURES_CALLBACK_PREFIX}|{ANALYZE_SPOT_CALLBACK_PREFIX}):",
    ))
    app.add_handler(CallbackQueryHandler(autoscan_auto_callback, pattern=r"^asauto:"))
    app.add_handler(CallbackQueryHandler(apikey_callback, pattern=r"^apikey:"))
    # Group 0: bắt tin nhắn nhập key/qty/đòn bẩy trước mọi handler text khác.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, autoscan_pending_message), group=0)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, symbol_message_handler), group=1)
    app.add_handler(MessageHandler(filters.COMMAND, symbol_message_handler), group=2)

    # Background job: check pending predictions every 30 minutes
    if app.job_queue is None:
        print("JobQueue is not available. Install python-telegram-bot[job-queue].")
    else:
        app.job_queue.run_repeating(job_check_predictions, interval=1800, first=300)
        try:
            # This job only wakes up to check whether a candle-close slot is due; it doesn't call Binance/LLM if the slot was already scanned.
            from analyze import AUTOSCAN_SCHEDULER_TICK_SECONDS
            app.job_queue.run_repeating(
                job_auto_scan,
                interval=AUTOSCAN_SCHEDULER_TICK_SECONDS,
                first=10,
                job_kwargs={"misfire_grace_time": 60},
            )
        except Exception as exc:
            print(f"Auto Scan job was not started: {exc}", flush=True)


def symbol_control_commands() -> list[BotCommand]:
    """Minimal user menu; maintenance/rarely-used commands can still be typed manually."""
    return [
        BotCommand("listsymbols", "Danh sách coin hỗ trợ"),
        BotCommand("history", "10 lệnh gần nhất"),
        BotCommand("onfutu", "Bật Auto Scan Futures"),
        BotCommand("onspot", "Bật Auto Scan Spot"),
        BotCommand("offfutu", "Tắt Auto Scan Futures"),
        BotCommand("offspot", "Tắt Auto Scan Spot"),
        BotCommand("autoscanstatus", "Trạng thái Auto Scan"),
        BotCommand("autoscanlogfutu", "Lệnh phiên Futures"),
        BotCommand("autoscanlogspot", "Lệnh phiên Spot"),
    ]
