import os
import sqlite3

from datetime import datetime, timedelta, timezone

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# The daily manual-analysis quota belongs to a Vietnamese user, so "hôm nay" must mean a Vietnam
# calendar day. date.today() would follow the container clock instead — on Railway that is UTC, so
# the quota silently rolled over at 07:00 VN and the user still saw yesterday's count all morning.
VN_TZ = timezone(timedelta(hours=7))


def vn_today() -> str:
    return datetime.now(VN_TZ).date().isoformat()

GET_USER_ID_CALLBACK = "get_user_id"
VERIFY_CALLBACK = "verify_account"
DB_PATH = os.getenv("DB_PATH", "bot.db")
KNOWN_COMMAND_PATTERN = (
    r"^/(start|whoami|adduser|removeuser|listusers|help|"
    r"addsymbol|removesymbol|listsymbols|setlimit|resetusage|stats|statsall|history|historyall|dashboard|dashboardall|clearhistory|checknow|autoscanon|autoscanoff|autoscanstatus|autoscanlog|exportdb)(@\w+)?(\s|$)"
)


def load_admin_ids() -> set[int]:
    raw_ids = os.getenv("ADMIN_USER_IDS", "")
    return {
        int(user_id.strip())
        for user_id in raw_ids.split(",")
        if user_id.strip().isdigit()
    }


ADMIN_USER_IDS = load_admin_ids()


DEFAULT_DAILY_LIMIT = 10


def ensure_whitelist_default_limit(conn: sqlite3.Connection) -> None:
    """Rebuild whitelist table if an old DB has daily_limit DEFAULT other than 10."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='whitelist'"
    ).fetchone()
    create_sql = row[0] if row else ""
    if "daily_limit INTEGER NOT NULL DEFAULT 10" in create_sql:
        return

    conn.execute("ALTER TABLE whitelist RENAME TO whitelist_old")
    conn.execute(f"""
        CREATE TABLE whitelist (
            user_id INTEGER PRIMARY KEY,
            daily_limit INTEGER NOT NULL DEFAULT {DEFAULT_DAILY_LIMIT},
            used_today INTEGER NOT NULL DEFAULT 0,
            last_reset_date TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.execute(f"""
        INSERT INTO whitelist (user_id, daily_limit, used_today, last_reset_date)
        SELECT
            user_id,
            COALESCE(daily_limit, {DEFAULT_DAILY_LIMIT}),
            COALESCE(used_today, 0),
            COALESCE(last_reset_date, '')
        FROM whitelist_old
    """)
    conn.execute("DROP TABLE whitelist_old")


def init_auth_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS whitelist (
                user_id INTEGER PRIMARY KEY,
                daily_limit INTEGER NOT NULL DEFAULT {DEFAULT_DAILY_LIMIT},
                used_today INTEGER NOT NULL DEFAULT 0,
                last_reset_date TEXT NOT NULL DEFAULT ''
            )
        """)
        for col, definition in [
            ("daily_limit", f"INTEGER NOT NULL DEFAULT {DEFAULT_DAILY_LIMIT}"),
            ("used_today", "INTEGER NOT NULL DEFAULT 0"),
            ("last_reset_date", "TEXT NOT NULL DEFAULT ''"),
        ]:
            try:
                conn.execute(f"ALTER TABLE whitelist ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass
        ensure_whitelist_default_limit(conn)
        conn.commit()


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_USER_IDS


def is_account_activated(user_id: int) -> bool:
    return is_user_whitelisted(user_id)




def add_whitelist_user(user_id: int) -> None:
    today = vn_today()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO whitelist (user_id, daily_limit, used_today, last_reset_date)
            VALUES (?, ?, 0, ?)
            """,
            (user_id, DEFAULT_DAILY_LIMIT, today),
        )
        conn.commit()


def remove_whitelist_user(user_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM whitelist WHERE user_id = ?",
            (user_id,),
        )
        conn.commit()


def is_user_whitelisted(user_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM whitelist WHERE user_id = ?",
            (user_id,),
        ).fetchone()

    return row is not None


def get_whitelist_users() -> list[tuple[int, int, int]]:
    """Return a list of (user_id, daily_limit, used_today)."""
    today = vn_today()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT user_id, daily_limit, used_today, last_reset_date FROM whitelist ORDER BY user_id"
        ).fetchall()

    result = []
    for user_id, daily_limit, used_today, last_reset_date in rows:
        if last_reset_date != today:
            used_today = 0
        result.append((user_id, daily_limit, used_today))
    return result


def start_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "Lấy User ID.",
                callback_data=GET_USER_ID_CALLBACK,
            ),
            InlineKeyboardButton(
                "Kích hoạt tài khoản.",
                callback_data=VERIFY_CALLBACK,
            ),
        ],
    ])


async def show_start_menu(update: Update) -> None:
    if update.effective_message:
        await update.effective_message.reply_text(
            "Chào mừng bạn đến với Teopard. Báo đời báo đốm thích phân tích coin."
        )
        await update.effective_message.reply_text(
            "Bạn vui lòng làm theo hướng dẫn sau để kích hoạt tài khoản:\n\n"
            "1. Nhấn nút 'Lấy User ID'.\n"
            "2. Cung cấp User ID vừa lấy cho admin.\n"
            "3. Sau khi admin cấp quyền thành công, nhấn nút 'Kích hoạt tài khoản'.",
            reply_markup=start_keyboard(),
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user

    if not user:
        return

    if is_account_activated(user.id):
        await update.effective_message.reply_text(
            "Tài khoản của bạn đã được kích hoạt thành công. Vui lòng gõ '/help' hoặc chọn 'Giúp đỡ' trên Menu để được hướng dẫn sử dụng."
        )
        return

    await show_start_menu(update)


async def get_user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user

    if not query or not user:
        return

    await query.answer()

    await query.message.reply_text(
        f"User ID của bạn là: {user.id}\n"
        "Vui lòng gửi User ID này cho admin để được thêm vào whitelist."
    )


async def verify_account(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user

    if not query or not user:
        return

    await query.answer()

    if is_account_activated(user.id):
        await query.message.reply_text(
            "Tài khoản của bạn đã được kích hoạt thành công. Vui lòng gõ '/help' chọn 'Giúp đỡ' trên Menu để được hướng dẫn sử dụng."
        )
        return

    if is_user_whitelisted(user.id):
        await query.message.reply_text(
            "Tài khoản của bạn đã được kích hoạt thành công. Vui lòng gõ '/help' chọn 'Giúp đỡ' trên Menu để được hướng dẫn sử dụng."
        )
        return

    await query.message.reply_text(
        "Kích hoạt tài khoản không thành công. Vui lòng gõ '/start' chọn 'Bắt đầu' trên Menu và làm theo hướng dẫn."
    )


async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    admin = update.effective_user

    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return

    if not context.args or not context.args[0].isdigit():
        await update.effective_message.reply_text("Cú pháp đúng: /adduser 123456789")
        return

    user_id = int(context.args[0])
    add_whitelist_user(user_id)

    await update.effective_message.reply_text(
        f"Đã thêm User ID {user_id} vào whitelist."
    )


async def remove_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    admin = update.effective_user

    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return

    if not context.args or not context.args[0].isdigit():
        await update.effective_message.reply_text("Cú pháp đúng: /removeuser 123456789")
        return

    user_id = int(context.args[0])
    remove_whitelist_user(user_id)
    await update.effective_message.reply_text(
        f"Đã xóa User ID {user_id} khỏi whitelist."
    )

    try:
        await context.bot.send_message(
            chat_id=user_id,
            text=(
                "Tài khoản của bạn đã bị thu hồi quyền truy cập chatbot. "
                "Vui lòng liên hệ admin để được xử lý."
            ),
            reply_markup=start_keyboard(),
        )
    except Exception:
        await update.effective_message.reply_text(
            "Không thể gửi thông báo cho user này. "
            "Có thể user chưa từng bấm Start hoặc đã chặn bot."
        )


async def list_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    admin = update.effective_user

    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return

    users = get_whitelist_users()

    if not users:
        await update.effective_message.reply_text("Whitelist hiện đang trống.")
        return

    lines = ["Danh sách whitelist:\n"]
    for user_id, daily_limit, used_today in users:
        lines.append(f"• {user_id} — {used_today}/{daily_limit} lượt hôm nay")

    await update.effective_message.reply_text("\n".join(lines))

def get_user_usage(user_id: int) -> tuple[int, int]:
    """Return (daily_limit, used_today) after auto-resetting if the date has rolled over."""
    today = vn_today()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT daily_limit, used_today, last_reset_date FROM whitelist WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if row is None:
            return DEFAULT_DAILY_LIMIT, 0

        daily_limit, used_today, last_reset_date = row

        if last_reset_date != today:
            conn.execute(
                "UPDATE whitelist SET used_today = 0, last_reset_date = ? WHERE user_id = ?",
                (today, user_id),
            )
            conn.commit()
            return daily_limit, 0

        return daily_limit, used_today


def increment_user_usage(user_id: int) -> None:
    """Increment used_today by 1."""
    today = vn_today()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE whitelist SET used_today = used_today + 1, last_reset_date = ? WHERE user_id = ?",
            (today, user_id),
        )
        conn.commit()


def decrement_user_usage(user_id: int) -> None:
    """Refund 1 lượt when analysis fails after pre-increment."""
    today = vn_today()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE whitelist SET used_today = MAX(0, used_today - 1), last_reset_date = ? WHERE user_id = ?",
            (today, user_id),
        )
        conn.commit()


def set_user_daily_limit(user_id: int, limit: int) -> bool:
    """Set daily_limit for a user. Returns False if the user doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "UPDATE whitelist SET daily_limit = ? WHERE user_id = ?",
            (limit, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def reset_user_usage(user_id: int) -> bool:
    """Reset used_today to 0 for a user. Returns False if the user doesn't exist."""
    today = vn_today()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "UPDATE whitelist SET used_today = 0, last_reset_date = ? WHERE user_id = ?",
            (today, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0

async def set_limit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    admin = update.effective_user

    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return

    if not context.args or len(context.args) < 2 or not context.args[0].isdigit() or not context.args[1].isdigit():
        await update.effective_message.reply_text("Cú pháp đúng: /setlimit 123456789 10")
        return

    user_id = int(context.args[0])
    limit = int(context.args[1])

    if set_user_daily_limit(user_id, limit):
        await update.effective_message.reply_text(
            f"Đã set giới hạn {limit} lượt/ngày cho User ID {user_id}."
        )
    else:
        await update.effective_message.reply_text(
            f"Không tìm thấy User ID {user_id} trong whitelist."
        )


async def reset_usage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    admin = update.effective_user

    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return

    if not context.args or not context.args[0].isdigit():
        await update.effective_message.reply_text("Cú pháp đúng: /resetusage 123456789")
        return

    user_id = int(context.args[0])

    if reset_user_usage(user_id):
        await update.effective_message.reply_text(
            f"Đã reset lượt sử dụng hôm nay cho User ID {user_id}."
        )
    else:
        await update.effective_message.reply_text(
            f"Không tìm thấy User ID {user_id} trong whitelist."
        )

async def handle_fallback_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    user = update.effective_user

    if not user:
        return

    if not is_account_activated(user.id):
        await show_start_menu(update)
        return

    await update.effective_message.reply_text(
        "Vui lòng gõ /help hoặc chọn 'Giúp đỡ' trên Menu để được hướng dẫn sử dụng."
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user

    if user:
        await update.effective_message.reply_text(
            f"User ID của bạn là: {user.id}\n"
            "Vui lòng gửi User ID này cho admin để được thêm vào whitelist."
        )


def register_auth_handlers(app: Application) -> None:
    init_auth_db()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("adduser", add_user))
    app.add_handler(CommandHandler("removeuser", remove_user))
    app.add_handler(CommandHandler("listusers", list_users))
    app.add_handler(CommandHandler("setlimit", set_limit))
    app.add_handler(CommandHandler("resetusage", reset_usage))  

    app.add_handler(CallbackQueryHandler(get_user_id, pattern=f"^{GET_USER_ID_CALLBACK}$"))
    app.add_handler(CallbackQueryHandler(verify_account, pattern=f"^{VERIFY_CALLBACK}$"))

    known_command_filter = filters.Regex(KNOWN_COMMAND_PATTERN)
    fallback_filter = (
        (filters.TEXT & ~filters.COMMAND)
        | (filters.COMMAND & ~known_command_filter)
    )

    app.add_handler(MessageHandler(fallback_filter, handle_fallback_message), group=10)


def auth_admin_commands() -> list[BotCommand]:
    """Compact admin menu: only the commonly used admin actions."""
    return [
        BotCommand("exportdb", "Tải snapshot bot.db"),
        BotCommand("adduser", "Thêm user vào whitelist"),
        BotCommand("removeuser", "Xóa user khỏi whitelist"),
        BotCommand("listusers", "Xem whitelist"),
        BotCommand("setlimit", "Đặt giới hạn manual/ngày"),
        BotCommand("resetusage", "Reset lượt manual hôm nay"),
        BotCommand("addsymbol", "Thêm symbol"),
        BotCommand("removesymbol", "Xóa symbol"),
        BotCommand("checknow", "Check prediction đang mở"),
    ]
