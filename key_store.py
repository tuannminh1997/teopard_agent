"""Lưu API key của user — MỘT BO DÙNG CHUNG cho futures và spot, mã hóa Fernet trước khi ghi DB.

Binance: 1 API key trade được cả Spot lẫn Futures (cùng 1 tài khoản), và hệ demo cũng vậy
(demo-fapi.binance.com cho futures, demo-api.binance.com cho spot — cùng 1 key). Nên key lưu
theo user, KHÔNG tách theo market.

Khóa lấy từ biến môi trường DATA_ENCRYPTION_KEY (đặt trên Railway Variables và .env local).
Fail-closed: thiếu khóa hoặc không giải mã được thì KHÔNG đọc/ghi, để flow báo lại user.
"""
import os
import sqlite3
from datetime import datetime, timezone

try:
    from cryptography.fernet import Fernet
except ImportError:  # pragma: no cover
    Fernet = None

DB_PATH = os.getenv("DB_PATH", "bot.db")
TABLE = "user_api_keys"


class KeyError_(RuntimeError):
    """Lỗi cấu hình key (thiếu env, không giải mã được...)."""


def _fernet():
    if Fernet is None:
        raise KeyError_("Thiếu thư viện cryptography (pip install cryptography).")
    key = os.getenv("DATA_ENCRYPTION_KEY", "").strip()
    if not key:
        raise KeyError_("Thiếu biến DATA_ENCRYPTION_KEY — thêm trên Railway Variables / .env local.")
    try:
        return Fernet(key.encode())
    except Exception as exc:
        raise KeyError_(f"DATA_ENCRYPTION_KEY không hợp lệ: {exc}") from exc


def _migrate_to_shared_key(conn: sqlite3.Connection) -> None:
    """Chuyển bảng legacy (user_id, market) → 1 key/user.

    Giữ dòng MỚI NHẤT của mỗi user (updated_at lớn nhất) để không mất key cũ đã nhập.
    DDL trong cùng transaction với copy nên nếu lỗi thì rollback, không mất dữ liệu.
    """
    tmp = f"{TABLE}_shared_tmp"
    conn.execute(f"DROP TABLE IF EXISTS {tmp}")
    conn.execute(f"""
        CREATE TABLE {tmp} (
            user_id    INTEGER PRIMARY KEY,
            api_key_enc TEXT NOT NULL,
            secret_enc  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    conn.execute(f"""
        INSERT INTO {tmp} (user_id, api_key_enc, secret_enc, updated_at)
        SELECT user_id, api_key_enc, secret_enc, updated_at
        FROM {TABLE}
        WHERE rowid IN (
            SELECT rowid FROM (
                SELECT rowid,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC, rowid DESC) AS rn
                FROM {TABLE}
            ) ranked WHERE rn = 1
        )
    """)
    conn.execute(f"DROP TABLE {TABLE}")
    conn.execute(f"ALTER TABLE {tmp} RENAME TO {TABLE}")


def init_key_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                user_id    INTEGER PRIMARY KEY,
                api_key_enc TEXT NOT NULL,
                secret_enc  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{TABLE}")')}
        if "market" in columns:
            _migrate_to_shared_key(conn)
        conn.commit()


def save_api_keys(user_id: int, api_key: str, secret: str) -> None:
    f = _fernet()  # fail-closed trước khi ghi
    api_key, secret = api_key.strip(), secret.strip()
    if len(api_key) < 20 or len(secret) < 20:
        raise KeyError_("API key/secret quá ngắn — kiểm tra lại bạn đã dán đủ chưa.")
    init_key_db()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE} (user_id, api_key_enc, secret_enc, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                api_key_enc=excluded.api_key_enc,
                secret_enc=excluded.secret_enc,
                updated_at=excluded.updated_at
            """,
            (user_id, f.encrypt(api_key.encode()).decode(),
             f.encrypt(secret.encode()).decode(), now),
        )
        conn.commit()


def get_api_keys(user_id: int) -> tuple[str, str] | None:
    """Trả (api_key, secret) dạng text; None nếu chưa lưu. Raise KeyError_ nếu lỗi cấu hình/giải mã."""
    f = _fernet()
    init_key_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"SELECT api_key_enc, secret_enc FROM {TABLE} WHERE user_id=?",
            (user_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        return f.decrypt(row[0].encode()).decode(), f.decrypt(row[1].encode()).decode()
    except Exception as exc:
        raise KeyError_("Không giải mã được API key đã lưu (DATA_ENCRYPTION_KEY bị đổi?) — nhập lại key mới.") from exc


def has_api_keys(user_id: int) -> bool:
    init_key_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"SELECT 1 FROM {TABLE} WHERE user_id=?", (user_id,)
        ).fetchone()
    return row is not None


def api_key_status(user_id: int) -> str:
    """'missing' | 'ok' | 'broken'.

    'broken' = dòng key CÓ trong DB nhưng không đọc/giải mã được (thiếu cryptography,
    thiếu DATA_ENCRYPTION_KEY, hoặc key đã bị đổi). Nếu chỉ dùng has_api_keys() thì UI vẫn
    hiện "Đã Thêm" trong khi mọi lần đặt lệnh đều abort — user không hiểu vì sao.
    """
    if not has_api_keys(user_id):
        return "missing"
    try:
        get_api_keys(user_id)
    except KeyError_:
        return "broken"
    return "ok"


def delete_api_keys(user_id: int) -> bool:
    """Gỡ API key của user (gỡ 1 lần là futures lẫn spot cùng mất). Trả True nếu có dòng bị xóa."""
    init_key_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            f"DELETE FROM {TABLE} WHERE user_id=?", (user_id,)
        )
        conn.commit()
        return cur.rowcount > 0
