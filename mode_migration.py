"""One-way migration of persisted analysis mode labels.

Legacy mode labels become `futures`/`spot`. Spot rows also get common Spot symbols where an
old Futures alias (for example 1000SHIBUSDT) was stored.
"""

from __future__ import annotations

import sqlite3


MODE_RENAMES = {
    "short": "futures",
    "intraday": "futures",
    "long": "spot",
    "swing": "spot",
    "spot": "spot",
}
FUTURES_SYMBOL_ALIASES = {
    "1000SHIB": "SHIB", "1000PEPE": "PEPE", "1000BONK": "BONK", "1000FLOKI": "FLOKI",
    "1000LUNC": "LUNC", "1000RATS": "RATS", "1000XEC": "XEC", "1000SATS": "SATS",
}


def migrate_mode_values(conn: sqlite3.Connection) -> int:
    """Rename legacy mode values in every existing table that has a `mode` column."""
    changed = 0
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for (table_name,) in tables:
        columns = {row[1].lower() for row in conn.execute(f'PRAGMA table_info("{table_name}")')}
        if "mode" not in columns:
            continue
        safe_table = str(table_name).replace('"', '""')
        if "symbol" in columns:
            for old, new in MODE_RENAMES.items():
                if new != "spot":
                    continue
                conn.execute(
                    f'UPDATE "{safe_table}" SET symbol=CASE '
                    + " ".join(f"WHEN upper(symbol)=? THEN ?" for _ in FUTURES_SYMBOL_ALIASES)
                    + " ELSE symbol END WHERE lower(trim(mode))=?",
                    tuple(item for key, value in FUTURES_SYMBOL_ALIASES.items() for item in (key + "USDT", value + "USDT")) + (old,),
                )
        for old, new in MODE_RENAMES.items():
            cursor = conn.execute(
                f'UPDATE "{safe_table}" SET mode=? WHERE lower(trim(mode))=?',
                (new, old),
            )
            changed += max(cursor.rowcount, 0)
    return changed
