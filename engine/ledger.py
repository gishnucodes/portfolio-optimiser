"""SQLite Ledger — persistent storage for trading decisions and daily summaries."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator

from config import DB_PATH


# ── Connection helper ────────────────────────────────────────────────────────

@contextmanager
def _connect(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Schema initialisation ───────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    window      TEXT    NOT NULL,
    ticker      TEXT    NOT NULL,
    agent       TEXT    NOT NULL,   -- 'rl', 'llm', 'ppo_momentum', etc.
    action      TEXT    NOT NULL,   -- 'BUY', 'SELL', 'HOLD'
    confidence  REAL,
    reasoning   TEXT,
    price       REAL    NOT NULL,
    mode        TEXT    NOT NULL DEFAULT 'backtest', -- 'backtest' or 'live'
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS daily_summary (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    date          TEXT    NOT NULL,
    agent         TEXT    NOT NULL,
    equity        REAL    NOT NULL,
    cash          REAL    NOT NULL,
    pnl           REAL    NOT NULL,
    sharpe        REAL,
    max_drawdown  REAL,
    win_rate      REAL,
    mode          TEXT    NOT NULL DEFAULT 'backtest', -- 'backtest' or 'live'
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the database tables if they don't exist."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        
        # Migration: Add mode column if missing (for existing DBs)
        try:
            conn.execute("ALTER TABLE decisions ADD COLUMN mode TEXT NOT NULL DEFAULT 'backtest'")
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        try:
            conn.execute("ALTER TABLE daily_summary ADD COLUMN mode TEXT NOT NULL DEFAULT 'backtest'")
        except sqlite3.OperationalError:
            pass


# ── Write operations ────────────────────────────────────────────────────────


def log_decision(
    *,
    date: str,
    window: str,
    ticker: str,
    agent: str,
    action: str,
    price: float,
    confidence: float | None = None,
    reasoning: str | None = None,
    mode: str = "backtest",
    db_path: Path = DB_PATH,
) -> None:
    """Insert a single trading decision into the ledger."""
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO decisions (date, window, ticker, agent, action,
               confidence, reasoning, price, mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date, window, ticker, agent, action, confidence, reasoning, price, mode),
        )


def log_daily_summary(
    *,
    date: str,
    agent: str,
    equity: float,
    cash: float,
    pnl: float,
    sharpe: float | None = None,
    max_drawdown: float | None = None,
    win_rate: float | None = None,
    mode: str = "backtest",
    db_path: Path = DB_PATH,
) -> None:
    """Insert an end-of-day summary row."""
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO daily_summary (date, agent, equity, cash, pnl,
               sharpe, max_drawdown, win_rate, mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date, agent, equity, cash, pnl, sharpe, max_drawdown, win_rate, mode),
        )


# ── Read operations ─────────────────────────────────────────────────────────


def get_decisions(
    agent: str | None = None,
    start: str | None = None,
    end: str | None = None,
    mode: str = "backtest",
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """Retrieve decision rows, filtered by mode, agent, and date range."""
    query = "SELECT * FROM decisions WHERE mode = ?"
    params: list[Any] = [mode]

    if agent:
        query += " AND agent = ?"
        params.append(agent)
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)

    query += " ORDER BY date, window, ticker"

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_equity_curve(
    agent: str,
    mode: str = "backtest",
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """Return chronologically ordered equity data for an agent in a specific mode."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT date, equity, cash, pnl FROM daily_summary "
            "WHERE agent = ? AND mode = ? ORDER BY date",
            (agent, mode),
        ).fetchall()
        return [dict(r) for r in rows]


def get_latest_summary(
    agent: str,
    mode: str = "backtest",
    db_path: Path = DB_PATH,
) -> dict[str, Any] | None:
    """Return the most recent daily summary for an agent/mode, or None."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM daily_summary WHERE agent = ? AND mode = ? ORDER BY date DESC LIMIT 1",
            (agent, mode),
        ).fetchone()
        return dict(row) if row else None
