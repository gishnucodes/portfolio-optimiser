"""Tests for engine.ledger — SQLite ledger round-trip and queries."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from engine.ledger import (
    get_decisions,
    get_equity_curve,
    get_latest_summary,
    init_db,
    log_daily_summary,
    log_decision,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Provide a fresh temporary database for each test."""
    path = tmp_path / "test_ledger.db"
    init_db(path)
    return path


# ── Tests ────────────────────────────────────────────────────────────────────


def test_log_and_retrieve_decision(db_path: Path) -> None:
    log_decision(
        date="2025-01-15",
        window="10:00",
        ticker="AAPL",
        agent="rl",
        action="BUY",
        price=150.0,
        confidence=0.85,
        reasoning=None,
        db_path=db_path,
    )
    rows = get_decisions(agent="rl", db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["action"] == "BUY"
    assert rows[0]["confidence"] == pytest.approx(0.85)


def test_log_and_retrieve_llm_decision(db_path: Path) -> None:
    log_decision(
        date="2025-01-15",
        window="13:00",
        ticker="MSFT",
        agent="llm",
        action="SELL",
        price=400.0,
        reasoning="RSI overbought at 78",
        db_path=db_path,
    )
    rows = get_decisions(agent="llm", db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["reasoning"] == "RSI overbought at 78"


def test_decision_date_filter(db_path: Path) -> None:
    for d in ["2025-01-10", "2025-01-15", "2025-01-20"]:
        log_decision(
            date=d, window="10:00", ticker="AAPL",
            agent="rl", action="HOLD", price=150.0, db_path=db_path,
        )
    rows = get_decisions(start="2025-01-12", end="2025-01-18", db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["date"] == "2025-01-15"


def test_daily_summary_round_trip(db_path: Path) -> None:
    log_daily_summary(
        date="2025-01-15", agent="rl",
        equity=10_200.0, cash=5_000.0, pnl=200.0,
        sharpe=1.5, max_drawdown=0.02, win_rate=0.6,
        db_path=db_path,
    )
    latest = get_latest_summary("rl", db_path=db_path)
    assert latest is not None
    assert latest["equity"] == pytest.approx(10_200.0)
    assert latest["sharpe"] == pytest.approx(1.5)


def test_equity_curve_order(db_path: Path) -> None:
    for d, eq in [("2025-01-10", 10_000), ("2025-01-11", 10_100), ("2025-01-12", 10_050)]:
        log_daily_summary(
            date=d, agent="rl", equity=eq, cash=5_000, pnl=eq - 10_000,
            db_path=db_path,
        )
    curve = get_equity_curve("rl", db_path=db_path)
    assert len(curve) == 3
    # Should be in chronological order
    assert curve[0]["date"] < curve[1]["date"] < curve[2]["date"]


def test_no_summary_returns_none(db_path: Path) -> None:
    assert get_latest_summary("rl", db_path=db_path) is None
