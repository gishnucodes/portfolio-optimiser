"""Tests for engine.portfolio — Portfolio position tracking and trade execution."""

from __future__ import annotations

import pytest

from engine.portfolio import Portfolio


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def portfolio() -> Portfolio:
    return Portfolio(initial_cash=10_000.0, trade_fraction=0.25)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_initial_state(portfolio: Portfolio) -> None:
    assert portfolio.cash == pytest.approx(10_000.0)
    assert portfolio.get_equity({"AAPL": 150.0}) == pytest.approx(10_000.0)


def test_buy_reduces_cash(portfolio: Portfolio) -> None:
    receipt = portfolio.execute_trade("AAPL", "BUY", 150.0)
    # Should invest 25% of 10k = $2500
    assert portfolio.cash == pytest.approx(7_500.0)
    assert receipt["shares_traded"] == pytest.approx(2_500.0 / 150.0)
    assert receipt["cost"] == pytest.approx(2_500.0)


def test_sell_with_no_holdings_is_noop(portfolio: Portfolio) -> None:
    receipt = portfolio.execute_trade("AAPL", "SELL", 150.0)
    assert portfolio.cash == pytest.approx(10_000.0)
    assert receipt["shares_traded"] == pytest.approx(0.0)


def test_hold_does_nothing(portfolio: Portfolio) -> None:
    portfolio.execute_trade("AAPL", "HOLD", 150.0)
    assert portfolio.cash == pytest.approx(10_000.0)


def test_buy_then_sell(portfolio: Portfolio) -> None:
    portfolio.execute_trade("AAPL", "BUY", 100.0)
    shares_after_buy = portfolio.get_position("AAPL")["shares"]
    assert shares_after_buy > 0

    portfolio.execute_trade("AAPL", "SELL", 110.0)
    shares_after_sell = portfolio.get_position("AAPL")["shares"]
    assert shares_after_sell < shares_after_buy
    # Cash should have increased from selling at a higher price
    assert portfolio.cash > 7_500.0


def test_equity_with_multiple_tickers(portfolio: Portfolio) -> None:
    portfolio.execute_trade("AAPL", "BUY", 150.0)
    portfolio.execute_trade("MSFT", "BUY", 300.0)
    prices = {"AAPL": 160.0, "MSFT": 310.0}
    equity = portfolio.get_equity(prices)
    # Equity should be higher than initial because prices rose
    assert equity > 10_000.0


def test_unrealised_pnl_positive(portfolio: Portfolio) -> None:
    portfolio.execute_trade("AAPL", "BUY", 100.0)
    pnl = portfolio.get_unrealised_pnl({"AAPL": 120.0})
    assert pnl > 0


def test_unrealised_pnl_negative(portfolio: Portfolio) -> None:
    portfolio.execute_trade("AAPL", "BUY", 100.0)
    pnl = portfolio.get_unrealised_pnl({"AAPL": 80.0})
    assert pnl < 0


def test_snapshot_contains_keys(portfolio: Portfolio) -> None:
    portfolio.execute_trade("AAPL", "BUY", 150.0)
    snap = portfolio.snapshot({"AAPL": 155.0})
    assert "cash" in snap
    assert "equity" in snap
    assert "holdings_value" in snap
    assert "unrealised_pnl" in snap
    assert "positions" in snap
    assert "AAPL" in snap["positions"]


def test_action_case_insensitive(portfolio: Portfolio) -> None:
    portfolio.execute_trade("AAPL", "buy", 100.0)
    assert portfolio.cash < 10_000.0
