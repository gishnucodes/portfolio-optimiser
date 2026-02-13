"""Tests for env.trade_gym — TradeGym custom Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.pipeline import INDICATOR_COLS, compute_indicators, n_features
from env.trade_gym import TradeGym


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Synthetic OHLCV+indicators DataFrame (200 rows, post-warmup clean)."""
    np.random.seed(0)
    n = 200
    close = 150 + np.cumsum(np.random.randn(n) * 0.3)
    df = pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.1,
            "High": close + abs(np.random.randn(n) * 0.4),
            "Low": close - abs(np.random.randn(n) * 0.4),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 50_000_000, size=n).astype(float),
        }
    )
    df = compute_indicators(df).dropna().reset_index(drop=True)
    return df


@pytest.fixture()
def env(sample_df: pd.DataFrame) -> TradeGym:
    return TradeGym(df=sample_df, initial_cash=10_000.0, trade_fraction=0.25)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_reset_returns_correct_shape(env: TradeGym) -> None:
    obs, info = env.reset(seed=42)
    assert obs.shape == (n_features(),)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_reset_restores_cash(env: TradeGym) -> None:
    # Do a buy, then reset
    env.reset(seed=42)
    env.step(TradeGym.BUY)
    obs, _ = env.reset(seed=42)
    # Cash should be the last-3 entry = initial_cash
    assert obs[-3] == pytest.approx(10_000.0, abs=1)


def test_step_hold_preserves_cash(env: TradeGym) -> None:
    obs, _ = env.reset(seed=42)
    cash_before = obs[-3]
    obs2, reward, terminated, truncated, info = env.step(TradeGym.HOLD)
    assert obs2[-3] == pytest.approx(cash_before, abs=1e-2)
    assert not terminated  # only 1 of 3 steps done


def test_step_buy_reduces_cash(env: TradeGym) -> None:
    obs, _ = env.reset(seed=42)
    cash_before = obs[-3]
    obs2, _, _, _, info = env.step(TradeGym.BUY)
    assert info["cash"] < cash_before


def test_step_sell_with_no_holdings_is_noop(env: TradeGym) -> None:
    obs, _ = env.reset(seed=42)
    cash_before = obs[-3]
    obs2, _, _, _, info = env.step(TradeGym.SELL)
    assert info["cash"] == pytest.approx(cash_before, abs=1e-2)


def test_episode_terminates_after_3_steps(env: TradeGym) -> None:
    env.reset(seed=42)
    for i in range(3):
        _, _, terminated, _, _ = env.step(TradeGym.HOLD)
        if i < 2:
            assert not terminated
        else:
            assert terminated


def test_reward_is_finite(env: TradeGym) -> None:
    env.reset(seed=42)
    for _ in range(3):
        _, reward, _, _, _ = env.step(TradeGym.BUY)
        assert np.isfinite(reward)


def test_buy_then_sell_changes_portfolio(env: TradeGym) -> None:
    env.reset(seed=42)
    _, _, _, _, info1 = env.step(TradeGym.BUY)
    assert info1["shares"] > 0
    _, _, _, _, info2 = env.step(TradeGym.SELL)
    assert info2["shares"] < info1["shares"]
