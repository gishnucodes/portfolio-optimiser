"""Tests for data.pipeline — indicator computation and state vector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.pipeline import (
    INDICATOR_COLS,
    build_state_vector,
    compute_indicators,
    n_features,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with 120 rows (enough for warmup)."""
    np.random.seed(42)
    n = 120
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.2,
            "High": close + abs(np.random.randn(n) * 0.5),
            "Low": close - abs(np.random.randn(n) * 0.5),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 50_000_000, size=n),
        }
    )
    return df


# ── Tests ────────────────────────────────────────────────────────────────────


def test_compute_indicators_adds_columns(sample_ohlcv: pd.DataFrame) -> None:
    df = compute_indicators(sample_ohlcv.copy())
    expected_cols = {"RSI_14", "MACD", "MACD_signal", "MACD_hist", "EMA_20", "EMA_50"}
    assert expected_cols.issubset(set(df.columns))


def test_compute_indicators_no_nan_after_warmup(sample_ohlcv: pd.DataFrame) -> None:
    df = compute_indicators(sample_ohlcv.copy())
    # After dropping the first 50 rows (warmup), there should be no NaN
    tail = df.iloc[50:]
    assert not tail[list(INDICATOR_COLS)].isna().any().any()


def test_build_state_vector_shape(sample_ohlcv: pd.DataFrame) -> None:
    df = compute_indicators(sample_ohlcv.copy()).dropna()
    row = df.iloc[0]
    vec = build_state_vector(row, cash=10_000.0, holdings=0.0, unrealized_pnl=0.0)
    assert vec.shape == (n_features(),)
    assert vec.dtype == np.float32


def test_build_state_vector_values(sample_ohlcv: pd.DataFrame) -> None:
    df = compute_indicators(sample_ohlcv.copy()).dropna()
    row = df.iloc[0]
    vec = build_state_vector(row, cash=5_000, holdings=3_000, unrealized_pnl=-200)
    # Last 3 entries should be portfolio state
    assert vec[-3] == pytest.approx(5_000, abs=1e-3)
    assert vec[-2] == pytest.approx(3_000, abs=1e-3)
    assert vec[-1] == pytest.approx(-200, abs=1e-3)


def test_n_features_matches_indicator_cols() -> None:
    assert n_features() == len(INDICATOR_COLS) + 3
