"""Tests for evaluation.metrics."""

import pandas as pd
import pytest

from evaluation.metrics import (
    calculate_alpha,
    calculate_cumulative_return,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_win_rate,
)


def test_sharpe_ratio():
    # Flat return = 0 sharpe
    rets = pd.Series([0.01, 0.01, 0.01])
    assert calculate_sharpe(rets) == 0.0
    
    # Positive sharpe
    rets = pd.Series([0.01, 0.02, 0.015, 0.01])
    sharpe = calculate_sharpe(rets)
    assert sharpe > 0


def test_max_drawdown():
    # 100 -> 110 -> 99 -> 120
    # Peak is 110. Drop to 99 is -10%. Max DD should be 0.10
    equity = pd.Series([100, 110, 99, 120])
    dd = calculate_max_drawdown(equity)
    assert dd == pytest.approx(0.10)


def test_win_rate():
    pnls = pd.Series([100, -50, 20, -10])
    # 2 positives out of 4 = 0.5
    assert calculate_win_rate(pnls) == 0.5


def test_cumulative_return():
    equity = pd.Series([100, 150])
    assert calculate_cumulative_return(equity) == 0.5


def test_alpha():
    agent = pd.Series([0.10])  # +10%
    bench = pd.Series([0.05])  # +5%
    # Simple alpha = 0.10 - 0.05 = 0.05
    assert calculate_alpha(agent, bench) == pytest.approx(0.05)
