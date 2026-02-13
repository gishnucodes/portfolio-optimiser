"""Evaluation metrics for trading performance."""

import numpy as np
import pandas as pd


def calculate_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualised Sharpe ratio (assuming 252 trading days)."""
    if len(daily_returns) < 2:
        return 0.0
    
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    
    if std_ret == 0:
        return 0.0
        
    dsr = (mean_ret - risk_free_rate) / std_ret
    return dsr * np.sqrt(252)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum peak-to-trough drawdown (as a positive percentage)."""
    if len(equity_curve) < 1:
        return 0.0
        
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return abs(drawdown.min())


def calculate_win_rate(daily_pnl: pd.Series) -> float:
    """Calculate percentage of days with positive PnL."""
    if len(daily_pnl) == 0:
        return 0.0
        
    wins = (daily_pnl > 0).sum()
    return wins / len(daily_pnl)


def calculate_cumulative_return(equity_curve: pd.Series) -> float:
    """Calculate total return percentage."""
    if len(equity_curve) < 1:
        return 0.0
        
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    
    if start_value == 0:
        return 0.0
        
    return (end_value - start_value) / start_value


def calculate_alpha(
    agent_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Calculate excess return over benchmark (simple Alpha)."""
    # Align dates
    df = pd.concat([agent_returns, benchmark_returns], axis=1, join="inner")
    if len(df) < 1:
        return 0.0
        
    agent_cum = (1 + df.iloc[:, 0]).prod() - 1
    bench_cum = (1 + df.iloc[:, 1]).prod() - 1
    
    return agent_cum - bench_cum
