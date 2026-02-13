"""Data pipeline: fetch market data via yfinance and engineer technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD


# ── Historical data ─────────────────────────────────────────────────────────


def fetch_historical(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download OHLCV history for *ticker* over the given *period*.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    """
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    # Flatten multi-level columns if present (yfinance >= 0.2.31)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ── Live / intraday data ────────────────────────────────────────────────────


def fetch_live(ticker: str) -> pd.DataFrame:
    """Fetch the most recent trading day's intraday data (5-min bars).

    Returns a DataFrame with the same OHLCV schema.
    """
    tk = yf.Ticker(ticker)
    df = tk.history(period="1d", interval="5m")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ── Indicator engineering ────────────────────────────────────────────────────


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append RSI(14), MACD(12,26,9), EMA(20), EMA(50) columns in-place.

    Any rows that fall in the warmup window (first ~50 rows) will contain NaN
    for some indicators.  The caller should drop or forward-fill as needed.
    """
    close = df["Close"]

    # RSI
    df["RSI_14"] = RSIIndicator(close=close, window=14).rsi()

    # MACD
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    # EMAs
    df["EMA_20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["EMA_50"] = EMAIndicator(close=close, window=50).ema_indicator()

    return df


# ── State vector construction ────────────────────────────────────────────────

# The columns that form the "market" part of the observation.
INDICATOR_COLS = [
    "Close", "Volume",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "EMA_20", "EMA_50",
]


def build_state_vector(
    df_row: pd.Series,
    cash: float,
    holdings: float,
    unrealized_pnl: float,
) -> np.ndarray:
    """Combine indicator values with portfolio state into a flat vector.

    Parameters
    ----------
    df_row : pd.Series
        A single row from a DataFrame that has been through `compute_indicators`.
    cash : float
        Current available cash.
    holdings : float
        Current total value of held positions.
    unrealized_pnl : float
        Current unrealised profit / loss.

    Returns
    -------
    np.ndarray
        1-D float32 vector of length ``len(INDICATOR_COLS) + 3``.
    """
    market = df_row[INDICATOR_COLS].values.astype(np.float32)
    portfolio = np.array([cash, holdings, unrealized_pnl], dtype=np.float32)
    return np.concatenate([market, portfolio])


def n_features() -> int:
    """Return the expected length of a state vector."""
    return len(INDICATOR_COLS) + 3  # +3 for cash, holdings, unrealised_pnl
