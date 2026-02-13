"""Scheduler — orchestrates the RL-vs-LLM trading loop.

Supports two modes:
- **Live mode**: Triggers at 10:00, 13:00, 15:00 EST using the ``schedule``
  library and fetches real-time data.
- **Backtest mode**: Replays historical data without waiting for real time.

Usage
-----
    # Live mode
    python -m engine.scheduler

    # Backtest mode
    python -m engine.scheduler --backtest --start 2025-01-01 --end 2025-01-31

    # RL-only (no LLM API calls)
    python -m engine.scheduler --backtest --start 2025-01-01 --end 2025-01-31 --rl-only
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import schedule

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm_agent import LLMTrader
from agents.rl_agent import RLTrader
from config import (
    BENCHMARK_TICKER,
    DB_PATH,
    INITIAL_CASH,
    MARKET_CLOSE,
    MODEL_DIR,
    TICKERS,
    TIMEZONE,
    TRADE_FRACTION,
    TRADING_WINDOWS,
)
from data.pipeline import (
    INDICATOR_COLS,
    build_state_vector,
    compute_indicators,
    fetch_historical,
    fetch_live,
)
from engine.ledger import init_db, log_daily_summary, log_decision
from engine.portfolio import Portfolio


# ── Action mapping ───────────────────────────────────────────────────────────

_RL_ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}


# ── Window execution ────────────────────────────────────────────────────────


def _run_window(
    window_label: str,
    date_str: str,
    ticker_data: dict[str, pd.Series],
    rl_trader: RLTrader,
    llm_trader: LLMTrader | None,
    rl_portfolio: Portfolio,
    llm_portfolio: Portfolio,
) -> None:
    """Execute a single trading window for all tickers."""

    for ticker, row in ticker_data.items():
        price = float(row["Close"])

        # ── RL agent ─────────────────────────────────────────────────────
        prices_dict = {t: float(d["Close"]) for t, d in ticker_data.items()}
        obs = build_state_vector(
            row,
            cash=rl_portfolio.cash,
            holdings=rl_portfolio.get_holdings_value(prices_dict),
            unrealized_pnl=rl_portfolio.get_unrealised_pnl(prices_dict),
        )
        rl_action_id, rl_confidence = rl_trader.predict(obs)
        rl_action = _RL_ACTION_MAP[rl_action_id]
        rl_portfolio.execute_trade(ticker, rl_action, price)

        log_decision(
            date=date_str,
            window=window_label,
            ticker=ticker,
            agent="rl",
            action=rl_action,
            confidence=rl_confidence,
            reasoning=None,
            price=price,
        )

        # ── LLM agent ───────────────────────────────────────────────────
        if llm_trader is not None:
            state_dict = {
                "ticker": ticker,
                "price": price,
                "volume": float(row["Volume"]),
                "rsi": float(row["RSI_14"]),
                "macd": float(row["MACD"]),
                "macd_signal": float(row["MACD_signal"]),
                "macd_hist": float(row["MACD_hist"]),
                "ema20": float(row["EMA_20"]),
                "ema50": float(row["EMA_50"]),
                "cash": llm_portfolio.cash,
                "holdings": llm_portfolio.get_holdings_value(prices_dict),
                "unrealised_pnl": llm_portfolio.get_unrealised_pnl(prices_dict),
            }
            llm_result = llm_trader.predict(state_dict)
            llm_action = llm_result["action"]
            llm_portfolio.execute_trade(ticker, llm_action, price)

            log_decision(
                date=date_str,
                window=window_label,
                ticker=ticker,
                agent="llm",
                action=llm_action,
                confidence=None,
                reasoning=llm_result["reasoning"],
                price=price,
            )

    prices_dict = {t: float(d["Close"]) for t, d in ticker_data.items()}
    print(
        f"  [{window_label}] RL equity: ${rl_portfolio.get_equity(prices_dict):,.2f}"
        + (
            f"  |  LLM equity: ${llm_portfolio.get_equity(prices_dict):,.2f}"
            if llm_trader
            else ""
        )
    )


# ── End-of-day settlement ───────────────────────────────────────────────────


def _settle_day(
    date_str: str,
    close_prices: dict[str, float],
    rl_portfolio: Portfolio,
    llm_portfolio: Portfolio,
    llm_active: bool,
) -> None:
    """Log daily summaries at market close."""
    for agent, port in [("rl", rl_portfolio)] + (
        [("llm", llm_portfolio)] if llm_active else []
    ):
        equity = port.get_equity(close_prices)
        pnl = equity - INITIAL_CASH
        log_daily_summary(
            date=date_str,
            agent=agent,
            equity=equity,
            cash=port.cash,
            pnl=pnl,
        )
    rl_eq = rl_portfolio.get_equity(close_prices)
    print(f"  📊 Day {date_str} settled — RL: ${rl_eq:,.2f}", end="")
    if llm_active:
        llm_eq = llm_portfolio.get_equity(close_prices)
        print(f"  |  LLM: ${llm_eq:,.2f}", end="")
    print()


# ── Backtest mode ────────────────────────────────────────────────────────────


def _run_backtest(
    tickers: list[str],
    start: str,
    end: str,
    rl_only: bool = False,
) -> None:
    """Replay historical data through the trading pipeline."""
    print(f"📦 Backtest mode: {start} → {end}  |  Tickers: {tickers}")
    print(f"   RL-only: {rl_only}\n")

    # Load RL model
    model_path = MODEL_DIR / "ppo_trader_v1"
    rl_trader = RLTrader.load(model_path)
    print(f"✅ Loaded RL model from {model_path}.zip")

    # Load LLM agent
    llm_trader: LLMTrader | None = None
    if not rl_only:
        llm_trader = LLMTrader()
        print("✅ Gemini LLM agent initialised")

    # Fetch historical data
    all_data: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        print(f"📥 Fetching {tk} …")
        df = fetch_historical(tk, period="5y")
        df = compute_indicators(df).dropna()
        # Filter to date range
        df.index = pd.to_datetime(df.index)
        df = df.loc[start:end]
        if len(df) == 0:
            print(f"  ⚠️  No data for {tk} in range {start}–{end}, skipping")
            continue
        all_data[tk] = df

    if not all_data:
        print("❌ No data available for any ticker in the given range.")
        return

    # Init DB
    init_db()

    # Init portfolios
    rl_portfolio = Portfolio()
    llm_portfolio = Portfolio()

    # Get unique trading days
    all_dates = sorted(
        set(d for df in all_data.values() for d in df.index.normalize().unique())
    )
    print(f"\n🗓️  {len(all_dates)} trading days found\n")

    for day in all_dates:
        date_str = day.strftime("%Y-%m-%d")
        print(f"📅 {date_str}")

        # For each day, sample up to 3 rows as "windows"
        for tk in all_data:
            day_rows = all_data[tk].loc[day.strftime("%Y-%m-%d")]
            if isinstance(day_rows, pd.Series):
                # Only one row for this day — use it for all windows
                all_data[tk]._day_rows_cache = pd.DataFrame([day_rows] * 3)
            else:
                n = len(day_rows)
                if n >= 3:
                    indices = [0, n // 2, n - 1]
                else:
                    indices = list(range(n)) + [n - 1] * (3 - n)
                all_data[tk]._day_rows_cache = day_rows.iloc[indices].reset_index(
                    drop=True
                )

        for w_idx, window_label in enumerate(TRADING_WINDOWS):
            ticker_data = {}
            for tk in all_data:
                cache = all_data[tk]._day_rows_cache
                if w_idx < len(cache):
                    ticker_data[tk] = cache.iloc[w_idx]
            if ticker_data:
                _run_window(
                    window_label=window_label,
                    date_str=date_str,
                    ticker_data=ticker_data,
                    rl_trader=rl_trader,
                    llm_trader=llm_trader,
                    rl_portfolio=rl_portfolio,
                    llm_portfolio=llm_portfolio,
                )
                # Avoid Gemini rate limits (free tier)
                if llm_trader:
                    time.sleep(6.0)

        # Settle at close (use last row of each ticker for close prices)
        close_prices = {}
        for tk in all_data:
            day_rows = all_data[tk].loc[day.strftime("%Y-%m-%d")]
            if isinstance(day_rows, pd.Series):
                close_prices[tk] = float(day_rows["Close"])
            else:
                close_prices[tk] = float(day_rows.iloc[-1]["Close"])

        _settle_day(date_str, close_prices, rl_portfolio, llm_portfolio, not rl_only)

    # Final summary
    final_prices = {
        tk: float(all_data[tk].iloc[-1]["Close"]) for tk in all_data
    }
    print(f"\n{'='*50}")
    print(f"🏁 Backtest Complete")
    print(f"   RL final equity:  ${rl_portfolio.get_equity(final_prices):,.2f}")
    if not rl_only:
        print(f"   LLM final equity: ${llm_portfolio.get_equity(final_prices):,.2f}")
    print(f"{'='*50}")


# ── Live mode ────────────────────────────────────────────────────────────────


def _run_live(tickers: list[str], rl_only: bool = False) -> None:
    """Run the scheduler in live mode, triggering at real EST windows."""
    tz = ZoneInfo(TIMEZONE)
    print(f"🔴 Live mode — windows: {TRADING_WINDOWS} EST  |  Tickers: {tickers}")

    # Load agents
    rl_trader = RLTrader.load(MODEL_DIR / "ppo_trader_v1")
    llm_trader = LLMTrader() if not rl_only else None

    init_db()
    rl_portfolio = Portfolio()
    llm_portfolio = Portfolio()

    def _window_job(window_label: str) -> None:
        now = datetime.now(tz)
        date_str = now.strftime("%Y-%m-%d")
        print(f"\n⏰ Window {window_label} triggered at {now.strftime('%H:%M:%S')}")

        ticker_data = {}
        for tk in tickers:
            try:
                live_df = fetch_live(tk)
                live_df = compute_indicators(live_df).dropna()
                if len(live_df) > 0:
                    ticker_data[tk] = live_df.iloc[-1]
            except Exception as e:
                print(f"  ⚠️  Failed to fetch {tk}: {e}")

        if ticker_data:
            _run_window(
                window_label, date_str, ticker_data,
                rl_trader, llm_trader, rl_portfolio, llm_portfolio,
            )

    def _close_job() -> None:
        now = datetime.now(tz)
        date_str = now.strftime("%Y-%m-%d")
        print(f"\n🔔 Market close at {now.strftime('%H:%M:%S')}")
        close_prices = {}
        for tk in tickers:
            try:
                live_df = fetch_live(tk)
                if len(live_df) > 0:
                    close_prices[tk] = float(live_df.iloc[-1]["Close"])
            except Exception as e:
                print(f"  ⚠️  Failed to fetch close for {tk}: {e}")
        if close_prices:
            _settle_day(date_str, close_prices, rl_portfolio, llm_portfolio, not rl_only)

    # Schedule windows
    for w in TRADING_WINDOWS:
        schedule.every().day.at(w, TIMEZONE).do(_window_job, window_label=w)
    schedule.every().day.at(MARKET_CLOSE, TIMEZONE).do(_close_job)

    print("⏳ Scheduler running. Press Ctrl+C to stop.\n")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n🛑 Scheduler stopped.")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="RL vs LLM Trading Scheduler")
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    parser.add_argument("--start", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument(
        "--tickers", nargs="+", default=TICKERS, help="Tickers to trade"
    )
    parser.add_argument(
        "--rl-only", action="store_true", help="Run RL agent only (skip LLM)"
    )
    args = parser.parse_args()

    if args.backtest:
        if not args.start or not args.end:
            parser.error("--backtest requires --start and --end")
        _run_backtest(args.tickers, args.start, args.end, rl_only=args.rl_only)
    else:
        _run_live(args.tickers, rl_only=args.rl_only)


if __name__ == "__main__":
    main()
