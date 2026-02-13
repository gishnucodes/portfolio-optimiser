#!/usr/bin/env python
"""CLI script to train the PPO RL agent on historical data.

Usage
-----
    python scripts/train_rl.py --tickers AAPL MSFT --period 5y --timesteps 500000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    INITIAL_CASH,
    MODEL_DIR,
    RL_BATCH_SIZE,
    RL_LEARNING_RATE,
    RL_N_EPOCHS,
    RL_N_STEPS,
    RL_TOTAL_TIMESTEPS,
    TICKERS,
    TRADE_FRACTION,
)
from data.pipeline import compute_indicators, fetch_historical
from env.trade_gym import TradeGym
from agents.rl_agent import train


def _build_combined_df(tickers: list[str], period: str) -> pd.DataFrame:
    """Fetch & combine indicator-enriched data for multiple tickers.

    Each ticker's data is stacked sequentially (not interleaved) so the
    environment can step through contiguous price histories.
    """
    frames: list[pd.DataFrame] = []
    for tk in tickers:
        print(f"📥  Fetching {tk} ({period}) …")
        df = fetch_historical(tk, period=period)
        df = compute_indicators(df)
        df = df.dropna()
        df["ticker"] = tk
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"✅  Combined dataset: {len(combined):,} rows across {len(tickers)} tickers")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PPO trading agent")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=TICKERS,
        help="Ticker symbols to train on (default: config.TICKERS)",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="yfinance period string, e.g. 1y, 5y (default: 5y)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=RL_TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {RL_TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(MODEL_DIR / "ppo_trader_v1"),
        help="Save path for the trained model (without .zip)",
    )
    args = parser.parse_args()

    # 1. Build dataset
    df = _build_combined_df(args.tickers, args.period)

    # 2. Create environment
    env = TradeGym(
        df=df,
        initial_cash=INITIAL_CASH,
        trade_fraction=TRADE_FRACTION,
    )

    # 3. Train
    print(f"\n🚀  Training PPO for {args.timesteps:,} timesteps …\n")
    train(
        env=env,
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        learning_rate=RL_LEARNING_RATE,
        n_steps=RL_N_STEPS,
        batch_size=RL_BATCH_SIZE,
        n_epochs=RL_N_EPOCHS,
    )

    print("\n🏁  Training complete!")


if __name__ == "__main__":
    main()
