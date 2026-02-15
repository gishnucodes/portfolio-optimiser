"""Training script for the RL agent using 5 years of historical data."""

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.rl_agent import train
from config import RL_TOTAL_TIMESTEPS, TICKERS
from data.pipeline import compute_indicators, fetch_historical
from env.trade_gym import TradeGym

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("🚀 Starting PPO Training on %d tickers...", len(TICKERS))
    
    # 1. Fetch Data
    ticker_data = {}
    for ticker in TICKERS:
        logger.info("📥 Fetching 5y history for %s...", ticker)
        try:
            df = fetch_historical(ticker, period="5y")
            df = compute_indicators(df).dropna()
            
            if len(df) > 100:  # profound sanity check
                ticker_data[ticker] = df
            else:
                logger.warning("Short data for %s (len=%d), skipping.", ticker, len(df))
        except Exception as e:
            logger.error("Failed to fetch %s: %s", ticker, e)

    if not ticker_data:
        logger.error("❌ No valid data found. Aborting.")
        return

    logger.info("✅ Loaded data for %d tickers.", len(ticker_data))

    # 2. Init Environment
    env = TradeGym(
        df=ticker_data,
        initial_cash=10_000.0,
        trade_fraction=0.20,
        windows_per_day=3,
    )

    # 3. Train
    logger.info("🧠 Training PPO agent for %d timesteps...", RL_TOTAL_TIMESTEPS)
    model = train(
        env,
        total_timesteps=RL_TOTAL_TIMESTEPS,
        save_path="models/ppo_trader_v1",
        verbose=1,
    )
    
    logger.info("🎉 Training complete! Model saved to models/ppo_trader_v1.zip")


if __name__ == "__main__":
    main()
