"""Training script for multiple RL strategies (Standard, Dip Buyer, Momentum)."""

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.rl_agent import train
from config import TICKERS
from data.pipeline import compute_indicators, fetch_historical
from env.trade_gym import TradeGym

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduced steps for faster "Battle Royale" training
STRATEGY_TIMESTEPS = 200_000


def main():
    logger.info("⚔️  Starting Strategy Battle Royale Training...")
    
    # 1. Fetch Data (Once for all strategies)
    ticker_data = {}
    for ticker in TICKERS:
        try:
            # Using 5y history
            df = fetch_historical(ticker, period="5y")
            df = compute_indicators(df).dropna()
            if len(df) > 100:
                ticker_data[ticker] = df
        except Exception as e:
            logger.error("Failed to fetch %s: %s", ticker, e)

    if not ticker_data:
        logger.error("❌ No valid data found. Aborting.")
        return

    logger.info("✅ Loaded data for %d tickers.", len(ticker_data))

    # 2. Train Each Strategy
    strategies = ["standard", "dip_buyer", "momentum"]
    
    for strat in strategies:
        logger.info(f"\n🧠 Training [{strat.upper()}] agent ({STRATEGY_TIMESTEPS} steps)...")
        
        env = TradeGym(
            df=ticker_data,
            initial_cash=10_000.0,
            trade_fraction=0.20,
            windows_per_day=3,
            strategy=strat,
        )

        model = train(
            env,
            total_timesteps=STRATEGY_TIMESTEPS,
            save_path=f"models/ppo_{strat}",
            verbose=0, # Less noise
        )
        
        logger.info(f"✅ Saved models/ppo_{strat}.zip")

    logger.info("\n🏆 All strategies trained! Ready for battle.")


if __name__ == "__main__":
    main()
