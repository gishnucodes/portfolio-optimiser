"""Backtest script for comparing multiple RL strategies."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.rl_agent import RLTrader
from config import INITIAL_CASH, MODEL_DIR, TICKERS, TRADE_FRACTION, TRADING_WINDOWS
from data.pipeline import build_state_vector, compute_indicators, fetch_historical
from engine.ledger import init_db, log_daily_summary, log_decision
from engine.portfolio import Portfolio

# ── Strategy Logic ───────────────────────────────────────────────────────────

def apply_strategy_constraints(
    strategy: str,
    action: str,
    price: float,
    cash: float,
    indicators: dict[str, float]
) -> str:
    """Enforce strategy-specific rules at inference time."""
    
    if strategy == "dip_buyer":
        # Rule: Maintain $2,000 cash reserve unless RSI < 30
        if action == "BUY" and cash < 2000.0:
            if indicators["RSI_14"] > 30:
                return "HOLD"
    
    elif strategy == "momentum":
        # Rule: Only buy if Price > EMA_50
        if action == "BUY" and price < indicators["EMA_50"]:
            return "HOLD"
        # Rule: Sell if Price < EMA_50 (Stop Loss)
        if action == "HOLD" and price < indicators["EMA_50"]:
             # Optional: Force sell? For now, let's just block buys to be safe.
             # Strict momentum would sell here.
             return "SELL"
             
    return action


# ── Backtest Loop ────────────────────────────────────────────────────────────

def run_battle(start: str, end: str):
    print(f"⚔️  Running Strategy Battle: {start} -> {end}")
    
    # 1. Load Models
    models = {}
    strategies = ["standard", "dip_buyer", "momentum"]
    
    for s in strategies:
        try:
            path = MODEL_DIR / f"ppo_{s}"
            models[s] = RLTrader.load(path)
            print(f"✅ Loaded {s}")
        except Exception:
            print(f"⚠️  Could not load {s}, skipping.")
    
    if not models:
        print("❌ No models found. Run training first!")
        return

    # 2. Setup Portfolios & DB
    # We use a fresh DB for the battle to avoid mixing with main logs?
    # Or just use distinct agent names.
    # Let's use the main DB but agent names like 'ppo_standard'
    init_db()
    
    portfolios = {s: Portfolio() for s in models}
    
    # 3. Fetch Data
    all_data = {}
    for tk in TICKERS:
        df = fetch_historical(tk, period="2y") # ample buffer
        df = compute_indicators(df).dropna()
        df = df.loc[start:end]
        if not df.empty:
            all_data[tk] = df
            
    trading_days = sorted(
        set(d for df in all_data.values() for d in df.index.normalize().unique())
    )
    
    print(f"🗓️  {len(trading_days)} trading days")

    # 4. Simulation
    for day in trading_days:
        date_str = day.strftime("%Y-%m-%d")
        
        # Intra-day windows
        for window in TRADING_WINDOWS:
            for s_name, model in models.items():
                port = portfolios[s_name]
                
                for ticker, df in all_data.items():
                    if date_str not in df.index:
                        continue
                    
                    # Get row (resample logic omitted for brevity, taking last close of day 
                    # as approximation for all windows is inaccurate but consistent for comparison.
                    # Ideally we fetch intraday, but we strictly use daily bars in pipeline.
                    # So we use the SAME daily bar for all 3 windows?
                    # Yes, standard backtest limitation unless we have intraday data.
                    # compute_indicators uses daily close.
                    
                    row = df.loc[date_str]
                    # If multiple rows? (shouldn't be with daily index)
                    if isinstance(row, pd.DataFrame): row = row.iloc[0]
                    
                    prices = {t: all_data[t].loc[date_str]["Close"] if date_str in all_data[t].index else 0.0 for t in all_data}
                    
                    # Build State
                    obs = build_state_vector(
                        row,
                        cash=port.cash,
                        holdings=port.get_holdings_value(prices), # approx
                        unrealized_pnl=port.get_unrealised_pnl(prices),
                    )
                    
                    # Predict
                    action_id, conf = model.predict(obs)
                    raw_action = {0: "HOLD", 1: "BUY", 2: "SELL"}[action_id]
                    
                    # Apply Constraints
                    indicators = {
                        "RSI_14": row["RSI_14"],
                        "EMA_50": row["EMA_50"],
                    }
                    final_action = apply_strategy_constraints(
                        s_name, raw_action, row["Close"], port.cash, indicators
                    )
                    
                    # Execute
                    port.execute_trade(ticker, final_action, row["Close"])
                    
                    # Log
                    log_decision(
                        date=date_str,
                        window=window,
                        ticker=ticker,
                        agent=f"ppo_{s_name}",
                        action=final_action,
                        confidence=conf,
                        reasoning=f"Strategy: {s_name} | Raw: {raw_action}",
                        price=row["Close"]
                    )

        # End of Day Settlement
        for s_name, port in portfolios.items():
            # Get close prices
            close_prices = {
                t: all_data[t].loc[date_str]["Close"] 
                for t in all_data 
                if date_str in all_data[t].index
            }
            equity = port.get_equity(close_prices)
            log_daily_summary(
                date=date_str,
                agent=f"ppo_{s_name}",
                equity=equity,
                cash=port.cash,
                pnl=equity - INITIAL_CASH
            )
            
    # Final Results
    print("\n🏁 Battle Royale Results:")
    for s_name, port in portfolios.items():
        if all_data:
             last_prices = {t: all_data[t].iloc[-1]["Close"] for t in all_data}
             eq = port.get_equity(last_prices)
             print(f"  {s_name:<15}: ${eq:,.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-02-14")
    args = parser.parse_args()
    
    run_battle(args.start, args.end)
