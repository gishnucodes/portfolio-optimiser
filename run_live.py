"""Live execution script for active experiments."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.rl_agent import RLTrader
from config import ACTIVE_EXPERIMENTS, MODEL_DIR, ROOT_DIR, TICKERS, TRADING_WINDOWS, INITIAL_CASH
from data.pipeline import build_state_vector, compute_indicators, fetch_historical
from engine.ledger import log_decision, log_daily_summary
from engine.portfolio import Portfolio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORTFOLIO_STATE_FILE = ROOT_DIR / "live_portfolios.json"

# ── Shared Strategy Logic ───────────────────────────────────────────────────

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
            rsi = indicators.get("RSI_14", 50)
            if rsi > 30:
                return "HOLD"
    
    elif strategy == "momentum":
        # Rule: Only buy if Price > EMA_50
        ema50 = indicators.get("EMA_50", 0)
        if action == "BUY" and price < ema50:
            return "HOLD"
             
    return action

# ── State Management ────────────────────────────────────────────────────────

def load_portfolios() -> dict:
    if PORTFOLIO_STATE_FILE.exists():
        try:
            with open(PORTFOLIO_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    return {}

def save_portfolios(state: dict):
    with open(PORTFOLIO_STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def restore_portfolio(state: dict) -> Portfolio:
    p = Portfolio(initial_cash=state.get("cash", INITIAL_CASH))
    # inject private attributes to restore state
    p.cash = state.get("cash", INITIAL_CASH)
    # Reconstruct positions
    # Portfolio uses _positions: dict[str, _Position]
    # We need to hack it back in or add a robust method.
    # Hack: Access private member for now.
    raw_pos = state.get("positions", {})
    from engine.portfolio import _Position
    p._positions = {
        k: _Position(shares=v["shares"], avg_cost=v["avg_cost"])
        for k, v in raw_pos.items()
        if v["shares"] > 0
    }
    return p

def serialize_portfolio(p: Portfolio) -> dict:
    return {
        "cash": p.cash,
        "positions": {
            k: {"shares": pos.shares, "avg_cost": pos.avg_cost}
            for k, pos in p._positions.items()
            if pos.shares > 0
        }
    }

# ── Live Runner ──────────────────────────────────────────────────────────────

def run_experiments():
    logger.info(f"🧪 Starting Live Run [{datetime.now()}]...")
    
    # 1. Load State
    state = load_portfolios()
    
    # 2. Fetch Data (Live/Recent)
    data_cache = {}
    for ticker in TICKERS:
        try:
            # Need enough history for indicators (200 days)
            df = fetch_historical(ticker, period="1y")
            df = compute_indicators(df).dropna()
            if not df.empty:
                # Use the very last row as "Live" data
                latest_date = df.index[-1].strftime("%Y-%m-%d")
                
                # Check if today matches data? (Market hours mismatch possible)
                # We assume fetch_historical gets up-to-date data.
                data_cache[ticker] = {
                    "row": df.iloc[-1],
                    "price": df.iloc[-1]["Close"],
                    "date": latest_date,
                    "indicators": {
                        "RSI_14": df.iloc[-1]["RSI_14"],
                        "EMA_50": df.iloc[-1]["EMA_50"],
                    }
                }
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")

    if not data_cache:
        logger.error("❌ No data available. Aborting.")
        return

    # Check date consistency (warn if stale)
    sample_date = next(iter(data_cache.values()))["date"]
    logger.info(f"📅 Market Data Date: {sample_date}")

    # 3. Iterate Experiments
    for exp_name, config in ACTIVE_EXPERIMENTS.items():
        if not config.get("enabled", False):
            continue
            
        logger.info(f"👉 Running {exp_name}...")
        
        # Load Model
        try:
            model_path = config['model_path']
            if not model_path.endswith('.zip'): model_path += ".zip"
            model_path = ROOT_DIR / model_path
            
            agent = RLTrader.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            continue

        # Restore Portfolio
        p_state = state.get(exp_name, {"cash": INITIAL_CASH, "positions": {}})
        port = restore_portfolio(p_state)
        
        # Trade Loop
        for window in TRADING_WINDOWS:
            for ticker in TICKERS:
                if ticker not in data_cache: continue
                
                d = data_cache[ticker]
                row = d["row"]
                price = d["price"]
                
                # Build State
                prices = {t: data_cache[t]["price"] for t in data_cache}
                obs = build_state_vector(
                    row,
                    cash=port.cash,
                    holdings=port.get_holdings_value(prices),
                    unrealized_pnl=port.get_unrealised_pnl(prices),
                )
                
                # Predict
                action_id, conf = agent.predict(obs)
                raw_action = {0: "HOLD", 1: "BUY", 2: "SELL"}[action_id]
                
                # Apply Constraints
                final_action = apply_strategy_constraints(
                    config["strategy"], raw_action, price, port.cash, d["indicators"]
                )
                
                # Execute
                receipt = port.execute_trade(ticker, final_action, price)
                
                # Log Decision (if actionable)
                # Logging HOLDs adds noise, maybe skip? No, keep logic consistent.
                # Only log unique HOLDs? No, DB handles it.
                log_decision(
                    date=d["date"], # Use market date
                    window=window,
                    ticker=ticker,
                    agent=exp_name,
                    action=final_action,
                    confidence=conf,
                    reasoning=f"Strategy: {config['strategy']} | Mode: Live",
                    price=price,
                    mode="live"
                )

        # Update State & Log Summary
        prices = {t: data_cache[t]["price"] for t in data_cache}
        equity = port.get_equity(prices)
        
        log_daily_summary(
            date=sample_date,
            agent=exp_name,
            equity=equity,
            cash=port.cash,
            pnl=equity - INITIAL_CASH,
            mode="live"
        )
        
        # Save State
        state[exp_name] = serialize_portfolio(port)
        save_portfolios(state)
        
        logger.info(f"   Equity: ${equity:,.2f} | Cash: ${port.cash:,.2f}")

    logger.info("✅ Live Run Complete.")

if __name__ == "__main__":
    run_experiments()
