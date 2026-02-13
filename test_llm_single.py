"""Test LLM agent with a single call to debug reasoning (Groq)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.llm_agent import LLMTrader

def test_single_call():
    print("🤖 Initialising LLM Trader (Groq)...")
    try:
        trader = LLMTrader()
    except Exception as e:
        print(f"❌ Failed to init: {e}")
        return

    # Dummy state for AAPL
    state = {
        "ticker": "AAPL",
        "price": 150.0,
        "volume": 50_000_000,
        "rsi": 30.0,          # Oversold
        "macd": -2.0,
        "macd_signal": -2.5,  # MACD > Signal (Bullish crossover)
        "macd_hist": 0.5,
        "ema20": 145.0,
        "ema50": 140.0,
        "cash": 10_000.0,
        "holdings": 0.0,
        "unrealised_pnl": 0.0,
    }

    print("\n📩 Sending prompt (Oversold RSI=30, Bullish MACD)...")
    result = trader.predict(state)
    
    print("\n📝 Result:")
    print(f"Action: {result['action']}")
    print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    test_single_call()
