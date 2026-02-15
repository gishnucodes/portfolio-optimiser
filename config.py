"""Central configuration for the portfolio-optimiser project."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
DB_PATH = ROOT_DIR / "daily_ledger.db"

# ── Live Experiments ────────────────────────────────────────────────────────

ACTIVE_EXPERIMENTS = {
    "live_momentum_v1": {
        "model_path": "models/ppo_momentum",
        "strategy": "momentum",
        "type": "rl",
        "enabled": True,
    },
    "live_dip_buyer_v1": {
        "model_path": "models/ppo_dip_buyer",
        "strategy": "dip_buyer",
        "type": "rl",
        "enabled": True,
    },
    "live_standard_v1": {
        "model_path": "models/ppo_standard",
        "strategy": "standard",
        "type": "rl",
        "enabled": True,
    },
}

# ── Ticker universe ─────────────────────────────────────────────────────────
TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "UNH",
]

# ── Trading schedule (EST) ──────────────────────────────────────────────────
TRADING_WINDOWS: list[str] = ["10:00", "13:00", "15:00"]
MARKET_CLOSE: str = "16:00"
TIMEZONE: str = "US/Eastern"

# ── Portfolio defaults ───────────────────────────────────────────────────────
INITIAL_CASH: float = 10_000.0
TRADE_FRACTION: float = 0.20  # fraction of cash/holdings per trade

# ── RL hyper-parameters ─────────────────────────────────────────────────────
RL_LEARNING_RATE: float = 3e-4
RL_N_STEPS: int = 2048
RL_BATCH_SIZE: int = 64
RL_N_EPOCHS: int = 10
RL_TOTAL_TIMESTEPS: int = 1_000_000

# ── LLM (Groq) ───────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# ── Benchmark ────────────────────────────────────────────────────────────────
BENCHMARK_TICKER: str = "^GSPC"  # S&P 500
