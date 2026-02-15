"""TradeGym — custom Gymnasium environment for the 3-window trading task."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from data.pipeline import INDICATOR_COLS, compute_indicators, n_features


class TradeGym(gym.Env):
    """A single-ticker, single-day trading environment.

    Each *episode* represents one trading day with exactly 3 decision windows.
    The agent chooses Hold (0), Buy (1), or Sell (2) at each window.

    Observations
    ------------
    A flat float32 vector containing technical indicators plus portfolio state
    (cash, holdings value, unrealised PnL).

    Rewards
    -------
    Log return of portfolio equity between steps, scaled by inverse rolling
    volatility so that calmer-market gains are weighted more than volatile
    swings.
    """

    metadata = {"render_modes": []}

    # Actions
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        df: pd.DataFrame | dict[str, pd.DataFrame],
        initial_cash: float = 10_000.0,
        trade_fraction: float = 0.20,
        windows_per_day: int = 3,
        strategy: str = "standard",
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame | dict[str, pd.DataFrame]
            OHLCV data with indicators. Can be a single DataFrame or a dict
            mapping ticker -> DataFrame.
        """
        super().__init__()

        self._dfs: list[pd.DataFrame] = []
        self._vols: list[np.ndarray] = []

        # Normalize input to a list of DataFrames
        input_dfs = df if isinstance(df, dict) else {"default": df}

        for _, d in input_dfs.items():
            # Clean data
            clean_df = d.dropna(subset=INDICATOR_COLS).reset_index(drop=True)
            if len(clean_df) >= windows_per_day:
                self._dfs.append(clean_df)
                
                # Pre-compute volatility
                log_ret = np.log(clean_df["Close"] / clean_df["Close"].shift(1))
                vol = log_ret.rolling(20).std().fillna(log_ret.std()).values
                self._vols.append(vol)

        assert len(self._dfs) > 0, "No valid DataFrames provided (check lengths)"

        self._initial_cash = initial_cash
        self._trade_fraction = trade_fraction
        self._windows_per_day = windows_per_day
        self._strategy = strategy
        
        # Current episode state
        self._full_df: pd.DataFrame = self._dfs[0]
        self._volatility: np.ndarray = self._vols[0]

        # Spaces
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features(),), dtype=np.float32,
        )

        # State variables
        self._cash: float = 0.0
        self._shares: float = 0.0
        self._avg_cost: float = 0.0
        self._step_in_day: int = 0
        self._global_idx: int = 0
        self._day_start_idx: int = 0
        self._prev_equity: float = 0.0

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Pick a random ticker/dataframe for this episode
        df_idx = self.np_random.integers(0, len(self._dfs))
        self._full_df = self._dfs[df_idx]
        self._volatility = self._vols[df_idx]

        self._cash = self._initial_cash
        self._shares = 0.0
        self._avg_cost = 0.0
        self._step_in_day = 0

        # Pick a random starting day (must leave room for windows_per_day rows)
        max_start = len(self._full_df) - self._windows_per_day
        self._day_start_idx = self.np_random.integers(0, max_start + 1)
        self._global_idx = self._day_start_idx

        self._prev_equity = self._equity()

        if options and "strategy" in options:
            self._strategy = options["strategy"]

        return self._obs(), {}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        price = self._current_price()
        
        # ── Strategy Overrides ───────────────────────────────────────────────
        if self._strategy == "dip_buyer":
            # If trying to BUY, check constraints
            if action == self.BUY:
                # Maintain ~20% cash reserve (assuming $10k initial) unless RSI < 30
                # Using hardcoded $2000 for simplicity as requested
                if self._cash < 2000.0:
                    # Check RSI (index 2 in INDICATOR_COLS list based on pipeline.py)
                    # "Close", "Volume", "RSI_14"... so index 2
                    rsi = self._full_df.iloc[self._global_idx]["RSI_14"]
                    if rsi > 30:
                        action = self.HOLD  # Block buy

        elif self._strategy == "momentum":
            # If trying to BUY, ensure trend is up (Price > EMA_50)
            if action == self.BUY:
                 ema50 = self._full_df.iloc[self._global_idx]["EMA_50"]
                 if price < ema50:
                     action = self.HOLD # Block buy against trend

        # ── Execution ────────────────────────────────────────────────────────
        if action == self.BUY:
            invest = self._cash * self._trade_fraction
            if invest > 0 and price > 0:
                bought = invest / price
                self._shares += bought
                self._avg_cost = (
                    (self._avg_cost * (self._shares - bought) + invest)
                    / self._shares
                    if self._shares > 0
                    else price
                )
                self._cash -= invest

        elif action == self.SELL:
            sell_shares = self._shares * self._trade_fraction
            if sell_shares > 0:
                self._cash += sell_shares * price
                self._shares -= sell_shares

        # Advance step
        self._step_in_day += 1
        self._global_idx = self._day_start_idx + self._step_in_day

        # Clamp global index to data bounds
        self._global_idx = min(self._global_idx, len(self._full_df) - 1)

        # Reward: risk-adjusted log return
        new_equity = self._equity()
        reward = self._compute_reward(new_equity)
        self._prev_equity = new_equity

        terminated = self._step_in_day >= self._windows_per_day
        truncated = False

        info = {
            "equity": new_equity,
            "cash": self._cash,
            "shares": self._shares,
            "price": self._current_price(),
            "step_in_day": self._step_in_day,
            "strategy": self._strategy,
        }

        return self._obs(), reward, terminated, truncated, info

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _current_price(self) -> float:
        return float(self._full_df.iloc[self._global_idx]["Close"])

    def _equity(self) -> float:
        return self._cash + self._shares * self._current_price()

    def _unrealised_pnl(self) -> float:
        if self._shares == 0:
            return 0.0
        return self._shares * (self._current_price() - self._avg_cost)

    def _obs(self) -> np.ndarray:
        row = self._full_df.iloc[self._global_idx]
        market = row[INDICATOR_COLS].values.astype(np.float32)
        portfolio = np.array(
            [self._cash, self._shares * self._current_price(), self._unrealised_pnl()],
            dtype=np.float32,
        )
        return np.concatenate([market, portfolio])

    def _compute_reward(self, new_equity: float) -> float:
        """Log return of equity, divided by recent volatility."""
        if self._prev_equity <= 0:
            return 0.0
        log_ret = float(np.log(new_equity / self._prev_equity))

        vol = self._volatility[self._global_idx]
        if vol > 0:
            return log_ret / vol
        return log_ret
