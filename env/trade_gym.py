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
        df: pd.DataFrame,
        initial_cash: float = 10_000.0,
        trade_fraction: float = 0.20,
        windows_per_day: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data **with indicators already computed** (via
            ``compute_indicators``).  Rows with NaN in any indicator column
            are automatically dropped.
        initial_cash : float
            Starting portfolio cash.
        trade_fraction : float
            Fraction of available cash (buy) / holdings (sell) per trade.
        windows_per_day : int
            Number of decision steps per episode (default 3).
        """
        super().__init__()

        # Clean data — drop warmup NaN rows
        self._full_df = df.dropna(subset=INDICATOR_COLS).reset_index(drop=True)
        assert len(self._full_df) >= windows_per_day, (
            f"Need at least {windows_per_day} rows after dropping NaN, "
            f"got {len(self._full_df)}"
        )

        self._initial_cash = initial_cash
        self._trade_fraction = trade_fraction
        self._windows_per_day = windows_per_day

        # Spaces
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features(),), dtype=np.float32,
        )

        # Rolling volatility (for reward scaling) — 20-period std of log returns
        log_ret = np.log(self._full_df["Close"] / self._full_df["Close"].shift(1))
        self._volatility = log_ret.rolling(20).std().fillna(log_ret.std()).values

        # State variables (initialised in reset)
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

        self._cash = self._initial_cash
        self._shares = 0.0
        self._avg_cost = 0.0
        self._step_in_day = 0

        # Pick a random starting day (must leave room for windows_per_day rows)
        max_start = len(self._full_df) - self._windows_per_day
        self._day_start_idx = self.np_random.integers(0, max_start + 1)
        self._global_idx = self._day_start_idx

        self._prev_equity = self._equity()

        return self._obs(), {}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        price = self._current_price()

        # Execute trade
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
