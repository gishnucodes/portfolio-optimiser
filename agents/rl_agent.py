"""RL Agent — PPO-based trader using Stable Baselines3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env.trade_gym import TradeGym


# ── Training callback ───────────────────────────────────────────────────────


class _LogCallback(BaseCallback):
    """Lightweight callback that prints training progress every *n* steps."""

    def __init__(self, log_every: int = 10_000, verbose: int = 1) -> None:
        super().__init__(verbose)
        self._log_every = log_every

    def _on_step(self) -> bool:
        if self.n_calls % self._log_every == 0 and self.verbose:
            mean_rew = np.mean(
                [ep["r"] for ep in self.model.ep_info_buffer]
            ) if self.model.ep_info_buffer else float("nan")
            print(
                f"[PPO] step {self.n_calls:>8,}  |  "
                f"mean episode reward: {mean_rew:+.4f}"
            )
        return True


# ── Train function ───────────────────────────────────────────────────────────


def train(
    env: TradeGym,
    total_timesteps: int = 500_000,
    save_path: str | Path = "models/ppo_trader_v1",
    *,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    verbose: int = 1,
) -> PPO:
    """Train a PPO agent on the supplied TradeGym environment.

    Parameters
    ----------
    env : TradeGym
        The custom Gymnasium trading environment.
    total_timesteps : int
        Total training timesteps.
    save_path : str | Path
        Where to save the model (without extension; SB3 appends ``.zip``).
    learning_rate, n_steps, batch_size, n_epochs
        PPO hyper-parameters.
    verbose : int
        Verbosity level for SB3 logger.

    Returns
    -------
    PPO
        The trained model instance.
    """
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        verbose=verbose,
        tensorboard_log=None,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=_LogCallback(log_every=10_000, verbose=verbose),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"✅  Model saved to {save_path}.zip")

    return model


# ── Inference wrapper ────────────────────────────────────────────────────────


class RLTrader:
    """Thin wrapper around a saved PPO model for inference.

    Usage
    -----
    >>> trader = RLTrader.load("models/ppo_trader_v1")
    >>> action, confidence = trader.predict(obs)
    """

    def __init__(self, model: PPO) -> None:
        self._model = model

    @classmethod
    def load(cls, path: str | Path) -> "RLTrader":
        """Load a saved PPO model from *path* (without ``.zip`` extension)."""
        model = PPO.load(str(path))
        return cls(model)

    def predict(
        self, obs: np.ndarray, deterministic: bool = True,
    ) -> tuple[int, float]:
        """Return ``(action, confidence)`` for a single observation.

        *confidence* is the probability the model assigns to its chosen action.
        """
        action, _states = self._model.predict(obs, deterministic=deterministic)
        action = int(action)

        # Extract action probabilities for confidence score
        obs_tensor = self._model.policy.obs_to_tensor(obs)[0]
        dist = self._model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy().flatten()
        confidence = float(probs[action])

        return action, confidence
