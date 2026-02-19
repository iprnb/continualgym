"""
Utilities for continual RL evaluation.

Includes:
- RunningStats        : online mean/std/min/max tracker
- EpisodeLogger       : logs pseudo-episode returns and lengths
- DriftTracker        : records drift snapshot over time
- plasticity_score    : measures how quickly an agent recovers after a change
- forgetting_score    : measures performance loss on earlier tasks
- plot_drift_overlay  : matplotlib helper (optional dep)
"""

from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Online statistics
# ---------------------------------------------------------------------------

class RunningStats:
    """
    Welford's online algorithm for mean and variance.

    Tracks min, max, count, mean, and population std in O(1) space.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self._M2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._M2 += delta * delta2
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self._M2 / (self.count - 1))

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }

    def __repr__(self) -> str:
        return (
            f"RunningStats({self.name}: n={self.count}, "
            f"mean={self.mean:.4f}±{self.std:.4f}, "
            f"range=[{self.min:.4f}, {self.max:.4f}])"
        )


# ---------------------------------------------------------------------------
# Episode logger
# ---------------------------------------------------------------------------

class EpisodeLogger:
    """
    Accumulates rewards across steps and logs pseudo-episode statistics.

    Works with ContinuingWrapper's info dict to detect pseudo-resets.

    Example
    -------
    >>> logger = EpisodeLogger(window=100)
    >>> obs, info = env.reset()
    >>> for _ in range(10_000):
    ...     obs, rew, done, trunc, info = env.step(env.action_space.sample())
    ...     logger.step(rew, info)
    >>> print(logger.summary())
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._current_return = 0.0
        self._current_length = 0
        self._returns: collections.deque = collections.deque(maxlen=window)
        self._lengths: collections.deque = collections.deque(maxlen=window)
        self._global_t_at_reset: List[int] = []
        self._total_steps = 0
        self._pseudo_episodes = 0

    def step(self, reward: float, info: Dict[str, Any]) -> None:
        self._current_return += reward
        self._current_length += 1
        self._total_steps += 1

        if info.get("pseudo_reset_occurred"):
            self._returns.append(self._current_return)
            self._lengths.append(self._current_length)
            self._global_t_at_reset.append(info.get("global_t", self._total_steps))
            self._pseudo_episodes += 1
            self._current_return = 0.0
            self._current_length = 0

    @property
    def recent_mean_return(self) -> float:
        if not self._returns:
            return float("nan")
        return float(np.mean(self._returns))

    @property
    def global_t_at_reset(self) -> float:
        return self._global_t_at_reset
    
    @property
    def returns(self) -> collections.deque:
        return self._returns

    @property
    def recent_std_return(self) -> float:
        if len(self._returns) < 2:
            return float("nan")
        return float(np.std(self._returns))

    def summary(self) -> Dict[str, Any]:
        return {
            "total_steps": self._total_steps,
            "pseudo_episodes": self._pseudo_episodes,
            "mean_return_last_N": self.recent_mean_return,
            "std_return_last_N": self.recent_std_return,
            "mean_length_last_N": float(np.mean(self._lengths)) if self._lengths else float("nan"),
            "window": self.window,
        }


# ---------------------------------------------------------------------------
# Drift tracker
# ---------------------------------------------------------------------------

class DriftTracker:
    """
    Records drift snapshots from info['nonstationarity'] for later analysis.

    Example
    -------
    >>> tracker = DriftTracker(record_every=1000)
    >>> for t in range(100_000):
    ...     obs, rew, done, trunc, info = env.step(env.action_space.sample())
    ...     tracker.step(t, info)
    >>> df = tracker.to_dataframe()  # requires pandas
    """

    def __init__(self, record_every: int = 1000):
        self.record_every = record_every
        self._records: List[Dict] = []

    def step(self, t: int, info: Dict[str, Any]) -> None:
        if t % self.record_every == 0:
            drift = info.get("nonstationarity", {})
            self._records.append({"t": t, **drift})

    @property
    def records(self) -> List[Dict]:
        return self._records

    def to_numpy(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (timesteps, values) arrays for a given drift key."""
        ts = np.array([r["t"] for r in self._records if key in r])
        vs = np.array([r[key] for r in self._records if key in r])
        return ts, vs

    def to_dataframe(self):
        """Convert records to a pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
            return pd.DataFrame(self._records)
        except ImportError as e:
            raise ImportError("pandas is required for to_dataframe()") from e


# ---------------------------------------------------------------------------
# Evaluation metrics for continual RL
# ---------------------------------------------------------------------------

def plasticity_score(
    returns_before: List[float],
    returns_after: List[float],
    n_recovery_episodes: int = 10,
) -> float:
    """
    Measure how quickly performance recovers after a distribution shift.

    Plasticity score = (mean_after - mean_before) / (|mean_before| + 1e-8)

    A positive value means the agent adapted upward; negative means it regressed.

    Parameters
    ----------
    returns_before : list[float]
        Episode returns from the N episodes just before the shift.
    returns_after : list[float]
        Episode returns from the N episodes just after the shift.
    n_recovery_episodes : int
        Number of episodes used for both windows (for documentation).

    Returns
    -------
    float
        Plasticity score.
    """
    before = np.mean(returns_before) if returns_before else 0.0
    after = np.mean(returns_after) if returns_after else 0.0
    return float((after - before) / (abs(before) + 1e-8))


def forward_transfer(
    returns_on_new_task: List[float],
    returns_random_baseline: List[float],
) -> float:
    """
    Measure whether experience on old tasks helps on new tasks.

    FT = (agent_mean - random_mean) / (|random_mean| + 1e-8)

    Parameters
    ----------
    returns_on_new_task : list[float]
        Agent's first-N-episode returns after task switch.
    returns_random_baseline : list[float]
        Random agent's returns on the new task.
    """
    agent = np.mean(returns_on_new_task) if returns_on_new_task else 0.0
    baseline = np.mean(returns_random_baseline) if returns_random_baseline else 0.0
    return float((agent - baseline) / (abs(baseline) + 1e-8))


def forgetting_score(
    peak_return: float,
    final_return: float,
) -> float:
    """
    Measure catastrophic forgetting.

    Forgetting = (peak - final) / (|peak| + 1e-8)

    A score near 0 means no forgetting; a large positive score means the
    agent has lost a lot of what it learned.
    """
    return float((peak_return - final_return) / (abs(peak_return) + 1e-8))


# ---------------------------------------------------------------------------
# Optional matplotlib helper
# ---------------------------------------------------------------------------

def plot_drift_overlay(
    tracker: DriftTracker,
    logger: Optional[EpisodeLogger] = None,
    keys: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Drift & Performance Over Time",
) -> Any:
    """
    Plot drift schedules and (optionally) learning curve in one figure.

    Requires matplotlib.

    Parameters
    ----------
    tracker : DriftTracker
    logger  : EpisodeLogger (optional, plots rolling return as background)
    keys    : drift keys to plot (default: all available)
    figsize : figure size
    title   : plot title

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError as e:
        raise ImportError("matplotlib is required for plot_drift_overlay()") from e

    records = tracker.records
    if not records:
        raise ValueError("DriftTracker has no records.")

    all_keys = [k for k in records[0].keys() if k != "t"]
    if keys is not None:
        all_keys = [k for k in keys if k in all_keys]

    n_panels = len(all_keys) + (1 if logger else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    row = 0
    if logger and logger.global_t_at_reset:
        ts = np.array(logger.global_t_at_reset)
        rets = np.array(list(logger.returns)[-len(ts):])
        axes[row].plot(ts, rets, color="steelblue", lw=1.5, label="Episode Return")
        axes[row].set_ylabel("Return")
        axes[row].legend(loc="upper right")
        axes[row].grid(True, alpha=0.3)
        row += 1

    for key in all_keys:
        ts, vs = tracker.to_numpy(key)
        axes[row].plot(ts, vs, lw=1.5, label=key)
        axes[row].set_ylabel(key, fontsize=9)
        axes[row].legend(loc="upper right")
        axes[row].grid(True, alpha=0.3)
        row += 1

    axes[-1].set_xlabel("Global Step")
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig
