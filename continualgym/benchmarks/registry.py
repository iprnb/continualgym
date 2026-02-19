"""
Pre-configured benchmark environments for continual RL research.

Each benchmark is a dict with:
  - 'id'          : string identifier
  - 'description' : what makes it interesting
  - 'factory'     : zero-arg callable returning a configured env
  - 'tags'        : list of characteristic tags

Use `make_benchmark(name)` to instantiate.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import gymnasium as gym
import numpy as np

from continualgym.drift import (
    LinearDrift,
    SinusoidalDrift,
    StepDrift,
    RandomWalkDrift,
    CompositeDrift,
    FunctionDrift,
)
from continualgym.wrappers.pipeline import make_continual


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

BenchmarkSpec = Dict  # {id, description, factory, tags}
_REGISTRY: Dict[str, BenchmarkSpec] = {}


def _register(spec: BenchmarkSpec) -> BenchmarkSpec:
    _REGISTRY[spec["id"]] = spec
    return spec


# ---------------------------------------------------------------------------
# CartPole benchmarks
# ---------------------------------------------------------------------------

_register({
    "id": "CartPole-Sinusoidal",
    "description": (
        "CartPole-v1 as a continuing task with sinusoidally drifting "
        "observation noise. Reward is shifted to remain centred at zero. "
        "No terminal signal — agent must learn to balance indefinitely."
    ),
    "factory": lambda: make_continual(
        "CartPole-v1",
        hide_done=True,
        pseudo_reset_penalty=-1.0,
        obs_noise_std_schedule=SinusoidalDrift(center=0.05, amplitude=0.04, period=30_000),
        reward_shift_schedule=SinusoidalDrift(center=0.0, amplitude=0.3, period=50_000),
    ),
    "tags": ["discrete-action", "sinusoidal", "obs-noise", "reward-drift"],
})

_register({
    "id": "CartPole-StepChanges",
    "description": (
        "CartPole-v1 with abrupt step-changes in observation noise std "
        "at 50k, 150k, 300k steps. Tests catastrophic forgetting / rapid adaptation."
    ),
    "factory": lambda: make_continual(
        "CartPole-v1",
        hide_done=True,
        pseudo_reset_penalty=-1.0,
        obs_noise_std_schedule=StepDrift(
            changepoints=[(50_000, 0.1), (150_000, 0.02), (300_000, 0.15)],
            initial=0.0,
        ),
    ),
    "tags": ["discrete-action", "step-changes", "non-markov"],
})

# ---------------------------------------------------------------------------
# MountainCar benchmarks
# ---------------------------------------------------------------------------

_register({
    "id": "MountainCarContinuous-LinearDrift",
    "description": (
        "MountainCarContinuous-v0 with linearly increasing action noise, "
        "making the task progressively harder. Reward is also slowly "
        "shifted downward to prevent reward hacking."
    ),
    "factory": lambda: make_continual(
        "MountainCarContinuous-v0",
        hide_done=True,
        action_noise_std_schedule=LinearDrift(start=0.0, end=0.5, duration=200_000),
        reward_shift_schedule=LinearDrift(start=0.0, end=-0.5, duration=200_000),
    ),
    "tags": ["continuous-action", "linear-drift", "action-noise"],
})

# ---------------------------------------------------------------------------
# Pendulum benchmarks
# ---------------------------------------------------------------------------

_register({
    "id": "Pendulum-RandomWalk",
    "description": (
        "Pendulum-v1 as a continuing task (no natural termination). "
        "Reward scale drifts as a slow random walk. "
        "Obs noise also follows a random walk. "
        "Tests slow, unpredictable non-stationarity."
    ),
    "factory": lambda: make_continual(
        "Pendulum-v1",
        hide_done=False,  # Pendulum doesn't naturally terminate
        reward_scale_schedule=RandomWalkDrift(lo=0.5, hi=1.5, step_size=0.0005, seed=42),
        obs_noise_std_schedule=RandomWalkDrift(lo=0.0, hi=0.1, step_size=0.0002, seed=99),
    ),
    "tags": ["continuous-action", "random-walk", "non-stationary-scale"],
})

_register({
    "id": "Pendulum-RewardSwitch",
    "description": (
        "Pendulum-v1 where the reward function switches every 100k steps "
        "between: (1) standard sparse-ish reward, (2) energy-minimising reward "
        "that penalises large actions, (3) velocity-maximising reward. "
        "Tests reward-function non-stationarity and goal change adaptation."
    ),
    "factory": lambda: make_continual(
        "Pendulum-v1",
        reward_functions=[
            # fn(obs, action, next_obs, reward, info) -> float
            lambda obs, a, nobs, r, i: r,                           # (0) standard
            lambda obs, a, nobs, r, i: r - 0.5 * float(np.sum(np.array(a)**2)),  # (1) energy
            lambda obs, a, nobs, r, i: float(obs[2]),               # (2) angular vel
        ],
        reward_fn_switch_schedule=StepDrift(
            changepoints=[(100_000, 1), (200_000, 2), (300_000, 0)],
            initial=0,
        ),
    ),
    "tags": ["continuous-action", "reward-switch", "goal-change"],
})

# ---------------------------------------------------------------------------
# LunarLander benchmarks
# ---------------------------------------------------------------------------

_register({
    "id": "LunarLander-Composite",
    "description": (
        "LunarLander-v3 with composite non-stationarity: sinusoidal obs noise "
        "PLUS linearly increasing action corruption probability. "
        "The agent faces both perceptual drift and actuation unreliability."
    ),
    "factory": lambda: make_continual(
        "LunarLander-v3",
        hide_done=True,
        pseudo_reset_penalty=-2.0,
        obs_noise_std_schedule=SinusoidalDrift(center=0.08, amplitude=0.06, period=40_000),
        action_corruption_prob_schedule=LinearDrift(start=0.0, end=0.15, duration=300_000),
    ),
    "tags": ["discrete-action", "composite-drift", "action-corruption"],
})

# ---------------------------------------------------------------------------
# Non-Markov / Hidden-state benchmarks
# ---------------------------------------------------------------------------

_register({
    "id": "CartPole-NonMarkov",
    "description": (
        "CartPole-v1 where the observation is randomly masked (sensor failure) "
        "with probability that drifts sinusoidally. The agent cannot form a "
        "full Markov state and must learn under partial observability that varies."
    ),
    "factory": lambda: make_continual(
        "CartPole-v1",
        hide_done=True,
        obs_mask_prob_schedule=SinusoidalDrift(center=0.25, amplitude=0.2, period=60_000),
    ),
    "tags": ["non-markov", "partial-observability", "sensor-failure"],
})

_register({
    "id": "MountainCar-BenchmarkSuite",
    "description": (
        "MountainCar-v0 with all drift mechanisms active at low intensity: "
        "obs noise, action corruption, reward shift. "
        "A comprehensive stress test for continual RL agents."
    ),
    "factory": lambda: make_continual(
        "MountainCar-v0",
        hide_done=True,
        pseudo_reset_penalty=-0.5,
        obs_noise_std_schedule=RandomWalkDrift(lo=0.0, hi=0.05, step_size=0.0001, seed=7),
        action_corruption_prob_schedule=SinusoidalDrift(center=0.05, amplitude=0.04, period=50_000),
        reward_shift_schedule=LinearDrift(start=0.0, end=-0.2, duration=500_000, cycle=True),
    ),
    "tags": ["stress-test", "all-mechanisms", "composite"],
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

BENCHMARKS: Dict[str, BenchmarkSpec] = _REGISTRY


def make_benchmark(name: str) -> gym.Env:
    """
    Instantiate a pre-configured continual RL benchmark environment.

    Parameters
    ----------
    name : str
        Benchmark ID. See `BENCHMARKS.keys()` for available options.

    Returns
    -------
    gym.Env
        The configured environment.

    Raises
    ------
    KeyError
        If `name` is not a known benchmark.

    Examples
    --------
    >>> import continualgym
    >>> env = continualgym.make_benchmark("Pendulum-RandomWalk")
    >>> obs, info = env.reset()
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Unknown benchmark '{name}'. Available benchmarks:\n"
            + "\n".join(f"  - {k}" for k in available)
        )
    return _REGISTRY[name]["factory"]()


def list_benchmarks(tag: str | None = None) -> List[str]:
    """
    List available benchmark IDs, optionally filtered by tag.

    Parameters
    ----------
    tag : str, optional
        If given, only return benchmarks that have this tag.

    Returns
    -------
    list[str]
    """
    if tag is None:
        return list(_REGISTRY.keys())
    return [k for k, v in _REGISTRY.items() if tag in v.get("tags", [])]
