# ContinualGym

> Transform any standard Gymnasium environment into a **non-stationary, continuing** task for continual reinforcement learning research.

---

## Why ContinualGym?

Most RL benchmarks are episodic and stationary. Continual RL research needs:

1. **Continuing tasks** — no episode resets, no terminal signal leaking task identity.
2. **Non-stationarity** — reward functions, observations, dynamics, and goals that drift over time.
3. **Non-Markov challenges** — sensor failure, time-varying partial observability.
4. **Drop-in compatibility** — works with any `gym.Env` without touching the base env code.

ContinualGym provides all of this via a thin, composable wrapper stack.

---

## Installation

```bash
pip install continualgym               # core (numpy + gymnasium)
pip install continualgym[viz]          # + matplotlib
pip install continualgym[mujoco]       # + MuJoCo envs
pip install continualgym[box2d]        # + Box2D envs (LunarLander etc.)
pip install continualgym[dev]          # + pytest, mypy, black
```

---

## Quick Start

```python
from continualgym import make_continual, SinusoidalDrift, LinearDrift

env = make_continual(
    "CartPole-v1",
    hide_done=True,                         # suppress terminal signal
    pseudo_reset_penalty=-1.0,              # small penalty at internal resets
    obs_noise_std_schedule=SinusoidalDrift( # observation noise drifts sinusoidally
        center=0.05, amplitude=0.04, period=30_000
    ),
    reward_shift_schedule=LinearDrift(      # reward slowly shifts downward
        start=0.0, end=-0.5, duration=200_000
    ),
)

obs, info = env.reset()
for t in range(100_000):
    action = env.action_space.sample()      # replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    # terminated and truncated are always False when hide_done=True
    # info["global_t"]          — total steps across pseudo-episodes
    # info["pseudo_episode"]    — how many internal resets have occurred
    # info["nonstationarity"]   — current drift values for logging
```

---

## Architecture

```
┌─────────────────────────────────┐
│      NonStationaryWrapper       │  ← Applies drift: obs noise, reward
│                                 │    scaling/shift, action corruption,
│  reward_scale_schedule ──────►  │    obs masking, physics param drift,
│  obs_noise_std_schedule ──────► │    reward function switching
│  action_corruption_prob ──────► │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│        ContinuingWrapper        │  ← Hides done signals, manages
│                                 │    pseudo-resets, maintains global_t
│  hide_done=True                 │    counter across pseudo-episodes
│  pseudo_reset_penalty=-1.0      │
│  max_steps_per_pseudo_ep=None   │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│         Base gym.Env            │  ← Any Gymnasium environment
│  (CartPole, HalfCheetah, ...)   │    unchanged
└─────────────────────────────────┘
```

---

## Drift Schedules

All drift is controlled by `DriftSchedule` objects that map `t → float`.

| Schedule | Description | Key Args |
|---|---|---|
| `LinearDrift` | Linear interpolation from `start` to `end` | `start`, `end`, `duration`, `cycle`, `delay` |
| `SinusoidalDrift` | `center + amplitude * sin(2π t / period)` | `center`, `amplitude`, `period`, `phase` |
| `StepDrift` | Piecewise constant; jumps at changepoints | `changepoints`, `initial` |
| `RandomWalkDrift` | Brownian motion clipped to `[lo, hi]` | `lo`, `hi`, `step_size`, `seed` |
| `CompositeDrift` | Add or multiply multiple schedules | `schedules`, `mode` |
| `FunctionDrift` | Wrap any `f(t) → float` callable | `fn` |

### Composing Schedules

```python
from continualgym import SinusoidalDrift, LinearDrift

# Drift that increases linearly AND oscillates on top
combined = LinearDrift(0, 0.5, 100_000) + SinusoidalDrift(0, 0.1, 10_000)

# Or multiply:
modulated = LinearDrift(0, 1.0, 200_000) * SinusoidalDrift(0.5, 0.5, 20_000)
```

---

## Drift Mechanisms

### 1. Observation Noise
```python
env = make_continual("CartPole-v1",
    obs_noise_std_schedule=SinusoidalDrift(center=0.05, amplitude=0.04))
```
Adds Gaussian noise `N(0, σ(t))` to every observation.

### 2. Observation Masking (Non-Markov / Sensor Failure)
```python
env = make_continual("CartPole-v1",
    obs_mask_prob_schedule=SinusoidalDrift(center=0.3, amplitude=0.25))
```
Each observation dimension independently zeroed out with probability `p(t)`.

### 3. Reward Scaling & Shifting
```python
env = make_continual("CartPole-v1",
    reward_scale_schedule=LinearDrift(1.0, 0.1, 200_000),   # shrinking scale
    reward_shift_schedule=LinearDrift(0.0, -1.0, 200_000))  # negative drift
```

### 4. Reward Function Switching
```python
reward_fns = [
    lambda obs, a, nobs, r, i: r,                                    # standard
    lambda obs, a, nobs, r, i: r - 0.5 * float(sum(a**2)),          # energy penalty
]
env = make_continual("Pendulum-v1",
    reward_functions=reward_fns,
    reward_fn_switch_schedule=StepDrift([(100_000, 1)], initial=0))
```

### 5. Action Corruption
```python
env = make_continual("CartPole-v1",
    action_corruption_prob_schedule=LinearDrift(0.0, 0.2, 300_000))
```
With probability `p(t)` the agent's action is replaced by a random sample.

### 6. Continuous Action Noise
```python
env = make_continual("HalfCheetah-v4",
    action_noise_std_schedule=SinusoidalDrift(center=0.1, amplitude=0.08))
```

### 7. Physics Parameter Drift (MuJoCo)
```python
env = make_continual("HalfCheetah-v4",
    physics_modifiers={
        "model.opt.gravity[1]": (LinearDrift(0, 1, 500_000), -15.0, -5.0),
    })
```

---

## Pre-configured Benchmarks

```python
from continualgym import make_benchmark, list_benchmarks

print(list_benchmarks())
# ['CartPole-Sinusoidal', 'CartPole-StepChanges', 'CartPole-NonMarkov',
#  'MountainCarContinuous-LinearDrift', 'Pendulum-RandomWalk',
#  'Pendulum-RewardSwitch', 'LunarLander-Composite',
#  'MountainCar-BenchmarkSuite']

env = make_benchmark("Pendulum-RandomWalk")
obs, info = env.reset()
```

Filter by tag:
```python
list_benchmarks(tag="non-markov")      # ['CartPole-NonMarkov']
list_benchmarks(tag="composite-drift") # ['LunarLander-Composite', ...]
```

---

## Evaluation Utilities

```python
from continualgym.utils import (
    EpisodeLogger,    # track pseudo-episode returns
    DriftTracker,     # record drift values over time
    RunningStats,     # online mean/std/min/max
    plasticity_score, # how fast agent recovers after shift
    forgetting_score, # how much performance degrades
    plot_drift_overlay,  # matplotlib visualization
)

logger = EpisodeLogger(window=100)
tracker = DriftTracker(record_every=500)

for t in range(500_000):
    obs, r, done, trunc, info = env.step(agent.act(obs))
    logger.step(r, info)
    tracker.step(t, info)

print(logger.summary())
# {'total_steps': 500000, 'pseudo_episodes': 312,
#  'mean_return_last_N': 42.3, 'std_return_last_N': 8.1, ...}

# Visualize
fig = plot_drift_overlay(tracker, logger)
fig.savefig("drift_overview.png")
```

---

## Advanced: Custom Wrapper Integration

`ContinuingWrapper` and `NonStationaryWrapper` can be stacked manually:

```python
import gymnasium as gym
from continualgym.wrappers import ContinuingWrapper, NonStationaryWrapper
from continualgym.drift import RandomWalkDrift

base = gym.make("Ant-v4")
cont = ContinuingWrapper(base, hide_done=True, pseudo_reset_penalty=-2.0)
env  = NonStationaryWrapper(
    cont,
    reward_scale_schedule=RandomWalkDrift(lo=0.7, hi=1.3, step_size=0.001),
    global_t_source="inner",
)
```

---

## Info Dict Reference

Every `step()` returns an augmented `info` dict:

| Key | Source | Description |
|---|---|---|
| `global_t` | ContinuingWrapper | Total env steps across all pseudo-episodes |
| `pseudo_episode` | ContinuingWrapper | Count of internal episode resets |
| `steps_in_pseudo_episode` | ContinuingWrapper | Steps in current pseudo-episode |
| `pseudo_reset_occurred` | ContinuingWrapper | `True` if this step triggered a reset |
| `nonstationarity` | NonStationaryWrapper | Dict of current drift values |
| `nonstationarity["reward_scale"]` | NonStationaryWrapper | Current reward multiplier |
| `nonstationarity["obs_noise_std"]` | NonStationaryWrapper | Current obs noise std |
| … | … | … |

---

## Running Tests

```bash
pip install continualgym[dev]
pytest tests/ -v
```

## Note
This repository contains early-stage research prototypes.
APIs are incomplete and subject to change.
The focus is on exploring system design and evaluation patterns.## Note
This repository contains early-stage research prototypes.
APIs are incomplete and subject to change.
The focus is on exploring system design and evaluation patterns.

---

## License

MIT
