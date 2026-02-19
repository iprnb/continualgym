"""
examples/quickstart.py
======================

Minimal end-to-end demonstration of ContinualGym.

Run with:
    python examples/quickstart.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from continualgym import (
    make_continual,
    SinusoidalDrift,
    LinearDrift,
    StepDrift,
    RandomWalkDrift,
    make_benchmark,
)
from continualgym.utils import EpisodeLogger, DriftTracker, RunningStats

TOTAL_STEPS = 5_000


def demo_basic():
    print("\n" + "="*60)
    print("DEMO 1 — CartPole as a Continuing Task (no drift)")
    print("="*60)

    env = make_continual("CartPole-v1", hide_done=True)
    obs, info = env.reset()
    logger = EpisodeLogger(window=50)
    stats = RunningStats("reward")

    for _ in range(TOTAL_STEPS):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        logger.step(reward, info)
        stats.update(reward)

    print(f"Total steps: {TOTAL_STEPS}")
    print(f"Pseudo-resets: {logger._pseudo_episodes}")
    print(f"Stats: {stats}")
    env.close()


def demo_sinusoidal_drift():
    print("\n" + "="*60)
    print("DEMO 2 — CartPole with Sinusoidal Obs Noise + Reward Shift")
    print("="*60)

    env = make_continual(
        "CartPole-v1",
        hide_done=True,
        obs_noise_std_schedule=SinusoidalDrift(center=0.05, amplitude=0.04, period=2000),
        reward_shift_schedule=SinusoidalDrift(center=0.0, amplitude=0.5, period=3000),
    )
    obs, info = env.reset()
    tracker = DriftTracker(record_every=100)

    for t in range(TOTAL_STEPS):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        tracker.step(t, info)

    records = tracker.records
    if records:
        ts, obs_noise = tracker.to_numpy("obs_noise_std")
        ts, rshift = tracker.to_numpy("reward_shift")
        print(f"Obs noise range: [{obs_noise.min():.4f}, {obs_noise.max():.4f}]")
        print(f"Reward shift range: [{rshift.min():.4f}, {rshift.max():.4f}]")
    env.close()


def demo_step_changes():
    print("\n" + "="*60)
    print("DEMO 3 — CartPole with Abrupt Step-Change Distribution Shifts")
    print("="*60)

    env = make_continual(
        "CartPole-v1",
        hide_done=True,
        obs_noise_std_schedule=StepDrift(
            changepoints=[(1000, 0.2), (3000, 0.0), (4500, 0.4)],
            initial=0.0,
        ),
    )
    env.reset()

    prev_std = -1
    for t in range(TOTAL_STEPS):
        _, _, _, _, info = env.step(env.action_space.sample())
        drift = info.get("nonstationarity", {})
        std = round(drift.get("obs_noise_std", 0.0), 4)
        if std != prev_std:
            print(f"  t={t:5d}: obs_noise_std changed → {std}")
            prev_std = std
    env.close()


def demo_reward_function_switch():
    print("\n" + "="*60)
    print("DEMO 4 — Pendulum with Switching Reward Functions")
    print("="*60)

    import numpy as np

    reward_fns = [
        lambda obs, a, nobs, r, i: r,                             # (0) standard
        lambda obs, a, nobs, r, i: r - 0.5 * float(np.sum(np.array(a)**2)),  # (1) energy
        lambda obs, a, nobs, r, i: float(nobs[2]),                # (2) angular vel
    ]
    fn_names = {0: "standard", 1: "energy-penalty", 2: "angular-vel"}

    env = make_continual(
        "Pendulum-v1",
        reward_functions=reward_fns,
        reward_fn_switch_schedule=StepDrift(
            changepoints=[(1500, 1), (3000, 2), (4500, 0)],
            initial=0,
        ),
    )
    env.reset()

    prev_fn = -1
    for t in range(TOTAL_STEPS):
        _, reward, _, _, info = env.step(env.action_space.sample())
    print("  Reward function switches logged (see step-drift changepoints above)")
    env.close()


def demo_benchmark():
    print("\n" + "="*60)
    print("DEMO 5 — Pre-configured Benchmark: Pendulum-RandomWalk")
    print("="*60)

    env = make_benchmark("Pendulum-RandomWalk")
    obs, info = env.reset()
    stats = RunningStats("reward")

    for _ in range(TOTAL_STEPS):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        stats.update(reward)

    print(f"Steps run: {TOTAL_STEPS}")
    print(f"Reward stats: {stats}")
    env.close()


def demo_non_markov():
    print("\n" + "="*60)
    print("DEMO 6 — Non-Markov: Sinusoidally Drifting Sensor Failure")
    print("="*60)

    env = make_continual(
        "CartPole-v1",
        hide_done=True,
        obs_mask_prob_schedule=SinusoidalDrift(center=0.3, amplitude=0.25, period=2000),
    )
    obs, info = env.reset()
    prev_p = -1
    for t in range(TOTAL_STEPS):
        obs, _, _, _, info = env.step(env.action_space.sample())
        p = round(info["nonstationarity"].get("obs_mask_prob", 0), 3)
        if abs(p - prev_p) > 0.05:
            n_zeros = (obs == 0).sum()
            print(f"  t={t:5d}: mask_prob={p:.3f}, zeros_in_obs={n_zeros}/{len(obs)}")
            prev_p = p
    env.close()


if __name__ == "__main__":
    demo_basic()
    demo_sinusoidal_drift()
    demo_step_changes()
    demo_reward_function_switch()
    demo_benchmark()
    demo_non_markov()
    print("\n✓ All demos complete.")
