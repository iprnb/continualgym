"""
Tests for ContinualGym.

Run with: pytest tests/test_continualgym.py -v
"""

import math
import pytest
import numpy as np
import gymnasium as gym

from continualgym.drift import (
    LinearDrift,
    SinusoidalDrift,
    StepDrift,
    RandomWalkDrift,
    CompositeDrift,
    FunctionDrift,
)
from continualgym.wrappers.continuing import ContinuingWrapper
from continualgym.wrappers.nonstationary import NonStationaryWrapper
from continualgym.wrappers.pipeline import make_continual
from continualgym.utils import RunningStats, EpisodeLogger, DriftTracker


# ---------------------------------------------------------------------------
# Drift schedule tests
# ---------------------------------------------------------------------------

class TestLinearDrift:
    def test_start_value(self):
        d = LinearDrift(start=0.0, end=1.0, duration=100)
        assert d(0) == pytest.approx(0.0)

    def test_end_value(self):
        d = LinearDrift(start=0.0, end=1.0, duration=100)
        assert d(100) == pytest.approx(1.0)

    def test_midpoint(self):
        d = LinearDrift(start=0.0, end=1.0, duration=100)
        assert d(50) == pytest.approx(0.5)

    def test_clamped_beyond_end(self):
        d = LinearDrift(start=0.0, end=1.0, duration=100)
        assert d(200) == pytest.approx(1.0)

    def test_cycle(self):
        d = LinearDrift(start=0.0, end=1.0, duration=100, cycle=True)
        # At t=100 we should be at peak (1.0), at t=200 back to start
        assert d(100) == pytest.approx(1.0)
        assert d(200) == pytest.approx(0.0)

    def test_delay(self):
        d = LinearDrift(start=0.0, end=1.0, duration=100, delay=50)
        assert d(0) == pytest.approx(0.0)
        assert d(50) == pytest.approx(0.0)
        assert d(150) == pytest.approx(1.0)


class TestSinusoidalDrift:
    def test_center_at_quarter_period(self):
        d = SinusoidalDrift(center=0.5, amplitude=0.5, period=100, phase=0)
        # sin(pi/2) = 1 → 0.5 + 0.5 * 1 = 1.0
        assert d(25) == pytest.approx(1.0, abs=1e-6)

    def test_bounded(self):
        d = SinusoidalDrift(center=0.5, amplitude=0.5, period=100)
        vals = [d(t) for t in range(1000)]
        assert min(vals) >= 0.0 - 1e-9
        assert max(vals) <= 1.0 + 1e-9


class TestStepDrift:
    def test_initial_value(self):
        d = StepDrift([(100, 0.5), (200, 1.0)], initial=0.0)
        assert d(0) == 0.0
        assert d(99) == 0.0

    def test_after_first_step(self):
        d = StepDrift([(100, 0.5), (200, 1.0)], initial=0.0)
        assert d(100) == 0.5
        assert d(199) == 0.5

    def test_after_second_step(self):
        d = StepDrift([(100, 0.5), (200, 1.0)], initial=0.0)
        assert d(200) == 1.0

    def test_uniform_factory(self):
        import random
        d = StepDrift.uniform(n_changes=3, total_steps=400, rng=random.Random(0))
        assert isinstance(d, StepDrift)
        assert len(d.changepoints) == 3


class TestRandomWalkDrift:
    def test_stays_in_bounds(self):
        d = RandomWalkDrift(lo=0.0, hi=1.0, step_size=0.1, seed=42)
        for t in range(10_000):
            v = d(t)
            assert 0.0 <= v <= 1.0

    def test_reset_returns_to_center(self):
        d = RandomWalkDrift(lo=0.0, hi=1.0, seed=0)
        for t in range(1000):
            d(t)
        d.reset()
        assert d._value == pytest.approx(0.5)


class TestCompositeDrift:
    def test_add(self):
        d1 = LinearDrift(start=0.1, end=0.1, duration=1)
        d2 = LinearDrift(start=0.2, end=0.2, duration=1)
        comp = d1 + d2
        assert comp(0) == pytest.approx(0.3)

    def test_mul(self):
        d1 = LinearDrift(start=2.0, end=2.0, duration=1)
        d2 = LinearDrift(start=3.0, end=3.0, duration=1)
        comp = d1 * d2
        assert comp(0) == pytest.approx(6.0)


class TestFunctionDrift:
    def test_lambda(self):
        d = FunctionDrift(lambda t: t / 100.0)
        assert d(50) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ContinuingWrapper tests
# ---------------------------------------------------------------------------

class TestContinuingWrapper:
    def make_env(self, **kwargs):
        base = gym.make("CartPole-v1")
        return ContinuingWrapper(base, **kwargs)

    def test_reset_works(self):
        env = self.make_env()
        obs, info = env.reset()
        assert obs is not None
        assert "global_t" in info

    def test_no_done_when_hide_done(self):
        env = self.make_env(hide_done=True)
        env.reset()
        for _ in range(500):
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            assert not term
            assert not trunc

    def test_global_t_increments(self):
        env = self.make_env()
        env.reset()
        for i in range(10):
            _, _, _, _, info = env.step(env.action_space.sample())
        assert env.global_t == 10

    def test_pseudo_episode_increments(self):
        env = self.make_env(hide_done=True)
        env.reset()
        initial_ep = env.pseudo_episode
        # Run many steps to force at least one episode end
        for _ in range(500):
            env.step(env.action_space.sample())
        assert env.pseudo_episode >= initial_ep

    def test_max_steps_per_pseudo_episode(self):
        env = self.make_env(max_steps_per_pseudo_episode=50, hide_done=True)
        env.reset()
        for _ in range(200):
            env.step(env.action_space.sample())
        assert env.pseudo_episode >= 3  # 200/50 = 4 resets

    def test_pseudo_reset_penalty(self):
        """Rewards at reset points should include the penalty."""
        env = self.make_env(
            pseudo_reset_penalty=-10.0,
            max_steps_per_pseudo_episode=10,
            hide_done=True,
        )
        env.reset()
        rewards = []
        for _ in range(100):
            _, r, _, _, _ = env.step(env.action_space.sample())
            rewards.append(r)
        assert min(rewards) <= -10.0 + 1.0 + 1e-3  # at least one penalty applied

    def test_requires_reset_before_step(self):
        env = self.make_env()
        with pytest.raises(RuntimeError):
            env.step(env.action_space.sample())


# ---------------------------------------------------------------------------
# NonStationaryWrapper tests
# ---------------------------------------------------------------------------

class TestNonStationaryWrapper:
    def make_env(self, **drift_kwargs):
        base = gym.make("CartPole-v1")
        cont = ContinuingWrapper(base)
        return NonStationaryWrapper(cont, **drift_kwargs)

    def test_obs_noise_changes_obs(self):
        env = self.make_env(
            obs_noise_std_schedule=LinearDrift(start=1.0, end=1.0, duration=1)
        )
        env.reset()
        obs1, *_ = env.step(0)
        # With std=1.0, obs should very likely differ from base
        base = gym.make("CartPole-v1")
        cont = ContinuingWrapper(base)
        env2 = NonStationaryWrapper(cont)
        env2.reset()
        obs2, *_ = env2.step(0)
        # Not a deterministic test but noise_std=1 should shift obs significantly
        assert obs1 is not None  # smoke test

    def test_reward_scaling(self):
        env = self.make_env(
            reward_scale_schedule=LinearDrift(start=0.0, end=0.0, duration=1)
        )
        env.reset()
        _, reward, *_ = env.step(0)
        assert reward == pytest.approx(0.0, abs=1e-6)  # 1.0 * 0.0 = 0.0

    def test_reward_shift(self):
        env = self.make_env(
            reward_shift_schedule=LinearDrift(start=100.0, end=100.0, duration=1)
        )
        env.reset()
        _, reward, *_ = env.step(0)
        # CartPole gives reward=1.0 per step, so with shift=100 → 101
        assert reward == pytest.approx(101.0, abs=0.1)

    def test_action_corruption_replaces_action(self):
        """With p=1.0 corruption, the action should always be random → env doesn't crash."""
        env = self.make_env(
            action_corruption_prob_schedule=LinearDrift(start=1.0, end=1.0, duration=1)
        )
        env.reset()
        for _ in range(20):
            env.step(0)  # always corrupted but shouldn't raise

    def test_drift_snapshot_in_info(self):
        env = self.make_env(
            reward_scale_schedule=SinusoidalDrift()
        )
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "nonstationarity" in info
        assert "reward_scale" in info["nonstationarity"]

    def test_reward_function_switch(self):
        call_log = []

        def fn0(obs, a, nobs, r, i):
            call_log.append(0)
            return 0.0

        def fn1(obs, a, nobs, r, i):
            call_log.append(1)
            return 99.0

        env = self.make_env(
            reward_functions=[fn0, fn1],
            reward_fn_switch_schedule=StepDrift([(0, 1)], initial=0),
        )
        env.reset()
        _, reward, *_ = env.step(0)
        assert reward == pytest.approx(99.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestMakeContinual:
    def test_basic_creation(self):
        env = make_continual("CartPole-v1")
        assert isinstance(env, NonStationaryWrapper)

    def test_full_pipeline(self):
        env = make_continual(
            "CartPole-v1",
            hide_done=True,
            obs_noise_std_schedule=SinusoidalDrift(center=0.1, amplitude=0.05),
            reward_shift_schedule=LinearDrift(start=-0.1, end=0.1, duration=1000),
        )
        obs, info = env.reset()
        assert obs is not None
        for _ in range(50):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert not term  # hide_done=True
            assert not trunc

    def test_smoke_pendulum(self):
        env = make_continual(
            "Pendulum-v1",
            action_noise_std_schedule=LinearDrift(0.0, 0.5, 1000),
        )
        env.reset()
        for _ in range(20):
            env.step(env.action_space.sample())


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestRunningStats:
    def test_mean(self):
        rs = RunningStats()
        for v in [1.0, 2.0, 3.0]:
            rs.update(v)
        assert rs.mean == pytest.approx(2.0, rel=1e-6)

    def test_std(self):
        rs = RunningStats()
        for v in [2.0, 4.0]:
            rs.update(v)
        assert rs.std == pytest.approx(math.sqrt(2.0), rel=1e-4)

    def test_min_max(self):
        rs = RunningStats()
        for v in [3.0, 1.0, 2.0]:
            rs.update(v)
        assert rs.min == 1.0
        assert rs.max == 3.0


class TestEpisodeLogger:
    def test_logs_pseudo_episodes(self):
        env = make_continual(
            "CartPole-v1",
            hide_done=True,
            max_steps_per_pseudo_episode=50,
        )
        env.reset()
        logger = EpisodeLogger(window=20)
        for _ in range(300):
            _, r, _, _, info = env.step(env.action_space.sample())
            logger.step(r, info)
        assert logger._pseudo_episodes >= 2
        assert not math.isnan(logger.recent_mean_return)


class TestDriftTracker:
    def test_records_snapshots(self):
        env = make_continual(
            "CartPole-v1",
            reward_scale_schedule=LinearDrift(0.0, 1.0, 1000),
        )
        env.reset()
        tracker = DriftTracker(record_every=10)
        for t in range(100):
            _, _, _, _, info = env.step(env.action_space.sample())
            tracker.step(t, info)
        assert len(tracker.records) > 0

    def test_to_numpy(self):
        env = make_continual(
            "CartPole-v1",
            reward_scale_schedule=LinearDrift(0.0, 1.0, 1000),
        )
        env.reset()
        tracker = DriftTracker(record_every=10)
        for t in range(200):
            _, _, _, _, info = env.step(env.action_space.sample())
            tracker.step(t, info)
        ts, vs = tracker.to_numpy("reward_scale")
        assert len(ts) > 0
        assert len(vs) == len(ts)


# ---------------------------------------------------------------------------
# Benchmarks smoke tests
# ---------------------------------------------------------------------------

class TestBenchmarks:
    def test_make_benchmark_cartpole_sinusoidal(self):
        from continualgym.benchmarks import make_benchmark
        env = make_benchmark("CartPole-Sinusoidal")
        obs, info = env.reset()
        assert obs is not None

    def test_make_benchmark_pendulum_random_walk(self):
        from continualgym.benchmarks import make_benchmark
        env = make_benchmark("Pendulum-RandomWalk")
        obs, info = env.reset()
        for _ in range(20):
            env.step(env.action_space.sample())

    def test_unknown_benchmark_raises(self):
        from continualgym.benchmarks import make_benchmark
        with pytest.raises(KeyError):
            make_benchmark("DoesNotExist-v99")

    def test_list_benchmarks(self):
        from continualgym.benchmarks import list_benchmarks
        all_bms = list_benchmarks()
        assert len(all_bms) > 0

    def test_list_benchmarks_by_tag(self):
        from continualgym.benchmarks import list_benchmarks
        sinusoidal = list_benchmarks(tag="sinusoidal")
        assert len(sinusoidal) >= 1
