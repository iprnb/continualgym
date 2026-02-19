"""
Drift injection primitives.

Every DriftSchedule maps a global timestep t -> a scalar value in [0, 1]
(or unbounded for some schedules). The NonStationaryWrapper uses these
to modulate environment parameters.
"""

from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional


class DriftSchedule(ABC):
    """Base class for all drift schedules."""

    @abstractmethod
    def __call__(self, t: int) -> float:
        """Return drift magnitude at global step t (typically in [0,1])."""

    def reset(self) -> None:
        """Called when the drift schedule itself should be reset (optional)."""

    def __add__(self, other: "DriftSchedule") -> "CompositeDrift":
        return CompositeDrift([self, other], mode="add")

    def __mul__(self, other: "DriftSchedule") -> "CompositeDrift":
        return CompositeDrift([self, other], mode="mul")


# ---------------------------------------------------------------------------
# Concrete schedules
# ---------------------------------------------------------------------------

class LinearDrift(DriftSchedule):
    """
    Linearly interpolates from `start` to `end` over `duration` steps,
    then stays at `end` (or cycles if `cycle=True`).

    Args:
        start:    Initial value (default 0.0).
        end:      Final value (default 1.0).
        duration: Steps over which to interpolate.
        cycle:    If True, oscillates back and forth indefinitely.
        delay:    Number of steps before drift begins.
    """

    def __init__(
        self,
        start: float = 0.0,
        end: float = 1.0,
        duration: int = 100_000,
        cycle: bool = False,
        delay: int = 0,
    ):
        self.start = start
        self.end = end
        self.duration = max(1, duration)
        self.cycle = cycle
        self.delay = delay

    def __call__(self, t: int) -> float:
        t = max(0, t - self.delay)
        if self.cycle:
            phase = (t % (2 * self.duration))
            if phase > self.duration:
                phase = 2 * self.duration - phase
            alpha = phase / self.duration
        else:
            alpha = min(t / self.duration, 1.0)
        return self.start + alpha * (self.end - self.start)


class SinusoidalDrift(DriftSchedule):
    """
    Sinusoidal oscillation.

    value(t) = center + amplitude * sin(2π * t / period + phase)

    Args:
        center:    Mean value (default 0.5).
        amplitude: Amplitude of oscillation (default 0.5).
        period:    Period in steps (default 50_000).
        phase:     Phase offset in radians (default 0.0).
    """

    def __init__(
        self,
        center: float = 0.5,
        amplitude: float = 0.5,
        period: int = 50_000,
        phase: float = 0.0,
    ):
        self.center = center
        self.amplitude = amplitude
        self.period = period
        self.phase = phase

    def __call__(self, t: int) -> float:
        return self.center + self.amplitude * math.sin(
            2 * math.pi * t / self.period + self.phase
        )


class StepDrift(DriftSchedule):
    """
    Piecewise-constant drift that jumps to a new value at specified changepoints.

    Args:
        changepoints: List of (step, value) pairs, sorted by step.
                      At step >= cp_step the value switches to cp_value.
        initial:      Value before the first changepoint (default 0.0).
    """

    def __init__(
        self,
        changepoints: List[tuple],
        initial: float = 0.0,
    ):
        self.changepoints = sorted(changepoints, key=lambda x: x[0])
        self.initial = initial

    def __call__(self, t: int) -> float:
        value = self.initial
        for step, val in self.changepoints:
            if t >= step:
                value = val
            else:
                break
        return value

    @classmethod
    def uniform(
        cls,
        n_changes: int,
        total_steps: int,
        values: Optional[List[float]] = None,
        initial: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> "StepDrift":
        """Factory: equally-spaced changepoints with optional random values."""
        rng = rng or random.Random()
        interval = total_steps // (n_changes + 1)
        if values is None:
            values = [rng.uniform(0, 1) for _ in range(n_changes)]
        cps = [(interval * (i + 1), v) for i, v in enumerate(values)]
        return cls(cps, initial=initial)


class RandomWalkDrift(DriftSchedule):
    """
    Discrete random walk clipped to [lo, hi].

    Args:
        lo:        Lower bound (default 0.0).
        hi:        Upper bound (default 1.0).
        step_size: Max absolute change per step (default 0.001).
        seed:      Random seed for reproducibility.
        update_every: Steps between walk updates (default 1).
    """

    def __init__(
        self,
        lo: float = 0.0,
        hi: float = 1.0,
        step_size: float = 0.001,
        seed: Optional[int] = None,
        update_every: int = 1,
    ):
        self.lo = lo
        self.hi = hi
        self.step_size = step_size
        self.update_every = update_every
        self._rng = random.Random(seed)
        self._value = (lo + hi) / 2.0
        self._last_t = -1

    def __call__(self, t: int) -> float:
        if t != self._last_t and t % self.update_every == 0:
            delta = self._rng.uniform(-self.step_size, self.step_size)
            self._value = max(self.lo, min(self.hi, self._value + delta))
            self._last_t = t
        return self._value

    def reset(self) -> None:
        self._value = (self.lo + self.hi) / 2.0
        self._last_t = -1


class CompositeDrift(DriftSchedule):
    """
    Combine multiple drift schedules via addition or multiplication.

    Args:
        schedules: List of DriftSchedule objects.
        mode:      'add' sums all values; 'mul' multiplies them; 'max' takes max.
    """

    def __init__(self, schedules: List[DriftSchedule], mode: str = "add"):
        assert mode in ("add", "mul", "max"), f"Unknown mode: {mode}"
        self.schedules = schedules
        self.mode = mode

    def __call__(self, t: int) -> float:
        vals = [s(t) for s in self.schedules]
        if self.mode == "add":
            return sum(vals)
        elif self.mode == "mul":
            result = 1.0
            for v in vals:
                result *= v
            return result
        else:  # max
            return max(vals)

    def reset(self) -> None:
        for s in self.schedules:
            s.reset()


class FunctionDrift(DriftSchedule):
    """
    Wrap any callable (int -> float) as a DriftSchedule.

    Example:
        d = FunctionDrift(lambda t: 0.5 + 0.1 * (t // 1000))
    """

    def __init__(self, fn: Callable[[int], float]):
        self.fn = fn

    def __call__(self, t: int) -> float:
        return self.fn(t)


# ---------------------------------------------------------------------------
# Parameter modifier helpers
# ---------------------------------------------------------------------------

class ParameterModifier:
    """
    Maps a drift value in [0,1] to a physical parameter value in [low, high].

    Supports linear and log-linear interpolation.
    """

    def __init__(
        self,
        low: float,
        high: float,
        schedule: DriftSchedule,
        log_scale: bool = False,
    ):
        self.low = low
        self.high = high
        self.schedule = schedule
        self.log_scale = log_scale

    def __call__(self, t: int) -> float:
        alpha = self.schedule(t)
        if self.log_scale:
            log_low = math.log(max(self.low, 1e-12))
            log_high = math.log(max(self.high, 1e-12))
            return math.exp(log_low + alpha * (log_high - log_low))
        return self.low + alpha * (self.high - self.low)
