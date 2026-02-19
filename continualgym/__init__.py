"""
ContinualGym: Transforms standard Gym environments into non-stationary,
continuing (non-episodic) environments for continual RL research.
"""

__version__ = "0.1.0"


def make_continual(*args, **kwargs):
    from continualgym.wrappers.pipeline import make_continual as _make_continual
    return _make_continual(*args, **kwargs)


def make_benchmark(*args, **kwargs):
    from continualgym.benchmarks.registry import make_benchmark as _make_benchmark
    return _make_benchmark(*args, **kwargs)


def list_benchmarks(*args, **kwargs):
    from continualgym.benchmarks.registry import list_benchmarks as _list_benchmarks
    return _list_benchmarks(*args, **kwargs)


@property
def BENCHMARKS():
    from continualgym.benchmarks.registry import BENCHMARKS as _BENCHMARKS
    return _BENCHMARKS


# Drift schedules — pure Python, no gymnasium dependency, safe to import eagerly
from continualgym.drift.schedules import (
    DriftSchedule,
    LinearDrift,
    SinusoidalDrift,
    StepDrift,
    RandomWalkDrift,
    CompositeDrift,
    FunctionDrift,
    ParameterModifier,
)

__all__ = [
    "make_continual",
    "make_benchmark",
    "list_benchmarks",
    "BENCHMARKS",
    "DriftSchedule",
    "LinearDrift",
    "SinusoidalDrift",
    "StepDrift",
    "RandomWalkDrift",
    "CompositeDrift",
    "FunctionDrift",
    "ParameterModifier",
]
