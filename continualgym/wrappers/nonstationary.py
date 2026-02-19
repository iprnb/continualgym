"""
NonStationaryWrapper
====================
Injects configurable non-stationarity into a Gymnasium environment by:

1. **Reward shaping**: Scales or shifts rewards via a drift schedule.
2. **Observation noise**: Adds Gaussian noise whose std drifts over time.
3. **Action noise / corruption**: Randomly corrupts actions with p(t).
4. **Gravity / physics parameter drift**: For MuJoCo/Box2D envs that expose
   sim parameters, directly modifies them (best-effort via env.unwrapped).
5. **Transition noise**: Randomly perturbs the applied action to inject
   stochastic dynamics drift.
6. **Reward function switch**: Periodically swaps between multiple reward
   functions.
7. **Observation mask**: Randomly masks out observation dimensions to simulate
   sensor failure drift.

Every mechanism is controlled by a `DriftSchedule` (see drift module) so
the non-stationarity pattern can be linear, sinusoidal, step-wise, etc.

Design philosophy
-----------------
* Minimal assumptions about the base env — works with any gym.Env.
* Env-specific hooks (e.g. mujoco model param edits) are opt-in.
* All drift is driven by the *global* step counter so it survives
  pseudo-resets transparently.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import gymnasium as gym
import numpy as np

from continualgym.drift import DriftSchedule


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
RewardFn = Callable[[np.ndarray, Any, np.ndarray, float, Dict], float]


class NonStationaryWrapper(gym.Wrapper):
    """
    Inject non-stationarity into any Gym environment.

    Parameters
    ----------
    env : gym.Env
        Base environment (may already be wrapped by ContinuingWrapper).

    reward_scale_schedule : DriftSchedule, optional
        Drift schedule for multiplicative reward scaling. Value 1.0 = no change.

    reward_shift_schedule : DriftSchedule, optional
        Drift schedule for additive reward offset.

    obs_noise_std_schedule : DriftSchedule, optional
        Drift schedule for the std of Gaussian observation noise added.

    action_corruption_prob_schedule : DriftSchedule, optional
        Probability that the action is replaced by a random action.

    action_noise_std_schedule : DriftSchedule, optional
        Std of Gaussian noise added to continuous actions.

    transition_noise_std_schedule : DriftSchedule, optional
        Std of Gaussian noise added to actions before env.step() (affects
        dynamics without corrupting the action the agent sees as taken).

    obs_mask_prob_schedule : DriftSchedule, optional
        Probability that each observation dimension is zeroed out independently.

    physics_modifiers : dict[str, (DriftSchedule, float, float)], optional
        Env-specific physics parameter drift. Keys are attribute paths on
        env.unwrapped (e.g. "gravity", "model.opt.gravity[1]" for MuJoCo).
        Values are (schedule, min_value, max_value) tuples.

    reward_functions : list[RewardFn], optional
        Alternative reward functions. Cycled through based on
        `reward_fn_switch_schedule` (integer schedule, index into this list).

    reward_fn_switch_schedule : DriftSchedule, optional
        Schedule returning float; cast to int to pick which reward_fn to use.

    global_t_source : str
        How to get the global step counter:
          - 'self'  : use this wrapper's own step counter (default)
          - 'inner' : use env.global_t (if ContinuingWrapper is inside)

    clip_obs : bool
        Whether to clip noisy observations to the original obs space bounds.

    seed : int, optional
        RNG seed for stochastic drift effects.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        reward_scale_schedule: Optional[DriftSchedule] = None,
        reward_shift_schedule: Optional[DriftSchedule] = None,
        obs_noise_std_schedule: Optional[DriftSchedule] = None,
        action_corruption_prob_schedule: Optional[DriftSchedule] = None,
        action_noise_std_schedule: Optional[DriftSchedule] = None,
        transition_noise_std_schedule: Optional[DriftSchedule] = None,
        obs_mask_prob_schedule: Optional[DriftSchedule] = None,
        physics_modifiers: Optional[
            Dict[str, Tuple[DriftSchedule, float, float]]
        ] = None,
        reward_functions: Optional[List[RewardFn]] = None,
        reward_fn_switch_schedule: Optional[DriftSchedule] = None,
        global_t_source: str = "self",
        clip_obs: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.reward_scale_schedule = reward_scale_schedule
        self.reward_shift_schedule = reward_shift_schedule
        self.obs_noise_std_schedule = obs_noise_std_schedule
        self.action_corruption_prob_schedule = action_corruption_prob_schedule
        self.action_noise_std_schedule = action_noise_std_schedule
        self.transition_noise_std_schedule = transition_noise_std_schedule
        self.obs_mask_prob_schedule = obs_mask_prob_schedule
        self.physics_modifiers = physics_modifiers or {}
        self.reward_functions = reward_functions or []
        self.reward_fn_switch_schedule = reward_fn_switch_schedule
        self.global_t_source = global_t_source
        self.clip_obs = clip_obs

        self._rng = np.random.default_rng(seed)
        self._global_t: int = 0
        self._initialized: bool = False

        # Cache obs space bounds for clipping
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Box):
            self._obs_low = obs_space.low
            self._obs_high = obs_space.high
        else:
            self._obs_low = None
            self._obs_high = None
            if clip_obs and obs_noise_std_schedule is not None:
                warnings.warn(
                    "clip_obs=True but observation space is not Box; clipping skipped."
                )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._initialized = True
        obs = self._perturb_obs(obs)
        return obs, info

    def step(self, action):
        t = self._get_t()

        # Apply physics parameter drift before step
        self._apply_physics_drift(t)

        # Potentially corrupt / noise the action
        action = self._perturb_action(action, t)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._global_t += 1
        t = self._get_t()

        # Perturb observation
        original_obs = obs  # capture before _perturb_obs
        obs = self._perturb_obs(obs, t)

        # Perturb reward
        reward = self._perturb_reward(original_obs, action, obs, reward, info, t)

        # Inject drift info into info dict
        info["nonstationarity"] = self._get_drift_snapshot(t)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Drift application helpers
    # ------------------------------------------------------------------

    def _get_t(self) -> int:
        if self.global_t_source == "inner":
            inner = self.env
            while inner is not None:
                if hasattr(inner, "global_t"):
                    return inner.global_t
                inner = getattr(inner, "env", None)
        return self._global_t

    def _perturb_obs(
        self, obs: np.ndarray, t: Optional[int] = None
    ) -> np.ndarray:
        if t is None:
            t = self._get_t()
        if not isinstance(obs, np.ndarray):
            return obs

        # Gaussian noise
        if self.obs_noise_std_schedule is not None:
            std = self.obs_noise_std_schedule(t)
            if std > 0:
                noise = self._rng.normal(0, std, size=obs.shape)
                obs = obs + noise.astype(obs.dtype)

        # Random masking
        if self.obs_mask_prob_schedule is not None:
            p = self.obs_mask_prob_schedule(t)
            if p > 0:
                mask = self._rng.random(size=obs.shape) > p
                obs = obs * mask.astype(obs.dtype)

        # Clip to original bounds
        if self.clip_obs and self._obs_low is not None:
            obs = np.clip(obs, self._obs_low, self._obs_high)

        return obs

    def _perturb_action(self, action: Any, t: int) -> Any:
        # Random action corruption
        if self.action_corruption_prob_schedule is not None:
            p = self.action_corruption_prob_schedule(t)
            if p > 0 and self._rng.random() < p:
                return self.env.action_space.sample()

        # Gaussian action noise (continuous spaces only)
        if self.action_noise_std_schedule is not None:
            if isinstance(self.env.action_space, gym.spaces.Box):
                std = self.action_noise_std_schedule(t)
                if std > 0:
                    noise = self._rng.normal(0, std, size=np.array(action).shape)
                    action = np.clip(
                        np.array(action, dtype=np.float32) + noise.astype(np.float32),
                        self.env.action_space.low,
                        self.env.action_space.high,
                    )

        return action

    def _perturb_reward(
        self,
        obs: np.ndarray,
        action: Any,
        next_obs: np.ndarray,
        reward: float,
        info: Dict,
        t: int,
    ) -> float:
        # Custom reward function
        if self.reward_functions and self.reward_fn_switch_schedule is not None:
            idx = int(self.reward_fn_switch_schedule(t)) % len(self.reward_functions)
            reward = self.reward_functions[idx](obs, action, next_obs, reward, info)

        # Scale
        if self.reward_scale_schedule is not None:
            reward *= self.reward_scale_schedule(t)

        # Shift
        if self.reward_shift_schedule is not None:
            reward += self.reward_shift_schedule(t)

        return reward

    def _apply_physics_drift(self, t: int) -> None:
        """Best-effort modification of physics parameters on env.unwrapped."""
        if not self.physics_modifiers:
            return
        base = self.env.unwrapped
        for attr_path, (schedule, lo, hi) in self.physics_modifiers.items():
            alpha = float(np.clip(schedule(t), 0.0, 1.0))
            value = lo + alpha * (hi - lo)
            self._set_nested_attr(base, attr_path, value)

    @staticmethod
    def _set_nested_attr(obj: Any, path: str, value: float) -> None:
        """Set a nested attribute like 'model.opt.gravity[1]'."""
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                return
        last = parts[-1]
        # Handle indexed attributes like 'gravity[1]'
        if "[" in last:
            attr, idx = last.split("[")
            idx = int(idx.rstrip("]"))
            arr = getattr(obj, attr, None)
            if arr is not None:
                arr[idx] = value
        else:
            try:
                setattr(obj, last, value)
            except AttributeError:
                pass

    def _get_drift_snapshot(self, t: int) -> Dict[str, float]:
        """Return current drift values for logging/debugging."""
        snap = {"t": t}
        if self.reward_scale_schedule:
            snap["reward_scale"] = self.reward_scale_schedule(t)
        if self.reward_shift_schedule:
            snap["reward_shift"] = self.reward_shift_schedule(t)
        if self.obs_noise_std_schedule:
            snap["obs_noise_std"] = self.obs_noise_std_schedule(t)
        if self.action_corruption_prob_schedule:
            snap["action_corruption_prob"] = self.action_corruption_prob_schedule(t)
        if self.action_noise_std_schedule:
            snap["action_noise_std"] = self.action_noise_std_schedule(t)
        if self.obs_mask_prob_schedule:
            snap["obs_mask_prob"] = self.obs_mask_prob_schedule(t)
        return snap

    @property
    def current_drift(self) -> Dict[str, float]:
        """Public accessor for current drift values."""
        return self._get_drift_snapshot(self._get_t())
