"""
ContinuingWrapper
=================
Converts an episodic Gymnasium environment into a *continuing* (non-episodic)
task following the continual RL setting.

Key ideas
---------
* Episode resets are replaced by *pseudo-resets*: the environment internally
  resets itself but the agent receives NO terminal signal (done/truncated is
  always False from the outside).
* The "reset state" at each pseudo-reset can itself drift over time (e.g.
  starting from different initial conditions).
* A global step counter `t` is maintained across pseudo-episodes.
* The observation at a pseudo-reset is injected transparently into the
  next step's return so the interface remains standard step/reset.

Non-Markov benchmarking mode
-----------------------------
When `hide_time_index=True` the wrapper drops any time-index feature from the
observation and does NOT expose the global step to the agent, creating a
partial-observability / non-Markov setting.

When `hide_done=True` (default), terminal signals are suppressed entirely so
the agent cannot infer episode boundaries.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class ContinuingWrapper(gym.Wrapper):
    """
    Wrap an episodic Gym environment as a continuing (non-episodic) task.

    Parameters
    ----------
    env : gym.Env
        The base episodic environment.
    hide_done : bool
        If True (default), always return done=False / truncated=False.
    hide_time_index : bool
        If True, strip any time-aware features from observations (best-effort).
        Primarily useful when the base env appends step count to obs.
    pseudo_reset_penalty : float
        Optional negative reward injected at each pseudo-reset to signal the
        agent that a "fall" occurred without giving an explicit done=True.
    max_steps_per_pseudo_episode : int | None
        If set, force a pseudo-reset every N steps regardless of the
        environment's terminal signal. Useful for non-terminating envs.
    obs_at_reset_is_next_obs : bool
        If True (default), the first observation after an internal reset is
        returned as the next obs from `step()` rather than requiring a
        separate `reset()` call.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env: gym.Env,
        hide_done: bool = True,
        hide_time_index: bool = False,
        pseudo_reset_penalty: float = 0.0,
        max_steps_per_pseudo_episode: Optional[int] = None,
        obs_at_reset_is_next_obs: bool = True,
    ):
        super().__init__(env)
        self.hide_done = hide_done
        self.hide_time_index = hide_time_index
        self.pseudo_reset_penalty = pseudo_reset_penalty
        self.max_steps_per_pseudo_episode = max_steps_per_pseudo_episode
        self.obs_at_reset_is_next_obs = obs_at_reset_is_next_obs

        # Global counters
        self._global_t: int = 0
        self._pseudo_episode: int = 0
        self._steps_in_pseudo_ep: int = 0

        # Buffered observation after a pseudo-reset
        self._pending_obs: Optional[np.ndarray] = None
        self._pending_info: Dict[str, Any] = {}

        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._process_obs(obs)
        self._initialized = True
        self._steps_in_pseudo_ep = 0
        info = self._augment_info(info)
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError(
                "ContinuingWrapper.reset() must be called before step()."
            )

        # If a pseudo-reset happened at the end of the last step, serve the
        # buffered observation now (avoids one wasted transition).
        if self._pending_obs is not None:
            obs = self._pending_obs
            info = self._pending_info
            self._pending_obs = None
            self._pending_info = {}
            reward = self.pseudo_reset_penalty
            self._steps_in_pseudo_ep += 1
            return obs, reward, False, False, info

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._global_t += 1
        self._steps_in_pseudo_ep += 1

        # Check if we should pseudo-reset
        force_reset = (
            self.max_steps_per_pseudo_episode is not None
            and self._steps_in_pseudo_ep >= self.max_steps_per_pseudo_episode
        )
        need_reset = terminated or truncated or force_reset

        if need_reset:
            reward += self.pseudo_reset_penalty
            self._pseudo_episode += 1
            self._steps_in_pseudo_ep = 0

            new_obs, new_info = self.env.reset()
            new_obs = self._process_obs(new_obs)
            new_info = self._augment_info(new_info, pseudo_reset=True)

            if self.obs_at_reset_is_next_obs:
                self._pending_obs = new_obs
                self._pending_info = new_info
        else:
            obs = self._process_obs(obs)

        info = self._augment_info(info, terminated=terminated, truncated=truncated)

        if self.hide_done:
            terminated = False
            truncated = False

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Properties exposed for wrappers stacked on top
    # ------------------------------------------------------------------

    @property
    def global_t(self) -> int:
        return self._global_t

    @property
    def pseudo_episode(self) -> int:
        return self._pseudo_episode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        """Hook for subclasses to modify observations (e.g. strip time dim)."""
        return obs

    def _augment_info(
        self,
        info: Dict[str, Any],
        terminated: bool = False,
        truncated: bool = False,
        pseudo_reset: bool = False,
    ) -> Dict[str, Any]:
        info = dict(info)
        info["global_t"] = self._global_t
        info["pseudo_episode"] = self._pseudo_episode
        info["steps_in_pseudo_episode"] = self._steps_in_pseudo_ep
        if terminated or truncated:
            info["pseudo_reset_occurred"] = True
        if pseudo_reset:
            info["is_post_pseudo_reset_obs"] = True
        return info
