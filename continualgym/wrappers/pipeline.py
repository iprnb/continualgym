"""
Pipeline factory
================
`make_continual` is the high-level entry point that stacks wrappers in the
correct order and wires together the global-step counter.

Wrapper order (outer → inner):
  NonStationaryWrapper  ← applies drift, sees global_t via inner chain
      ContinuingWrapper ← hides done signals, manages pseudo-resets
          <base env>
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym

from continualgym.drift import DriftSchedule
from continualgym.wrappers.continuing import ContinuingWrapper
from continualgym.wrappers.nonstationary import NonStationaryWrapper


def make_continual(
    env_id: str,
    *,
    # --- ContinuingWrapper params ---
    hide_done: bool = True,
    pseudo_reset_penalty: float = 0.0,
    max_steps_per_pseudo_episode: Optional[int] = None,
    # --- NonStationaryWrapper params ---
    reward_scale_schedule: Optional[DriftSchedule] = None,
    reward_shift_schedule: Optional[DriftSchedule] = None,
    obs_noise_std_schedule: Optional[DriftSchedule] = None,
    action_corruption_prob_schedule: Optional[DriftSchedule] = None,
    action_noise_std_schedule: Optional[DriftSchedule] = None,
    obs_mask_prob_schedule: Optional[DriftSchedule] = None,
    physics_modifiers: Optional[Dict[str, Tuple[DriftSchedule, float, float]]] = None,
    reward_functions: Optional[List[Any]] = None,
    reward_fn_switch_schedule: Optional[DriftSchedule] = None,
    clip_obs: bool = True,
    seed: Optional[int] = None,
    # --- General ---
    env_kwargs: Optional[Dict[str, Any]] = None,
    render_mode: Optional[str] = None,
) -> NonStationaryWrapper:
    """
    Create a fully configured continual RL environment.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID (e.g. 'CartPole-v1', 'HalfCheetah-v4').
    hide_done : bool
        Suppress terminal signals (default True).
    pseudo_reset_penalty : float
        Reward penalty at internal episode resets.
    max_steps_per_pseudo_episode : int | None
        Force pseudo-reset every N steps.
    reward_scale_schedule : DriftSchedule | None
        Multiplicative reward drift.
    reward_shift_schedule : DriftSchedule | None
        Additive reward drift.
    obs_noise_std_schedule : DriftSchedule | None
        Observation noise std drift.
    action_corruption_prob_schedule : DriftSchedule | None
        Action corruption probability drift.
    action_noise_std_schedule : DriftSchedule | None
        Continuous action noise std drift.
    obs_mask_prob_schedule : DriftSchedule | None
        Observation masking probability drift.
    physics_modifiers : dict | None
        Physics parameter drift (env-specific).
    reward_functions : list | None
        Alternative reward functions for reward-function drift.
    reward_fn_switch_schedule : DriftSchedule | None
        Schedule selecting which reward function to use.
    clip_obs : bool
        Clip perturbed observations to original space bounds.
    seed : int | None
        RNG seed.
    env_kwargs : dict | None
        Extra keyword arguments passed to gym.make().
    render_mode : str | None
        Render mode for the base environment.

    Returns
    -------
    NonStationaryWrapper
        Fully stacked environment ready for continual RL training.

    Examples
    --------
    >>> from continualgym import make_continual, SinusoidalDrift
    >>> env = make_continual(
    ...     "CartPole-v1",
    ...     obs_noise_std_schedule=SinusoidalDrift(center=0.05, amplitude=0.04),
    ...     reward_shift_schedule=SinusoidalDrift(center=0.0, amplitude=0.5, period=20_000),
    ...     hide_done=True,
    ... )
    >>> obs, info = env.reset()
    >>> for _ in range(1000):
    ...     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """
    kwargs = env_kwargs or {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    base_env = gym.make(env_id, **kwargs)

    continuing_env = ContinuingWrapper(
        base_env,
        hide_done=hide_done,
        pseudo_reset_penalty=pseudo_reset_penalty,
        max_steps_per_pseudo_episode=max_steps_per_pseudo_episode,
    )

    any_drift = any([
        reward_scale_schedule,
        reward_shift_schedule,
        obs_noise_std_schedule,
        action_corruption_prob_schedule,
        action_noise_std_schedule,
        obs_mask_prob_schedule,
        physics_modifiers,
        reward_functions,
    ])

    if any_drift:
        env = NonStationaryWrapper(
            continuing_env,
            reward_scale_schedule=reward_scale_schedule,
            reward_shift_schedule=reward_shift_schedule,
            obs_noise_std_schedule=obs_noise_std_schedule,
            action_corruption_prob_schedule=action_corruption_prob_schedule,
            action_noise_std_schedule=action_noise_std_schedule,
            obs_mask_prob_schedule=obs_mask_prob_schedule,
            physics_modifiers=physics_modifiers,
            reward_functions=reward_functions,
            reward_fn_switch_schedule=reward_fn_switch_schedule,
            global_t_source="inner",
            clip_obs=clip_obs,
            seed=seed,
        )
    else:
        # Still wrap for consistent API but no drift applied
        env = NonStationaryWrapper(
            continuing_env,
            global_t_source="inner",
            seed=seed,
        )

    return env
