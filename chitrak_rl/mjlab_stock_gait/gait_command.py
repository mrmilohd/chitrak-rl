"""Gait-conditioned velocity command.

Extends mjlab's stock UniformVelocityCommand (velocity_command.py) by
subclassing it directly -- all existing behavior (lin_vel_x/y/ang_vel_z
sampling, heading control, standing/forward/world-frame envs, the GUI/
debug-vis hooks) is inherited as-is via super() calls, not reimplemented.
What's added here is purely the gait-clock machinery: 5 extra command
dimensions (frequency, phase, offset, bound, duration) and the per-step
phase-accumulator -> per-foot phase -> desired_contact_states pipeline,
ported from Walk These Ways' _step_contact_targets() (legged_robot.py).

Gait sampling has NO curriculum -- one of 4 named gaits (trot/pace/bound/
pronk) is chosen with equal 1/4 probability each resample, and phase/offset/
bound are then drawn directly from that gait's target sub-range. This is
mathematically identical to WTW's own "sample uniform over the full range,
then remap" two-step (confirmed: x/2 + 0.25 applied to Uniform(0,1) IS
Uniform(0.25, 0.75)) -- just without the redundant intermediate step.
`duration` is fixed at 0.5, matching WTW's actual published config
(gait_duration_cmd_range = [0.5, 0.5], not randomized despite the name).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class GaitConditionedCommand(UniformVelocityCommand):
  cfg: GaitConditionedCommandCfg

  def __init__(self, cfg: GaitConditionedCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n = self.num_envs
    device = self.device

    # Gait-shape parameters, written by _resample_command, read by _update_command.
    self.gait_frequency = torch.full((n,), sum(cfg.gait_frequency_range) / 2, device=device)
    self.gait_duration = torch.full((n,), cfg.gait_duration, device=device)
    self.gait_phase = torch.zeros(n, device=device)
    self.gait_offset = torch.zeros(n, device=device)
    self.gait_bound = torch.zeros(n, device=device)

    # Non-gait-category command dims (jump/footswing/pose/stance) -- per the
    # confirmed WTW behavior, these are sampled completely independently of
    # which gait category got picked, just plain uniform draws each resample.
    self.body_height = torch.zeros(n, device=device)
    self.footswing_height = torch.zeros(n, device=device)
    self.pitch = torch.zeros(n, device=device)
    self.roll = torch.zeros(n, device=device)
    self.stance_width = torch.zeros(n, device=device)
    self.stance_length = torch.zeros(n, device=device)

    # Persistent phase accumulator (WTW's gait_indices) -- advanced every step,
    # never reset by _resample_command (only by episode reset, via `reset()`).
    self.gait_indices = torch.zeros(n, device=device)

    # Outputs -- what reward/observation terms will actually read.
    # Order is [FR, FL, RR, RL] -- matching every sensor/site already in this
    # codebase (feet_ground_contact, foot_height_scan, foot site ordering),
    # NOT WTW's own literal array order ([FL, FR, RL, RR] in
    # _step_contact_targets()). The per-foot FORMULA assignment below is
    # still exactly WTW's (FL always gets phase+offset+bound combined; FR
    # gets offset alone; RL gets bound alone; RR gets phase alone) -- only
    # the array slot each foot's result lands in has been changed, so it
    # lines up with the rest of this codebase instead of WTW's own.
    self.foot_phase = torch.zeros(n, 4, device=device)
    self.desired_contact_states = torch.zeros(n, 4, device=device)
    self.clock_inputs = torch.zeros(n, 4, device=device)

  def reset(self, env_ids: torch.Tensor) -> dict:
    # CommandTerm.reset() returns a metrics dict the manager logs (.items()
    # is called on it) -- this override silently returned None instead until
    # caught by actually running the task, breaking CommandManager.reset().
    extras = super().reset(env_ids)
    self.gait_indices[env_ids] = 0.0
    return extras

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Velocity / heading / standing / forward-env sampling: unchanged, inherited.
    super()._resample_command(env_ids)

    n = len(env_ids)
    r = torch.empty(n, device=self.device)

    self.gait_frequency[env_ids] = r.uniform_(*self.cfg.gait_frequency_range)
    self.gait_duration[env_ids] = self.cfg.gait_duration  # fixed, not randomized.

    # Plain independent uniform draws, no gait-category involvement at all
    # (confirmed: WTW's gaitwise_curricula override only ever touches
    # phase/offset/bound -- indices 5,6,7 -- never these).
    self.body_height[env_ids] = r.uniform_(*self.cfg.body_height_range)
    self.footswing_height[env_ids] = r.uniform_(*self.cfg.footswing_height_range)
    self.pitch[env_ids] = r.uniform_(*self.cfg.pitch_range)
    self.roll[env_ids] = r.uniform_(*self.cfg.roll_range)
    self.stance_width[env_ids] = r.uniform_(*self.cfg.stance_width_range)
    self.stance_length[env_ids] = r.uniform_(*self.cfg.stance_length_range)

    # Choose one of 4 named gaits, 1/4 probability each.
    category = torch.randint(0, 4, (n,), device=self.device)  # 0=trot 1=pace 2=bound 3=pronk
    trot_ids = env_ids[category == 0]
    pace_ids = env_ids[category == 1]
    bound_ids = env_ids[category == 2]
    pronk_ids = env_ids[category == 3]

    # Default: all three offset terms at 0 (overwritten below per-category).
    self.gait_phase[env_ids] = 0.0
    self.gait_offset[env_ids] = 0.0
    self.gait_bound[env_ids] = 0.0

    if len(trot_ids) > 0:
      self.gait_phase[trot_ids] = torch.empty(len(trot_ids), device=self.device).uniform_(0.25, 0.75)
    if len(pace_ids) > 0:
      self.gait_offset[pace_ids] = torch.empty(len(pace_ids), device=self.device).uniform_(0.25, 0.75)
    if len(bound_ids) > 0:
      self.gait_bound[bound_ids] = torch.empty(len(bound_ids), device=self.device).uniform_(0.25, 0.75)
    if len(pronk_ids) > 0:
      m = len(pronk_ids)
      self.gait_phase[pronk_ids] = torch.empty(m, device=self.device).uniform_(-0.25, 0.25) % 1.0
      self.gait_offset[pronk_ids] = torch.empty(m, device=self.device).uniform_(-0.25, 0.25) % 1.0
      self.gait_bound[pronk_ids] = torch.empty(m, device=self.device).uniform_(-0.25, 0.25) % 1.0

  def _update_command(self) -> None:
    # Heading control + standing-env zeroing: unchanged, inherited.
    super()._update_command()

    # Advance the shared per-env clock.
    self.gait_indices = torch.remainder(self.gait_indices + self._env.step_dt * self.gait_frequency, 1.0)

    # Per-foot raw phase, [FR, FL, RR, RL] (see __init__ note on ordering).
    raw_phase = torch.stack(
      [
        self.gait_indices + self.gait_offset,  # FR
        self.gait_indices + self.gait_phase + self.gait_offset + self.gait_bound,  # FL
        self.gait_indices + self.gait_phase,  # RR
        self.gait_indices + self.gait_bound,  # RL
      ],
      dim=1,
    )  # [B, 4]
    raw_phase = torch.remainder(raw_phase, 1.0)

    # Duration warp: stretch [0, duration) -> [0, 0.5) (stance), [duration, 1) -> [0.5, 1) (swing).
    duration = self.gait_duration.unsqueeze(1)  # [B, 1], broadcasts against [B, 4]
    stance_mask = raw_phase < duration
    warped = torch.where(
      stance_mask,
      raw_phase * (0.5 / duration),
      0.5 + (raw_phase - duration) * (0.5 / (1.0 - duration)),
    )
    self.foot_phase = warped
    self.clock_inputs = torch.sin(2 * math.pi * warped)

    # Smoothed (von Mises CDF) target contact state: ~1 in [0,0.5) stance, ~0 in [0.5,1) swing.
    kappa = self.cfg.kappa_gait_probs
    cdf = torch.distributions.normal.Normal(0, kappa).cdf
    self.desired_contact_states = cdf(warped) * (1 - cdf(warped - 0.5)) + cdf(warped - 1) * (
      1 - cdf(warped - 0.5 - 1)
    )


@dataclass(kw_only=True)
class GaitConditionedCommandCfg(UniformVelocityCommandCfg):
  gait_frequency_range: tuple[float, float] = (2.0, 4.0)
  """Sampling range for commanded step frequency (cycles/second)."""
  gait_duration: float = 0.5
  """Fixed stance-fraction of the gait cycle (matches WTW's published config -- not randomized)."""
  kappa_gait_probs: float = 0.07
  """Sharpness of the stance/swing transition smoothing (von Mises CDF), matches WTW's default."""

  # All six ranges below are WTW's actual published values (scripts/train.py),
  # sampled with zero gait-category dependence -- see _resample_command.
  body_height_range: tuple[float, float] = (-0.25, 0.15)
  footswing_height_range: tuple[float, float] = (0.03, 0.35)
  pitch_range: tuple[float, float] = (-0.4, 0.4)
  roll_range: tuple[float, float] = (-0.0, 0.0)  # WTW's own range is degenerate (always 0).
  stance_width_range: tuple[float, float] = (0.10, 0.45)
  stance_length_range: tuple[float, float] = (0.35, 0.45)

  def build(self, env: ManagerBasedRlEnv) -> GaitConditionedCommand:
    return GaitConditionedCommand(self, env)
