"""Stock Go1 flat-terrain task, with mjlab's own reward set replaced by a
faithful port of Walk These Ways' reward functions (go1_gym/envs/rewards/corl_rewards.py),
plus a GaitConditionedCommand (gait_command.py) replacing the plain 3-dim
velocity command.

All ~19 of WTW's active reward functions are now ported (the full list from
the "exact observation/reward space" breakdown earlier in this build), now
that GaitConditionedCommand supplies everything the gait-dependent ones need
(desired_contact_states, foot_phase, and the body-height/footswing/pitch/
roll/stance command dims).

Every weight below is WTW's *actual final published* value (scripts/train.py,
last write wins over go1_config.py and the library defaults).
"""

from __future__ import annotations

import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_apply_yaw,
  quat_conjugate,
  quat_from_angle_axis,
  quat_mul,
)

from .gait_command import GaitConditionedCommandCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# ---------------------------------------------------------------------------
# Reward functions -- ported from corl_rewards.py, mjlab func(env, **params) form.
# ---------------------------------------------------------------------------


def wtw_tracking_lin_vel(
  env, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, tracking_sigma: float = 0.25
) -> torch.Tensor:
  """Port of CoRLRewards._reward_tracking_lin_vel."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  lin_vel_error = torch.sum(torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
  return torch.exp(-lin_vel_error / tracking_sigma)


def wtw_tracking_ang_vel(
  env, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, tracking_sigma_yaw: float = 0.25
) -> torch.Tensor:
  """Port of CoRLRewards._reward_tracking_ang_vel."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-ang_vel_error / tracking_sigma_yaw)


def wtw_lin_vel_z(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Port of CoRLRewards._reward_lin_vel_z."""
  asset = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def wtw_ang_vel_xy(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Port of CoRLRewards._reward_ang_vel_xy."""
  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def wtw_torques(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Port of CoRLRewards._reward_torques.

  WTW sums squared torque over all 12 DOF; mjlab's per-actuator applied force
  (`asset.data.actuator_force`) is the equivalent quantity.
  """
  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.actuator_force), dim=1)


class wtw_dof_acc:
  """Port of CoRLRewards._reward_dof_acc.

  Needs the previous step's joint velocity, so this is a stateful class-based
  term (same __init__(cfg, env) / __call__(env, **params) shape as mjlab's own
  feet_swing_height) rather than a plain function.
  """

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    asset = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
    self.last_dof_vel = asset.data.joint_vel.clone()
    self.step_dt = env.step_dt

  def __call__(self, env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    dof_vel = asset.data.joint_vel
    cost = torch.sum(torch.square((self.last_dof_vel - dof_vel) / self.step_dt), dim=1)
    self.last_dof_vel = dof_vel.clone()
    return cost

  def reset(self, env_ids: torch.Tensor) -> None:
    # Avoid a one-step acceleration spike from comparing pre-reset and
    # post-reset velocity for envs that just got reset.
    pass  # joint_vel is read fresh from asset.data each call; nothing to clear here.


def wtw_action_rate(env) -> torch.Tensor:
  """Port of CoRLRewards._reward_action_rate.

  mjlab's ActionManager already tracks .action/.prev_action natively (no
  custom state needed), zeroed on reset -- same role as WTW's last_actions.
  """
  return torch.sum(torch.square(env.action_manager.prev_action - env.action_manager.action), dim=1)


def wtw_action_smoothness_1(env) -> torch.Tensor:
  """Port of CoRLRewards._reward_action_smoothness_1.

  WTW diffs joint_pos_target (post action-scale); this uses raw policy
  actions instead (the scale is a constant per-joint factor, so the shape of
  the penalty is preserved, just not pixel-identical) -- and does not
  replicate WTW's first-step zero-masking.
  """
  return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def wtw_action_smoothness_2(env) -> torch.Tensor:
  """Port of CoRLRewards._reward_action_smoothness_2 (second-order action diff)."""
  return torch.sum(
    torch.square(
      env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action
    ),
    dim=1,
  )


def wtw_collision(env, sensor_name: str, force_threshold: float = 0.1) -> torch.Tensor:
  """Port of CoRLRewards._reward_collision.

  Counts how many penalized geoms (thigh/calf, matching WTW's
  penalize_contacts_on = ["thigh", "calf"]) are currently in contact above
  force_threshold -- a count, not a netted force, same as the original.
  """
  sensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force_mag = torch.norm(sensor.data.force, dim=-1)  # [B, N]
  return torch.sum((force_mag > force_threshold).float(), dim=1)


def wtw_dof_pos_limits(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Port of CoRLRewards._reward_dof_pos_limits, against mjlab's resolved soft limits."""
  asset = env.scene[asset_cfg.name]
  dof_pos = asset.data.joint_pos
  limits = asset.data.soft_joint_pos_limits  # [..., 2] = (lower, upper)
  out_of_limits = -(dof_pos - limits[..., 0]).clip(max=0.0)
  out_of_limits += (dof_pos - limits[..., 1]).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


def wtw_dof_vel(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Port of CoRLRewards._reward_dof_vel."""
  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel), dim=1)


class wtw_feet_slip:
  """Port of CoRLRewards._reward_feet_slip.

  Needs last step's contact state for the contact_filt = logical_or(contact,
  last_contacts) trick -- stateful for the same reason as wtw_dof_acc.
  """

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    sensor = env.scene[cfg.params["sensor_name"]]
    assert sensor.data.found is not None
    self.last_contacts = torch.zeros_like(sensor.data.found, dtype=torch.bool)

  def __call__(
    self, env, sensor_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
  ) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    asset = env.scene[asset_cfg.name]
    assert sensor.data.found is not None
    contact = sensor.data.found > 0
    contact_filt = torch.logical_or(contact, self.last_contacts)
    self.last_contacts = contact
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    foot_speed_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)
    return torch.sum(contact_filt.float() * foot_speed_sq, dim=1)


# ---------------------------------------------------------------------------
# Gait-dependent reward functions -- need GaitConditionedCommand's
# desired_contact_states / foot_phase, or the body-height/footswing/pose/
# stance command dims it samples. All read the command term directly via
# env.command_manager.get_term(command_name) for these (named attributes,
# not WTW's flat command-vector indices -- see gait_command.py).
# ---------------------------------------------------------------------------


def wtw_jump(
  env, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, base_height_target: float = 0.30
) -> torch.Tensor:
  """Port of CoRLRewards._reward_jump.

  Returns a NEGATIVE quantity, exactly like WTW's own function -- paired
  with WTW's positive published weight (10.0), net effect is a penalty.
  """
  asset = env.scene[asset_cfg.name]
  gait_term = env.command_manager.get_term(command_name)
  body_height = asset.data.root_link_pos_w[:, 2]
  jump_height_target = gait_term.body_height + base_height_target
  return -torch.square(body_height - jump_height_target)


def wtw_tracking_contacts_shaped_force(
  env, command_name: str, sensor_name: str, gait_force_sigma: float = 100.0
) -> torch.Tensor:
  """Port of CoRLRewards._reward_tracking_contacts_shaped_force.

  Penalizes nonzero foot contact force while that foot's gait phase says it
  should be in swing. `feet_ground_contact`'s force data and
  desired_contact_states are both in [FR, FL, RR, RL] order -- consistent,
  see gait_command.py.
  """
  gait_term = env.command_manager.get_term(command_name)
  sensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  foot_forces = torch.norm(sensor.data.force, dim=-1)  # [B, 4]
  desired_contact = gait_term.desired_contact_states  # [B, 4]
  cost = -(1 - desired_contact) * (1 - torch.exp(-(foot_forces**2) / gait_force_sigma))
  return torch.sum(cost, dim=1) / 4


def wtw_tracking_contacts_shaped_vel(
  env,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  gait_vel_sigma: float = 10.0,
) -> torch.Tensor:
  """Port of CoRLRewards._reward_tracking_contacts_shaped_vel.

  Penalizes nonzero foot velocity while that foot's gait phase says it
  should be in stance.
  """
  gait_term = env.command_manager.get_term(command_name)
  asset = env.scene[asset_cfg.name]
  foot_vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]  # [B, 4, 3]
  foot_speed = torch.norm(foot_vel, dim=-1)  # [B, 4]
  desired_contact = gait_term.desired_contact_states
  cost = -(desired_contact * (1 - torch.exp(-(foot_speed**2) / gait_vel_sigma)))
  return torch.sum(cost, dim=1) / 4


def wtw_feet_clearance_cmd_linear(
  env, command_name: str, height_sensor_name: str, foot_radius_offset: float = 0.02
) -> torch.Tensor:
  """Port of CoRLRewards._reward_feet_clearance_cmd_linear.

  Tracks a commanded swing-foot-height trajectory (triangular profile,
  peaking mid-swing at the commanded footswing_height), active only while
  that foot is in swing (gated by (1 - desired_contact_states)).
  """
  gait_term = env.command_manager.get_term(command_name)
  height_sensor = env.scene[height_sensor_name]
  foot_height = height_sensor.data.heights  # [B, 4], FR/FL/RR/RL order.
  phase = gait_term.foot_phase  # [B, 4], same order.
  triangular = 1 - torch.abs(1.0 - torch.clip(phase * 2.0 - 1.0, 0.0, 1.0) * 2.0)
  target_height = gait_term.footswing_height.unsqueeze(1) * triangular + foot_radius_offset
  cost = torch.square(target_height - foot_height) * (1 - gait_term.desired_contact_states)
  return torch.sum(cost, dim=1)


def wtw_orientation_control(
  env, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Port of CoRLRewards._reward_orientation_control.

  Tracks commanded body pitch/roll, by comparing actual projected gravity
  against the projected gravity a body at the COMMANDED pitch/roll would have.
  """
  asset = env.scene[asset_cfg.name]
  gait_term = env.command_manager.get_term(command_name)
  pitch_cmd = gait_term.pitch
  roll_cmd = gait_term.roll

  x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
  y_axis = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(env.num_envs, 3)
  quat_roll = quat_from_angle_axis(-roll_cmd, x_axis)
  quat_pitch = quat_from_angle_axis(-pitch_cmd, y_axis)
  desired_base_quat = quat_mul(quat_roll, quat_pitch)

  gravity_w = asset.data.gravity_vec_w
  desired_projected_gravity = quat_apply_inverse(desired_base_quat, gravity_w)
  actual_projected_gravity = quat_apply_inverse(asset.data.root_link_quat_w, gravity_w)

  return torch.sum(
    torch.square(actual_projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1
  )


def wtw_raibert_heuristic(
  env, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Port of CoRLRewards._reward_raibert_heuristic.

  Foot order here is [FR, FL, RR, RL] throughout -- matching this codebase's
  sensor/site convention (and matching WTW's OWN comment on its
  nominal-position arrays, "# nominal positions: [FR, FL, RR, RL]"). Note:
  WTW's original function pairs that FR/FL/RR/RL-ordered nominal-position
  array against `self.foot_indices`, which its own _step_contact_targets()
  fills in a DIFFERENT order (FL/FR/RL/RR) -- an internal inconsistency in
  the original code. This port does not replicate that: gait_term.foot_phase
  is already in FR/FL/RR/RL order (see gait_command.py), consistent with
  everything else used here.

  Sign convention also corrected for mjlab's y-left coordinate convention
  (confirmed from go1.xml: FR/RR sit at negative body-frame y, FL/RL at
  positive y -- deep-dive doc, Section 1b). WTW's own +/- signs were
  calibrated for Isaac Gym's coordinate convention and are flipped here
  rather than copied literally -- copying them as-is would make the policy
  try to spread its feet in the wrong left/right direction.
  """
  gait_term = env.command_manager.get_term(command_name)
  asset = env.scene[asset_cfg.name]

  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 4, 3]
  base_pos_w = asset.data.root_link_pos_w.unsqueeze(1)  # [B, 1, 3]
  base_quat = asset.data.root_link_quat_w  # [B, 4]

  translated = foot_pos_w - base_pos_w  # [B, 4, 3]
  base_quat_conj = quat_conjugate(base_quat).unsqueeze(1).repeat(1, 4, 1)  # [B, 4, 4]
  footsteps_b = quat_apply_yaw(base_quat_conj, translated)  # [B, 4, 3]

  stance_width = gait_term.stance_width.unsqueeze(1)
  stance_length = gait_term.stance_length.unsqueeze(1)

  # [FR, FL, RR, RL]. FR/RR = right side = negative y; FL/RL = left side = positive y.
  ys_nom = torch.cat(
    [-stance_width / 2, stance_width / 2, -stance_width / 2, stance_width / 2], dim=1
  )
  xs_nom = torch.cat(
    [stance_length / 2, stance_length / 2, -stance_length / 2, -stance_length / 2], dim=1
  )

  phase = torch.abs(1.0 - (gait_term.foot_phase * 2.0)) - 0.5  # [B, 4]
  frequencies = gait_term.gait_frequency.unsqueeze(1).clamp(min=1e-3)
  x_vel_des = gait_term.command[:, 0:1]
  yaw_vel_des = gait_term.command[:, 2:3]
  y_vel_des = yaw_vel_des * stance_length / 2

  ys_offset = phase * y_vel_des * (0.5 / frequencies)
  ys_offset[:, 2:4] *= -1  # Rear legs (RR, RL) get opposite sign.
  xs_offset = phase * x_vel_des * (0.5 / frequencies)

  desired_xs = xs_nom + xs_offset
  desired_ys = ys_nom + ys_offset
  desired_xy = torch.stack([desired_xs, desired_ys], dim=2)  # [B, 4, 2]

  err = torch.abs(desired_xy - footsteps_b[:, :, :2])
  return torch.sum(torch.square(err), dim=(1, 2))


def gait_clock_inputs(env, command_name: str) -> torch.Tensor:
  """clock_inputs observation -- reads GaitConditionedCommand's per-foot sin(2*pi*phase)."""
  gait_term = env.command_manager.get_term(command_name)
  return gait_term.clock_inputs


# ---------------------------------------------------------------------------
# Env config: stock Go1 flat task, rewards swapped out wholesale.
# ---------------------------------------------------------------------------


def mjlab_stock_gait_env_cfg(play: bool = False):
  cfg = unitree_go1_flat_env_cfg(play=play)

  # --- Commands: upgrade "twist" from UniformVelocityCommand to
  # GaitConditionedCommand, reusing the stock task's already-correct
  # velocity/heading settings rather than hardcoding them again.
  old_twist = cfg.commands["twist"]
  cfg.commands["twist"] = GaitConditionedCommandCfg(
    entity_name=old_twist.entity_name,
    resampling_time_range=old_twist.resampling_time_range,
    heading_command=old_twist.heading_command,
    heading_control_stiffness=old_twist.heading_control_stiffness,
    rel_standing_envs=old_twist.rel_standing_envs,
    rel_heading_envs=old_twist.rel_heading_envs,
    rel_forward_envs=old_twist.rel_forward_envs,
    debug_vis=old_twist.debug_vis,
    ranges=old_twist.ranges,
  )

  # --- Curriculum: reuse mjlab's own stock commands_vel curriculum
  # unchanged (Section 7f, deep-dive doc) -- same 3-stage schedule the stock
  # Go1 task itself uses, just for the velocity/yaw-rate dims. No curriculum
  # at all for the 5 gait-shape dims (frequency/phase/offset/bound/duration)
  # -- those are plain uniform/fixed sampling in GaitConditionedCommand,
  # matching WTW's actual published behavior (num_bins=1 on every gait dim).
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
        {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
        {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
      ],
    },
  )

  # WTW's collision reward needs a contact sensor on the penalized bodies
  # (thigh + calf) -- not present in the stock flat config (only the rough
  # variant builds it, for its illegal_contact termination). Same geom-naming
  # pattern as env_cfgs.py's thigh_ground_cfg/shank_ground_cfg.
  foot_names = ("FR", "FL", "RR", "RL")
  penalized_geom_names = tuple(f"{leg}_thigh_collision{i}" for leg in foot_names for i in (1, 2, 3)) + tuple(
    f"{leg}_calf_collision{i}" for leg in foot_names for i in (1, 2)
  )
  penalized_contact_cfg = ContactSensorCfg(
    name="penalized_contact",
    primary=ContactMatch(mode="geom", pattern=penalized_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (penalized_contact_cfg,)

  # --- Observations: clock_inputs, the one new term the gait clock adds.
  clock_term = ObservationTermCfg(func=gait_clock_inputs, params={"command_name": "twist"})
  cfg.observations["actor"].terms["clock_inputs"] = clock_term
  cfg.observations["critic"].terms["clock_inputs"] = clock_term

  # Replace mjlab's own reward set wholesale -- this task is meant to be the
  # WTW-faithful reward table, not a hybrid of mjlab's + WTW's.
  cfg.rewards = {
    "wtw_tracking_lin_vel": RewardTermCfg(
      func=wtw_tracking_lin_vel, weight=1.0, params={"command_name": "twist"}
    ),
    "wtw_tracking_ang_vel": RewardTermCfg(
      func=wtw_tracking_ang_vel, weight=0.5, params={"command_name": "twist"}
    ),
    "wtw_lin_vel_z": RewardTermCfg(func=wtw_lin_vel_z, weight=-0.02),
    "wtw_ang_vel_xy": RewardTermCfg(func=wtw_ang_vel_xy, weight=-0.001),
    "wtw_torques": RewardTermCfg(func=wtw_torques, weight=-1e-4),
    "wtw_dof_acc": RewardTermCfg(func=wtw_dof_acc, weight=-2.5e-7),
    "wtw_action_rate": RewardTermCfg(func=wtw_action_rate, weight=-0.01),
    "wtw_action_smoothness_1": RewardTermCfg(func=wtw_action_smoothness_1, weight=-0.1),
    "wtw_action_smoothness_2": RewardTermCfg(func=wtw_action_smoothness_2, weight=-0.1),
    "wtw_collision": RewardTermCfg(
      func=wtw_collision, weight=-5.0, params={"sensor_name": "penalized_contact"}
    ),
    "wtw_dof_pos_limits": RewardTermCfg(func=wtw_dof_pos_limits, weight=-10.0),
    "wtw_dof_vel": RewardTermCfg(func=wtw_dof_vel, weight=-1e-4),
    "wtw_feet_slip": RewardTermCfg(
      func=wtw_feet_slip,
      weight=-0.04,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=foot_names),
      },
    ),
    "wtw_jump": RewardTermCfg(func=wtw_jump, weight=10.0, params={"command_name": "twist"}),
    "wtw_tracking_contacts_shaped_force": RewardTermCfg(
      func=wtw_tracking_contacts_shaped_force,
      weight=4.0,
      params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
    ),
    "wtw_tracking_contacts_shaped_vel": RewardTermCfg(
      func=wtw_tracking_contacts_shaped_vel,
      weight=4.0,
      params={"command_name": "twist", "asset_cfg": SceneEntityCfg("robot", site_names=foot_names)},
    ),
    "wtw_feet_clearance_cmd_linear": RewardTermCfg(
      func=wtw_feet_clearance_cmd_linear,
      weight=-30.0,
      params={"command_name": "twist", "height_sensor_name": "foot_height_scan"},
    ),
    "wtw_orientation_control": RewardTermCfg(
      func=wtw_orientation_control, weight=-5.0, params={"command_name": "twist"}
    ),
    "wtw_raibert_heuristic": RewardTermCfg(
      func=wtw_raibert_heuristic,
      weight=-10.0,
      params={"command_name": "twist", "asset_cfg": SceneEntityCfg("robot", site_names=foot_names)},
    ),
  }

  return cfg


# ---------------------------------------------------------------------------
# Foot ordering used throughout this file and gait_command.py: [FR, FL, RR, RL].
# This matches the sensor/site convention already established in env_cfgs.py
# (feet_ground_contact, foot_height_scan), NOT WTW's own internal array order
# in _step_contact_targets() (which is [FL, FR, RL, RR]) -- see gait_command.py
# for the full explanation of the reorder and why it matters.
# ---------------------------------------------------------------------------
