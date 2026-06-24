"""Goalkeeper-specific reward functions for training.

Matches the Humanoid-Goalkeeper paper's reward set (24 terms), prioritized
to 7 goalkeeper-specific functions. The remaining regularization terms
(action_rate, joint_pos_limits, etc.) are reused from existing modules.

State tracking follows the pattern in training_rewards.py: per-environment
tensors stored on the env object via _gk_get_or_init_state, reset via a
dedicated reset event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.entity import Entity


def _gk_get_or_init_state(
  env: ManagerBasedRlEnv, name: str, default: float, dtype=torch.float32,
) -> torch.Tensor:
  t = getattr(env, name, None)
  if t is None or t.shape[0] != env.num_envs:
    t = torch.full((env.num_envs,), default, dtype=dtype, device=env.device)
    setattr(env, name, t)
  return t


def _resolve_ee_indices(
  robot: Entity, ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Look up body indices for named end-effector bodies at runtime."""
  return torch.as_tensor(
    robot.find_bodies(ee_body_names, preserve_order=True)[0],
    device=robot.data.body_link_pos_w.device,
  )


# -- Reset event ---------------------------------------------------------------


def _reset_gk_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
) -> None:
  """Reset per-environment GK tracking state on episode start."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  for key in ("_gk_max_ball_speed", "_gk_block_awarded", "_gk_conceded", "_gk_clear_awarded"):
    t = getattr(env, key, None)
    if t is not None and t.shape[0] == env.num_envs:
      t[env_ids] = 0.0


# -- Reward functions ----------------------------------------------------------


def goalkeeper_ee_reach(
  env: ManagerBasedRlEnv,
  std: float = 0.3,
  planar: bool = False,
  near_gate_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
) -> torch.Tensor:
  """End-effector reach reward: exponential distance to ball.

  Paper's 'eereach' — encourages hands and feet to reach toward the ball,
  rewarding proximity of 4 end-effectors (both hands + both feet) via an
  exponential decay: sum_i exp(-||ee_i - ball||² / σ²).

  If ``planar`` is True, only the lateral/vertical (y, z) components of the
  offset are used (the forward x axis is ignored). This is important for the
  goalkeeper: the ball starts 3-5m in front, so a full 3D reach reward pulls the
  keeper into a forward lunge that topples it. Rewarding y-z alignment instead
  makes the keeper stay on its line (x≈0) and move a hand/foot to the ball's
  crossing point — the actual blocking behavior.

  If ``near_gate_x`` > 0, the reward is gated to fire only while the ball is
  within ``near_gate_x`` metres of the keeper plane (|ball_x_rel| < near_gate_x).
  Combined with full 3D distance (planar=False), this rewards an actual
  interception at the moment the ball arrives — it cannot be gamed by loose
  alignment, and being gated it does not pull the keeper into an early lunge at
  the far ball.

  Weight: 10.0 (matches paper).
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]

  ball_pos = ball.data.root_link_pos_w  # (B, 3)
  ee_indices = _resolve_ee_indices(robot, ee_body_names)
  ee_pos = robot.data.body_link_pos_w[:, ee_indices]  # (B, N, 3)

  delta = ee_pos - ball_pos.unsqueeze(1)  # (B, N, 3)
  if planar:
    delta = delta[..., 1:]  # keep (y, z) only — ignore forward x
  dist_sq = torch.sum(delta * delta, dim=-1)  # (B, N)
  reward_per_ee = torch.exp(-dist_sq / (std * std))  # (B, N)
  reward = reward_per_ee.sum(dim=-1)  # (B,)

  if near_gate_x > 0.0:
    ball_x_rel = ball_pos[:, 0] - env.scene.env_origins[:, 0]
    gate = (ball_x_rel.abs() < near_gate_x).to(reward.dtype)
    reward = reward * gate

  return reward


def goalkeeper_body_intercept(
  env: ManagerBasedRlEnv,
  std: float = 0.35,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Dense whole-body blocking reward: exp(-min_dist(ball, ANY body link)²/σ²).

  A block is ANY body part intersecting the ball, so rewarding the minimum
  distance from the ball to any of the robot's body links is directly aligned
  with blocking and dense whenever the ball is near the keeper.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  ball_pos = ball.data.root_link_pos_w           # (B,3)
  body_pos = robot.data.body_link_pos_w          # (B,L,3)
  d2 = (body_pos - ball_pos.unsqueeze(1)).pow(2).sum(dim=-1)
  return torch.exp(-d2.min(dim=1).values / (std * std))


def goalkeeper_intercept_point(
  env: ManagerBasedRlEnv,
  std: float = 0.4,
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
  ),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Dense reward: nearest end-effector close to the ball's INTERCEPTION POINT.

  We extrapolate the ball's trajectory (ballistic, with gravity) to the keeper's
  plane (x_rel = 0) to get the exact (y, z) where the ball will cross, then
  reward the nearest hand/foot for being at that 3D point. Unlike rewarding
  proximity to the ball's *current* position (which pulls a forward lunge) or
  loose planar alignment (which the keeper games without intercepting), this
  gives a STABLE, directional target over the whole flight, so the keeper learns
  to pre-position a limb exactly where the ball will arrive — the actual block.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  bp = ball.data.root_link_pos_w           # (B,3) world
  bv = ball.data.root_link_lin_vel_w       # (B,3)
  org = env.scene.env_origins              # (B,3)

  ball_x_rel = bp[:, 0] - org[:, 0]
  vx = bv[:, 0]
  # time for the ball to reach the keeper plane (x_rel = 0); 0 if already past.
  t = torch.clamp(-ball_x_rel / (vx - 1e-3), min=0.0, max=2.0)
  g = 9.81
  cross = torch.empty_like(bp)
  cross[:, 0] = org[:, 0]                              # keeper plane (x_rel = 0)
  cross[:, 1] = bp[:, 1] + bv[:, 1] * t
  cross[:, 2] = torch.clamp(bp[:, 2] + bv[:, 2] * t - 0.5 * g * t * t, min=0.0)

  # Whole-body: reward ANY body link being at the interception point (a block is
  # any body part on the ball, not just a hand). This is the precise, adaptive
  # target the keeper should reach — independent of which fixed dive it uses.
  body = robot.data.body_link_pos_w                   # (B,L,3)
  d2 = (body - cross.unsqueeze(1)).pow(2).sum(dim=-1)  # (B,L)
  return torch.exp(-d2.min(dim=1).values / (std * std))


def goalkeeper_stop_ball(
  env: ManagerBasedRlEnv,
  velocity_drop_threshold: float = 2.0,
  behind_robot_x_threshold: float = 0.0,
  goal_x: float = -0.5,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """One-shot block reward when the ball is deflected before entering the goal.

  Coordinate system (see goalkeeper_ball_reset.py): robot faces +x; ball comes
  from +x (front) toward -x; the goal plane is at x = goal_x (-0.5). A successful
  save = the ball is decelerated significantly while it is still in front of the
  goal plane (i.e. the keeper has deflected/stopped it before it scores).

  Tracks per-environment max ball speed and awards 1.0 when:
    1. max_speed - current_speed > velocity_drop_threshold (2.0 m/s), AND
    2. the ball has NOT yet crossed the goal plane (ball_x > goal_x).

  The reward is awarded at most once per episode per environment.
  (Was previously gated on ``ball_x > robot_x`` — an inverted condition copied
  from a stale "ball comes from -x" coordinate convention.)

  Weight: 100.0 (matches paper's stopball=100).
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]

  ball_pos = ball.data.root_link_pos_w  # (B, 3)
  ball_vel = ball.data.root_link_lin_vel_w  # (B, 3)
  robot_pos = robot.data.root_link_pos_w  # (B, 3)

  current_speed = torch.norm(ball_vel, dim=-1)  # (B,)

  max_speed = _gk_get_or_init_state(env, "_gk_max_ball_speed", 0.0)
  block_awarded = _gk_get_or_init_state(env, "_gk_block_awarded", 0.0)

  # Update per-env max speed.
  max_speed = torch.maximum(max_speed, current_speed)
  setattr(env, "_gk_max_ball_speed", max_speed)

  # Ball origin is in env-local x; env_origins offset is added in world frame, so
  # compare against the goal plane in the robot/world-relative x of the ball.
  ball_x_rel = ball_pos[:, 0] - env.scene.env_origins[:, 0]
  speed_drop = (max_speed - current_speed) > velocity_drop_threshold
  ball_in_play = ball_x_rel > goal_x  # not yet scored
  block_detected = speed_drop & ball_in_play & (block_awarded < 0.5)

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  if torch.any(block_detected):
    ids = torch.nonzero(block_detected, as_tuple=False).squeeze(-1)
    reward[ids] = 1.0
    block_awarded[ids] = 1.0
    setattr(env, "_gk_block_awarded", block_awarded)

  return reward


def goalkeeper_clear_ball(
  env: ManagerBasedRlEnv,
  min_clear_vx: float = 0.5,
  near_goal_x: float = 0.25,
  goal_x: float = -0.5,
  goal_half_width: float = 1.5,
  goal_height: float = 1.8,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """One-shot reward for clearing the ball away from the goal.

  A normal block can merely slow the ball near the keeper. For harder shots we
  also want an active deflection: after contact, the ball should move back
  toward +x or leave the goal frame before it crosses the goal plane. This term
  rewards that safe clearance once per episode.
  """
  ball: Entity = env.scene[ball_cfg.name]
  ball_pos = ball.data.root_link_pos_w
  ball_vel = ball.data.root_link_lin_vel_w
  ball_x_rel = ball_pos[:, 0] - env.scene.env_origins[:, 0]
  ball_y_rel = ball_pos[:, 1] - env.scene.env_origins[:, 1]

  in_front_of_goal = ball_x_rel > goal_x
  near_keeper_or_goal = ball_x_rel < near_goal_x
  moving_out = ball_vel[:, 0] > min_clear_vx
  out_of_frame = (ball_y_rel.abs() > goal_half_width) | (ball_pos[:, 2] > goal_height)
  clear_detected = in_front_of_goal & near_keeper_or_goal & (moving_out | out_of_frame)

  clear_awarded = _gk_get_or_init_state(env, "_gk_clear_awarded", 0.0)
  fire = clear_detected & (clear_awarded < 0.5)

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(fire):
    ids = torch.nonzero(fire, as_tuple=False).squeeze(-1)
    reward[ids] = 1.0
    clear_awarded[ids] = 1.0
    setattr(env, "_gk_clear_awarded", clear_awarded)
  return reward


def goalkeeper_strike_through(
  env: ManagerBasedRlEnv,
  contact_std: float = 0.16,
  min_limb_vx: float = 0.2,
  near_x: float = 0.35,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
) -> torch.Tensor:
  """Dense reward for forceful limb motion through the ball toward +x.

  The ball travels from +x to -x. A stronger save should put a hand/foot near the
  ball and move that limb back toward +x, so the contact does not only absorb
  energy but actively clears the ball out. The distance gate keeps this from
  rewarding unrelated arm swinging far from the ball.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]

  ball_pos = ball.data.root_link_pos_w
  ball_x_rel = ball_pos[:, 0] - env.scene.env_origins[:, 0]
  ee_indices = _resolve_ee_indices(robot, ee_body_names)
  ee_pos = robot.data.body_link_pos_w[:, ee_indices]
  ee_vel = robot.data.body_link_lin_vel_w[:, ee_indices]

  dist_sq = torch.sum((ee_pos - ball_pos.unsqueeze(1)) ** 2, dim=-1)
  proximity = torch.exp(-dist_sq / (contact_std * contact_std))
  outward_vx = torch.clamp(ee_vel[:, :, 0] - min_limb_vx, min=0.0)
  per_limb = proximity * outward_vx

  near_plane = (ball_x_rel.abs() < near_x).to(per_limb.dtype)
  return torch.max(per_limb, dim=1).values * near_plane


def goalkeeper_goal_conceded(
  env: ManagerBasedRlEnv,
  goal_x: float = -0.5,
  goal_half_width: float = 1.5,
  goal_height: float = 1.8,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """One-shot penalty (returns 1.0 once) when the ball enters the goal.

  Mirrors the eval block criterion exactly (ball crosses x ≤ goal_x inside the
  |y| ≤ goal_half_width, z ≤ goal_height frame), so optimizing -weight·this term
  directly optimizes the evaluated block rate. Awarded at most once per episode.
  Intended for PPO fine-tuning of an already-competent (distilled) policy, where
  it provides a precisely-aligned learning signal that dense shaping rewards lack.
  """
  ball: Entity = env.scene[ball_cfg.name]
  ball_pos = ball.data.root_link_pos_w
  ball_x_rel = ball_pos[:, 0] - env.scene.env_origins[:, 0]
  ball_y_rel = ball_pos[:, 1] - env.scene.env_origins[:, 1]
  in_goal = (
    (ball_x_rel <= goal_x)
    & (ball_y_rel.abs() <= goal_half_width)
    & (ball_pos[:, 2] <= goal_height)
  )

  conceded = _gk_get_or_init_state(env, "_gk_conceded", 0.0)
  fire = in_goal & (conceded < 0.5)
  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(fire):
    ids = torch.nonzero(fire, as_tuple=False).squeeze(-1)
    reward[ids] = 1.0
    conceded[ids] = 1.0
    setattr(env, "_gk_conceded", conceded)
  return reward


def goalkeeper_stay_on_line(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize lateral deviation from goal center.

  Paper's 'stayonline' — discourages the robot from drifting sideways
  away from the center of the goal line.

  Weight: -2.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  robot_y = robot.data.root_link_pos_w[:, 1]
  return torch.abs(robot_y)


def goalkeeper_no_retreat(
  env: ManagerBasedRlEnv,
  goal_line_x: float = 0.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize retreating behind the goal line.

  Paper's 'noretreat' — discourages the robot from backing into its own goal.
  Coordinate system (see goalkeeper_ball_reset.py): robot faces +x at x≈0, the
  ball comes from +x, and the goal is behind at -x. "Retreating" therefore means
  moving toward -x. We penalize robot_x dropping below goal_line_x.
  (Was previously ``clamp(robot_x - goal_line_x)`` — penalizing forward motion
  toward the ball, the opposite of the intent, from a stale coordinate frame.)

  Weight: -2.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  robot_x = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  retreat = torch.clamp(goal_line_x - robot_x, min=0.0)
  return retreat * retreat


def goalkeeper_feet_slippage(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  foot_body_names: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
  contact_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Penalize foot sliding when feet are in ground contact.

  Paper's 'feet_slippage' — returns the sum of squared xy-velocities
  for feet that are currently touching the ground. This discourages
  slipping while moving laterally.

  Weight: -3.0 (exp(-10*vel) → direct penalty).
  """
  robot: Entity = env.scene[robot_cfg.name]
  foot_indices = _resolve_ee_indices(robot, foot_body_names)
  foot_vel_w = robot.data.body_link_lin_vel_w[:, foot_indices]  # (B, 2, 3)
  foot_vel_xy_sq = torch.sum(foot_vel_w[:, :, :2] ** 2, dim=-1)  # (B, 2)

  # Check foot-ground contact via sensor.
  # feet_ground_contact uses reduce="netforce", single slot.
  contact = env.scene.sensors.get(contact_sensor_name, None)
  if contact is not None:
    found = contact.data.force[:, 0]  # (B, 3) — net force on feet
    contact_found = torch.norm(found, dim=-1) > 1.0  # (B,)
  else:
    contact_found = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  slip = foot_vel_xy_sq.sum(dim=-1)  # (B,)
  return slip * contact_found.float()


def goalkeeper_posture_orientation(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward upright posture via projected gravity.

  Paper's 'postorientation'. ``projected_gravity_b`` is the (normalized) gravity
  direction expressed in the robot base frame, so its z-component is ≈ -1.0 when
  the robot is perfectly upright and rises toward 0 as it tilts to horizontal.
  We therefore reward ``clamp(-grav_z, 0, 1)`` → 1.0 upright, 0.0 on its side.
  (Was ``clamp(grav_z, 0, 1)``, which is ≈0 for an upright robot and never fired
  — a sign error that removed the dense "stay upright" signal.)

  Weight: 3.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  grav_z = robot.data.projected_gravity_b[:, 2]
  return torch.clamp(-grav_z, 0.0, 1.0)


def goalkeeper_ang_vel_xy(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize base angular velocity in the xy-plane.

  Paper's 'ang_vel_xy' — high angular velocity in pitch/roll axes
  indicates instability during reactive movements.

  Weight: -0.1 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  ang_vel = robot.data.root_link_ang_vel_b[:, :2]  # (B, 2)
  return torch.sum(ang_vel * ang_vel, dim=-1)  # (B,)
