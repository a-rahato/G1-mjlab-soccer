"""Train a frozen-base residual goalkeeper.

The keeper's useful diving behavior is already encoded in a base policy. Direct
PPO fine-tuning of the whole actor is fragile, so this script freezes the base
and trains only a small bounded residual head:

  action = base_policy(obs) + residual_scale * tanh(residual(obs))

The reward can include block-rate terms plus clear/strike shaping, but the base
policy remains intact and every eval block rolls back if the deterministic
policy gets worse.
"""
from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from src.tasks.soccer import mdp
from src.tasks.soccer.mdp.goalkeeper_rewards import (
  _reset_gk_state,
  goalkeeper_body_intercept,
  goalkeeper_clear_ball,
  goalkeeper_goal_conceded,
  goalkeeper_intercept_point,
  goalkeeper_posture_orientation,
  goalkeeper_stop_ball,
  goalkeeper_strike_through,
)
from src.tasks.soccer.mdp.shooter_rewards import action_rate_l2_clip


@dataclass
class Cfg:
  base: str = "src/assets/soccer/weight/goalkeeper_polished_v2.pt"
  out: str = "logs/experiments/residual_keeper.pt"
  num_envs: int = 512
  warmup: int = 20
  block_iters: int = 10
  blocks: int = 20
  eval_resets: int = 2
  lr: float = 3.0e-4
  std: float = 0.05
  residual_scale: float = 0.15
  residual_head: tuple[int, ...] = (512, 256, 128)
  train_regions: tuple[int, ...] = ()
  w_conceded: float = 20.0
  w_intercept: float = 2.0
  w_body: float = 1.0
  w_sharp: float = 0.5
  sharp_std: float = 0.12
  w_cross: float = 4.0
  w_stop: float = 1.0
  w_clear: float = 0.3
  w_strike: float = 0.05
  w_posture: float = 0.5
  clear_min_vx: float = 0.5
  strike_min_vx: float = 0.2
  seed: int = 2810
  device: str = "cuda:0"


def _crossing_contact(env, gate=0.20, std=0.10):
  ball = env.scene["ball"]
  robot = env.scene["robot"]
  bp = ball.data.root_link_pos_w
  bx = bp[:, 0] - env.scene.env_origins[:, 0]
  body = robot.data.body_link_pos_w
  d2 = (body - bp.unsqueeze(1)).pow(2).sum(-1).min(1).values
  return (bx.abs() < gate).float() * torch.exp(-d2 / (std * std))


def _nan_termination(env):
  robot = env.scene["robot"]
  bad = torch.isnan(robot.data.joint_pos).any(-1) | torch.isinf(robot.data.joint_pos).any(-1)
  root_pos = robot.data.root_link_pos_w
  return bad | torch.isnan(root_pos).any(-1) | torch.isinf(root_pos).any(-1)


def _sanitize_step(env_wrapper):
  inner = env_wrapper.unwrapped
  orig = inner.step

  def safe(action):
    obs, rew, term, trunc, extras = orig(action)
    for key in list(obs.keys()):
      obs[key] = torch.nan_to_num(obs[key], nan=0.0, posinf=0.0, neginf=0.0)
    rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
    return obs, rew, term, trunc, extras

  inner.step = safe


def _eval(env, policy, ball, n_steps=150, n_resets=2):
  num_envs = env.unwrapped.num_envs
  org = env.unwrapped.scene.env_origins
  blocked = 0
  total = 0
  with torch.inference_mode():
    for _ in range(n_resets):
      obs, _ = env.reset()
      entered = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
      for _ in range(n_steps):
        obs = env.step(policy(obs))[0]
        bp = ball.data.root_link_pos_w
        entered |= ((bp[:, 0] - org[:, 0]) <= -0.5) & ((bp[:, 1] - org[:, 1]).abs() <= 1.5) & (bp[:, 2] <= 1.8)
      blocked += int((~entered).sum())
      total += num_envs
  return blocked / total


def _runner_cfg(cfg: Cfg):
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="src.tasks.soccer.modules.gk_residual.GoalkeeperResidual",
      hidden_dims=cfg.residual_head,
      activation="elu",
      obs_normalization=False,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": cfg.std, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 256), activation="elu", obs_normalization=False),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.1,
      entropy_coef=0.0,
      num_learning_epochs=3,
      num_mini_batches=4,
      learning_rate=cfg.lr,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.003,
      max_grad_norm=0.5,
    ),
    experiment_name="g1_goalkeeper_residual",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def main(cfg: Cfg):
  configure_torch_backends()
  torch.manual_seed(cfg.seed)

  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  if cfg.train_regions:
    import copy as _copy

    rb = env_cfg.events["reset_ball"]
    vc = _copy.copy(rb.params["vel_cfg"])
    vc.regions = [vc.regions[i] for i in cfg.train_regions]
    rb.params["vel_cfg"] = vc
    print(f"[INFO] specialist training on regions {tuple(cfg.train_regions)}", flush=True)

  env_cfg.rewards = {
    "goal_conceded": RewardTermCfg(func=goalkeeper_goal_conceded, weight=-cfg.w_conceded, params={}),
    "intercept": RewardTermCfg(func=goalkeeper_intercept_point, weight=cfg.w_intercept, params={"std": 0.4}),
    "body": RewardTermCfg(func=goalkeeper_body_intercept, weight=cfg.w_body, params={"std": 0.35}),
    "sharp": RewardTermCfg(func=goalkeeper_body_intercept, weight=cfg.w_sharp, params={"std": cfg.sharp_std}),
    "cross": RewardTermCfg(func=_crossing_contact, weight=cfg.w_cross, params={"gate": 0.20, "std": 0.10}),
    "stop_ball": RewardTermCfg(func=goalkeeper_stop_ball, weight=cfg.w_stop,
                               params={"velocity_drop_threshold": 2.0, "goal_x": -0.5}),
    "clear_ball": RewardTermCfg(func=goalkeeper_clear_ball, weight=cfg.w_clear,
                                params={"min_clear_vx": cfg.clear_min_vx, "goal_x": -0.5}),
    "strike_through": RewardTermCfg(func=goalkeeper_strike_through, weight=cfg.w_strike,
                                    params={"min_limb_vx": cfg.strike_min_vx}),
    "posture": RewardTermCfg(func=goalkeeper_posture_orientation, weight=cfg.w_posture),
    "action_rate": RewardTermCfg(func=action_rate_l2_clip, weight=-0.1),
  }
  env_cfg.events["reset_gk_state"] = EventTermCfg(func=_reset_gk_state, mode="reset", params={})
  env_cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan_guard": TerminationTermCfg(func=_nan_termination, time_out=False),
  }

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device), clip_actions=100.0)
  _sanitize_step(env)

  import src.tasks.soccer.modules.gk_residual as gkr

  gkr.BASE_CKPT = cfg.base
  gkr.BASE_HIDDEN = (1024, 512, 256)
  gkr.RESIDUAL_SCALE = cfg.residual_scale
  runner = MjlabOnPolicyRunner(env, asdict(_runner_cfg(cfg)), device=cfg.device)
  with torch.no_grad():
    runner.alg.actor.distribution.std_param.fill_(cfg.std)
  print(f"[INFO] frozen base {cfg.base}", flush=True)
  print(f"[INFO] residual scale {cfg.residual_scale}, head {tuple(cfg.residual_head)}", flush=True)

  ball = env.unwrapped.scene["ball"]
  policy = runner.get_inference_policy(device=cfg.device)

  if cfg.warmup > 0:
    for p in runner.alg.actor.parameters():
      p.requires_grad_(False)
    runner.learn(num_learning_iterations=cfg.warmup, init_at_random_ep_len=True)
    for p in runner.alg.actor.residual.parameters():
      p.requires_grad_(True)
    with torch.no_grad():
      runner.alg.actor.distribution.std_param.fill_(cfg.std)

  os.makedirs(os.path.dirname(cfg.out), exist_ok=True)

  def save():
    saved = runner.alg.save()
    saved["iter"] = 0
    saved["infos"] = {"env_state": {"common_step_counter": 0}}
    torch.save(saved, cfg.out)

  best = _eval(env, policy, ball, n_resets=cfg.eval_resets)
  best_state = copy.deepcopy(runner.alg.actor.state_dict())
  save()
  print(f"[EVAL] init block {100 * best:.1f}%", flush=True)
  for block in range(cfg.blocks):
    runner.learn(num_learning_iterations=cfg.block_iters, init_at_random_ep_len=False)
    with torch.no_grad():
      runner.alg.actor.distribution.std_param.clamp_(min=1e-3, max=cfg.std)
    score = _eval(env, policy, ball, n_resets=cfg.eval_resets)
    tag = ""
    if score >= best:
      best = score
      best_state = copy.deepcopy(runner.alg.actor.state_dict())
      save()
      tag = " *best* (saved)"
    elif score < best - 0.02:
      runner.alg.actor.load_state_dict(best_state)
      tag = " rollback"
    print(f"[EVAL] block {block+1}/{cfg.blocks}: {100 * score:.1f}%  (best {100 * best:.1f}%){tag}", flush=True)

  runner.alg.actor.load_state_dict(best_state)
  save()
  print(f"[INFO] saved residual keeper (best {100 * best:.1f}%) to {cfg.out}", flush=True)
  env.close()


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa

  main(tyro.cli(Cfg, prog="train_residual_keeper"))
