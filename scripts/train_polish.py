"""RL polish of the distilled diving student toward the EXACT block objective.

Past RL here collapsed (forgetting / deterministic-flop / cold critic). This
version fixes all three:
  - init the actor from the good diving student, critic warm-up first;
  - ClampStd keeps the action std tiny so the DETERMINISTIC mean is what learns
    (local refinement, not high-variance flailing);
  - a BC ANCHOR pulls the actor toward the repair-oracle's blocking actions every
    update, preventing drift/forgetting;
  - EVAL every few iters on the full eval distribution + ROLLBACK to the best
    checkpoint if block rate drops. So it can only keep or improve.
Reward = exact goal_conceded (the eval metric) + dense intercept/body shaping.
"""
from __future__ import annotations
import copy, os, sys
from dataclasses import asdict, dataclass
import torch, tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from src.tasks.soccer import mdp
from src.tasks.soccer.config.g1.gk_train_cfg import goalkeeper_train_runner_cfg
from src.tasks.soccer.mdp.goalkeeper_rewards import (
  _reset_gk_state, goalkeeper_goal_conceded, goalkeeper_intercept_point,
  goalkeeper_body_intercept, goalkeeper_clear_ball, goalkeeper_stop_ball,
  goalkeeper_posture_orientation, goalkeeper_strike_through)
from src.tasks.soccer.mdp.shooter_rewards import action_rate_l2_clip


@dataclass
class Cfg:
  init: str = "logs/repairs/base_r2.pt"      # actor init (the diving student)
  bc_data: str = "logs/repairs/fix_r2.pt"    # BC anchor dataset; empty disables anchor
  out: str = "logs/repairs/polished.pt"
  num_envs: int = 1024
  warmup: int = 50            # critic-only warmup iterations
  block_iters: int = 20       # train iters between evals
  blocks: int = 40            # number of train/eval blocks
  eval_resets: int = 2        # eval averages this many resets (less rollback noise)
  lr: float = 1.0e-4
  std: float = 0.06
  bc_coef: float = 0.5
  sym_coef: float = 0.0       # left/right equivariance loss (fixes L<R asymmetry)
  region_weights: tuple[float, ...] = ()  # per-region conceded weight [RMid,LMid,RUp,LUp,RLow,LLow]
  train_regions: tuple[int, ...] = ()  # restrict ball reset to these regions (specialist training)
  w_conceded: float = 10.0
  w_intercept: float = 2.0
  w_body: float = 2.0
  w_sharp: float = 0.0        # sharp contact reward (body_intercept, small std) — closes near-misses
  sharp_std: float = 0.12
  w_cross: float = 0.0        # crossing-instant sharp contact reward (targets timing near-misses)
  w_stop: float = 1.0
  w_clear: float = 0.0        # reward ball deflection away from goal (+x / out of frame)
  w_strike: float = 0.0       # reward forceful hand/foot velocity through ball toward +x
  clear_min_vx: float = 0.5
  strike_min_vx: float = 0.2
  w_posture: float = 0.0
  seed: int = 2810
  device: str = "cuda:0"


def _conceded_weighted(env, region_weights=(1, 1, 1, 1, 1, 1), goal_x=-0.5,
                       goal_half_width=1.5, goal_height=1.8):
  """goal_conceded penalty, but scaled per ball region so RL can focus on the
  weak high/mid balls without sacrificing the easy ones (region index in
  env._gk_region, set by the ball reset)."""
  from src.tasks.soccer.mdp.goalkeeper_rewards import _gk_get_or_init_state
  ball = env.scene["ball"]
  bp = ball.data.root_link_pos_w
  bx = bp[:, 0] - env.scene.env_origins[:, 0]
  by = bp[:, 1] - env.scene.env_origins[:, 1]
  in_goal = (bx <= goal_x) & (by.abs() <= goal_half_width) & (bp[:, 2] <= goal_height)
  conceded = _gk_get_or_init_state(env, "_gk_conceded", 0.0)
  fire = in_goal & (conceded < 0.5)
  reward = torch.zeros(env.num_envs, device=env.device)
  if torch.any(fire):
    ids = torch.nonzero(fire, as_tuple=False).squeeze(-1)
    reg = getattr(env, "_gk_region", None)
    w = torch.tensor(region_weights, device=env.device, dtype=torch.float32)
    if reg is not None:
      reward[ids] = w[reg[ids].long().clamp(0, 5)]
    else:
      reward[ids] = 1.0
    conceded[ids] = 1.0
    setattr(env, "_gk_conceded", conceded)
  return reward


def _crossing_contact(env, gate=0.20, std=0.10):
  """Sharp reward for ANY blocking link being on the ball EXACTLY when the ball
  crosses the keeper plane (|ball_x_rel| < gate). Targets high-ball TIMING
  near-misses: rewards being at the right place at the right instant, not just
  close over the trajectory."""
  ball = env.scene["ball"]; robot = env.scene["robot"]
  bp = ball.data.root_link_pos_w
  bx = bp[:, 0] - env.scene.env_origins[:, 0]
  body = robot.data.body_link_pos_w
  d2 = (body - bp.unsqueeze(1)).pow(2).sum(-1).min(1).values
  at_plane = (bx.abs() < gate).float()
  return at_plane * torch.exp(-d2 / (std * std))


def _nan_termination(env):
  """True for envs whose physics state went NaN/Inf (so mjlab resets them)."""
  r = env.scene["robot"]
  bad = torch.isnan(r.data.joint_pos).any(-1) | torch.isinf(r.data.joint_pos).any(-1)
  rp = r.data.root_link_pos_w
  bad = bad | torch.isnan(rp).any(-1) | torch.isinf(rp).any(-1)
  return bad


def _sanitize_step(env_wrapper):
  """Wrap the inner env.step to scrub any residual NaN/Inf from obs and reward."""
  inner = env_wrapper.unwrapped
  orig = inner.step

  def safe(action):
    obs, rew, term, trunc, extras = orig(action)
    for k in list(obs.keys()):
      obs[k] = torch.nan_to_num(obs[k], nan=0.0, posinf=0.0, neginf=0.0)
    rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
    return obs, rew, term, trunc, extras

  inner.step = safe


def _eval(env, policy, ball, n_steps=150, n_resets=3):
  """Average block rate over several resets (n_resets*num_envs trials) so the
  rollback decision isn't driven by single-batch eval noise."""
  N = env.unwrapped.num_envs
  org = env.unwrapped.scene.env_origins
  blocked = 0; total = 0
  with torch.inference_mode():
    for _ in range(n_resets):
      obs, _ = env.reset()
      entered = torch.zeros(N, dtype=torch.bool, device=env.unwrapped.device)
      for _ in range(n_steps):
        a = policy(obs)
        obs = env.step(a)[0]
        bp = ball.data.root_link_pos_w
        entered |= ((bp[:, 0] - org[:, 0]) <= -0.5) & ((bp[:, 1] - org[:, 1]).abs() <= 1.5) & (bp[:, 2] <= 1.8)
      blocked += int((~entered).sum()); total += N
    return blocked / total


def main(cfg: Cfg):
  configure_torch_backends()
  dev = cfg.device
  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs; env_cfg.seed = cfg.seed
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  if cfg.region_weights:
    conceded_term = RewardTermCfg(func=_conceded_weighted, weight=-cfg.w_conceded,
                                  params={"region_weights": tuple(cfg.region_weights)})
  else:
    conceded_term = RewardTermCfg(func=goalkeeper_goal_conceded, weight=-cfg.w_conceded, params={})
  env_cfg.rewards = {
    "goal_conceded": conceded_term,
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
  if cfg.train_regions:  # specialist: restrict ball reset to a subset of regions
    import copy as _copy
    rb = env_cfg.events["reset_ball"]; vc = _copy.copy(rb.params["vel_cfg"])
    vc.regions = [vc.regions[i] for i in cfg.train_regions]
    rb.params["vel_cfg"] = vc
    print(f"[INFO] specialist training on regions {tuple(cfg.train_regions)}", flush=True)
  env_cfg.events["reset_gk_state"] = EventTermCfg(func=_reset_gk_state, mode="reset", params={})
  # NaN guard: on some GPUs (e.g. RTX 4090/Ada) the warp sim occasionally produces
  # NaN qpos under PPO exploration. Terminate those envs so mjlab resets them
  # *within* step() (before obs/history are computed) — keeps the run alive.
  env_cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan_guard": TerminationTermCfg(func=_nan_termination, time_out=False),
  }
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev), clip_actions=100.0)
  _sanitize_step(env)   # belt-and-suspenders: scrub any residual NaN in obs/rew

  agent = goalkeeper_train_runner_cfg()
  agent.algorithm.class_name = "src.tasks.soccer.modules.bc_anchor_ppo.BCAnchorPPO"
  agent.algorithm.learning_rate = cfg.lr
  agent.algorithm.entropy_coef = 0.0
  agent.algorithm.clip_param = 0.1
  agent.algorithm.desired_kl = 0.005
  agent.actor.distribution_cfg["init_std"] = cfg.std
  runner = MjlabOnPolicyRunner(env, asdict(agent), device=dev)

  ck = torch.load(cfg.init, map_location=dev, weights_only=False)
  runner.alg.actor.load_state_dict(ck["actor_state_dict"])
  print(f"[INFO] actor init from {cfg.init}", flush=True)
  alg = runner.alg
  alg.std_clamp = cfg.std; alg.bc_coef = cfg.bc_coef; alg.sym_coef = cfg.sym_coef
  with torch.no_grad():
    runner.alg.actor.distribution.std_param.fill_(cfg.std)

  if cfg.bc_coef > 0.0 and cfg.bc_data:
    d = torch.load(cfg.bc_data, map_location="cpu", weights_only=False)
    bk = d["blocked"].bool()
    alg.bc_obs = d["obs"][bk].to(dev); alg.bc_act = d["act"][bk].to(dev)
    print(f"[INFO] BC anchor {alg.bc_obs.shape[0]} pairs", flush=True)
  else:
    alg.bc_obs = None; alg.bc_act = None; alg.bc_coef = 0.0
    print("[INFO] BC anchor disabled.", flush=True)

  ball = env.unwrapped.scene["ball"]
  policy = runner.get_inference_policy(device=dev)

  # Critic warm-up (actor frozen) so PPO has a sane value function before it edits
  # the precious actor.
  if cfg.warmup > 0:
    for p in runner.alg.actor.parameters(): p.requires_grad_(False)
    bc0 = alg.bc_coef; alg.bc_coef = 0.0
    runner.learn(num_learning_iterations=cfg.warmup, init_at_random_ep_len=True)
    for p in runner.alg.actor.parameters(): p.requires_grad_(True)
    alg.bc_coef = bc0
    with torch.no_grad(): runner.alg.actor.distribution.std_param.fill_(cfg.std)

  os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
  def _save():
    saved = runner.alg.save(); saved["iter"] = 0
    saved["infos"] = {"env_state": {"common_step_counter": 0}}
    torch.save(saved, cfg.out)

  best = _eval(env, policy, ball, n_resets=cfg.eval_resets); best_state = copy.deepcopy(runner.alg.actor.state_dict())
  _save()   # persist immediately so a kill never loses the best so far
  print(f"[EVAL] init block {100*best:.1f}%", flush=True)
  for b in range(cfg.blocks):
    runner.learn(num_learning_iterations=cfg.block_iters, init_at_random_ep_len=False)
    with torch.no_grad(): runner.alg.actor.distribution.std_param.clamp_(max=cfg.std)
    r = _eval(env, policy, ball, n_resets=cfg.eval_resets)
    tag = ""
    if r >= best:
      best = r; best_state = copy.deepcopy(runner.alg.actor.state_dict())
      _save(); tag = " *best* (saved)"
    elif r < best - 0.02:
      runner.alg.actor.load_state_dict(best_state); tag = " rollback"
    print(f"[EVAL] block {b+1}/{cfg.blocks}: {100*r:.1f}%  (best {100*best:.1f}%){tag}", flush=True)

  runner.alg.actor.load_state_dict(best_state); _save()
  print(f"[INFO] saved polished (best {100*best:.1f}%) to {cfg.out}", flush=True)
  env.close()


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa
  main(tyro.cli(Cfg, prog="train_polish"))
