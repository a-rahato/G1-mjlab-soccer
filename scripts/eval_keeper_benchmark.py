"""Unified goalkeeper benchmark.

This script evaluates different keeper implementations under the same rollout
and scoring code. It is intended for branch-to-branch comparison: native MLP
checkpoints, MoE6 bundles, and frozen-base residual checkpoints all go through
the same environment seeds, goal test, per-region report, and optional JSON/CSV
output.
"""
from __future__ import annotations

import csv
import json
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from src.tasks.soccer.config.g1.gk_train_cfg import goalkeeper_train_runner_cfg


REGION_NAMES = ["Right-Mid", "Left-Mid", "Right-Up", "Left-Up", "Right-Low", "Left-Low"]
BLOCK_LINKS = (
  "left_wrist_yaw_link", "right_wrist_yaw_link",
  "left_elbow_link", "right_elbow_link",
  "left_shoulder_roll_link", "right_shoulder_roll_link",
  "left_ankle_roll_link", "right_ankle_roll_link",
  "left_knee_link", "right_knee_link",
  "torso_link", "pelvis",
)


@dataclass
class Cfg:
  policy_type: str = "native"  # native | moe6 | residual | moe_residual
  checkpoint: str = "src/assets/soccer/weight/goalkeeper_polished_v2.pt"
  residual_base: str = ""
  moe_bundle: str = "src/assets/soccer/weight/goalkeeper_moe6.pt"
  residual_scale: float = 0.03
  residual_head: tuple[int, ...] = (512, 256, 128)
  suite: str = "official"  # official | hard | custom
  num_envs: int = 256
  batches: int = 16
  seeds: tuple[int, ...] = (2810,)
  steps: int = 150
  regions: tuple[int, ...] = ()
  hard_t_flight: tuple[float, float] = (0.45, 0.75)
  hard_regions: tuple[int, ...] = (0, 1, 2, 3)
  clear_vx: float = 0.5
  contact_threshold: float = 0.12
  output_json: str = ""
  output_csv: str = ""
  device: str = "cuda:0"


class MoE6Policy:
  def __init__(self, bundle_path: str, env, device: str):
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    self.z_low = bundle.get("z_low", 0.85)
    self.z_up = bundle.get("z_up", 1.35)
    self.vz_low = bundle.get("vz_low", -5.0)
    self.latch_hi = bundle.get("latch_hi", 5.0)
    self.land_x = bundle.get("land_x", 0.0)
    self.ball = env.unwrapped.scene["ball"]
    self.org = env.unwrapped.scene.env_origins
    self.num_envs = env.unwrapped.num_envs
    self.device = device
    self.g = 9.81

    tmp = tempfile.mkdtemp(prefix="bench_moe6_")
    self.experts = []
    for idx, state in enumerate(bundle["sr"]):
      path = f"{tmp}/sr{idx}.pt"
      torch.save(state, path)
      runner = MjlabOnPolicyRunner(env, asdict(goalkeeper_train_runner_cfg()), device=device)
      runner.load(path, load_cfg={"actor": True})
      self.experts.append(runner.get_inference_policy(device=device))

    mirror_map = bundle.get("mirror_map", "")
    if mirror_map:
      from src.tasks.soccer.modules.symmetry import mirror_action, mirror_obs

      base = list(self.experts)

      def mirror(policy):
        return lambda obs: mirror_action(policy({"actor": mirror_obs(obs["actor"])}))

      for pair in mirror_map.split(","):
        dst, src = (int(x) for x in pair.split(":"))
        self.experts[dst] = mirror(base[src])
        print(f"[INFO] MoE expert {dst} := mirror(expert {src})", flush=True)
    self.reset()

  def reset(self):
    self.latched = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

  def route(self):
    bp = self.ball.data.root_link_pos_w
    bv = self.ball.data.root_link_lin_vel_w
    bx = bp[:, 0] - self.org[:, 0]
    vx = bv[:, 0]
    valid = (vx < -1.0) & (bx > 0.2) & (bx < self.latch_hi)
    t = torch.clamp(-(bx - self.land_x) / (vx - 1e-3), 0.0, 2.0)
    cy = (bp[:, 1] - self.org[:, 1]) + bv[:, 1] * t
    cz = bp[:, 2] + bv[:, 2] * t - 0.5 * self.g * t * t
    base = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    base = torch.where(cz < self.z_low, torch.full_like(base, 4), base)
    base = torch.where(cz > self.z_up, torch.full_like(base, 2), base)
    vz_cross = bv[:, 2] - self.g * t
    base = torch.where(vz_cross < self.vz_low, torch.full_like(base, 4), base)
    return base + (cy < 0).long(), valid

  def __call__(self, obs):
    region, valid = self.route()
    latch_now = valid & (self.latched < 0)
    self.latched = torch.where(latch_now, region, self.latched)
    use = torch.where(self.latched < 0, torch.zeros_like(self.latched), self.latched)
    acts = torch.stack([expert(obs) for expert in self.experts], 0)
    return acts[use, torch.arange(self.num_envs, device=self.device)]


def _residual_runner_cfg(cfg: Cfg):
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="src.tasks.soccer.modules.gk_residual.GoalkeeperResidual",
      hidden_dims=cfg.residual_head,
      activation="elu",
      obs_normalization=False,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.05, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 256), activation="elu", obs_normalization=False),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.1,
      entropy_coef=0.0, num_learning_epochs=3, num_mini_batches=4,
      learning_rate=1e-4, schedule="adaptive", gamma=0.99, lam=0.95,
      desired_kl=0.003, max_grad_norm=0.5,
    ),
    experiment_name="g1_goalkeeper_residual_eval",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=1,
  )


def load_policy(cfg: Cfg, env):
  if cfg.policy_type == "moe6":
    return MoE6Policy(cfg.checkpoint, env, cfg.device)
  if cfg.policy_type == "native":
    runner = MjlabOnPolicyRunner(env, asdict(goalkeeper_train_runner_cfg()), device=cfg.device)
    runner.load(cfg.checkpoint, load_cfg={"actor": True})
    return runner.get_inference_policy(device=cfg.device)
  if cfg.policy_type == "residual":
    import src.tasks.soccer.modules.gk_residual as gkr

    gkr.BASE_CKPT = cfg.residual_base or None
    gkr.BASE_HIDDEN = (1024, 512, 256)
    gkr.RESIDUAL_SCALE = cfg.residual_scale
    runner = MjlabOnPolicyRunner(env, asdict(_residual_runner_cfg(cfg)), device=cfg.device)
    runner.load(cfg.checkpoint, load_cfg={"actor": True})
    return runner.get_inference_policy(device=cfg.device)
  if cfg.policy_type == "moe_residual":
    import src.tasks.soccer.modules.gk_moe_residual as gkmoe

    gkmoe.ENV = env
    gkmoe.BUNDLE_PATH = cfg.moe_bundle
    gkmoe.RESIDUAL_SCALE = cfg.residual_scale
    runner_cfg = _residual_runner_cfg(cfg)
    runner_cfg.actor.class_name = "src.tasks.soccer.modules.gk_moe_residual.GoalkeeperMoEResidual"
    runner = MjlabOnPolicyRunner(env, asdict(runner_cfg), device=cfg.device)
    runner.load(cfg.checkpoint, load_cfg={"actor": True})
    return runner.get_inference_policy(device=cfg.device)
  raise ValueError(f"unknown policy_type: {cfg.policy_type}")


def configure_ball_distribution(env_cfg, cfg: Cfg):
  rb = env_cfg.events["reset_ball"]
  vel_cfg = rb.params["vel_cfg"]
  if cfg.suite == "hard":
    vel_cfg.t_flight_range = tuple(cfg.hard_t_flight)
    regions = cfg.regions or cfg.hard_regions
  elif cfg.suite == "custom":
    regions = cfg.regions
  elif cfg.suite == "official":
    regions = cfg.regions
  else:
    raise ValueError(f"unknown suite: {cfg.suite}")

  if regions:
    vel_cfg.regions = [vel_cfg.regions[i] for i in regions]
    lows_y = [r["width"][0] for r in vel_cfg.regions]
    highs_y = [r["width"][1] for r in vel_cfg.regions]
    lows_z = [r["height"][0] for r in vel_cfg.regions]
    highs_z = [r["height"][1] for r in vel_cfg.regions]
    vel_cfg.ball_start_y_range = (min(lows_y), max(highs_y))
    vel_cfg.ball_start_z_range = (min(lows_z), max(highs_z))


def make_env(cfg: Cfg, seed: int):
  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = seed
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  configure_ball_distribution(env_cfg, cfg)
  return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device), clip_actions=100.0)


def rollout_batch(cfg: Cfg, env, policy, seed: int, batch_idx: int):
  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]
  if hasattr(policy, "reset"):
    policy.reset()

  raw = env.unwrapped
  ball = raw.scene["ball"]
  robot = raw.scene["robot"]
  org = raw.scene.env_origins
  block_ids = torch.as_tensor(robot.find_bodies(BLOCK_LINKS, preserve_order=True)[0], device=cfg.device)
  true_region = getattr(raw, "_gk_region", torch.zeros(cfg.num_envs, device=cfg.device)).clone().long()
  entered = torch.zeros(cfg.num_envs, dtype=torch.bool, device=cfg.device)
  min_dist = torch.full((cfg.num_envs,), 1e9, device=cfg.device)
  max_out_vx = torch.full((cfg.num_envs,), -1e9, device=cfg.device)
  contacted = torch.zeros(cfg.num_envs, dtype=torch.bool, device=cfg.device)

  # Do not use torch.inference_mode() here: mjlab reuses/reset internal sensor
  # buffers between batches, and inference tensors reject later in-place reset.
  with torch.no_grad():
    for _ in range(cfg.steps):
      action = policy(obs)
      obs = env.step(action)[0]
      bp = ball.data.root_link_pos_w
      bv = ball.data.root_link_lin_vel_w
      bx = bp[:, 0] - org[:, 0]
      by = bp[:, 1] - org[:, 1]
      in_goal = (bx <= -0.5) & (by.abs() <= 1.5) & (bp[:, 2] <= 1.8)
      entered |= in_goal

      body = robot.data.body_link_pos_w[:, block_ids]
      dist = (body - bp.unsqueeze(1)).pow(2).sum(-1).min(1).values.sqrt()
      min_dist = torch.minimum(min_dist, dist)
      contacted |= dist < cfg.contact_threshold

      near_keeper = (bx > -0.5) & (bx < 0.5)
      max_out_vx = torch.maximum(max_out_vx, torch.where(near_keeper, bv[:, 0], max_out_vx))

  blocked = ~entered
  cleared = blocked & contacted & (max_out_vx > cfg.clear_vx)
  rows = []
  for idx in range(cfg.num_envs):
    region = int(true_region[idx])
    rows.append({
      "seed": seed,
      "batch": batch_idx,
      "env": idx,
      "region": region,
      "region_name": REGION_NAMES[region] if 0 <= region < len(REGION_NAMES) else str(region),
      "blocked": bool(blocked[idx].item()),
      "cleared": bool(cleared[idx].item()),
      "contacted": bool(contacted[idx].item()),
      "min_dist": float(min_dist[idx].item()),
      "max_out_vx": float(max_out_vx[idx].item()),
    })
  return rows


def summarize(rows: list[dict]):
  def rate(key, subset):
    return sum(1 for r in subset if r[key]) / len(subset) if subset else 0.0

  out = {
    "n": len(rows),
    "block_rate": rate("blocked", rows),
    "clear_rate": rate("cleared", rows),
    "contact_rate": rate("contacted", rows),
    "mean_min_dist": sum(r["min_dist"] for r in rows) / len(rows),
    "mean_max_out_vx": sum(r["max_out_vx"] for r in rows) / len(rows),
    "regions": {},
  }
  grouped = defaultdict(list)
  for row in rows:
    grouped[row["region_name"]].append(row)
  for name in REGION_NAMES:
    subset = grouped.get(name, [])
    if not subset:
      continue
    out["regions"][name] = {
      "n": len(subset),
      "block_rate": rate("blocked", subset),
      "clear_rate": rate("cleared", subset),
      "contact_rate": rate("contacted", subset),
      "mean_min_dist": sum(r["min_dist"] for r in subset) / len(subset),
      "mean_max_out_vx": sum(r["max_out_vx"] for r in subset) / len(subset),
    }
  return out


def write_outputs(cfg: Cfg, rows: list[dict], summary: dict):
  if cfg.output_csv:
    path = Path(cfg.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
      writer.writeheader()
      writer.writerows(rows)
  if cfg.output_json:
    path = Path(cfg.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(cfg), "summary": summary}
    with path.open("w") as f:
      json.dump(payload, f, indent=2)


def print_summary(cfg: Cfg, summary: dict):
  print("\n" + "=" * 72)
  print(f"KEEPER BENCHMARK | policy={cfg.policy_type} suite={cfg.suite} checkpoint={cfg.checkpoint}")
  print("=" * 72)
  print(f"trials:       {summary['n']}")
  print(f"block_rate:   {100 * summary['block_rate']:.1f}%")
  print(f"clear_rate:   {100 * summary['clear_rate']:.1f}%")
  print(f"contact_rate: {100 * summary['contact_rate']:.1f}%")
  print(f"mean_min_dist {summary['mean_min_dist']:.3f} m")
  print(f"mean_out_vx:  {summary['mean_max_out_vx']:.3f} m/s")
  print("\nregion        n    block   clear  contact  min_d   out_vx")
  for name in REGION_NAMES:
    stats = summary["regions"].get(name)
    if not stats:
      continue
    print(f"{name:<11} {stats['n']:4d}  {100*stats['block_rate']:6.1f}%"
          f" {100*stats['clear_rate']:6.1f}% {100*stats['contact_rate']:7.1f}%"
          f"  {stats['mean_min_dist']:.3f}  {stats['mean_max_out_vx']:.3f}")
  print("=" * 72 + "\n")


def main(cfg: Cfg):
  configure_torch_backends()
  all_rows: list[dict] = []
  for seed in cfg.seeds:
    torch.manual_seed(seed)
    env = make_env(cfg, seed)
    policy = load_policy(cfg, env)
    for batch in range(cfg.batches):
      rows = rollout_batch(cfg, env, policy, seed, batch)
      all_rows.extend(rows)
      partial = summarize(all_rows)
      print(f"seed {seed} batch {batch+1}/{cfg.batches}: "
            f"{sum(r['blocked'] for r in all_rows)}/{len(all_rows)} = {100*partial['block_rate']:.1f}%",
            flush=True)
    env.close()
  summary = summarize(all_rows)
  print_summary(cfg, summary)
  write_outputs(cfg, all_rows, summary)


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa

  main(tyro.cli(Cfg, prog="eval_keeper_benchmark"))
