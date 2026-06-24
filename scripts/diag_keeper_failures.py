"""Diagnose *why* a goalkeeper concedes on a given suite.

This reuses ``eval_keeper_benchmark``'s env + policy loading so the rollout,
seeds, and ball distribution match the unified benchmark exactly. The point is
not to produce another block-rate number, but to split conceded balls into the
two qualitatively different failure modes, with data:

  - reachability (min_dist): did *any* blocking link ever get near the ball?
      * touched   (min_dist < contact_threshold): a real contact, yet conceded
                  -> the save was made but did not stop/clear the ball
                  (control / clearing-quality problem).
      * near-miss (contact_threshold <= min_dist < reach_thresh): got close but
                  not close enough.
      * far       (min_dist >= reach_thresh): never got there at all
                  (reachability / timing problem).

  - reaction timing: the first control step at which a hand or foot moves faster
    than ``move_thresh``, measured relative to the step the ball first crosses
    the keeper's plane (x <= 0). Reacting only just before arrival points at a
    late-decision problem; reacting early but still ending up "far" points at a
    physical-reach limit that no amount of extra perception can fix.

It also bins block rate by launch ball speed, so we can see how fast the balls
that beat the keeper actually are.

The motivation is to decide, empirically, whether the bottleneck on fast
mid/high balls is perception/memory (which an LSTM could plausibly help) or
control/reachability (which it cannot). The actor already receives a 10-frame
ball-position history, so ball velocity is recoverable by finite differences;
this script is what tells us whether that information is the limiting factor.

Example (run on the server):

    MUJOCO_GL=egl WANDB_MODE=disabled MPLCONFIGDIR=/tmp/mpl \\
    .venv/bin/python scripts/diag_keeper_failures.py \\
      --policy-type moe6 \\
      --checkpoint src/assets/soccer/weight/goalkeeper_moe6.pt \\
      --suite hard --num-envs 256 --batches 8 \\
      --output-csv logs/experiments/diag_moe6_hard.csv
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import csv
import torch
import tyro

from mjlab.utils.torch import configure_torch_backends

# Reuse the benchmark's env/policy machinery verbatim so this diagnostic and the
# headline benchmark agree on rollout, seeds, and ball distribution.
from eval_keeper_benchmark import (
  BLOCK_LINKS,
  REGION_NAMES,
  Cfg as BenchCfg,
  load_policy,
  make_env,
)

# Links whose motion counts as the keeper "reacting" (hands + feet).
REACT_LINKS = (
  "left_wrist_yaw_link", "right_wrist_yaw_link",
  "left_ankle_roll_link", "right_ankle_roll_link",
)


@dataclass
class DiagCfg:
  policy_type: str = "moe6"  # native | moe6 | residual
  checkpoint: str = "src/assets/soccer/weight/goalkeeper_moe6.pt"
  residual_base: str = ""
  residual_scale: float = 0.03
  residual_head: tuple[int, ...] = (512, 256, 128)
  suite: str = "hard"  # official | hard | custom
  num_envs: int = 256
  batches: int = 8
  seeds: tuple[int, ...] = (2810,)
  steps: int = 150
  regions: tuple[int, ...] = ()
  hard_t_flight: tuple[float, float] = (0.45, 0.75)
  hard_regions: tuple[int, ...] = (0, 1, 2, 3)
  clear_vx: float = 0.5
  contact_threshold: float = 0.12
  device: str = "cuda:0"

  # -- Diagnostic-specific knobs ---------------------------------------------
  reach_thresh: float = 0.25  # min_dist below which we say the keeper "reached"
  move_thresh: float = 1.0    # m/s; hand/foot speed that counts as "reacting"
  late_steps: int = 6         # reacting within this many steps of arrival = late
  speed_bins: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0)  # |launch v| bin edges
  output_csv: str = ""

  def to_bench(self) -> BenchCfg:
    """Project onto the benchmark Cfg so we can reuse make_env / load_policy."""
    return BenchCfg(
      policy_type=self.policy_type,
      checkpoint=self.checkpoint,
      residual_base=self.residual_base,
      residual_scale=self.residual_scale,
      residual_head=self.residual_head,
      suite=self.suite,
      num_envs=self.num_envs,
      batches=self.batches,
      seeds=self.seeds,
      steps=self.steps,
      regions=self.regions,
      hard_t_flight=self.hard_t_flight,
      hard_regions=self.hard_regions,
      clear_vx=self.clear_vx,
      contact_threshold=self.contact_threshold,
      device=self.device,
    )


def rollout_batch(cfg: DiagCfg, env, policy, seed: int, batch_idx: int) -> list[dict]:
  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]
  if hasattr(policy, "reset"):
    policy.reset()

  raw = env.unwrapped
  ball = raw.scene["ball"]
  robot = raw.scene["robot"]
  org = raw.scene.env_origins
  n = cfg.num_envs
  dev = cfg.device

  block_ids = torch.as_tensor(robot.find_bodies(BLOCK_LINKS, preserve_order=True)[0], device=dev)
  react_ids = torch.as_tensor(robot.find_bodies(REACT_LINKS, preserve_order=True)[0], device=dev)
  true_region = getattr(raw, "_gk_region", torch.zeros(n, device=dev)).clone().long()

  # Launch velocity is set by the ball reset; read it before stepping.
  launch_v = ball.data.root_link_lin_vel_w.clone()
  launch_speed = launch_v.norm(dim=-1)

  entered = torch.zeros(n, dtype=torch.bool, device=dev)
  contacted = torch.zeros(n, dtype=torch.bool, device=dev)
  min_dist = torch.full((n,), 1e9, device=dev)
  max_out_vx = torch.full((n,), -1e9, device=dev)
  arrival_step = torch.full((n,), cfg.steps, dtype=torch.long, device=dev)  # ball crosses x<=0
  reaction_step = torch.full((n,), cfg.steps, dtype=torch.long, device=dev)  # keeper first moves

  with torch.no_grad():
    for step in range(cfg.steps):
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

      # First step the ball reaches the keeper's plane.
      crossed = (bx <= 0.0) & (arrival_step == cfg.steps)
      arrival_step = torch.where(crossed, torch.full_like(arrival_step, step), arrival_step)

      # First step a hand/foot moves faster than move_thresh.
      ee_speed = robot.data.body_link_lin_vel_w[:, react_ids].norm(dim=-1).max(dim=1).values
      moving = (ee_speed > cfg.move_thresh) & (reaction_step == cfg.steps)
      reaction_step = torch.where(moving, torch.full_like(reaction_step, step), reaction_step)

  blocked = ~entered
  cleared = blocked & contacted & (max_out_vx > cfg.clear_vx)

  rows = []
  for i in range(n):
    region = int(true_region[i])
    md = float(min_dist[i].item())
    blk = bool(blocked[i].item())
    react = int(reaction_step[i].item())
    arr = int(arrival_step[i].item())
    # reaction margin: how many steps before the ball arrives did the keeper move
    # (positive = reacted early; <= late_steps = reacted late / not at all).
    react_margin = arr - react
    if md < cfg.contact_threshold:
      reach_class = "touched"
    elif md < cfg.reach_thresh:
      reach_class = "near_miss"
    else:
      reach_class = "far"
    rows.append({
      "seed": seed,
      "batch": batch_idx,
      "env": i,
      "region": region,
      "region_name": REGION_NAMES[region] if 0 <= region < len(REGION_NAMES) else str(region),
      "blocked": blk,
      "cleared": bool(cleared[i].item()),
      "contacted": bool(contacted[i].item()),
      "min_dist": md,
      "reach_class": reach_class,
      "launch_speed": float(launch_speed[i].item()),
      "launch_vx": float(launch_v[i, 0].item()),
      "arrival_step": arr,
      "reaction_step": react,
      "react_margin": react_margin,
      "late_reaction": bool(react_margin <= cfg.late_steps),
    })
  return rows


def _rate(rows: list[dict], key: str) -> float:
  return sum(1 for r in rows if r[key]) / len(rows) if rows else 0.0


def print_report(cfg: DiagCfg, rows: list[dict]) -> None:
  conceded = [r for r in rows if not r["blocked"]]
  blocked = [r for r in rows if r["blocked"]]
  n = len(rows)

  print("\n" + "=" * 72)
  print(f"KEEPER FAILURE DIAGNOSIS | policy={cfg.policy_type} suite={cfg.suite}")
  print(f"checkpoint={cfg.checkpoint}")
  print("=" * 72)
  print(f"trials:        {n}")
  print(f"block_rate:    {100 * _rate(rows, 'blocked'):.1f}%   "
        f"({len(blocked)} blocked / {len(conceded)} conceded)")
  print(f"clear_rate:    {100 * _rate(rows, 'cleared'):.1f}%")
  print(f"contact_rate:  {100 * _rate(rows, 'contacted'):.1f}%")

  # -- Conceded-ball reach breakdown -----------------------------------------
  print("\n-- conceded balls, by how close the keeper got --")
  if conceded:
    for cls in ("touched", "near_miss", "far"):
      sub = [r for r in conceded if r["reach_class"] == cls]
      frac = len(sub) / len(conceded)
      mean_md = sum(r["min_dist"] for r in sub) / len(sub) if sub else 0.0
      print(f"  {cls:<9} {len(sub):4d}  ({100*frac:5.1f}% of conceded)  mean_min_dist {mean_md:.3f} m")
    far = [r for r in conceded if r["reach_class"] == "far"]
    if far:
      late = [r for r in far if r["late_reaction"]]
      print(f"\n-- of the {len(far)} 'far' (never reached) conceded balls --")
      print(f"  late reaction (moved <= {cfg.late_steps} steps before arrival): "
            f"{len(late)} ({100*len(late)/len(far):.1f}%)  -> late-decision suspect")
      print(f"  reacted early but still too far:                              "
            f"{len(far)-len(late)} ({100*(len(far)-len(late))/len(far):.1f}%)  -> physical-reach limit")
      mean_margin = sum(r["react_margin"] for r in far) / len(far)
      print(f"  mean reaction margin on 'far' balls: {mean_margin:.1f} steps "
            f"(arrival_step - reaction_step; larger = reacted earlier)")
  else:
    print("  (none)")

  # -- Block rate vs launch speed --------------------------------------------
  print("\n-- block rate by launch ball speed --")
  edges = list(cfg.speed_bins)
  bins: dict[str, list[dict]] = defaultdict(list)
  bounds = [0.0] + edges + [float("inf")]
  labels = []
  for j in range(len(bounds) - 1):
    lo, hi = bounds[j], bounds[j + 1]
    labels.append(f"{lo:.0f}-{'inf' if hi == float('inf') else f'{hi:.0f}'} m/s")
  for r in rows:
    s = r["launch_speed"]
    for j in range(len(bounds) - 1):
      if bounds[j] <= s < bounds[j + 1]:
        bins[labels[j]].append(r)
        break
  print("  speed range       n    block")
  for lab in labels:
    sub = bins.get(lab, [])
    if not sub:
      continue
    print(f"  {lab:<14} {len(sub):4d}  {100*_rate(sub, 'blocked'):6.1f}%")

  # -- Per-region reach summary ----------------------------------------------
  print("\n-- per-region (block / reached-but-conceded / never-reached) --")
  grouped: dict[str, list[dict]] = defaultdict(list)
  for r in rows:
    grouped[r["region_name"]].append(r)
  print("  region        n    block   touched&conceded  far&conceded")
  for name in REGION_NAMES:
    sub = grouped.get(name, [])
    if not sub:
      continue
    conc = [r for r in sub if not r["blocked"]]
    touched_conc = sum(1 for r in conc if r["reach_class"] in ("touched", "near_miss"))
    far_conc = sum(1 for r in conc if r["reach_class"] == "far")
    print(f"  {name:<11} {len(sub):4d}  {100*_rate(sub, 'blocked'):6.1f}%"
          f"  {touched_conc:4d}              {far_conc:4d}")
  print("=" * 72 + "\n")


def write_csv(path: str, rows: list[dict]) -> None:
  if not path or not rows:
    return
  p = Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  with p.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"[INFO] wrote per-trial rows to {path}")


def main(cfg: DiagCfg) -> None:
  configure_torch_backends()
  bench = cfg.to_bench()
  all_rows: list[dict] = []
  for seed in cfg.seeds:
    torch.manual_seed(seed)
    env = make_env(bench, seed)
    policy = load_policy(bench, env)
    for batch in range(cfg.batches):
      rows = rollout_batch(cfg, env, policy, seed, batch)
      all_rows.extend(rows)
      blk = sum(r["blocked"] for r in all_rows)
      print(f"seed {seed} batch {batch+1}/{cfg.batches}: "
            f"{blk}/{len(all_rows)} = {100*blk/len(all_rows):.1f}%", flush=True)
    env.close()
  print_report(cfg, all_rows)
  write_csv(cfg.output_csv, all_rows)


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa

  main(tyro.cli(DiagCfg, prog="diag_keeper_failures"))
