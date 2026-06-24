"""Residual-RL actor on top of a FROZEN qbs MoE6 goalkeeper.

The strongest keeper we have is the 6-region MoE6 bundle (official ~91%). A
unified-benchmark failure diagnosis showed that on the hard suite (fast mid/up
balls) ~73% of conceded goals are "reached but not stopped" (a hand/foot is
within ~0.1-0.16 m of the ball yet it still goes in), not a perception or
reachability-of-decision problem (the keeper reacts ~0.47 s early on every
missed ball). That is exactly the regime a small, bounded residual can help:
keep the MoE6 diving behavior intact, and learn a correction that squares up the
contact / pushes the ball out, trained and selected on the hard distribution.

  action = MoE6(obs)  +  residual_scale * tanh(residual(obs))

The MoE6 base is frozen and its hand-coded router uses the privileged ground
truth ball state (same as the benchmark MoE6Policy); the trainable residual sees
only the 960-D actor observation, so the learned correction stays deployable.
At init the residual last layer is zero, so the policy == MoE6 and can only be
refined from there.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable

# Set by the training/eval script before the runner is constructed (the runner
# cfg dataclass can't carry these extra fields).
BUNDLE_PATH = None
EXPERT_HIDDEN = (1024, 512, 256)
RESIDUAL_SCALE = 0.15
ENV = None  # the RslRlVecEnvWrapper, needed for the privileged router


class _MoE6Base(nn.Module):
  """Frozen 6-region MoE6: 6 expert MLPs + hand-coded privileged router + latch.

  Mirrors scripts/eval_keeper_benchmark.py:MoE6Policy so the base action here is
  identical to the benchmarked MoE6. Kept self-contained (no script import) so it
  can live in a committable module.
  """

  def __init__(self, bundle_path, env, obs, obs_groups, obs_set, output_dim, activation, dcfg):
    super().__init__()
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    self.group = obs_groups[obs_set][0]
    hidden = tuple(bundle.get("hidden", EXPERT_HIDDEN))
    self.z_low = bundle.get("z_low", 0.85)
    self.z_up = bundle.get("z_up", 1.35)
    self.vz_low = bundle.get("vz_low", -5.0)
    self.latch_hi = bundle.get("latch_hi", 5.0)
    self.land_x = bundle.get("land_x") or 0.0
    self.g = 9.81

    self.experts = nn.ModuleList()
    for state in bundle["sr"]:
      m = MLPModel(obs, obs_groups, obs_set, output_dim, hidden, activation, False, dict(dcfg))
      m.load_state_dict(state["actor_state_dict"])
      for p in m.parameters():
        p.requires_grad_(False)
      m.eval()
      self.experts.append(m)

    # mirror_map "dst:src,dst:src" -> expert[dst] := mirror(expert[src]).
    self.mirror: dict[int, int] = {}
    mm = bundle.get("mirror_map", "")
    if mm:
      for pair in mm.split(","):
        dst, src = (int(x) for x in pair.split(":"))
        self.mirror[dst] = src
        print(f"[MoE6Base] expert {dst} := mirror(expert {src})", flush=True)

    self.ball = env.unwrapped.scene["ball"]
    self.org = env.unwrapped.scene.env_origins
    self.num_envs = env.unwrapped.num_envs
    self.register_buffer("latched", torch.full((self.num_envs,), -1, dtype=torch.long))

  def reset(self, dones=None):
    if dones is None:
      self.latched.fill_(-1)
    else:
      idx = dones.nonzero(as_tuple=False).squeeze(-1)
      self.latched[idx] = -1

  def route(self):
    bp = self.ball.data.root_link_pos_w
    bv = self.ball.data.root_link_lin_vel_w
    bx = bp[:, 0] - self.org[:, 0]
    vx = bv[:, 0]
    valid = (vx < -1.0) & (bx > 0.2) & (bx < self.latch_hi)
    t = torch.clamp(-(bx - self.land_x) / (vx - 1e-3), 0.0, 2.0)
    cy = (bp[:, 1] - self.org[:, 1]) + bv[:, 1] * t
    cz = bp[:, 2] + bv[:, 2] * t - 0.5 * self.g * t * t
    base = torch.zeros(self.num_envs, dtype=torch.long, device=bp.device)
    base = torch.where(cz < self.z_low, torch.full_like(base, 4), base)
    base = torch.where(cz > self.z_up, torch.full_like(base, 2), base)
    vz_cross = bv[:, 2] - self.g * t
    base = torch.where(vz_cross < self.vz_low, torch.full_like(base, 4), base)
    return base + (cy < 0).long(), valid

  def _expert_action(self, idx, obs):
    if idx in self.mirror:
      from src.tasks.soccer.modules.symmetry import mirror_action, mirror_obs
      src = self.mirror[idx]
      mobs = {self.group: mirror_obs(obs[self.group])}
      return mirror_action(self.experts[src].forward(mobs, stochastic_output=False))
    return self.experts[idx].forward(obs, stochastic_output=False)

  def forward(self, obs):
    region, valid = self.route()
    latch_now = valid & (self.latched < 0)
    self.latched = torch.where(latch_now, region, self.latched)
    use = torch.where(self.latched < 0, torch.zeros_like(self.latched), self.latched)
    acts = torch.stack([self._expert_action(i, obs) for i in range(len(self.experts))], 0)
    return acts[use, torch.arange(self.num_envs, device=use.device)]


class GoalkeeperMoEResidual(nn.Module):
  is_recurrent: bool = False

  def __init__(
    self, obs, obs_groups, obs_set, output_dim,
    hidden_dims=(512, 256, 128), activation="elu", obs_normalization=False,
    distribution_cfg=None, bundle_path=None, expert_hidden=(1024, 512, 256),
    residual_scale=0.15, **kwargs,
  ):
    super().__init__()
    self.group = obs_groups[obs_set][0]
    if bundle_path is None:
      bundle_path = BUNDLE_PATH
      expert_hidden = EXPERT_HIDDEN
      residual_scale = RESIDUAL_SCALE
    self.residual_scale = residual_scale
    dcfg = dict(distribution_cfg or {"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"})

    # Frozen MoE6 base.
    assert ENV is not None, "gk_moe_residual.ENV must be set before constructing the runner."
    self.base = _MoE6Base(bundle_path, ENV, obs, obs_groups, obs_set, output_dim, activation, dict(dcfg))

    # Trainable residual (zero-initialized last layer -> starts at the MoE6 base).
    in_dim = obs[self.group].shape[-1]
    self.residual = MLP(in_dim, output_dim, hidden_dims, activation)
    for m in reversed([mod for mod in self.residual.modules() if isinstance(mod, nn.Linear)]):
      nn.init.zeros_(m.weight); nn.init.zeros_(m.bias); break

    dist_class: type[Distribution] = resolve_callable(dcfg.pop("class_name"))
    self.distribution: Distribution = dist_class(output_dim, **dcfg)

  def _mean(self, obs):
    with torch.no_grad():
      base = self.base.forward(obs)
    res = self.residual(obs[self.group])
    return base + self.residual_scale * torch.tanh(res)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
    mean = self._mean(obs)
    self.distribution.update(mean)
    return self.distribution.sample() if stochastic_output else self.distribution.deterministic_output(mean)

  # rsl_rl interface
  def reset(self, dones=None, hidden_state=None):
    self.base.reset(dones)

  def get_hidden_state(self): return None
  def detach_hidden_state(self, dones=None): pass
  def update_normalization(self, obs): pass
  @property
  def output_mean(self): return self.distribution.mean
  @property
  def output_std(self): return self.distribution.std
  @property
  def output_entropy(self): return self.distribution.entropy
  @property
  def output_distribution_params(self): return self.distribution.params
  def get_output_log_prob(self, outputs): return self.distribution.log_prob(outputs)
  def get_kl_divergence(self, old, new): return self.distribution.kl_divergence(old, new)
