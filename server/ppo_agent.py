"""PPO actor-critic network, rollout buffer, and update logic.

Architecture:
  Shared MLP trunk (74 → 256 → 256) with tanh activations.
  Actor head:
    - Continuous branch: linear → tanh → (move_x, move_y) direction mean;
      log-std is a learned parameter vector (not input-dependent).
    - Discrete branch: linear → Bernoulli logits for (split, eject).
  Critic head: linear → scalar value estimate.

Actions are represented as a single flat tensor of shape (4,):
  [move_x, move_y, split, eject]  where move_x/y ∈ ℝ (pre-tanh Gaussian sample)
  and split/eject ∈ {0, 1} (Bernoulli sample).

Rollout buffer collects N_STEPS × N_BOTS transitions, then computes GAE
advantage estimates before each PPO update.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli

# ---------------------------------------------------------------------------
# Observation / action dimensions
# ---------------------------------------------------------------------------

N_OBS       = 74   # v2 perception inputs (matches _build_inputs_batch_v2 output)
N_CONT      = 2    # move_x, move_y (Gaussian)
N_DISC      = 2    # split, eject  (Bernoulli)
N_ACTIONS   = N_CONT + N_DISC   # 4 — same layout as NEAT outputs

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic for the agar.io bot."""

    def __init__(self, n_obs: int = N_OBS, hidden: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            _layer_init(nn.Linear(n_obs, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        # Continuous actor (move direction)
        self.actor_mean    = _layer_init(nn.Linear(hidden, N_CONT), std=0.01)
        self.actor_logstd  = nn.Parameter(torch.zeros(N_CONT))
        # Discrete actor (split, eject)
        self.actor_logits  = _layer_init(nn.Linear(hidden, N_DISC), std=0.01)
        # Critic
        self.critic        = _layer_init(nn.Linear(hidden, 1), std=1.0)

    # ------------------------------------------------------------------
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.shared(obs))

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) actions given observations.

        Returns (action, log_prob, entropy, value) all with batch dim B.
        action shape: (B, 4) — [move_x, move_y, split, eject]
        """
        features = self.shared(obs)

        # Continuous
        mean      = self.actor_mean(features)   # (B, 2) — unbounded; tanh applied at game time
        std       = self.actor_logstd.exp().expand_as(mean)
        dist_cont = Normal(mean, std)

        # Discrete
        logits    = self.actor_logits(features)  # (B, 2)
        dist_disc = Bernoulli(logits=logits)

        if action is None:
            cont_act  = dist_cont.sample()
            disc_act  = dist_disc.sample()
        else:
            cont_act  = action[:, :N_CONT]
            disc_act  = action[:, N_CONT:]

        log_prob = (
            dist_cont.log_prob(cont_act).sum(-1)      # sum over move dims
            + dist_disc.log_prob(disc_act).sum(-1)    # sum over action dims
        )
        entropy = (
            dist_cont.entropy().sum(-1)
            + dist_disc.entropy().sum(-1)
        )
        value  = self.critic(features).squeeze(-1)
        action = torch.cat([cont_act, disc_act], dim=-1)
        return action, log_prob, entropy, value

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | os.PathLike, device: torch.device | str = "cpu") -> "ActorCritic":
        model = cls()
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        return model


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-size circular buffer for on-policy rollouts.

    Stores experience from N_BOTS bots over N_STEPS ticks.
    After fill(), call compute_returns_and_advantages() then iterate
    minibatches via get_minibatches().
    """

    def __init__(self, n_steps: int, n_bots: int, n_obs: int = N_OBS,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 device: torch.device | str = "cpu") -> None:
        self.n_steps    = n_steps
        self.n_bots     = n_bots
        self.n_obs      = n_obs
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = torch.device(device)
        self._ptr       = 0
        self._full      = False

        # Pre-allocate tensors on CPU; moved to device during update
        self.obs      = torch.zeros(n_steps, n_bots, n_obs,      dtype=torch.float32)
        self.actions  = torch.zeros(n_steps, n_bots, N_ACTIONS,  dtype=torch.float32)
        self.logprobs = torch.zeros(n_steps, n_bots,             dtype=torch.float32)
        self.rewards  = torch.zeros(n_steps, n_bots,             dtype=torch.float32)
        self.dones    = torch.zeros(n_steps, n_bots,             dtype=torch.float32)
        self.values   = torch.zeros(n_steps, n_bots,             dtype=torch.float32)
        # Computed post-collection
        self.advantages: torch.Tensor | None = None
        self.returns:    torch.Tensor | None = None

    def add(
        self,
        step: int,
        obs:      np.ndarray,   # (n_bots, n_obs)
        actions:  torch.Tensor, # (n_bots, 4)
        logprobs: torch.Tensor, # (n_bots,)
        rewards:  np.ndarray,   # (n_bots,)
        dones:    np.ndarray,   # (n_bots,) — 1.0 if bot died this tick
        values:   torch.Tensor, # (n_bots,)
    ) -> None:
        self.obs[step]      = torch.as_tensor(obs,     dtype=torch.float32)
        self.actions[step]  = actions.detach().cpu()
        self.logprobs[step] = logprobs.detach().cpu()
        self.rewards[step]  = torch.as_tensor(rewards, dtype=torch.float32)
        self.dones[step]    = torch.as_tensor(dones,   dtype=torch.float32)
        self.values[step]   = values.detach().cpu()

    def compute_returns_and_advantages(self, last_values: torch.Tensor, last_dones: np.ndarray) -> None:
        """GAE-lambda advantage estimation.  Call after the rollout is complete."""
        last_values_cpu = last_values.detach().cpu()
        last_dones_t    = torch.as_tensor(last_dones, dtype=torch.float32)
        advantages = torch.zeros_like(self.rewards)
        last_gae   = torch.zeros(self.n_bots)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones_t
                next_values       = last_values_cpu
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values       = self.values[t + 1]

            delta    = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns    = advantages + self.values

    def get_minibatches(self, minibatch_size: int):
        """Yield flat (T*N, ...) minibatch tensors for PPO update."""
        assert self.advantages is not None, "Call compute_returns_and_advantages first"
        T, N = self.n_steps, self.n_bots
        # Flatten to (T*N, ...) and move to device
        b_obs        = self.obs.reshape(-1, self.n_obs).to(self.device)
        b_actions    = self.actions.reshape(-1, N_ACTIONS).to(self.device)
        b_logprobs   = self.logprobs.reshape(-1).to(self.device)
        b_advantages = self.advantages.reshape(-1).to(self.device)
        b_returns    = self.returns.reshape(-1).to(self.device)  # type: ignore[union-attr]

        # Normalise advantages per minibatch (common practice)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total = T * N
        indices = torch.randperm(total)
        for start in range(0, total, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            yield (
                b_obs[mb_idx],
                b_actions[mb_idx],
                b_logprobs[mb_idx],
                b_advantages[mb_idx],
                b_returns[mb_idx],
            )


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Wraps an ActorCritic and runs PPO update steps."""

    def __init__(
        self,
        model:            ActorCritic,
        lr:               float = 3e-4,
        n_epochs:         int   = 4,
        minibatch_size:   int   = 2048,
        clip_coef:        float = 0.2,
        value_coef:       float = 0.5,
        entropy_coef:     float = 0.01,
        max_grad_norm:    float = 0.5,
        target_kl:        float | None = None,
        device:           torch.device | str = "cpu",
    ) -> None:
        self.model          = model
        self.optimizer      = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.n_epochs       = n_epochs
        self.minibatch_size = minibatch_size
        self.clip_coef      = clip_coef
        self.value_coef     = value_coef
        self.entropy_coef   = entropy_coef
        self.max_grad_norm  = max_grad_norm
        self.target_kl      = target_kl
        self.device         = torch.device(device)

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run PPO update epochs over the filled rollout buffer.

        Returns a dict of training stats for logging.
        """
        total_pg_loss  = 0.0
        total_vf_loss  = 0.0
        total_ent_loss = 0.0
        total_approx_kl = 0.0
        n_updates      = 0

        self.model.train()
        for _ in range(self.n_epochs):
            for obs_mb, act_mb, logp_old_mb, adv_mb, ret_mb in buffer.get_minibatches(self.minibatch_size):
                _, logp_new, entropy, value_new = self.model.get_action_and_value(obs_mb, act_mb)

                # Policy loss (clipped surrogate)
                log_ratio  = logp_new - logp_old_mb
                ratio      = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    total_approx_kl += approx_kl

                pg_loss1 = -adv_mb * ratio
                pg_loss2 = -adv_mb * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                vf_loss = 0.5 * (value_new - ret_mb).pow(2).mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss + self.value_coef * vf_loss - self.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss  += pg_loss.item()
                total_vf_loss  += vf_loss.item()
                total_ent_loss += ent_loss.item()
                n_updates      += 1

            if self.target_kl is not None and (total_approx_kl / max(n_updates, 1)) > self.target_kl:
                break  # early stop this epoch

        return {
            'pg_loss':    total_pg_loss  / max(n_updates, 1),
            'vf_loss':    total_vf_loss  / max(n_updates, 1),
            'entropy':    total_ent_loss / max(n_updates, 1),
            'approx_kl':  total_approx_kl / max(n_updates, 1),
        }
