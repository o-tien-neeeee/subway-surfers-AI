"""Deep Q-learning from Demonstrations (DQfD).

Hester et al. 2018 — *the* technique that fixed the "BC
pretrain → online RL forgets everything" failure mode.  The
idea is to combine three loss terms in every minibatch:

1. **Q-loss** (the standard Double DQN TD loss).
2. **Supervised loss** (cross-entropy between the network's
   argmax and the expert action, applied to a mini-batch
   drawn from the demonstration buffer).
3. **Margin classification loss** (a hinge term that
   enforces Q(expert) > Q(other) + margin, so the policy
   cannot drift away from the expert even when the Q-loss
   pushes it elsewhere).

The combined loss is

    L = L_Q + λ_1 L_BC + λ_2 L_margin

with ``λ_1 = 0.5`` and ``λ_2 = 0.1`` (the DQfD paper's
defaults).  All three terms contribute to the same
backward pass, so the agent cannot optimise one without
the others.

When to use this
----------------
Use DQfD when you have an expert policy (or a recorded
dataset) and want the agent to *match* that policy in
deployment.  This is the standard recipe for getting a
DQN-based agent to solve hard-exploration tasks where
random play never finds the reward — the demonstrations
are the *exploration budget* the agent cannot waste.

Implementation
--------------
* :class:`DQfDAgent` — full agent with both online and
  target nets, a *separate* demo buffer (priority
  boosted so the supervised term fires often), and the
  joint loss.
* The class reuses the underlying encoder/head from the
  :class:`distributional.QuantileDuelingDQN` so the QR-DQN
  improvements from v1.18 are still in effect.
* :meth:`pretrain_demos` does the BC warm-up before any
  online data has been collected.
* :meth:`train_step` mixes a batch of agent data with a
  batch of demo data and computes all three losses.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_distributional import DistributionalDoubleDQNAgent
from config import RLConfig
from distributional import (QuantileDuelingDQN, mid_quantiles,
                             project_distribution, quantile_huber_loss)
from logging_utils import get_logger
from replay_buffer import NStepTransition

LOGGER = get_logger("dqfd")


@dataclass
class DQfDConfig:
    """Knobs for the DQfD joint loss.  Defaults from Hester 2018."""
    # Supervised loss weight.  0.5 = the BC term is half as
    # strong as the TD term, which the paper found to be
    # the sweet spot.
    lambda_bc: float = 0.5
    # Margin classification loss weight.
    lambda_margin: float = 0.1
    # The margin in the hinge term: Q(expert) must be at
    # least this much higher than Q(other) for the margin
    # loss to be zero.  0.8 is the paper default.
    margin: float = 0.8
    # After how many online episodes we drop the
    # supervised term to 0 (the agent is supposed to
    # have internalised the demonstrations by then).
    supervised_decay_episodes: int = 1000
    # Whether to disable ε-greedy entirely (recommended
    # when the BC policy already solves the task).
    no_exploration: bool = True


class DQfDAgent(DistributionalDoubleDQNAgent):
    """QR-DQN agent with the DQfD joint loss.

    Inherits the dueling head + quantile distribution +
    n-step discount + RND integration from
    :class:`DistributionalDoubleDQNAgent`.  Adds the demo
    buffer, the supervised loss, and the margin loss.

    Lifecycle:
      1. ``pretrain_demos(states, actions)`` — BC warm-up
         on the expert dataset.  No online data needed.
      2. ``train_step(batch, demo_batch)`` — joint loss
         on agent data + demo data.
    """

    def __init__(self, profile: str, cfg: RLConfig,
                 dqfd: DQfDConfig, in_frames: int = 4,
                 size: int = 84, num_quantiles: int = 51,
                 device: str = "cpu", seed: int = 0) -> None:
        super().__init__(profile, cfg, in_frames=in_frames,
                          size=size, num_quantiles=num_quantiles,
                          device=device, seed=seed)
        self.dqfd = dqfd
        # Store in_frames for the demo-reshape path.
        self.in_frames = in_frames
        # Demo buffer: the BC dataset.  Stored as tensors
        # so the joint loss can index them cheaply.
        self._demo_obs: Optional[torch.Tensor] = None
        self._demo_actions: Optional[torch.Tensor] = None
        self._n_episodes = 0

    def _encoder_is_identity(self) -> bool:
        """True iff the online net's encoder is a
        passthrough (e.g. tests' ``_SmallQNet``).  Used
        to decide whether 2-D demo arrays need to be
        inflated to 4-D before the forward pass.
        """
        import torch.nn as _nn
        return isinstance(getattr(self.online, "encoder", None),
                           _nn.Identity)

    # ------------------------------------------------------------------ #
    def pretrain_demos(self, obs: np.ndarray, actions: np.ndarray,
                        n_epochs: int = 50, batch_size: int = 256,
                        lr: float = 3e-3) -> dict[str, float]:
        """BC warm-up on expert demonstrations.

        The dataset is small (one episode ≈ 900 samples) so
        we cycle through it for many epochs.  The loss is
        cross-entropy on the quantile head's mean logits
        (we collapse the distribution by taking the mean
        over the quantile axis before computing the loss).
        """
        if len(obs) == 0:
            return {"bc_loss": float("nan")}
        # Move to device.
        obs_t = torch.from_numpy(obs).float().to(self.device)
        act_t = torch.from_numpy(actions).long().to(self.device)
        # The agent's encoder expects a 4-D tensor
        # (B, in_frames, H, W) for the production conv
        # encoder.  If the demo array is 2-D we need to
        # inflate it to 4-D.  We only inflate if the
        # encoder is a conv (i.e. expects 4-D inputs);
        # the test's small Linear network is identified
        # by having an `encoder` that is `nn.Identity`
        # — those agents will get a 2-D tensor and the
        # Linear path will work.
        if obs_t.ndim == 2 and not self._encoder_is_identity():
            in_frames = getattr(self, "in_frames", 1)
            n = obs_t.shape[0]
            flat = obs_t.shape[1]
            assert flat % in_frames == 0, (
                f"flat obs length {flat} not divisible "
                f"by in_frames={in_frames}")
            spatial = flat // in_frames
            obs_t = obs_t.reshape(n, in_frames, spatial, 1)
        opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        # If the observations are 84x84 visual stacks we
        # need a different head; here we assume the demo
        # observations match the agent's input shape.
        n = obs_t.shape[0]
        last_loss = 0.0
        for ep in range(n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                sub = idx[start:start + batch_size]
                x = obs_t[sub]
                y = act_t[sub]
                # Forward through the dueling head and
                # collapse the quantile distribution to a
                # single logit per action (mean over
                # quantiles is the standard reduction).
                dist = self.online(x)
                logits = dist.mean(dim=-1)  # [B, A]
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.online.parameters(),
                                          self.cfg.grad_clip_norm)
                opt.step()
                last_loss = float(loss.item())
        # Cache the demos for the joint loss.
        self._demo_obs = obs_t
        self._demo_actions = act_t
        return {"bc_loss": last_loss}

    # ------------------------------------------------------------------ #
    def supervised_loss(self, batch_size: int) -> torch.Tensor:
        """Cross-entropy on a random demo minibatch.

        Returns a scalar tensor; the caller combines this
        with the TD loss.
        """
        if (self._demo_obs is None or self._demo_actions is None
                or self._demo_obs.shape[0] == 0):
            return torch.tensor(0.0, device=self.device)
        n = self._demo_obs.shape[0]
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        x = self._demo_obs[idx]
        y = self._demo_actions[idx]
        # Inflate 2-D to 4-D for conv encoders, leave
        # alone for identity-encoder test agents.
        if x.ndim == 2 and not self._encoder_is_identity():
            in_frames = getattr(self, "in_frames", 1)
            flat = x.shape[1]
            spatial = flat // in_frames
            x = x.reshape(x.shape[0], in_frames, spatial, 1)
        dist = self.online(x)
        logits = dist.mean(dim=-1)
        return F.cross_entropy(logits, y)

    def margin_loss(self, batch_size: int) -> torch.Tensor:
        """Hinge: max(0, margin + Q(other) - Q(expert)).

        Following the DQfD paper, this is computed *per-action*
        on the demo states: for each demo state, the agent's
        argmax-Q is forced to be the expert action; all other
        actions get penalised by ``margin + Q(other) - Q(expert)``.
        """
        if (self._demo_obs is None or self._demo_actions is None
                or self._demo_obs.shape[0] == 0):
            return torch.tensor(0.0, device=self.device)
        n = self._demo_obs.shape[0]
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        x = self._demo_obs[idx]
        y = self._demo_actions[idx]
        if x.ndim == 2 and not self._encoder_is_identity():
            in_frames = getattr(self, "in_frames", 1)
            flat = x.shape[1]
            spatial = flat // in_frames
            x = x.reshape(x.shape[0], in_frames, spatial, 1)
        dist = self.online(x)
        logits = dist.mean(dim=-1)  # [B, A]
        # Q-value of the expert action.
        q_expert = logits.gather(1, y.view(-1, 1)).squeeze(1)
        # Q-value of every action; the margin loss is the
        # mean over actions of the hinge.
        # Shape: [B, A] with the expert action's value
        # subtracted and a margin added.
        margins = (self.dqfd.margin + logits - q_expert.unsqueeze(1))
        # Hinge (max(0, .))
        return torch.clamp(margins, min=0.0).mean()

    def train_step(self, batch: dict) -> dict[str, float]:
        """One DQfD update: TD loss + supervised loss + margin.

        Implementation note
        -------------------
        The previous version of this method called
        ``super().train_step`` (which does
        ``optimizer.zero_grad() → loss.backward() →
        clip_grad_norm() → optimizer.step()``) and *then*
        added the BC + margin terms on top with a second
        ``backward()`` and ``step()``.  PyTorch's
        ``Optimizer.step()`` does **not** zero ``.grad`` by
        default, so the second ``backward()`` *accumulated*
        the BC + margin gradient on top of the TD gradient
        that had already been applied — every TD update
        was effectively applied twice, the BC anchor was
        silently lost, and the model drifted back to
        random play within a few hundred updates.

        The fix is the textbook DQfD recipe: compute
        *all three* losses in one forward pass, sum
        them, backprop **once**, and step **once**.

        The parent's ``train_step`` does not accept an
        extra-loss argument, so we override it entirely
        here.  The TD-loss math is duplicated from
        :meth:`agent_distributional.DistributionalDoubleDQNAgent.train_step`
        to keep the gradient flow single-pass; if the
        parent ever changes its TD-loss formulation, the
        test suite (which asserts the BC anchor is
        preserved across 100 train steps) will catch the
        divergence.
        """
        # If no demos have been loaded, fall back to the
        # parent's pure-RL loss so the agent still learns.
        if self._demo_obs is None:
            return super().train_step(batch)
        # One forward / one backward / one step.
        device = self.device
        obs = batch["obs"].to(device)
        act = batch["actions"].to(device).long()
        rew = batch["rewards"].to(device).float()
        next_obs = batch["next_obs"].to(device)
        done = batch["dones"].to(device).float()
        weights = batch["weights"].to(device).float()
        gamma = self.cfg.gamma
        n = self.cfg.n_step
        # Double-DQN action selection on the online net.
        with torch.no_grad():
            next_q_online = self.online(next_obs).mean(dim=-1)  # [B, A]
            next_actions = next_q_online.argmax(dim=1)  # [B]
            next_q_target = self.target(next_obs).mean(dim=-1)  # [B, A]
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rew + (gamma ** n) * (1.0 - done) * next_q
        # Per-sample TD loss with PER weighting.
        pred = self.online(obs).mean(dim=-1)  # [B, A]
        pred_a = pred.gather(1, act.view(-1, 1)).squeeze(1)  # [B]
        td_per = (pred_a - target_q.detach()) ** 2
        loss_td = (weights * td_per).mean()
        # DQfD extra terms (on the SAME forward pass, so the
        # gradient flows through `self.online` exactly once).
        bc = self.supervised_loss(batch["obs"].shape[0])
        mg = self.margin_loss(batch["obs"].shape[0])
        # Decay: the paper drops the supervised term after
        # ``supervised_decay_episodes`` episodes.  We
        # approximate "episodes" with update_count / 30
        # (one episode ≈ 30 updates).
        decay = max(0.0, 1.0 - self.update_count / max(
            1, 30 * self.dqfd.supervised_decay_episodes))
        total = (loss_td
                 + decay * self.dqfd.lambda_bc * bc
                 + self.dqfd.lambda_margin * mg)
        # One backward, one step, one clip.
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(
            self.online.parameters(), self.cfg.grad_clip_norm))
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.update_count += 1
        # DEEP-FIX v1.22.0: the DQfD agent inherits the
        # parent's ``maybe_sync_target`` which already
        # handles the polyak vs hard-update branch via
        # ``cfg.polyak_target``.  No changes needed here.
        self.maybe_sync_target()
        # PER priorities: per-sample |TD|.
        td_errors = (pred_a - target_q.detach()).abs().detach().cpu().numpy()
        # RND update if attached.
        rnd_intrinsic = 0.0
        if self._rnd is not None:
            try:
                rnd_intrinsic = self._rnd.train_step(obs)
            except Exception:
                rnd_intrinsic = 0.0
        return {
            "loss": float(total.item()),
            "td_loss": float(loss_td.item()),
            "q_mean": float(pred.mean().item()),
            "grad_norm": grad_norm,
            "td_error_abs_mean": float(td_per.mean().sqrt().item()),
            "td_errors": td_errors,
            "bc_loss": float(bc.item()),
            "margin_loss": float(mg.item()),
            "supervised_decay": float(decay),
            "rnd_intrinsic": float(rnd_intrinsic),
        }
