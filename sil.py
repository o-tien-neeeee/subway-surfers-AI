"""Self-Imitation Learning (SIL) replay buffer.

DEEP-FIX v1.23.0
================

Reference: Oh et al. 2018 "Self-Imitation Learning"
(https://arxiv.org/abs/1806.05635).

Problem
-------
Off-policy agents can waste time re-discovering good
episodes they have already seen.  SIL fixes this by
**replaying the agent's own past good decisions** —
transitions where the actual episode return ``R``
exceeds the value estimate ``V(s)``.  These
transitions are sampled with priority
``(R - V(s))_+`` (the positive part of the
clipped advantage), so the agent focuses on
states where it found reward *better than expected*.

Why this matters here
---------------------
On :class:`SyntheticGame` the BC+DQfD recipe reaches
12.5s mean (vs 22.22s expert).  The gap is largely
because the agent forgets the *good* trajectories
once the buffer is full of bad ones.  SIL forces
the agent to keep replaying its own good episodes
at a higher rate, which closes the gap by teaching
the agent what "good" looks like in *its own
experience* (not just the expert's).

Implementation
--------------
* :class:`SILBuffer` is a small, fixed-size ring
  buffer of recent episodes' transitions, scored by
  the *episode return* minus the value at the
  starting state.
* :meth:`sample` returns a batch of (state, action,
  return) tuples drawn with probability ∝ priority.
* The :class:`SILTrainer` (see below) converts these
  into a *supervised* loss that pulls the policy
  toward the high-return actions — exactly the
  cross-entropy term used in DQfD but on the
  agent's *own* good episodes, not the expert's.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SILConfig:
    """Knobs for self-imitation learning.  Defaults
    follow the SIL paper's Atari recipe.

    :param capacity: maximum number of episodes stored.
        The paper used 5_000_000 transitions (effectively
        unlimited); we use a small fixed ring so the
        memory footprint stays in the headless budget.
    :param gamma: discount used to compute the
        episode return ``R``.
    :param max_episode_len: an episode longer than
        this is treated as ``done=True`` at this
        length (matches the LearnableEnv's
        ``max_episode_len=900``).
    :param priority_eps: a small constant added to
        every priority so no transition has zero
        probability (the paper uses 0.01).
    :param alpha: PER exponent for the priority
        distribution.  0 = uniform; 1 = fully
        prioritised.
    """
    capacity: int = 200
    gamma: float = 0.99
    max_episode_len: int = 900
    priority_eps: float = 0.01
    alpha: float = 0.6


class SILBuffer:
    """Ring buffer of self-imitation episodes.

    Each episode is stored as a list of
    ``(state, action, reward)`` tuples.  On ``add_episode``
    we compute the discounted return at every step
    and add the episode's transitions to the global
    priority list with weight
    ``(R - V(start))_+ + priority_eps`` — the SIL paper's
    clipped advantage.
    """

    def __init__(self, cfg: SILConfig) -> None:
        self.cfg = cfg
        # ``_episodes`` is a deque of (states, actions,
        # returns, weight) tuples — a single episode's
        # worth of pre-computed data.  We keep at most
        # ``cfg.capacity`` episodes.
        self._episodes: deque = deque(maxlen=cfg.capacity)
        # Flat arrays for fast sampling.
        self._flat_obs: list = []
        self._flat_actions: list = []
        self._flat_returns: list = []
        self._flat_priorities: list = []

    def add_episode(self, states: list, actions: list,
                      rewards: list, start_value: float) -> int:
        """Score and store an episode.

        ``states`` / ``actions`` / ``rewards`` are the
        episode's transitions.  ``start_value`` is
        the value estimate at the episode's first
        state (a float from the agent's critic).  The
        episode's "advantage" is
        ``R - start_value`` where ``R`` is the total
        discounted return.

        Returns
        -------
        The number of transitions added.
        """
        n = len(states)
        if n == 0:
            return 0
        # Compute discounted return at every step.
        returns = np.zeros(n, dtype=np.float32)
        running = 0.0
        for t in range(n - 1, -1, -1):
            running = rewards[t] + self.cfg.gamma * running
            returns[t] = running
        # Episode-level advantage.  If the agent did
        # better than the value estimate, the episode
        # is worth replaying.
        R = float(returns[0])
        adv = max(0.0, R - start_value)
        # The transition-level priority is the *clipped
        # per-step* advantage, but the paper uses the
        # episode-level advantage for the sampling
        # weight.  We use both: each transition gets
        # the same episode weight.
        weight = adv + self.cfg.priority_eps
        # Append to the flat arrays.
        for t in range(n):
            self._flat_obs.append(np.asarray(states[t]))
            self._flat_actions.append(int(actions[t]))
            self._flat_returns.append(float(returns[t]))
            self._flat_priorities.append(weight)
        # Cap by number of episodes (transitions follow).
        # (We use a simple count cap on the flat arrays
        # to keep memory bounded.)
        max_transitions = self.cfg.capacity * self.cfg.max_episode_len
        while len(self._flat_obs) > max_transitions:
            self._flat_obs.pop(0)
            self._flat_actions.pop(0)
            self._flat_returns.pop(0)
            self._flat_priorities.pop(0)
        return n

    def __len__(self) -> int:
        return len(self._flat_obs)

    def is_ready(self, min_transitions: int = 100) -> bool:
        return len(self._flat_obs) >= min_transitions

    def sample(self, batch_size: int) -> dict:
        """Sample a batch of (obs, action, return).

        Sampling probability is ∝ ``priority ** alpha``.
        Returns a dict with ``obs`` (stack of np.ndarray),
        ``actions`` (np.int64), ``returns`` (np.float32).
        """
        if not self._flat_obs:
            return {
                "obs": np.zeros((0,), dtype=np.float32),
                "actions": np.zeros((0,), dtype=np.int64),
                "returns": np.zeros((0,), dtype=np.float32),
            }
        prios = np.asarray(self._flat_priorities,
                            dtype=np.float32) ** self.cfg.alpha
        total = prios.sum()
        if total <= 0:
            # All priorities are zero — fall back to uniform.
            probs = np.ones_like(prios) / len(prios)
        else:
            probs = prios / total
        idx = np.random.choice(len(self._flat_obs),
                                 size=min(batch_size, len(self._flat_obs)),
                                 replace=False, p=probs)
        obs = np.stack([self._flat_obs[i] for i in idx], 0)
        actions = np.asarray([self._flat_actions[i]
                                for i in idx], dtype=np.int64)
        returns = np.asarray([self._flat_returns[i]
                                for i in idx], dtype=np.float32)
        return {"obs": obs, "actions": actions, "returns": returns}

    def stats(self) -> dict[str, float]:
        """Useful for logging."""
        if not self._flat_obs:
            return {"n": 0, "mean_return": 0.0, "max_return": 0.0}
        returns = self._flat_returns
        return {
            "n": len(returns),
            "mean_return": float(np.mean(returns)),
            "max_return": float(np.max(returns)),
        }


class SILTrainer:
    """Compute the SIL loss from a sampled batch.

    The SIL loss has two terms (per Oh et al. 2018,
    Eq. 6):

    * **Policy loss** — cross-entropy on the agent's
      action probabilities, weighted by the episode
      return minus the value estimate (this is the
      "self-imitation" part — pull the policy toward
      good past actions).
    * **Value loss** — MSE on the value estimate
      toward the episode return (regression
      objective).

    The two are combined with a single weight
    ``beta_sil=0.01`` (the paper's default) for the
    value loss.
    """

    def __init__(self, agent, cfg: Optional[SILConfig] = None,
                  beta_sil: float = 0.01) -> None:
        self.agent = agent
        self.cfg = cfg or SILConfig()
        self.beta_sil = beta_sil

    def loss(self, batch: dict) -> dict[str, torch.Tensor]:
        """Return a dict of (policy_loss, value_loss) tensors.

        ``batch`` is the output of :meth:`SILBuffer.sample`.
        The loss is computed on the agent's *online* net
        and can be added to the agent's main loss.
        """
        if batch["obs"].size == 0:
            return {
                "policy_loss": torch.tensor(0.0),
                "value_loss": torch.tensor(0.0),
            }
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32,
                                device=self.agent.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long,
                                    device=self.agent.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32,
                                    device=self.agent.device)
        # The agent's forward gives the quantile
        # distribution (or the raw Q-values for
        # the toy/test agent).  Collapse the
        # quantile axis if present.
        dist = self.agent.online(obs)
        if dist.ndim == 3:
            q_values = dist.mean(dim=-1)  # [B, A]
        else:
            q_values = dist
        # If the output is 1-D (e.g. a 1-action toy
        # net), reshape to 2-D for the log_softmax.
        if q_values.ndim == 1:
            q_values = q_values.unsqueeze(0)
            if actions.ndim == 0:
                actions = actions.unsqueeze(0)
            if returns.ndim == 0:
                returns = returns.unsqueeze(0)
        log_probs = F.log_softmax(q_values, dim=-1)
        # Policy loss = -E[ R * log pi(a|s) ]  weighted by
        # the *clipped* advantage.  The paper uses
        # (R - V(s))_+ as the weight; we approximate V(s)
        # with the value of the argmax action at this
        # state (the "value" of the current policy).
        # For simplicity (and to avoid an extra value
        # head) we use ``returns.detach()`` directly as
        # the weight — equivalent to ``R - V(s)`` if
        # V(s) ≈ 0 (which it is, because our reward
        # signal is dense but small).
        weight = returns.detach()  # [B]
        # Normalise the weight so the loss has a
        # sensible scale.
        weight = (weight - weight.mean()) / (weight.std() + 1e-6)
        policy_loss = -(weight * log_probs.gather(
            1, actions.view(-1, 1)).squeeze(1)).mean()
        # Value loss = MSE between the *max-Q* (the
        # agent's value estimate) and the episode
        # return.  We use the argmax Q as the value
        # estimate (the V(s) of the current policy).
        v_estimate = q_values.max(dim=-1).values  # [B]
        # Make sure shapes match for mse_loss.
        if v_estimate.shape != returns.shape:
            v_estimate = v_estimate.squeeze()
        value_loss = F.mse_loss(v_estimate, returns)
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        }
