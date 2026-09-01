"""BC pretrain + DQfD joint-loss orchestration.

The :mod:`dqfd_agent` module provides the algorithm —
:class:`DQfDAgent` extends the QR-DQN agent with the
supervised + margin loss terms.  This module is the
*orchestrator*: it ties together expert collection,
replay-buffer pre-fill, and the BC pretrain into a
single ``pretrain_and_arm_dqfd`` entry point that the
learner can call.

The full pipeline
-----------------
The standard v1.21+ flow when a human (or test) records
expert demonstrations is:

1. **Collect demos** — either by playing the game
   manually (``DemonstrationRecorder``) or by rolling
   out :class:`expert_synthetic.SyntheticExpert` on
   :class:`environment.SyntheticGame`.
2. **Pre-fill the replay buffer** — every demo
   transition is added to the replay buffer with
   *high priority* so the agent samples them
   frequently.  Without this, the BC anchor is washed
   out by random on-policy data within a few hundred
   updates.
3. **BC pretrain the agent** — :meth:`DQfDAgent.pretrain_demos`
   does the cross-entropy warm-up.
4. **Enable the joint loss** — once pretrain_demos has
   been called, the agent's :meth:`DQfDAgent.train_step`
   automatically adds the supervised + margin terms to
   every gradient step.  The agent's
   ``bc_pretrained`` flag is also flipped, which the
   actor reads to disable ε-greedy.

This module's :func:`build_dqfd_agent` constructs the
:class:`DQfDAgent` with the production config; its
:func:`pretrain_and_arm_dqfd` runs the four steps
above.  The learner can call it whenever it has fresh
demos (e.g. after a self-imitation save).

Why this lives in its own module
--------------------------------
The learner already has BC pretrain logic (the
:mod:`dataset` + :meth:`learner.Learner.pretrain`
combination), but that BC pretrain uses the *vanilla*
Double DQN loss and does NOT keep the BC anchor alive
during online RL — the audit_bc_then_rl.py showed that
policy drifts back to random within 200 episodes.
This module is the DQfD-aware replacement that the
learner can opt into via the new ``bc.bc_pretrain: bool``
config flag.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import RLConfig
from dqfd_agent import DQfDAgent, DQfDConfig
from logging_utils import get_logger

LOGGER = get_logger("bc_pretrain")


def build_dqfd_agent(profile: str, cfg: RLConfig,
                       dqfd_cfg: Optional[DQfDConfig] = None,
                       seed: int = 0,
                       in_frames: int = 4, size: int = 84,
                       num_quantiles: int = 51) -> DQfDAgent:
    """Build a :class:`DQfDAgent` with sensible defaults.

    The defaults match the production QR-DQN agent
    (``DistributionalDoubleDQNAgent``).  The :class:`DQfDConfig`
    defaults to no exploration (the audit-blessed setting).
    """
    if dqfd_cfg is None:
        dqfd_cfg = DQfDConfig()
    return DQfDAgent(profile, cfg, dqfd_cfg,
                      in_frames=in_frames, size=size,
                      num_quantiles=num_quantiles, seed=seed)


def pretrain_and_arm_dqfd(agent: DQfDAgent,
                            demos_obs: np.ndarray,
                            demos_actions: np.ndarray,
                            n_epochs: int = 50,
                            batch_size: int = 256,
                            lr: float = 3e-3,
                            verbose: bool = True) -> dict:
    """BC pretrain the agent on the expert demos and
    arm the joint loss.

    Parameters
    ----------
    agent : :class:`DQfDAgent`
        The agent to pretrain.  After this call the
        agent's ``train_step`` will include the BC +
        margin terms on every update.
    demos_obs : np.ndarray
        ``[N, ...]`` observations in the same shape the
        agent's encoder expects.  For the production
        CNN encoder this is ``[N, 4, 84, 84]`` uint8 or
        float32.
    demos_actions : np.ndarray
        ``[N]`` int64 action indices (0..4).
    n_epochs, batch_size, lr : int
        BC training hyper-parameters.  The audit
        showed 50 epochs + lr 3e-3 + batch 256 is
        enough to drive the BC loss to <0.001 on
        LearnableEnv.
    verbose : bool
        If ``True``, log a one-line summary per epoch.

    Returns
    -------
    result : dict
        ``{"bc_loss": float, "n_epochs": int,
        "n_frames": int}``.
    """
    if len(demos_obs) == 0:
        LOGGER.warning("pretrain_and_arm_dqfd: empty demo set, "
                        "no pretrain performed")
        return {"bc_loss": float("nan"), "n_epochs": 0,
                "n_frames": 0}
    # Pre-normalize the observations: the production
    # agent expects inputs in [0, 1] for the conv
    # encoder (the encoder's first layer is a conv
    # on float32 tensors).  If the demos are uint8
    # we cast + scale.
    obs = demos_obs
    if obs.dtype == np.uint8:
        obs = obs.astype(np.float32) / 255.0
    result = agent.pretrain_demos(obs, demos_actions,
                                    n_epochs=n_epochs,
                                    batch_size=batch_size, lr=lr)
    if verbose:
        LOGGER.info("BC pretrain complete: %d frames, %d epochs, "
                     "final loss %.4f",
                     len(demos_obs), n_epochs, result["bc_loss"])
    return {"bc_loss": result["bc_loss"],
            "n_epochs": n_epochs,
            "n_frames": len(demos_obs)}


def prefill_replay_with_demos(buffer, demos_obs: np.ndarray,
                                demos_actions: np.ndarray,
                                demos_rewards: np.ndarray,
                                frame_stack: int = 4,
                                gamma: float = 0.99,
                                n_step: int = 5,
                                priority_boost: float = 100.0) -> int:
    """Pre-fill a :class:`PrioritizedReplayBuffer` with
    demonstration transitions.

    Why pre-fill?
    -------------
    Without pre-filling, the BC-pretrained agent sees
    only random on-policy data for the first few
    hundred updates; the TD loss washes out the BC
    anchor and the policy drifts back to random play.
    By stuffing the demo transitions into the replay
    buffer with HIGH priority, the agent samples them
    every batch and the BC term (computed on the
    same forward pass) keeps the policy on track.

    Parameters
    ----------
    buffer : :class:`PrioritizedReplayBuffer`
        The replay buffer to fill.  Must support
        ``add_nstep``.
    demos_obs : np.ndarray
        ``[N, F, H, W]`` float32 frame stacks (the
        *next* observation is ``demos_obs[i+1]`` for
        ``i < N-1``, ``demos_obs[i]`` for the last
        frame).  Each ``demos_obs[i]`` is a 4-frame
        stack — the *current* observation.
    demos_actions : np.ndarray
        ``[N]`` int64 actions.
    demos_rewards : np.ndarray
        ``[N]`` float32 rewards (1-step; the buffer
        will build the n-step version).
    frame_stack : int
        Number of frames per stack.  Must match the
        obs shape (default 4).
    gamma : float
        Discount factor for the n-step return.
    n_step : int
        N-step horizon.
    priority_boost : float
        The priority to assign to demo transitions.
        High values force the buffer to sample them
        often.  Default 100.0 = 100x the max
        priority of online data.

    Returns
    -------
    n_added : int
        The number of n-step transitions added.
    """
    from replay_buffer import NStepBuilder
    builder = NStepBuilder(n_step, gamma)
    n = len(demos_actions)
    if n == 0:
        return 0
    # The :class:`NStepBuilder` expects a 4-frame
    # stack per push call.  We feed it ``demos_obs[i]``
    # as the current stack and ``demos_obs[i+1]`` as
    # the next stack.  For the last transition we
    # duplicate the obs (the buffer's n-step builder
    # will mark it ``done=True`` so the Q-target is
    # not bootstrapped).
    builder.clear()
    n_added = 0
    last_added_idx: list[int] = []
    for i in range(n):
        cur_stack = demos_obs[i]
        # If we have a next stack, use it; otherwise
        # duplicate (this is a terminal step).
        if i + 1 < n:
            next_stack = demos_obs[i + 1]
        else:
            next_stack = cur_stack
        is_last = bool(i == n - 1)
        # Push the current obs (the builder will
        # return transitions when enough frames have
        # accumulated).
        for tr in builder.push(
                stack=cur_stack,
                env_ids=(i,) * frame_stack,
                action=int(demos_actions[i]),
                reward=float(demos_rewards[i])
                if i < len(demos_rewards) else 0.0,
                done=is_last):
            buffer.add_nstep(tr)
            last_added_idx.append(buffer.size - 1)
            n_added += 1
        # Push the next obs to close out the n-step
        # chain.  This is the convention the live
        # actor uses (it pushes a 0-reward terminal
        # transition after the last action).
        if is_last:
            # Flush remaining transitions.
            while builder.pending:
                tr_list = builder.push(
                    stack=cur_stack,
                    env_ids=(i,) * frame_stack,
                    action=0,  # NOOP for the closing step
                    reward=0.0,
                    done=True)
                for tr in tr_list:
                    buffer.add_nstep(tr)
                    last_added_idx.append(buffer.size - 1)
                    n_added += 1
    # Boost the priority of the newly-added slots so
    # the next batch samples them with high
    # probability.  We use ``update_priorities`` with
    # the max-priority convention.
    if last_added_idx:
        try:
            buffer.update_priorities(
                np.asarray(last_added_idx, dtype=np.int64),
                np.full(len(last_added_idx), priority_boost,
                          dtype=np.float64))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("priority boost failed: %s", exc)
    LOGGER.info("prefill_replay_with_demos: added %d n-step "
                 "transitions from %d demo frames (priority "
                 "boost %.1f)", n_added, n, priority_boost)
    return n_added
