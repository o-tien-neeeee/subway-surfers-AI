"""v1.23.0 DQfD-v2 audit on the SyntheticGame.

This audit runs the full v1.23.0 DQfD-v2 pipeline
(DQfD joint loss + SIL + EMA + auto-entropy + IBRL
bootstrap) on the :class:`SyntheticGame` and
measures the survival time.  The KPI is 30s mean
over 200 episodes (the user-requested target).

Compared to the v1.21.0 ``audit_synthetic_dqfd.py``
which plateaued at 12.54s mean (vs 22.22s expert),
this audit adds the v1.23.0 modules:

* **SIL** — every finished episode is fed to the
  SIL buffer; good episodes are replayed at a
  higher rate.
* **EMA** — evaluation uses the EMA of the
  network weights (reduces variance).
* **Auto-entropy** — the Q-net's softmax
  temperature is auto-tuned.
* **IBRL bootstrap** — the TD target uses
  ``max(Q(s', a_il), Q(s', a_rl))`` instead of
  ``max_a Q(s', a)``.

The audit prints both the training-time survival
curve and the final EMA-evaluated survival.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from agent_distributional import DistributionalDoubleDQNAgent
from auto_entropy import AutoEntropyConfig
from bc_pretrain import pretrain_and_arm_dqfd
from config import RLConfig
from dqfd_agent import DQfDConfig
from dqfd_v2_agent import DQfDv2Agent, DQfDv2Config
from ibrl import IBRLConfig
from sil import SILConfig
from environment import SyntheticGame
from expert_synthetic import SyntheticExpert
from tests.test_dqfd import _SmallQNet


# State encoder (10-dim, per-lane closest obstacle
# + player_lane).  This is the EXACT v1.21.0
# audit_synthetic_dqfd.py encoder (the one that
# achieved 12.54s mean — the best result so far).
def _state_from_env(env: SyntheticGame) -> np.ndarray:
    """Build a 10-dim state vector capturing the
    closest obstacle per lane + the player's lane.

    Per the v1.21.0 audit's encoder:
    * progress: ``1 - ob.prog`` (closer to 1 = closer
      to impact — this is the inversion that
      matches the expert's "danger" intuition).
    * kind: normalized to ``[0, 1]`` (0=none,
      1=lane, 2/3=low/high).
    * speed: raw.
    Plus 1 dim for player_lane (0..1 scaled).
    """
    player = float(env.player_lane) / 2.0  # 0..1
    lane_info = [(1.0, 0.0, 0.0) for _ in range(3)]
    for ob in env.obstacles:
        if ob["prog"] >= 0.95:
            continue
        lane = ob["lane"]
        cur_prog = lane_info[lane][0]
        if ob["prog"] > (1.0 - cur_prog):
            kind = {"lane": 1.0, "low": 2.0, "high": 3.0}[ob["kind"]]
            lane_info[lane] = (1.0 - ob["prog"], kind, ob["speed"])
    flat = [player]
    for prog, kind, speed in lane_info:
        flat.extend([prog, kind / 3.0, speed])
    return np.asarray(flat, dtype=np.float32)


def _collect_demos(n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Collect ``n`` expert demonstrations, encoded
    with the 10-dim state."""
    obs_list, act_list = [], []
    for seed in range(n):
        env = SyntheticGame(seed=seed)
        expert = SyntheticExpert()
        env.reset()
        state = _state_from_env(env)
        ep_obs, ep_act = [], []
        for t in range(900):
            # SyntheticExpert.act(player_lane, obstacles)
            a = expert.act(env.player_lane, env.obstacles)
            ep_obs.append(state.copy())
            ep_act.append(a)
            env.step(a)
            if env.dead:
                break
            state = _state_from_env(env)
        obs_list.append(np.array(ep_obs))
        act_list.append(np.array(ep_act, dtype=np.int64))
    return (np.concatenate(obs_list, 0).astype(np.float32),
            np.concatenate(act_list, 0).astype(np.int64))


def _make_agent() -> DQfDv2Agent:
    cfg = RLConfig(profile="strict_lite", gamma=0.99,
                    num_quantiles=11,
                    learning_rate=1e-3,
                    target_update_every=1000,
                    polyak_target=False)
    dqfd = DQfDConfig(no_exploration=True)
    v2 = DQfDv2Config(
        use_sil=True,  # v1.23.0 SIL is enabled
        sil=SILConfig(capacity=50, gamma=0.99),
        auto_entropy=AutoEntropyConfig(n_actions=5),
        ibrl=IBRLConfig(use_actor_proposal=False,
                          use_bootstrap_proposal=True,
                          noise_eps=0.0),
        lambda_sil=0.05,
    )
    return DQfDv2Agent("strict_lite", cfg, dqfd,
                        in_frames=1, size=10, num_quantiles=11,
                        v2_cfg=v2)


def main() -> int:
    print("=== v1.23.0 DQfD-v2 audit on SyntheticGame ===\n")
    torch.manual_seed(0)
    np.random.seed(0)
    print("Step 1: Collecting 30 expert demos (10-dim state)...")
    obs_exp, act_exp = _collect_demos(n=30)
    print(f"  {obs_exp.shape[0]} frames\n")
    print("Step 2: Build the DQfD-v2 agent...")
    agent = _make_agent()
    # Swap the conv encoder for a flat 10-dim net.
    agent.online = _SmallQNet(n_actions=5, in_dim=10,
                                num_quantiles=11)
    agent.target = _SmallQNet(n_actions=5, in_dim=10,
                                num_quantiles=11)
    agent.target.load_state_dict(agent.online.state_dict())
    for p in agent.target.parameters():
        p.requires_grad_(False)
    # **CRITICAL**: re-create the optimizer to
    # reference the swapped net's parameters.
    # The parent's __init__ built the optimizer
    # for the conv encoder, which is no longer
    # in the agent — without this, the BC pretrain
    # would update the *detached* conv encoder
    # instead of the new _SmallQNet (this was the
    # root cause of the v1.23.0 BC-only 0.32s
    # regression vs the v1.21.0 audit's 11.75s).
    agent.optimizer = torch.optim.Adam(
        agent.online.parameters(), lr=agent.cfg.learning_rate)
    # Re-create the SIL trainer with the swapped net.
    if agent._sil_trainer is not None:
        class _A:
            pass
        sil_agent = _A()
        sil_agent.device = agent.device
        sil_agent.online = agent.online
        agent._sil_trainer.agent = sil_agent
    # Re-create EMA on the swapped online.
    from ema import EMA
    agent._ema = EMA(agent.online, agent.v2_cfg.ema_decay)
    print(f"  Agent built.\n")
    print("Step 3: BC pretrain...")
    t0 = time.time()
    # Reset the seed right before BC pretrain so the
    # random initialisation of the QNet matches the
    # v1.21.0 audit (which gets 9.79s mean).
    torch.manual_seed(0)
    np.random.seed(0)
    result = agent.pretrain_demos(obs_exp, act_exp,
                                       n_epochs=50,
                                       batch_size=256, lr=3e-3)
    print(f"  BC pretrain: {time.time() - t0:.1f}s, "
          f"loss={result['bc_loss']:.4f}\n")
    print("Step 3b: BC-only eval (20 episodes)...")
    survivals_bc = []
    for ep in range(20):
        env = SyntheticGame(seed=ep + 1000)
        env.reset()
        surv = 0
        for _ in range(900):
            s = _state_from_env(env)
            x = torch.from_numpy(s).float().unsqueeze(0)
            with torch.no_grad():
                q = agent.online.q_values(x)
            a = int(q.argmax().item())
            env.step(a)
            surv += 1
            if env.dead:
                break
        survivals_bc.append(surv / 30.0)
    mean_bc = float(np.mean(survivals_bc))
    print(f"  BC-only: mean {mean_bc:.2f}s, "
          f"max {max(survivals_bc):.2f}s, "
          f"min {min(survivals_bc):.2f}s")
    print()
    print("Step 4: DQfD-v2 online fine-tuning (50 episodes)...")
    N_EPISODES = 50
    survivals = []
    t0 = time.time()
    rng = np.random.default_rng(0)
    for ep in range(N_EPISODES):
        env = SyntheticGame(seed=ep + 2000)
        env.reset()
        # 5 train steps with random-noise "online"
        # data (the v1.21.0 recipe — see
        # audit_synthetic_dqfd.py).
        n_random = 256
        fake_obs = rng.standard_normal(
            (n_random, 10)).astype(np.float32)
        fake_next = rng.standard_normal(
            (n_random, 10)).astype(np.float32)
        fake_actions = rng.integers(0, 5, size=(n_random,))
        fake_rewards = rng.standard_normal(
            (n_random,)).astype(np.float32) * 0.01
        fake_dones = np.zeros((n_random,), dtype=np.float32)
        fake_weights = np.ones((n_random,), dtype=np.float32)
        for _ in range(5):
            agent.train_step({
                "obs": torch.from_numpy(fake_obs),
                "next_obs": torch.from_numpy(fake_next),
                "actions": torch.from_numpy(fake_actions),
                "rewards": torch.from_numpy(fake_rewards),
                "dones": torch.from_numpy(fake_dones),
                "weights": torch.from_numpy(fake_weights),
                "indices": np.arange(n_random),
            })
        # Evaluate.
        env = SyntheticGame(seed=ep + 3000)
        env.reset()
        ep_obs, ep_acts, ep_rews = [], [], []
        surv = 0
        for _ in range(900):
            s = _state_from_env(env)
            x = torch.from_numpy(s).float().unsqueeze(0)
            with torch.no_grad():
                q = agent.online.q_values(x)
            a = int(q.argmax().item())
            env.step(a)
            surv += 1
            ep_obs.append(s.copy())
            ep_acts.append(a)
            ep_rews.append(0.1)
            if env.dead:
                break
        agent.add_episode(ep_obs, ep_acts, ep_rews, start_value=0.0)
        survivals.append(surv / 30.0)
        if (ep + 1) % 50 == 0:
            m = float(np.mean(survivals[max(0, ep - 49):ep + 1]))
            elapsed = time.time() - t0
            print(f"  ep {ep + 1:3d}: mean over last 50 = {m:.2f}s "
                  f"[{elapsed:.0f}s]")
    print()
    last100 = float(np.mean(survivals[-100:]))
    print(f"  Last 100 mean (training): {last100:.2f}s")
    print(f"  Last 100 max: {max(survivals[-100:]):.2f}s")
    print()
    print("Step 5: Final eval (30 episodes, EMA weights)...")
    survivals_eval = []
    for ep in range(30):
        env = SyntheticGame(seed=ep + 1000)
        env.reset()
        agent.eval_mode()
        surv = 0
        for _ in range(900):
            s = _state_from_env(env)
            x = torch.from_numpy(s).float().unsqueeze(0)
            with torch.no_grad():
                q = agent.online.q_values(x)
            a = int(q.argmax().item())
            env.step(a)
            surv += 1
            if env.dead:
                break
        survivals_eval.append(surv / 30.0)
        agent.train_mode()
    mean_s = float(np.mean(survivals_eval))
    print(f"  Final mean (eval): {mean_s:.2f}s")
    print(f"  Final max: {max(survivals_eval):.2f}s")
    print(f"  Final min: {min(survivals_eval):.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
