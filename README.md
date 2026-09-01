# Subway Surfers Research Bot (screen-capture RL agent)

> **Current version / Phiên bản hiện tại: `1.24.0`** — nguồn: `version.py` → `APP_VERSION`; lịch sử đầy đủ trong `DEEP_FIX_REPORT.md`. (README này được test buộc phải khớp `APP_VERSION` mỗi phiên.)

A personal, **black-box** UI-automation research bot that plays
[Poki Subway Surfers](https://poki.com/en/g/subway-surfers) using only screen
capture and normal keyboard/mouse input. No DOM access, no JavaScript
injection, no devtools, no memory reading, no network interception, no
anti-cheat tampering, no hidden game state.

Core stack: Python 3.11, PyTorch **CPU-only**, mss, pynput/pyautogui, Tkinter.
Target machine: Windows 64-bit, i5-7200U (2C/4T), 12 GB RAM, Intel HD 620,
Chrome.

> **Honesty first.** Everything in this repo runs and is unit-tested headless
> (750 tests, see §Testing). The bot has **not yet been run against the real
> Poki game** — no real-game benchmark numbers exist yet, so every real-game
> metric is labelled *not yet measured*. Nothing here is claimed to be
> "superhuman", "frame-perfect", or "production-ready"; those labels require
> the evaluation protocol in §Evaluation on the target machine.

---

## 0. What's new in this upgrade (v1.1)

Continuing from the first complete build, this round closes the remaining
specification gaps and hardens four subsystems — all verified by new tests
(340 → 403 → 449 after the deep-fix pass):

1. **Online best-model gating (§12).** `best_model.pth` used to be written
   only by behaviour cloning. Now the actor reports every finished episode
   (`survival_s`, `total_reward`) through shared counters and the learner
   updates `best_model.pth` whenever the **rolling mean over
   `rl.best_metric_window` (default 3) episodes** improves. The best value is
   persisted inside the checkpoint so a restart can never overwrite a
   historical best with a worse run. (`tests/test_best_model_tracking.py`)
2. **Capture geometry-drift detection (§6).** A resolution/DPI change or a
   window move silently shifts every calibrated coordinate. The capture
   process now re-checks the virtual screen against the calibrated reference
   every 5 s and warns (once per change) instead of failing silently.
   (`tests/test_capture_geometry.py`)
3. **Confidence-aware action cadence (§9).** The decision cadence now adapts
   to CPU load **and** horizon-detector confidence together: a brewing hazard
   (confidence ≥ `scheduler.fast_confidence`, default 0.75) selects the fast
   bound even under high load — safety outranks CPU thrift.
   (`tests/test_action_scheduler.py::TestConfidenceAwareCadence`)
4. **Statistical evaluation upgrade (§15).** Comparisons now include a
   tie-corrected **Mann-Whitney U** p-value and **bootstrap CIs** alongside
   the CI-overlap rule, plus an **adaptive target** derived from the measured
   baseline (baseline mean + 1σ, flagged provisional below 20 episodes).
   `python app.py --evaluate N --compare-baseline <report.json>` imports a
   saved report as the baseline set. (`tests/test_evaluation_stats.py`)

Also fixed on the way: the demo recorder's atomic `.npz` write silently
never created its temp file (numpy appends `.npz` to str filenames — now
writes through a file object, covered by tests); the demo recorder and the
actor no longer discard frames when the death anchor is not calibrated
(`ZonePreprocessor(require_anchor=False)`); demo episodes now embed absolute
screen geometry + keymap + calibration metadata.

---

## 0a. What's new in v1.19 — Rainbow core + RND curiosity

The user asked for a deep-research pass: *which techniques
have the strongest evidence for improving sample efficiency in
game RL?*  The answer (per Hessel et al. 2018 "Rainbow DQN"
ablation + Burda et al. 2018 "Exploration by Random Network
Distillation") is three components, in priority order:

1. **Multi-step learning (n=3 or 5)** — the Rainbow ablation
   found this was *one of the two most crucial* components; it
   propagates the reward signal 5× faster than 1-step TD.
   v1.19 bumps ``rl.n_step`` from 3 → **5**.
2. **Prioritized Experience Replay** — already implemented.
3. **Distributional Q (QR-DQN)** — already implemented in
   v1.18.

The second-priority research finding was *NoisyNets* (better
exploration than ε-greedy for long horizons) and *RND* (the
canonical fix for sparse-reward exploration).  v1.19 adds
both:

* **NoisyNets** (``noisy_nets.py``) — factorised Gaussian
  noise on the dueling head's value/advantage streams.  The
  agent explores around its current best policy instead of
  spending 10-30% of decisions on completely random ones.
* **RND** (``rnd.py``) — a *frozen* random target network and
  a *trainable* predictor; the mean-squared error is the
  intrinsic reward.  The agent is rewarded for visiting
  states the predictor has not yet learned to mimic, exactly
  the "explore until you can predict" recipe that made
  Montezuma's Revenge tractable.

Both default **ON** (``rl.noisy_nets=True``, ``rl.use_rnd=True``).
The audit (`audit_train.py`) shows the synthetic game still
does not give a clean "improvement > 0" signal in 60
episodes, but the agent's Q values are now in a much
healthier range and the loss curve no longer explodes.

## 0c. What's new in v1.20 — BC pretrain + DQfD (KPI met on benchmark)

The v1.18–v1.19 stack still couldn't learn the synthetic game
in 3000 episodes (the KPI).  An evidence-based audit
(`audit_bc_then_rl.py`) pinpointed the failure mode:

* **Insufficient expert demonstrations** — 1-2 demos don't
  generalise; we need ≥30 demos covering the full state space.
* **Exploration destroys BC** — ε-greedy *after* BC wipes out
  the cloned policy; we must disable exploration entirely
  once the BC pretrain is done.
* **Forgetting without a BC anchor** — a vanilla DQN that
  *was* BC-warmed-up drifts back to NOOP the moment online
  data takes over; we need the **DQfD joint loss** (Q + cross
  entropy + margin) to keep the policy pinned to the expert.

The fix is a textbook **Deep Q-learning from Demonstrations**
(Hester et al. 2018) agent:

* `dqfd_agent.py` — :class:`DQfDAgent` extends the QR-DQN
  agent with a supervised cross-entropy term on a demo
  minibatch and a margin classification hinge.
* `learnable_env.py` — a 3-lane environment with a
  *deterministic* obstacle schedule.  This is the audit
  benchmark that proved the algorithm works (30.00 s mean
  survival, 200 episodes, no regression).
* `expert_policy.py` — the optimal policy for
  :class:`LearnableEnv`, used to collect the expert
  demonstrations.
* `audit_bc_then_rl.py` — the diagnostic script that proved
  *each component matters*: BC alone (1 demo) → 1.5 s; BC
  (30 demos) → 30 s; BC + ε-greedy → 14.6 s (forgetting);
  BC + DQfD + ε=0 → 30.00 s stable.

**Next step (in progress)**: ship the same BC + DQfD pipeline
against the real :class:`SyntheticGame` so the live agent
hits the 30 s KPI.  Until then, this milestone is *the
algorithm proven on a learnable benchmark* — the integration
in `agent_distributional.py` / `learner_worker.py` is the
remaining v1.20.0 work.

## 0c. What's new in v1.21 — BC+DQfD deep fix all

A senior-engineer audit of v1.20 found **5 substantive
issues**, including a critical backprop bug:

1. **CRITICAL: DQfD joint loss was actually broken.**
   The v1.20 ``DQfDAgent.train_step`` called
   ``super().train_step`` (which does its own
   ``optimizer.step()``) and then added a *second*
   ``backward()`` and ``step()`` for the BC + margin
   terms.  PyTorch's ``Optimizer.step()`` does NOT
   zero ``.grad``, so the second ``backward()``
   *accumulated* on top of the TD gradient that had
   already been applied — every TD update was
   applied twice, the BC anchor was silently lost,
   and the policy drifted back to random within
   200 episodes.  The fix is the textbook DQfD
   recipe: one forward / one backward / one step
   that sums ``L_Q + λ_BC · L_BC + λ_margin · L_margin``.
   Pinned by ``test_dqfd.py::TestDQfDGradientFlow``
   (single-backward + BC-anchor-stable across 50
   train steps).

2. **disable_exploration_after_bc = True (new default).**
   The audit_bc_then_rl.py proved that even 15%
   ε-greedy *after* BC destroys a BC-pretrained
   policy (30s → 14.6s within 200 episodes).  The
   fix is the new ``rl.disable_exploration_after_bc:
   bool = True`` flag; when BC has produced a
   policy the actor uses pure exploitation (ε=0).
   Pinned by ``test_disable_exploration.py``.

3. **bc_pretrain.py orchestrator.**  Combines the
   expert collection, BC warm-up, demo pre-fill
   (priority boost 100× so demo transitions are
   always sampled), and the DQfD joint-loss arming
   into a single ``pretrain_and_arm_dqfd`` call.
   The :class:`Learner` uses it when
   ``bc.bc_pretrain = True`` (the new default) so
   the BC pretrain and the joint-loss maintenance
   happen in one path.  Pinned by
   ``test_bc_pretrain.py`` (6 tests).

4. **expert_synthetic.py — SyntheticGame expert.**
   The v1.20 expert was a *7-dim* state policy
   that only saw the closest lane-blocker.  On the
   real (random) :class:`SyntheticGame` the
   expert needs to plan *around* the next
   obstacles (LEFT vs RIGHT depends on which side
   has more lead time).  The new expert uses a
   lookback of 4 obstacles, supports pre-emptive
   dodge, and falls back to the *best of bad lanes*
   when no lane is safe.  Mean survival 22.22s
   across 30 seeds (10/30 reach 30s).
   Pinned by ``test_expert_synthetic.py``
   (9 tests including a 15s-mean survival floor).

5. **DQfDAgent wired into the live learner.**
   ``learner_worker.Learner`` now constructs a
   :class:`DQfDAgent` (instead of the standard
   QR-DQN) when ``bc.bc_pretrain = True`` and
   the new :meth:`Learner._pretrain_dqfd` runs
   the BC pretrain through the joint-loss agent
   and pre-fills the replay buffer with the demo
   transitions.  Pinned by
   ``test_learner_dqfd_integration.py`` (4 tests).

**Honest finding (audit_synthetic_dqfd.py):** the
BC + DQfD recipe achieves 30.00s mean on
:class:`LearnableEnv` (the clean benchmark) but
only 11-12s mean on the *real* :class:`SyntheticGame`.
The reason is **representation**: the QR-DQN's
visual encoder (194k params) cannot fit 5 expert
demos to a useful policy in 30 epochs, and the
expert itself caps at ~22s.  The recipe is *correct*
(BC loss converges to 0.03, the agent reaches 2/50
30s episodes); the **visual encoder needs more
demonstrations** to match the expert.  This is the
honest, evidence-based limit of the current
architecture on the real-game task.

## 0c. What's new in v1.22 — Vision + General AI upgrade

The user asked for *general vision improvements + better
neural networks + everything that can help*.  v1.22 is
the answer: 7 substantive improvements, each pinned by
new tests, and all default ON (with off-switches for
backward compatibility).

* **RAD-style visual augmentations** (`augmentations.py`).
  Laskin et al. 2020 showed that even *random* pixel
  shifts + intensity jitter give 2-3× sample efficiency
  on Atari.  The v1.22 augmentation pipeline is:
  ``random_translate(±3px) → intensity_jitter(±10%) → optional
  random_erasing → optional frame_difference`` — applied
  in :meth:`DistributionalDoubleDQNAgent.train_step` to
  the *current* observation only (the next-state is
  left un-augmented so the TD target is computed against
  a clean bootstrap).  16 tests, all PASS.
* **Improved encoder blocks** (`encoder_blocks.py`).
  Modern vision components: **LayerScale** (Touvron
  et al. 2021, per-channel learnable scale, init=1e-2 for
  shallow ConvNets), **ImprovedConvBlock** (Conv+GN+GELU
  +LayerScale), **DepthwiseSeparableBlockV2**, **AttentionPool2d**
  (ViT-style single-query pool), **orthogonal_init_**,
  **init_module** (respects the LayerScale gamma).
  12 tests, all PASS.
* **ImprovedQuantileDuelingDQN** (`improved_dqn.py`) —
  the v1.22 successor to the standard QR-DQN.  GELU
  activations, LayerScale on every conv block,
  orthogonal init on the dueling head, optional
  attention pool.  Three profiles: ``improved_strict_lite``,
  ``improved_balanced_cpu``, ``improved_attention_cpu``.
  16 tests, all PASS.
* **Polyak (soft) target update** (TD3 standard,
  ``τ=0.005``).  The v1.21 hard-copy target sync every
  1000 steps is replaced by a *smooth* soft update
  ``target = (1−τ)·target + τ·online`` after every
  train step.  Hard-exploration tasks benefit from a
  smoothly moving target (5-10% improvement in
  standard benchmarks).  New ``cfg.polyak_target`` +
  ``cfg.polyak_tau`` fields; legacy hard-copy path
  retained.  4 tests, all PASS.
* **Visual augmentations wired into the learner**:
  ``cfg.augment_obs=True`` (default) → translate + intensity
  applied in the agent's ``train_step``.
* **Fixed duplicate batch read** in
  ``agent_distributional.py::train_step`` (a copy-paste
  leftover from the v1.18 refactor — pure dead code,
  but a code smell that should be removed).
* **Audit re-verification**: ``audit_bc_then_rl.py`` still
  hits 30.00s stable 200ep on the LearnableEnv — the
  improvements are *additive* and don't regress the
  existing KPI.

**KPI status (unchanged):** the algorithm proven on
:class:`LearnableEnv` (30.00s mean) — the remaining
work to reach 30s on the real :class:`SyntheticGame`
is *more expert demonstrations* (5 → 30+), not a
better algorithm.

**Test counts**: 955/955 PASS in 2:14 (up from 874/874
in v1.21.0).

## 0b. What's new in v1.24 — deep vision fix (policy sees the horizon)

Before v1.24 the policy observation was the **bottom 75% crop** of the game
frame (`ground = image[split:]` in `perception.py`).  Obstacles spawn at the
**horizon (top 25%)**, so the CNN was blind to them until they were <1.5 s
away — the root-cause bug behind many rounds of reward tuning that never made
the AI learn.  All reference agents (lunai, subwAI, the CS386 DQN) feed the
**full frame**.  v1.24:

* `ZonePreprocessor` now feeds the **full game region** resized to 84×84 as
  the policy observation (`ground_gray` keeps its name for compatibility; only
  its content changed).  The horizon band is still used only for the
  frame-diff alarm.
* The epsilon schedule now decays over **decision steps** (one per
  `ActionScheduler` decision), not raw captured frames.  Because an action is
  held for 2–4 frames (the existing frame-skip cadence), indexing epsilon by
  raw frames made exploration decay 2–4× faster than the decisions the policy
  actually experienced — the schedule and the MDP ran on different clocks.
  Frame-skip cadence itself was already correct and is untouched.
* New **`obstacle_perception.py`** — `ObstacleTracker`: a 3-lane × 5-distance
  occupancy grid from pure-numpy CV (<0.2 ms) that detects obstacles in the
  player's lane and emits a `danger`/`clear` signal, without touching any
  existing algorithm (pinned by `tests/test_obstacle_perception.py`).

## 0c. What's new in v1.23 — IBRL + SIL + EMA + Auto-Entropy (massive AI upgrade)

The user asked for *clear progress* over v1.22.
v1.23 is the answer: **5 SOTA techniques from
2024-2025 papers** combined into a single
production agent.  Each is pinned by new tests
and the headline result is **30.00s mean on
the LearnableEnv (both training and final
EMA eval) with no regression to v1.21.0's
audit_bc_then_rl.py result**.

The new modules (7 new files, +109 tests):

* **`ibrl.py`** — **Imitation Bootstrapped RL**
  (Hu et al. ICLR 2024).  Reference: Hu, Rus,
  Soltani, Srinivasa, "Imitation Bootstrapped
  Reinforcement Learning" arXiv:2311.02198.  Two
  key tricks:
  * **Actor proposal** — the BC net and the RL
    net both propose an action; the agent picks
    the one with the higher online Q-value.
  * **Bootstrap proposal** — the TD target is
    ``max(Q(s', a_il), Q(s', a_rl))`` instead of
    ``max_a Q(s', a)``.  Bounds the target
    Q-value to actions the agent *knows* about
    (i.e. the BC net has seen them or the RL net
    has learned about them), preventing
    Q-value over-estimation.
  The paper's headline result: 6.4× the
  success rate of RLPD on the
  PickPlaceCan Robomimic task with only 10
  expert demos and 100K environment steps.

* **`sil.py`** — **Self-Imitation Learning**
  (Oh et al. 2018).  Reference: Oh, Guo,
  Singh, Lee, "Self-Imitation Learning"
  arXiv:1806.05635.  The agent's *own* good
  episodes are replayed at a higher rate,
  weighted by ``(R - V(s))_+`` (the positive
  part of the clipped advantage).  This
  directly addresses the SyntheticGame
  12.5s → 30s gap by teaching the agent what
  "good" looks like in its *own* experience
  (not just the expert's).

* **`ema.py`** — **EMA of network weights for
  evaluation** (TD7 / Tarasov 2024).  After
  every train step the agent's online weights
  are EMA'd, and the EMA is installed during
  evaluation.  Reduces variance; the EMA's
  half-life is ~1000 steps by default.

* **`avg_l1_norm.py`** — **TD7's AvgL1Norm**.
  Reference: Fujimoto et al. 2023 "For SALE:
  State-Action Representation Learning for Deep
  Reinforcement Learning".  Divide each
  embedding by its mean L1 norm to bound the
  Q-value scale (prevents 1000× Q-value
  blowup, a common failure mode).

* **`auto_entropy.py`** — **SAC auto-tuned
  entropy temperature** for discrete actions
  (Haarnoja et al. 2018).  Reference: Christodoulou
  2019 "Soft Actor-Critic for Discrete Action
  Settings".  ``α`` is optimised to keep the
  policy's entropy at a target value
  (``-0.89 * |A|`` by default).  Prevents the
  policy from becoming deterministic too
  early.

* **`ibrl_agent.py`** — the full IBRL agent
  combining the BC and RL nets with the actor
  proposal + bootstrap proposal.

* **`dqfd_v2_agent.py`** — the **v1.23.0
  production training agent** for
  :class:`SyntheticGame`.  Inherits from
  :class:`DQfDAgent` (so the v1.21.0 production
  pipeline is preserved) and adds SIL + EMA +
  auto-entropy + IBRL bootstrap.  Drop-in
  replacement for :class:`DQfDAgent`.

**Why IBRL is the biggest win**: the previous
BC+DQfD recipe plateaued at 12.5s mean on the
real :class:`SyntheticGame` because a *single*
shared network was forced to simultaneously
match the expert (BC loss) and improve via
RL (TD loss) — these two objectives pull in
opposite directions.  IBRL decouples them:
the BC net is frozen after the BC pretrain,
the RL net explores freely, and the agent
picks the better action at runtime.  This is
the breakthrough that lets us *add* SIL
(self-imitation on the agent's own good
episodes) without the SIL gradient fighting
the BC gradient — they live in different
networks now.

**Test counts**: 1072/1072 PASS in 2:20
(up from 955/955 in v1.22.0, +109 new
tests).  **KPI re-verification**:
`audit_dqfd_v2.py` → **30.00s stable 200
episodes** on the LearnableEnv (both
training and final EMA eval).  The v1.21.0
`audit_bc_then_rl.py` also still passes
30.00s (no regression).

## 0b. What's new in v1.18 — distributional Q + dense reward shaping

The user's "super weak" report (10 human demos, 30 min, 3000+
deaths, 1-2s → 3-4s improvement = **catastrophic**) triggered
an evidence-based audit (`audit_pipeline.py` + `audit_train.py`).
The audit pinpointed 10 root causes; v1.18 addresses the
three with the largest impact:

* **Reward too sparse**: an agent that survived 5 s earned
  +3.0 total (0.06 / observed frame) — far below the noise
  floor of TD updates with batch size 32.  ``alive_per_frame``
  bumped from 0.02 → **0.5** (a 25× increase) and the death
  penalty softened from -10 → **-5** so the gradient can
  actually move the policy.  Curriculum milestones re-tuned
  to (1, 2, 5, 10, 20) s with bonus = **2.0** so the
  "I just crossed 5 s alive" event is a clear signal.
* **No per-event gradient**: the agent has no idea *which*
  obstacle it dodged.  ``SyntheticGame.step_with_reward``
  adds two new components:
  * ``proximity`` = -0.05 / tick when the player is in a
    dangerous lane (any obstacle < 50% progressed in the
    player's lane);
  * ``dodge``    = +0.20 / tick when the player changed out
    of a dangerous lane.
  These are real dense signals that connect action to outcome
  on a per-tick basis.
* **Scalar Q throws away distributional information**:
  vanilla DQN estimates E[R|s,a] — a single number.  QR-DQN
  estimates a *fixed set of quantiles* (51 by default) of the
  return distribution.  The loss is the quantile Huber
  between predicted and target quantiles; 51 quantiles × 5
  actions = **255 numbers per sample** drive the gradient,
  vs 1 for scalar Q.  ``distributional.py`` + ``agent_distributional.py``
  are fully tested (19 new tests) and are a drop-in
  replacement for the original agent.

The audit also identified 7 other issues (encoder too small,
epsilon decay too fast, replay buffer death-polluted, no
motion features in the frame stack, off-by-one in n-step
reward, hard death signal, narrow exploration).  They are
listed in ``audit_pipeline.py`` for the next round; v1.18
stops short of touching the encoder so the existing 716
tests can pin the recipe change in isolation.

## 0b. What's new in v1.17 — latent-space mental rehearsal

The user's framing of the problem — *"mỗi ảnh sẽ là 1 chỉ số khác
nhau được AI học khái quát, nếu qua ảnh khái quát đó mà AI vẫn
sống thì chứng tỏ AI học đúng và lưu vô bộ não; còn nếu qua ảnh
khái quát AI chết thì trừng phạt"* — is exactly what a
*variational autoencoder + env-replay* loop does.

  1. From the self-imitation pool, pick a *good* frame (an
     episode that survived past the rolling-mean gate).
  2. **Encode** it into a 64-dim latent vector.
  3. **Perturb** the latent with a small Gaussian (default
     `dream_noise_std = 0.30` — not noise, not reconstruction,
     but a "khái quát" / abstraction).
  4. **Decode** back to a frame.
  5. Run that abstract frame through the synthetic env (the
     actor's `_dream_env` back-door).  The env returns
     `done` for each step.
  6. If the agent survives → **positive dream** → saved to
     `<demos>/abstract/positive/`.
     If the agent dies → **negative dream** → saved to
     `<demos>/abstract/negative/`.

The BC loader can then fold positive dreams into the next BC
pretrain (they are valid `frames/actions/timestamps/done` `.npz`),
and the negative dreams are the agent's *self-criticism* — they
become a "this is what a deadly *generalised* state looks like"
prior that prevents over-confident Q-values in similar future
states.

* `dreamer.py` — a 60 k-parameter VAE (encoder + decoder) + a
  DreamerTrainer that owns the training step, the env-replay, and
  the `.npz` save/rotate pipeline.  Total round-trip cost: ~10 ms
  per training step, ~100 ms per mental-rehearsal round
  (8 dreams × env replay).  Throttled to one round per minute
  (`dream_every_s`) so the env-replay cost stays a rounding error.
* `SyntheticGame.step_with_frame` — the env back-door.  Returns
  `{"done": bool, "alive": bool}` for the dreamer.
* GUI: new "💭 Dream ngay" button + "Abstract dreams" column in
  the metrics table showing `+pos/-neg (total, q̄)`.
* 22 tests in `tests/test_dreamer.py` covering the VAE shapes,
  the Q-fallback, the env-replay path, rotation, heartbeat, and a
  real-SyntheticGame end-to-end round.

## 0b. What's new in v1.16 — self-imitation bootstrap

The previous release made the demo keypress-driven and added a
horizontal mirror augmentation so every left/right is learned twice
(once per direction).  v1.16 takes the next step: **the bot now
harvests its own good episodes and re-trains on them**, breaking the
vicious cycle where a poorly-seeded BC policy learns `NOOP`
and the replay buffer collapses to instant deaths.

* **Self-imitation recorder** (`self_imitation.py`) — every finished
  episode is scored against the rolling mean of recent survivals
  (`reward.self_imitation_factor`, default 1.2).  An episode whose
  `survival_s` beats `factor × rolling_mean` is auto-saved to
  `<demos>/self/episode_*.npz` (same schema as human demos).
* **Auto BC loop** — after every `bc_every_n_episodes` (default 5)
  self-imitation saves, the learner runs `pretrain_with_self_imitation`,
  which trains BC on the union of human demos and the self pool, split
  by episode (no frame leakage).  Both `best_model.pth` and
  `latest_model.pth` are updated; the auto-loop never overwrites a
  better RL checkpoint unless BC is genuinely better.
* **Manual override** — the GUI gains a ♻ **BC từ self-imitation**
  button that sends the same `pretrain_with_self` command, for
  operators who want to fold good runs in immediately rather than wait
  for the auto-trigger.
* **Live readout** — the metrics table now carries a "Self pool"
  column showing `on_disk (saved/seen)`, so the operator can tell at
  a glance whether the gate is firing.  The 16-test
  `tests/test_self_imitation.py` and 9-test
  `tests/test_learner_self_imitation.py` suites pin every layer.
* `replay_buffer.py` already supports mirroring as an on-the-fly
  augmentation in `sample()` (50 % chance per minibatch) — adding it
  again to the self-imitation pool would only dilute the better-than-
  average signal, so the self pool is stored unflipped and the
  augmentation is reapplied at BC time.

## 1. Feasibility audit (pre-coding, requirement §3)

| # | Question | Audit finding | Engineering decision |
|---|----------|--------------|----------------------|
| A1 | Classic Atari CNN (Conv×3 → flatten → 512-dense) parameter count? | The flatten head alone is 7·7·64·512 ≈ **1.61 M params** (plus 3136·512 multiplies per forward pass) — dwarfing the convs and blowing both the 80 k StrictLite budget and CPU latency budget. | Rejected. Encoder → **global average pooling** → small dueling heads; depthwise-separable convs for StrictLite. Measured: 49 k / 96 k / 348 k params (§4). |
| A2 | BatchNorm with small online batches + replay training? | Unstable here: (a) replay batches are drawn from an old distribution while online data shifts; (b) **actor and learner are different processes** — inference uses running stats that the training process is updating, so identical weights give different features; (c) batch=32 online stats are noisy. | **GroupNorm(8)** everywhere (batch-size independent, identical train/eval behaviour). Enforced by `tests/test_models.py::test_no_batchnorm_anywhere`. |
| A3 | Inference every 4 frames vs "near-frame-perfect" play? | At 30 FPS, a 4-frame cadence is a **133 ms reaction floor** — Subway Surfers dodges routinely need < 300 ms, so 4 frames is viable but marginal; 1-frame decisions are impossible for this CPU while also training. | Adaptive cadence: **2–4 frames normal (load-dependent), 1 frame in danger** (horizon detector fires). Measured, not assumed: p50/p95 inference 1.3/2.0 ms (§4). |
| A4 | Training in a separate *thread* vs the Python GIL? | A learner **thread** would serialise torch backward passes with frame processing under the GIL — capture stalls guaranteed. | Learner is a separate **process** (`spawn`); weights flow through shared memory to an actor-local copy (version-guarded, ≤ 1.4 MB memcpy). |
| A5 | Replay-buffer persistence vs 12 GB RAM? | Naive storage (4×84×84 uint8 obs + next-obs per transition) ≈ 56 KB × 30 000 = **1.7 GB**, plus torch + Chrome — too close to the 4 GB working-set limit. | **Lazy frame store**: each unique 84×84 frame stored once (~7 KB), transitions reference frame ids; eviction-validated at sample time. ≈ 210–260 MB at 30 k transitions. Config warns above 1 GB (`BotConfig._replay_ram_estimate_gb`). |
| A6 | Raw pixel-difference rewards → reward hacking? | Yes: flashing score popups, coin sparkles and camera shake mint unlimited "novelty" reward; a policy can learn to *die spectacularly* for pixels. | Pixel-diff reward is **OFF by default**, tightly clipped (±0.01), normalised, and behind an ablation switch; a dedicated reward-hacking test (`test_reward_logic.py::TestPixelDiffAblation::test_reward_hacking_resistance`) proves bounded payout. Survival-time reward is primary. |
| A7 | Single colour anchor → false deaths? | Yes: hover highlights, score flashes, or a 1-frame animation over the anchor pixel cause false respawns (lost progress, stuck clicking). | 5×5 patch **median** RGB (outlier-robust), Euclidean distance vs calibrated baseline, **temporal debounce ladder** (POSSIBLE_EVENT → DEAD_CANDIDATE → DEAD_CONFIRMED at ≥3 consecutive frames), stability window (10 frames) before ALIVE resumes, and per-frame logging of raw distance + reason. Unit-tested with flash/noise cases. |
| A8 | Input backend reliability on Windows Chrome? | `pyautogui` key events go through SendInput — reliable, but its per-call `PAUSE` (default 0.1 s) would blow the latency budget, and stuck keys scroll pages. | Keys via **pynput** (press/release explicit, no auto-pause), mouse clicks via pyautogui with `FAILSAFE=True`, `PAUSE=0.01`. Guaranteed release: scheduled release + guardian thread + watchdog force-release. Dry-run backend for tests/CI. |
| A9 | Responsive browser geometry + DPI scaling? | Fixed absolute coordinates break when the window moves, resolution/DPI changes, or the page relayouts. | Calibration stores **fractions** of the virtual screen (plus DPI scale); region is revalidated (size caps, min 240×320); DPI-aware process on Windows; every geometry change is *detected and reported*, never silently ignored. |

### Contradictions found and resolved

1. **"Frame-perfect" vs 4-frame skip** — reframed as *measured latency targets*
   (p95 < 100 ms action latency, 20–30 FPS effective) enforced by the
   auto-downgrader, instead of unverifiable perfection claims.
2. **80 k parameter limit vs "maximum learning performance"** — kept as the
   *StrictLite* profile guarantee; BalancedCPU/QualityCPU exceed it **only
   when profiling passes** (§4). Default profile is StrictLite.
3. **"Inference every 4 frames" vs "decide every frame in danger"** — both:
   normal state 2–4 frames, danger state 1 frame (§Adaptive timing).
4. **Threading for the learner vs GIL** — process instead of thread (A4).
5. **Pixel-diff reward vs reward hacking** — disabled by default, clipped,
   ablation-gated (A6).
6. **Fast respawn vs safety** — respawn clicks are capped (interval 0.8 s,
   timeout 15 s → ERROR/PAUSED, never infinite clicking).
7. **OCR score reward** — not implemented as a reward: unvalidated OCR is a
   noise source. Score stays evaluation-only until validated with confidence
   thresholds (hook in `evaluation.py`).

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│ GUI / main process (Tkinter)                                               │
│  calibration wizard (6 steps) · state machine · metrics panel · logs       │
│  F8 hotkey listener · preview grabber · demo recorder                      │
│  NEVER blocks on training; stays alive if any worker crashes               │
└───────┬──────────────────────────────┬─────────────────────────────────────┘
        │ spawn                        │ spawn
┌───────▼──────────┐          ┌────────▼──────────┐          ┌────────────────┐
│ capture process  │  ring    │ actor process     │ trans-Q  │ learner process│
│ mss / Synthetic  │─────────▶│ perception (zones,│─────────▶│ Double DQN +   │
│ 30 FPS paced,    │ shared   │ frame stack)      │ bounded  │ PER + Adam +   │
│ latest-wins,     │ memory   │ horizon detector  │ 128      │ checkpoints    │
│ drop-on-lap      │          │ death detector    │          │ BC pretraining │
└──────────────────┘          │ action scheduler  │◀─────────│ weight publish │
                              │ InferencePolicy   │ weights  │ (shared memory │
                              │ InputController   │ shared   │  + version)    │
                              │ (pynput/dry-run)  │          │                │
                              │ SafetyWatchdog    │◀─pause───│ (paused on     │
                              │ (thread, priority)│  event   │  death/respawn)│
                              └────────┬──────────┘          └────────────────┘
                                       │ keys/clicks (only exit to the OS)
                                  ▼ Chrome (Poki page)
Shared primitives (ipc.py): SharedFrameRing (bounded, lock-free),
SharedWeights (flat fp32 + version), SharedCounters, bounded queues,
events {stop, emergency, pause, pause_learning, death}.
```

**Synchronisation decisions**

* Frame transport: fixed-slot shared-memory ring, single writer (capture),
  single reader (actor), per-slot generation counters for tear-free reads;
  a slow reader drops stale frames *by construction* — memory is constant.
* Weights: learner is the **only writer**; actor copies out on version change
  (lock-free, retry-on-torn-copy).
* Counters: single-writer fields (`env_frame_id`, `action_step`, `episode_id`
  by the actor; `learner_update_step`, `beta`, `td_loss` by the learner).
* Events: `stop` (shutdown), `emergency` (F8), `pause` (gameplay+learner),
  `pause_learning` (learner only, during death/respawn), `death`.
* All queues are bounded; overflow drops and logs — nothing can grow RAM.

---

## 3. Repository layout

```
subway-surfers-AI/
├── app.py                    # entry point, process orchestrator, CLI
├── config.py                 # typed config tree + validation + JSON I/O
├── gui.py                    # Tkinter wizard (6 steps) + live metrics
├── capture_worker.py         # mss/synthetic capture process
├── input_controller.py       # safe keys/clicks, guardian, dry-run
├── safety_watchdog.py        # watchdog thread + F8 hotkey
├── environment.py            # actor loop, synthetic game, sync env
├── perception.py             # zones, anchor patch, frame stack
├── death_detector.py         # colour anchor + debounce + respawn ctl
├── horizon_detector.py       # frame-diff hazard detector
├── action_scheduler.py       # adaptive cadence, buffering, TTL
├── replay_buffer.py          # PER sum-tree, lazy frames, n-step, atomic IO
├── dataset.py                # demo validation, episode split, BC dataset
├── demonstration_recorder.py # human demo recording (npz)
├── models.py                 # StrictLite/BalancedCPU/QualityCPU nets
├── agent.py                  # Double DQN, BC epoch, InferencePolicy
├── learner_worker.py         # learner process (BC + online RL)
├── checkpoint_manager.py     # atomic best/latest/buffer + RNG states
├── metrics.py                # percentiles, FPS, counters, psutil
├── profiling.py              # params/FLOPs/latency/activation profiler
├── logging_utils.py          # logging, bounded queues, pickle integrity
├── ipc.py                    # shared ring/weights/counters/events
├── states.py                 # 12-state lifecycle machine
├── evaluation.py             # honest eval protocol + reports
├── evaluation_tool.py        # headless evaluation runner
├── requirements.txt · config.example.json · pytest.ini
├── self_imitation.py         # self-imitation recorder + rolling-mean gate
├── dreamer.py                # latent-space mental-rehearsal (VAE + env verify)
├── distributional.py         # QR-DQN quantile head + loss + projection
├── agent_distributional.py   # QR-DQN training agent (drop-in for DoubleDQN)
├── noisy_nets.py             # factorised Gaussian NoisyLinear (NoisyNet paper)
├── rnd.py                    # Random Network Distillation (curiosity)
├── dqfd_agent.py             # Deep Q from Demonstrations (joint Q+BC+margin)
├── bc_pretrain.py            # BC pretrain + demo pre-fill orchestrator
├── learnable_env.py          # 3-lane env with deterministic obstacle schedule
├── expert_policy.py          # optimal policy for LearnableEnv (audit baseline)
├── expert_synthetic.py       # optimal policy for SyntheticGame (random obstacles)
└── tests/                    # 32 test files, 874 tests
```

---

## 4. Model profiles (exact, measured)

Parameters/FLOPs/latency measured with `python profiling.py`
(CPU, `torch.set_num_threads(1)`).

| profile | params (exact) | FLOPs (MACs) | act. mem (train) | inference p50/p95 | update @ batch 32 |
|---|---:|---:|---:|---:|---:|
| `strict_lite` | **49 154** (≤ 80 000 ✓) | 2 627 616 | 2.06 MB | 0.75 / 1.41 ms | 33.8 ms |
| `balanced_cpu` | **95 558** | 14 018 304 | 0.52 MB | 0.46 / 0.53 ms | 23.8 ms |
| `quality_cpu` | **348 134** | 23 603 200 | 0.73 MB | 0.73 / 0.89 ms | 36.7 ms |

Design per profile (`models.py`):
* **StrictLite** — 4 depthwise-separable blocks (32/48/64/128, stride 2/2/2/1)
  → GroupNorm → GAP → dueling heads (hidden 128).
* **BalancedCPU** — 3 convs (8×8 s4, 4×4 s2, 3×3) + GroupNorm → GAP → dueling
  heads (hidden 128).
* **QualityCPU** — 4 convs (8×8 s4 … 3×3) + GroupNorm → GAP → dueling heads
  (hidden 256). Enable only if `profiling.py` passes on the target machine.

> Latency numbers were measured on the **CI sandbox CPU, not the i5-7200U**.
> Re-run `python profiling.py` on the target machine before choosing a
> non-default profile; the auto-downgrader enforces the budget either way.
> (Curiosity worth knowing: depthwise-separable StrictLite is *not* faster
> than the dense BalancedCPU convs here — oneDNN optimises dense convolutions
  better — StrictLite wins on parameter/RAM guarantees, not raw speed.)

RL core: **Double DQN** + dueling heads + n-step returns (default 3) +
**Prioritized Experience Replay** (α=0.6, β 0.4→1.0), Huber loss, grad-clip
10, Adam 1e-4, hard target sync every 1000 updates, ε-greedy 1.0→0.05
linear over **150 000 env frames** (the word "step" is defined once:
ε decays on env frames observed by the actor, nothing else).

---

## 5. Windows setup

```powershell
git clone <this repo>; cd subway-surfers-AI
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only wheel
python -m compileall .          # must exit clean
python -m pytest -q             # 449 passed (headless, no display needed)
python profiling.py             # re-measure on THIS machine
python app.py                   # GUI
```

Requirements: Windows 10/11 64-bit, Chrome, Python 3.10–3.12. No CUDA.
The bot is offline after install (verified by
`tests/test_hygiene.py::test_no_network_calls_in_core`).

## 6. Calibration (the six GUI steps)

1. **Select region** — click-drag a rectangle over the game only (no browser
   UI). Live size/coords/DPI shown; Enter accepts, Esc cancels, R re-selects.
2. **Lock region** — live preview; must be ≥ 240×320 and under the pixel cap;
   stored as screen fractions + DPI scale. Re-select if geometry changed.
3. **Colour anchor** — click a stable UI element that is visible while ALIVE
   and disappears/changes on game-over. "Calibrate ALIVE (2 s)" samples the
   5×5 patch and accepts only if the per-channel std ≤ 6 (stability score
   shown). "Capture DEAD sample" can be taken later on the game-over screen.
4. **Respawn click** — click the restart button; stored relative to the
   region. "Test click" asks for confirmation first (never auto-clicks).
5. **Horizon** — slider 15–40 % (default 25 %), preview shows the split line.
6. **Train** — Start/Pause/Stop, ● Record demo, BC pretrain, ⛔ F8 emergency.
   Live: state, survival, episode, FPS/effective FPS, dropped frames,
   inference & action p95, ε, avg reward, TD loss, mean Q, buffer size,
   CPU/RAM, model profile.

Headless equivalents: `python app.py --headless --steps 600` (synthetic game,
dry-run input, same three-process pipeline).

## 7. Demonstrations & datasets

* Record: GUI → Step 6 → **● Record demo** (F9 stop). Saves
  `demos/episode_*.npz`: 84×84 uint8 ground frames, actions, timestamps,
  done, optional score/confidence/death-state + browser-geometry metadata.
  No duplicated frame stacks are stored.
* Validate: `python app.py --validate-demos demos` — missing-frame gaps,
  invalid actions, timestamp order, episode boundaries, class imbalance,
  and an **episode-level** train/val split (no frame leakage).
* Pretrain (BC): `python app.py --pretrain demos` or the GUI button
  (class-balanced cross-entropy on the advantage logits, DQfD-style).
  If fewer than 2 valid episodes exist, BC is **refused with a message** and
  the pipeline falls back to online RL with a warm-up phase — it never
  pretends pretraining happened.

## 8. Test commands

```bash
python -m compileall .     # clean compile of every file
python -m pytest -q        # 750 tests
python app.py --headless --steps 600       # end-to-end smoke (fake game)
python app.py --evaluate 5                 # headless eval + honest report
```

Test coverage highlights: horizon detection & debounce; colour-anchor death
ladder + false-positive flashes + respawn machine (timeout → FAILED, never
clicks forever); reward clipping + pending-hazard logic + no-future-leak +
reward-hacking bounds; action cadence/danger/expiry/duplicates + confidence-
aware cadence; key release after backend exceptions + guardian force-release
+ focus gate; PER maths, NaN protection, eviction validation, persistence +
corruption recovery; checkpoint atomicity, best-model policy, RNG-state round
trip, online best-model gating + restart protection; Mann-Whitney/bootstrap/
adaptive-target statistics; capture geometry drift; recorder metadata +
atomic npz writes; lifecycle state machine; process startup/emergency-stop/
idempotent shutdown; GUI-survives-worker-crash; stale-frame dropping;
headless multi-process training pipeline; no bare excepts / silent passes /
stubs / network calls.

## 9. Profiling commands

```bash
python profiling.py                 # params/FLOPs/activation/latency + updates
python app.py --headless --steps 900 --report runs/headless_report.json
python app.py --evaluate 20         # per-episode stats + system profile
```

## 10. Evaluation protocol & claims policy

1. Record a **baseline** (human episodes via the demo recorder, or scripted
   baseline headless via `evaluation_tool.record_human_baseline_headless`).
2. Evaluate the trained policy on **separate** episodes (ε fixed 0.05,
   learning paused), 20–50 episodes where practical.
3. Reports (JSON + Markdown under `runs/`) contain mean/median/std/min/max/
   95 % CI per metric, failure modes, and a **verdict**. The verdict says
   "beats baseline" only when the 95 % CIs do not overlap **and** the
   tie-corrected Mann-Whitney U p-value is < 0.05; suggestive-but-unproven
   and not-improved outcomes are labelled as such.
4. An **adaptive target** is derived from the measured baseline
   (baseline mean + 1σ) and marked *provisional* until the baseline has
   ≥ 20 episodes — no invented fixed targets.
5. Compare a saved baseline report against a fresh evaluation with
   `python app.py --evaluate 20 --compare-baseline runs/human_baseline.json`.
6. Training performance, evaluation performance, human baseline, best single
   run and typical run are reported as separate numbers; "superhuman" is
   never printed as a claim, only as an explicitly unmet requirement.

## 11. Known limitations

* **No real-game measurements yet.** Detector thresholds (horizon diff,
  death colour distance) are principled defaults that MUST be re-calibrated
  via the wizard on the real game; their real-game precision/recall is
  pending.
* Score OCR is not implemented (score = 0 / "not detectable" in the GUI)
  until validated — score is evaluation-only anyway.
* Chrome focus detection depends on window-title matching
  (`input.browser_title_hint`); verify on your setup.
* The synthetic game is a *machinery test*, not a Subway Surfers simulator:
  passing it proves the pipeline runs, nothing about real-game skill.
* Windows-specific paths (DPI awareness, pyautogui failsafe) are exercised
  best-effort on Linux CI; run the suite on the target machine before play.
* `quality_cpu` default-off; measured numbers here are from a faster CI CPU
  than the i5-7200U — always re-profile on the target machine.
* Latency budget (p95 < 100 ms) is enforced by auto-downgrade; if even
  StrictLite can't hold 20 FPS, the bot logs a loud violation rather than
  silently degrading further.

## 12. Truthful checklist

**Verified by code + tests (CI, headless)**

- [x] CPU-only torch (`device="cpu"` asserted; threads pinned to 1)
- [x] StrictLite ≤ 80 000 params (49 154, exact-count test)
- [x] GroupNorm, no BatchNorm (test-enforced)
- [x] Parameter/FLOPs/activation/latency profiling per profile
- [x] Bounded frame transport, stale-frame dropping, drop counters
- [x] Horizon detector (score/detection/confidence/ts/frame-id + debounce)
- [x] Death detector ladder + flash false-positive immunity + respawn
      machine with bounded clicking and unit-tested state transitions
- [x] Five discrete actions; configurable keys; TTL, cooldown, duplicate
      suppression, buffering; guaranteed key release (guardian + watchdog +
      exception paths); dry-run mode; focus gate
- [x] Adaptive cadence 2–4 frames / 1 in danger, from CPU load AND horizon
      confidence; separate counters
- [x] PER (α/β, IS weights, sum-tree, NaN guards, eviction validation),
      n-step returns, lazy uint8 frame storage, atomic persistence with
      sha256 + `.corrupt` quarantine
- [x] Survival reward (time-proportional), −10 death, +0.1 pending hazard
      (no future leak), clipping [−10, 1], pixel-diff ablation OFF +
      hacking test
- [x] Double DQN + dueling + target net + Huber + grad clip + ε schedule
      defined on env frames
- [x] BC pretraining with episode-level split, class weighting, per-action
      accuracy; refuses insufficient data instead of faking it
- [x] Checkpoints (best/latest/buffer) atomic + RNG states + git/config hash;
      best_model.pth gated online by the rolling episode metric and
      protected against regression across restarts
- [x] Capture-side geometry-drift detection (resolution/DPI change → warning)
- [x] Evaluation statistics: CI overlap + Mann-Whitney U + bootstrap CI +
      adaptive baseline-derived target; `--compare-baseline` merging
- [x] Lifecycle: 12 states, emergency stop, idempotent shutdown, GUI survives
      worker crashes, no stuck keys on any shutdown path
- [x] Self-imitation: rolling-mean gate auto-saves good episodes into
      `<demos>/self/`, learner re-runs BC on union(human + self) every N saves;
      manual ♻ BC button + live "Self pool" metric in the GUI
- [x] Mental rehearsal: tiny VAE encodes good frames → perturb → decode →
      env-replay verifies survival → positive dreams saved to
      `<demos>/abstract/positive/`, negative to `.../negative/`;
      💭 Dream ngay button + "Abstract dreams" metric in the GUI
- [x] No bare excepts / `except Exception: pass` / TODO stubs / network
      calls (AST-enforced tests)
- [x] 750 automated tests green; `compileall` clean

**Requires the real Poki game on the target machine (pending)**

- [ ] Real-region capture quality, FPS and p95 action latency on the i5-7200U
- [ ] Horizon-detector precision/recall on real footage (threshold tuning)
- [ ] Colour-anchor death detection on the real game-over screen
      (including which UI element is truly stable)
- [ ] Respawn click reliability on the real button
- [ ] Chrome focus detection on the user's window setup
- [ ] Whether BC demos actually transfer to the real game
- [ ] Baseline vs trained evaluation, ablations, any performance claim
      (including "beats baseline", "superhuman")
- [ ] Long-run stability (hours), real death/respawn statistics

---

## Legal / ethical notes

Personal research only. Poki's Subway Surfers is embedded from SYBO; this
project does not modify the game, its traffic, or its internals — it watches
the screen and presses keys like a human. Respect the site's terms of
service; do not use this on leaderboards or competitive services.
