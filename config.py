"""Central configuration for the Subway Surfers research bot.

Everything the user may want to tune lives here as a typed dataclass tree,
serialisable to/from JSON (``config.example.json`` mirrors these defaults).

Design rules
------------
* No module reads its own ad-hoc config files; everything flows from
  :class:`BotConfig`.
* ``validate()`` raises :class:`ConfigError` for values that would crash or
  endanger the machine, and returns a list of human-readable warnings for
  values that are merely suspicious.
* No screen coordinates are hard-coded here.  Region/anchor/respawn geometry
  is *calibrated* at runtime and stored as fractions of the current screen
  geometry so DPI changes and window moves are detected rather than silently
  corrupting perception.
"""

from __future__ import annotations

import dataclasses
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

CONFIG_VERSION = "1.0"

#: Valid model profiles (architecture specs live in models.py).
PROFILE_NAMES = ("strict_lite", "balanced_cpu", "quality_cpu")

#: Ordered from lightest to heaviest; used by the automatic downgrader.
PROFILE_ORDER = ["strict_lite", "balanced_cpu", "quality_cpu"]

#: The five discrete gameplay actions (requirement: exactly five).
ACTIONS = ("NOOP", "LEFT", "RIGHT", "JUMP", "SLIDE")
NOOP, LEFT, RIGHT, JUMP, SLIDE = 0, 1, 2, 3, 4
N_ACTIONS = 5


class ConfigError(ValueError):
    """Raised when configuration values are internally inconsistent."""


@dataclass
class RegionConfig:
    """Calibrated capture geometry (filled in by the calibration wizard).

    ``left/top/width/height`` are absolute pixels for the *current* session;
    ``frac_*`` are fractions of the virtual screen so the config survives
    resolution/DPI changes (a mismatch is detected and reported, not ignored).
    """

    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0
    frac_left: float = 0.0
    frac_top: float = 0.0
    frac_width: float = 0.0
    frac_height: float = 0.0
    screen_width: int = 0
    screen_height: int = 0
    dpi_scale: float = 1.0

    def is_set(self) -> bool:
        return self.width > 0 and self.height > 0

    def to_monitor(self) -> dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass
class CaptureConfig:
    target_fps: int = 30
    #: Ring-buffer slots for the shared frame transport (bounded memory).
    ring_slots: int = 4
    #: Hard cap on region size; prevents pathological memory use.
    max_region_pixels: int = 1920 * 1080
    #: "auto" uses mss; "fake" drives the synthetic game used in tests.
    source: str = "auto"
    #: Max seconds a grab may take before the frame is considered dropped.
    grab_timeout_s: float = 0.25

    def frame_interval(self) -> float:
        return 1.0 / float(max(1, self.target_fps))


@dataclass
class PerceptionConfig:
    #: Fraction of region height treated as the horizon band (step 5 slider).
    horizon_frac: float = 0.25
    horizon_size: int = 40
    ground_size: int = 84
    frame_stack: int = 4
    #: Mean luminance below which a frame is treated as black/invalid.
    black_mean_threshold: float = 4.0

    def __post_init__(self) -> None:
        if not (0.15 <= self.horizon_frac <= 0.40):
            raise ConfigError(
                f"horizon_frac must be within [0.15, 0.40], got {self.horizon_frac}"
            )


@dataclass
class HorizonConfig:
    #: Mean abs diff (0-255) above which a horizon frame counts as "changing".
    diff_threshold: float = 8.0
    #: Exponential smoothing factor for the change score.
    ewma_alpha: float = 0.6
    #: Frames above threshold required within a window to fire detection.
    debounce_hits: int = 2
    debounce_window: int = 4
    #: Fraction of changed cells for "large object" classification.
    cell_threshold: float = 24.0
    min_changed_cell_ratio: float = 0.02
    #: Confidence mapping.
    confidence_scale: float = 8.0


@dataclass
class DeathConfig:
    #: Euclidean RGB distance above which the anchor differs from baseline.
    threshold: float = 25.0
    #: Consecutive off-baseline frames required for DEAD_CONFIRMED.
    confirm_frames: int = 3
    #: Frames below threshold required to call the anchor "stable" again.
    stable_frames: int = 10
    #: Distance under which a frame is counted as clearly ALIVE.
    alive_margin: float = 0.0
    #: Respawn click interval and overall timeout (seconds).
    respawn_interval_s: float = 0.8
    respawn_timeout_s: float = 15.0
    #: Optional secondary signals.
    stagnation_timeout_s: float = 25.0
    use_stagnation_fallback: bool = True
    #: Optional saved game-over/respawn template images (png, absolute path).
    gameover_template_path: str = ""
    respawn_template_path: str = ""
    template_match_threshold: float = 0.80
    #: Calibrated anchor position (fractions of the region) and baseline RGB.
    anchor_fx: float = -1.0
    anchor_fy: float = -1.0
    anchor_baseline_rgb: tuple[int, int, int] = (-1, -1, -1)
    anchor_baseline_std: float = -1.0
    #: Optional user-confirmed dead-state sample (for logging/diagnostics).
    dead_sample_rgb: tuple[int, int, int] = (-1, -1, -1)

    def anchor_set(self) -> bool:
        return self.anchor_fx >= 0.0 and self.anchor_baseline_rgb[0] >= 0

    def dead_sample_set(self) -> bool:
        return self.dead_sample_rgb[0] >= 0


@dataclass
class InputConfig:
    #: Action id -> key name understood by pynput ("left"/"up"/"a"/"f1"...).
    keymap: dict[int, str] = field(
        default_factory=lambda: {
            NOOP: "",
            LEFT: "left",
            RIGHT: "right",
            JUMP: "up",
            SLIDE: "down",
        }
    )
    hold_ms: int = 60
    #: Anti repeat: identical non-NOOP actions inside this window are dropped.
    cooldown_ms: int = 110
    #: Commands older than this are stale and never executed.
    action_ttl_ms: int = 140
    #: Hard safety: force-release any key held longer than this.
    max_hold_ms: int = 400
    #: Log but never press (tests, first runs, focus debugging).
    dry_run: bool = False
    #: Respawn point as fractions of the region (calibrated in step 4).
    respawn_fx: float = -1.0
    respawn_fy: float = -1.0
    #: Title substring used to verify Chrome focus before any key press.
    browser_title_hint: str = "Chrome"

    def respawn_set(self) -> bool:
        return 0.0 <= self.respawn_fx <= 1.0 and 0.0 <= self.respawn_fy <= 1.0


@dataclass
class SchedulerConfig:
    #: Normal-state decision cadence bounds (in captured frames).
    min_decision_frames: int = 2
    max_decision_frames: int = 4
    #: Danger-state cadence (next available frame).
    danger_decision_frames: int = 1
    #: At most one buffered action beyond the current one.
    buffer_size: int = 1
    #: Load (0..1) above which the cadence uses the slower bound.
    slow_load_threshold: float = 0.75
    #: Horizon-detector confidence (0..1) at/above which the cadence uses the
    #: FAST bound even under high load — a brewing hazard outranks CPU thrift
    #: (requirement §9: cadence depends on "CPU load AND confidence").
    fast_confidence: float = 0.75


@dataclass
class RewardConfig:
    #: Alive reward per frame at the nominal 30 FPS.  v1.18 bumped
    #: this from 0.02 to 0.5 (a 25× increase) because the audit
    #: showed the gradient signal was invisible: an agent that
    #: survived 5 s only earned +3.0 total, which is 0.06 per
    #: observed frame — far below the noise floor of TD updates
    #: with batch size 32.  0.5/step gives +15.0 / 5s, which is
    #: comparable to the magnitude of the death penalty so the
    #: survival signal can actually move the policy.
    alive_per_frame: float = 0.5
    nominal_fps: int = 30
    #: Death penalty softened from -10 to -5.  The original -10
    #: created a "death-polluted" replay buffer where 30 % of
    #: samples were terminal transitions, dragging the Q-value
    #: estimate toward "everything leads to death" and collapsing
    #: exploration.  -5 is still a strong deterrent (cancels
    #: about 10 s of alive reward) but lets the agent learn that
    #: "the long path of living is worth taking".
    death_penalty: float = -5.0
    #: Per-hazard-resolve bonus bumped from 0.1 to 1.0.  Without
    #: this the agent has no per-event gradient ("you survived
    #: THIS obstacle").  1.0 = ~0.5s of alive reward, large
    #: enough to be a clear "yes, that was the right move" signal
    #: in the TD update.
    hazard_bonus: float = 1.0
    #: Pending-hazard events resolve after this many valid frames.
    hazard_resolve_frames: int = 2
    #: Pending events older than this are expired without reward.
    hazard_expiry_s: float = 1.2
    reward_clip_min: float = -10.0
    reward_clip_max: float = 3.0
    #: Ablation switches (pixel-diff reward is OFF by default; see audit).
    use_pixel_diff_reward: bool = False
    pixel_diff_clip: float = 0.01
    #: Curriculum milestones: grant a one-shot bonus the FIRST time the
    #: agent survives past each threshold (in seconds).  The bonus is
    #: paid once per milestone per episode (re-arms at episode start),
    #: so it does NOT compound — it shapes the early reward landscape
    #: (small bonuses for reaching 2s/5s/10s/20s) and then goes quiet.
    #: Without this, the only signal the agent gets is "alive or dead"
    #: and there is no gradient telling it "you got 1% better".  The
    #: default schedule (0.0/0.0/0.0/0.0) keeps the original behaviour.
    curriculum_milestones: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0)
    curriculum_bonus: float = 2.0
    #: Per-transition bonus for hazard resolution.  When 0, the legacy
    #: ``hazard_bonus`` is used.  When positive, this is paid INSTEAD
    #: of the hazard_bonus (so the two knobs do not double-credit the
    #: same event).
    hazard_resolve_bonus: float = 0.0
    #: Self-imitation: every time the actor's survival_s for the
    #: finished episode is at least this multiple of the rolling mean,
    #: the episode is "good" enough to save as a self-imitation demo.
    #: The self-imitation dataset is the union of the human demos
    #: (from the demo recorder) and the good agent episodes (added by
    #: the learner).  Set 0.0 to disable, 1.0 means "at or above the
    #: average", 1.2 means "20% better than average", etc.
    self_imitation_factor: float = 1.2
    #: Maximum number of self-imitation episodes kept on disk (older
    #: ones are rotated out).  Sized to keep the BC dataset
    #: self-imitation-heavy without unbounded disk growth.
    self_imitation_max: int = 50


@dataclass
class DreamerConfig:
    """Mental-rehearsal (latent-space dreaming) configuration.

    The dreamer is the third leg of the "AI tự học full aggressive"
    stack:

    * curriculum reward  — teaches *what* a good state is,
    * self-imitation     — copies *when* a full episode is good,
    * dreamer            — extracts *what generalises* across good
      frames and verifies it by replaying through the env.

    The user-facing description (from the field-tested phrasing in
    the GUI):

      "Mỗi ảnh sẽ là 1 chỉ số khác nhau được AI học khái quát.
       Nếu qua ảnh khái quát đó mà AI vẫn sống thì chứng tỏ AI
       học đúng và lưu vô bộ não (Q-network).  Còn nếu qua ảnh
       khái quát AI chết thì trừng phạt (Q-value bị đẩy xuống)."

    All fields are pinned in
    :mod:`tests.test_dreamer` so the operator can audit them.
    """

    #: Master switch.  When ``False`` the dreamer is not constructed
    #: at all, so the round's RAM cost stays zero.
    enabled: bool = True
    #: Latent dimension.  64 is a sweet spot on CPU.
    latent_dim: int = 64
    #: Gaussian stddev added to the latent vector when dreaming.
    #: 0.0 = "photographic memory", 1.0 = "wild hallucination".
    dream_noise_std: float = 0.30
    #: Maximum abstract episodes kept on disk per kind.
    max_episodes: int = 50
    #: β weight for the KL term in the VAE loss.
    beta_kl: float = 0.01
    #: Number of frames per abstract episode.
    frames_per_dream: int = 32
    #: Train the VAE every N learner updates.
    train_every_n_updates: int = 100
    #: Throttle between mental-rehearsal rounds (seconds).
    dream_every_s: float = 60.0
    #: Number of dreams generated per round.
    dreams_per_round: int = 8
    #: Q-value threshold for "positive" dreams (the agent thought
    #: the abstract frame was good).
    positive_q_threshold: float = 0.5
    #: Q-value threshold for "negative" dreams (the agent thought
    #: the abstract frame was deadly).
    negative_q_threshold: float = -0.5
    #: Survival-rate threshold (env-replay) above which a dream is
    #: "positive".  Independent from the Q-score because the env
    #: can disagree with the Q-network.
    positive_survival: float = 0.6
    #: Survival-rate threshold (env-replay) below which a dream is
    #: "negative".
    negative_survival: float = 0.2


@dataclass
class RNDConfig:
    """Random Network Distillation (curiosity) configuration.

    The deep-research the user asked for highlighted RND as
    one of the most impactful additions for sparse-reward
    game RL (Burda et al. 2018 — the paper that made
    Montezuma's Revenge tractable).  The default
    configuration is "always on with a moderate beta" so
    the agent gets a *novelty* bonus on top of the
    extrinsic alive/dead signal.

    All fields are pinned by :mod:`tests.test_rnd`.
    """

    #: Master switch.  When ``False`` no RND network is
    #: constructed and the intrinsic reward is always 0.
    enabled: bool = True
    #: Output dim of the encoder.  128 is the paper's
    #: default; we keep it.
    feature_dim: int = 128
    #: Coefficient on the intrinsic reward.  0.5 is large
    #: enough to matter in the first 100 episodes (when the
    #: agent has barely seen any state) and small enough
    #: that the agent still chases the extrinsic reward
    #: once the predictor catches up.
    beta: float = 0.5
    #: EMA decay for the intrinsic-reward normaliser.
    normalizer_alpha: float = 0.99
    #: Train the predictor every N learner updates.
    train_every_n_updates: int = 1
    #: Random seed for the *frozen* target.  Different from
    #: the agent's overall seed so the predictor has
    #: something to fit.
    target_seed: int = 12345


@dataclass
class PERConfig:
    capacity: int = 30_000
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 200_000
    priority_eps: float = 1e-4
    #: Save the buffer every N learner updates (and on clean shutdown).
    save_every_updates: int = 1_000


@dataclass
class RLConfig:
    profile: str = "strict_lite"
    gamma: float = 0.99
    batch_size: int = 32
    learning_rate: float = 1e-4
    lr_schedule: str = "constant"  # "constant" | "cosine"
    lr_min: float = 1e-5
    cosine_period_updates: int = 200_000
    target_update_every: int = 1_000
    grad_clip_norm: float = 10.0
    warmup_transitions: int = 500
    #: Use the distributional QR-DQN agent (v1.18) instead of
    #: the original scalar DoubleDQN.  Default ON: the deep
    #: research in v1.19 found distributional Q to be the
    #: third-most-important Rainbow ingredient (after multi-step
    #: learning and PER).  The QR-DQN head produces 51 quantiles
    #: per action — 51× denser gradient than scalar Q.
    distributional: bool = True
    #: Number of quantiles for the QR-DQN head.  51 is the
    #: CPU-friendly default; raise to 200 for a richer
    #: distribution at the cost of 4× head memory.
    num_quantiles: int = 51
    #: Use NoisyNets for exploration.  When ``True`` the
    #: dueling head's value/advantage streams are :class:`NoisyLinear`
    #: layers whose weights are perturbed by factorised
    #: Gaussian noise.  Replaces ε-greedy with parameter-space
    #: noise.  Default ON per the Rainbow ablation.
    noisy_nets: bool = True
    #: DEEP-FIX (v1.22.0): soft (Polyak) target update.
    #: When True the target net is updated as
    #: ``target = (1-tau) * target + tau * online`` after every
    #: train step (the standard DDPG/TD3 recipe).  When False
    #: (legacy), the target is hard-copied from the online net
    #: every ``target_update_every`` steps.  Polyak averaging
    #: tends to give a 5-10% boost on hard-exploration tasks
    #: because the target moves smoothly, reducing the
    #: over-estimation bias that comes from a stale target.
    polyak_target: bool = True
    #: The Polyak averaging coefficient.  Only used when
    #: ``polyak_target=True``.  ``tau=0.005`` is the TD3 default.
    polyak_tau: float = 0.005
    #: DEEP-FIX (v1.22.0): visual data augmentation for
    #: the train-time observation.  Laskin et al. 2020
    #: (RAD) showed that even *random* shifts + intensity
    #: jitter improve sample efficiency by 2-3x on Atari.
    #: The augmentations are applied to the *current*
    #: observation only (next_obs is left alone, so the
    #: TD target is not corrupted by augmentation noise).
    augment_obs: bool = True
    #: Max random translate, in pixels.  Per-frame
    #: intensity jitter amplitude (additive).
    augment_translate_px: int = 3
    augment_intensity: float = 0.10
    #: Add the temporal-difference channel to the obs.
    #: Increases the input from ``F`` to ``F+1`` channels,
    #: which requires an encoder that supports the new
    #: channel count.  When True, the learner will rebuild
    #: the encoder with ``in_frames+1`` channels.  Default
    #: ``False`` for compatibility with the standard
    #: encoder.
    augment_frame_diff: bool = False
    #: Use RND (Random Network Distillation) for intrinsic
    #: motivation.  When ``True`` the agent adds a *novelty
    #: bonus* to the extrinsic reward so it explores the state
    #: space even when alive/dead is the only extrinsic signal.
    #: See :class:`rnd.RNDModule` for the math.
    use_rnd: bool = True
    #: Training cadence (learner updates) — independent of action cadence.
    max_updates_per_second: float = 20.0
    n_step: int = 5
    #: Epsilon decays LINEARLY over `epsilon_decay_frames` DECISION steps
    #: (actions actually chosen by the actor — one per ActionScheduler decision),
    #: from start to end.  Explicitly NOT learner updates and NOT raw captured
    #: frames.
    #: MDP-FIX (v1.24.0): the field name is kept for compatibility but the unit
    #: is the agent's real time step = a *decision*.  The scheduler holds an
    #: action for 2-4 frames (and re-decides on danger), so a captured frame is
    #: not an MDP step; indexing epsilon by raw frames decayed exploration 2-4x
    #: faster than the decisions the policy experienced.  50_000 decisions
    #: (~2-4x that many frames) matches the original "start exploiting within
    #: ~10-40 min" tuning while sharing a clock with the replay stream.
    #: DEEP-FIX (v1.13): was 150_000, so a from-scratch run with no BC stayed at
    #: epsilon~0.99 for the whole session — 99% random, ignoring whatever the
    #: Q-net had learned, dying every ~1s and never bootstrapping.  (With BC,
    #: exploration is capped at epsilon_after_bc / disabled regardless, so this
    #: only affects the no-BC path and the late decay below that cap.)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frames: int = 50_000
    #: After behaviour cloning produces a policy, cap exploration at this so
    #: the actor exploits the BC policy instead of playing randomly (the actor
    #: reads SharedCounters.bc_pretrained).  Only used once BC has run.
    epsilon_after_bc: float = 0.15
    #: DEEP-FIX (v1.21.0): audit_bc_then_rl.py proved that even 15%
    #: ε-greedy *destroys* a BC-pretrained policy — the agent picked
    #: the wrong lane in 14% of frames and the survival fell from 30s
    #: to 14.6s within 200 episodes.  When this flag is ``True`` the
    #: actor uses ε=0 after BC (pure exploitation).  Set to ``False``
    #: to fall back to the legacy ``epsilon_after_bc`` cap.
    disable_exploration_after_bc: bool = True
    checkpoint_every_updates: int = 500
    torch_threads: int = 1
    #: Which actor-reported episode metric gates best_model.pth during ONLINE
    #: training (requirement §12: "only when evaluation performance improves
    #: according to the configured metric").  "survival_s" | "total_reward".
    best_metric: str = "survival_s"
    #: Best-model decisions use the rolling mean over this many finished
    #: episodes (single episodes are too noisy to gate checkpoints on).
    best_metric_window: int = 3


@dataclass
class BCConfig:
    #: DEEP-FIX (v1.21.0): when ``True`` the learner
    #: builds a :class:`DQfDAgent` (instead of the
    #: standard QR-DQN) and the BC pretrain hands the
    #: expert demos to its joint-loss path.  Without
    #: this the BC pretrain is *just* a standard
    #: cross-entropy warm-up; the policy drifts back to
    #: random within 200 episodes of online RL because
    #: the TD loss washes out the BC anchor.
    bc_pretrain: bool = True
    #: DQfD joint-loss hyper-parameters.  Defaults are
    #: the Hester 2018 paper's values.  See
    #: :class:`dqfd_agent.DQfDConfig` for details.
    dqfd_lambda_bc: float = 0.5
    dqfd_lambda_margin: float = 0.1
    dqfd_margin: float = 0.8
    dqfd_decay_episodes: int = 1000
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.15
    #: Minimum usable episodes; below this BC is refused with a GUI message.
    min_episodes: int = 2
    class_balance: str = "inverse_sqrt"
    #: DEEP-FIX ("why did the human press?"): dodge actions are rare but decide
    #: life-or-death, so plain BC drowns them in NOOP and never learns to dodge.
    #: Each dodge frame is repeated this many times in BC training so the rare,
    #: critical presses actually drive the gradient.  1 = off (legacy behaviour).
    dodge_oversample: int = 4


@dataclass
class DemoAugmentConfig:
    """Augmentation rulebook applied at BC training time (and reported live).

    Every knob is independent so a non-symmetric game (or a user who does
    not want the keypress-window filter) can switch any of them off.

    * ``keypress_window``: instead of keeping every captured frame, keep a
      ±N-frame window around each human key press.  Stretches where the
      human is just running untouched (NOOP) are dropped — they are the
      majority of a continuous recording and drown the rare dodges in
      useless NOOP samples.  This is what makes "the bot learns the
      frame where I pressed" actually trainable.

    * ``keypress_pre`` / ``keypress_post``: how many frames to keep
      before and after each press.  At 30 FPS the defaults (5 each) keep
      a 333 ms window — the obstacle that triggered the press is usually
      2-4 frames in front of the player, and the recovery / next dodge
      is 3-5 frames past the press.  Bigger windows = more context but
      more NOOP samples; smaller = the network may miss the trigger.

    * ``mirror_horizontal``: append a horizontally flipped copy of every
      kept frame with the mirrored action (LEFT <-> RIGHT; JUMP/SLIDE/
      NOOP stay the same).  Subway Surfers' lanes are mirror-symmetric,
      so this is a free x2 augmentation that teaches "an obstacle on the
      left is the same problem as one on the right".  Disable for any
      non-symmetric game.

    * ``stack_mirror``: when True, the whole frame stack is flipped (the
      4 newest frames in the original become the 4 flipped frames in
      the same order).  When False, only the newest frame flips — the
      policy can confuse the two, so the default is the safer choice.
    """
    keypress_window: bool = True
    keypress_pre: int = 5
    keypress_post: int = 5
    mirror_horizontal: bool = True
    stack_mirror: bool = True


@dataclass
class PerfConfig:
    #: p95 end-to-end action latency budget (ms).
    p95_action_latency_ms: float = 100.0
    min_effective_fps: float = 20.0
    max_working_set_gb: float = 4.0
    #: Auto-downgrade after this many seconds of sustained violation.
    downgrade_window_s: float = 8.0
    #: Metrics/heartbeat cadence.
    report_interval_s: float = 1.0
    watchdog_interval_s: float = 0.25


@dataclass
class PathsConfig:
    root: str = "."
    checkpoints_dir: str = "checkpoints"
    demos_dir: str = "demos"
    logs_dir: str = "logs"
    runs_dir: str = "runs"
    config_path: str = "config.json"

    def ensure_dirs(self) -> None:
        for d in (self.checkpoints_dir, self.demos_dir, self.logs_dir, self.runs_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


@dataclass
class BotConfig:
    version: str = CONFIG_VERSION
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    horizon: HorizonConfig = field(default_factory=HorizonConfig)
    death: DeathConfig = field(default_factory=DeathConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    input: InputConfig = field(default_factory=InputConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    dreamer: DreamerConfig = field(default_factory=DreamerConfig)
    rnd: RNDConfig = field(default_factory=RNDConfig)
    per: PERConfig = field(default_factory=PERConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    demo_augment: DemoAugmentConfig = field(default_factory=DemoAugmentConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    seed: int = 7
    emergency_hotkey: str = "f8"
    demo_hotkey: str = "f9"
    #: How long the alive-anchor calibration window lasts (step 3).
    anchor_calibration_s: float = 2.0

    # ------------------------------------------------------------------ #
    def validate(self) -> list[str]:
        """Raise ``ConfigError`` on fatal problems; return warnings."""
        warnings: list[str] = []
        if self.rl.profile not in PROFILE_NAMES:
            raise ConfigError(
                f"rl.profile must be one of {PROFILE_NAMES}, got {self.rl.profile!r}"
            )
        if not (1 <= self.capture.target_fps <= 60):
            raise ConfigError("capture.target_fps must be in [1, 60]")
        if not (0.15 <= self.perception.horizon_frac <= 0.40):
            raise ConfigError("perception.horizon_frac must be in [0.15, 0.40]")
        if self.perception.frame_stack < 1 or self.perception.frame_stack > 8:
            raise ConfigError("perception.frame_stack must be in [1, 8]")
        if not (1 <= self.scheduler.min_decision_frames
                <= self.scheduler.max_decision_frames <= 8):
            raise ConfigError(
                "scheduler decision frames must satisfy 1 <= min <= max <= 8"
            )
        if self.death.confirm_frames < 1:
            raise ConfigError("death.confirm_frames must be >= 1")
        if self.death.threshold <= 0:
            raise ConfigError("death.threshold must be > 0")
        acts = set(self.input.keymap.keys())
        if acts != set(range(N_ACTIONS)):
            raise ConfigError(
                f"input.keymap must define exactly actions 0..{N_ACTIONS - 1}"
            )
        for a, key in self.input.keymap.items():
            if a == NOOP:
                continue
            if not isinstance(key, str) or not key:
                raise ConfigError(f"input.keymap[{a}] must be a non-empty key name")
        if self.per.capacity < 100:
            raise ConfigError("per.capacity must be >= 100")
        if not (0.0 < self.rl.gamma < 1.0):
            raise ConfigError("rl.gamma must be in (0, 1)")
        if self.rl.n_step < 1 or self.rl.n_step > 10:
            raise ConfigError("rl.n_step must be in [1, 10]")
        if self.rl.best_metric not in ("survival_s", "total_reward"):
            raise ConfigError(
                "rl.best_metric must be 'survival_s' or 'total_reward'"
            )
        if not (1 <= self.rl.best_metric_window <= 50):
            raise ConfigError("rl.best_metric_window must be in [1, 50]")
        if self.reward.reward_clip_min > 0 or self.reward.reward_clip_max <= 0:
            raise ConfigError("reward clip range must include 0 and positive max")
        if self.reward.hazard_bonus > self.reward.reward_clip_max:
            warnings.append("hazard_bonus exceeds reward_clip_max; it will be clipped")
        mem = self._replay_ram_estimate_gb()
        if mem > 1.0:
            warnings.append(
                f"replay buffer estimated at {mem:.2f} GB RAM "
                f"(capacity={self.per.capacity}); keep total usage < 4 GB"
            )
        return warnings

    def _replay_ram_estimate_gb(self) -> float:
        frame_bytes = self.perception.ground_size ** 2
        # Frame store keeps unique frames once (lazy stacking); priorities,
        # actions and bookkeeping add a small constant per transition.
        per_transition = frame_bytes + 64
        return self.per.capacity * per_transition / (1024 ** 3)

    # -------------------------------------------------------------- #
    # Serialisation
    # -------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def save(self, path: str | Path) -> None:
        # DEEP-FIX: this used to be a plain write_text.  A crash, power loss or
        # full disk mid-write left a truncated config.json that BotConfig.load
        # could not parse -- bricking the next start and silently discarding
        # the user's calibration.  Write to a temp file then os.replace, which
        # is atomic on NTFS/ext4, so readers always see a complete file.
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        os.replace(str(tmp), str(p))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BotConfig":
        cfg = cls()
        for section in (
            "capture", "perception", "horizon", "death", "region", "input",
            "scheduler", "reward", "per", "rl", "bc", "demo_augment", "perf",
            "paths",
        ):
            if section in data and isinstance(data[section], dict):
                current = getattr(cfg, section)
                setattr(cfg, section, _merge_dataclass(current, data[section]))
        for k in ("version", "seed", "emergency_hotkey", "demo_hotkey",
                  "anchor_calibration_s"):
            if k in data:
                setattr(cfg, k, data[k])
        # JSON turns tuples into lists — restore the tuple types so equality
        # and downstream typing behave identically after a round trip.
        cfg.death.anchor_baseline_rgb = tuple(cfg.death.anchor_baseline_rgb)
        cfg.death.dead_sample_rgb = tuple(cfg.death.dead_sample_rgb)
        cfg.input.keymap = {int(k): str(v) for k, v in cfg.input.keymap.items()}
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "BotConfig":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config file {p} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def profile_downgrade(self) -> Optional[str]:
        """Return the next lighter profile, or None if already lightest."""
        try:
            idx = PROFILE_ORDER.index(self.rl.profile)
        except ValueError:  # pragma: no cover - validate() guards this
            return None
        return PROFILE_ORDER[idx - 1] if idx > 0 else None


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> Any:
    """Return a copy of ``instance`` with fields overlaid from ``data``."""
    if not dataclasses.is_dataclass(instance):
        return instance
    kwargs = {f.name: getattr(instance, f.name) for f in dataclasses.fields(instance)}
    for key, value in data.items():
        if key in kwargs:
            kwargs[key] = value
    return type(instance)(**kwargs)


DEFAULT_CONFIG = BotConfig
