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
    #: Alive reward per frame at the nominal 30 FPS (~0.6/second).
    alive_per_frame: float = 0.02
    nominal_fps: int = 30
    death_penalty: float = -10.0
    hazard_bonus: float = 0.1
    #: Pending-hazard events resolve after this many valid frames.
    hazard_resolve_frames: int = 2
    #: Pending events older than this are expired without reward.
    hazard_expiry_s: float = 1.2
    reward_clip_min: float = -10.0
    reward_clip_max: float = 1.0
    #: Ablation switches (pixel-diff reward is OFF by default; see audit).
    use_pixel_diff_reward: bool = False
    pixel_diff_clip: float = 0.01


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
    #: Training cadence (learner updates) — independent of action cadence.
    max_updates_per_second: float = 20.0
    n_step: int = 3
    #: Epsilon decays LINEARLY over `epsilon_decay_frames` ENV frames
    #: (frames observed by the actor), from start to end.  Explicitly NOT
    #: learner updates and NOT action steps.
    #: DEEP-FIX: was 150_000 (~83 min), so a from-scratch run with no BC stayed
    #: at epsilon~0.99 for the whole session — 99% random, ignoring whatever the
    #: Q-net had learned, dying every ~1s and never bootstrapping.  50_000 lets
    #: the no-BC path start exploiting its improving policy within ~10 min so it
    #: can actually learn something from random play.  (With BC, exploration is
    #: capped at epsilon_after_bc regardless, so this only affects the no-BC path
    #: and the late decay below that cap.)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frames: int = 50_000
    #: After behaviour cloning produces a policy, cap exploration at this so
    #: the actor exploits the BC policy instead of playing randomly (the actor
    #: reads SharedCounters.bc_pretrained).  Only used once BC has run.
    epsilon_after_bc: float = 0.15
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
    per: PERConfig = field(default_factory=PERConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    bc: BCConfig = field(default_factory=BCConfig)
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
            "scheduler", "reward", "per", "rl", "bc", "perf", "paths",
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
