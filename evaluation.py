"""Honest evaluation protocol (requirement §15).

Rules enforced here:
* Nothing is labelled "superhuman" or "improved" without a recorded baseline
  and a statistical comparison on SEPARATE episodes.
* Training-time performance, evaluation performance, human baseline, best
  single run and typical run are reported as distinct numbers.
* Any metric never measured is reported as ``"not yet measured"`` — never
  fabricated.  (Real-game numbers require running the Poki game, which this
  repo cannot do on CI.)
* Confidence intervals use a normal-approximation 95% (reported with n so the
  reader can judge; for n<20 treat the interval as indicative).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

NOT_MEASURED = "not yet measured"


@dataclass
class EpisodeRecord:
    episode_id: int
    survival_s: float
    total_reward: float
    steps: int
    env_frames: int
    fps: float
    action_latency_p95_ms: float
    inference_p95_ms: float
    score: float = 0.0
    kind: str = "eval"  # "train" | "eval" | "human_baseline"
    ts: float = field(default_factory=time.time)


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"status": NOT_MEASURED}
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        sd = math.sqrt(var)
        sem = sd / math.sqrt(n)
        ci95 = 1.96 * sem
    else:
        sd = 0.0
        ci95 = 0.0
    xs = sorted(values)
    median = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    return {
        "status": "ok",
        "n": n,
        "mean": mean,
        "median": median,
        "std": sd,
        "min": xs[0],
        "max": xs[-1],
        "ci95": ci95,
        "best_single_run": xs[-1],
    }


class EvaluationReport:
    """Collects episode records and produces an honest comparison report."""

    def __init__(self) -> None:
        self.records: list[EpisodeRecord] = []

    def add(self, rec: EpisodeRecord) -> None:
        self.records.append(rec)

    def of_kind(self, kind: str) -> list[EpisodeRecord]:
        return [r for r in self.records if r.kind == kind]

    def _col(self, kind: str, key: str) -> list[float]:
        return [getattr(r, key) for r in self.of_kind(kind)]

    def summary(self) -> dict[str, Any]:
        return {
            "generated_at": time.time(),
            "train": {k: summarize(self._col("train", k))
                      for k in ("survival_s", "total_reward", "score", "fps",
                                "action_latency_p95_ms")},
            "eval": {k: summarize(self._col("eval", k))
                     for k in ("survival_s", "total_reward", "score", "fps",
                               "action_latency_p95_ms")},
            "human_baseline": {
                k: summarize(self._col("human_baseline", k))
                for k in ("survival_s", "total_reward", "score", "fps",
                          "action_latency_p95_ms")
            },
            "failure_modes": self.failure_modes(),
        }

    def failure_modes(self) -> dict[str, int]:
        modes: dict[str, int] = {}
        for r in self.records:
            if r.survival_s < 5.0:
                modes["early_death_lt_5s"] = modes.get("early_death_lt_5s", 0) + 1
            if r.action_latency_p95_ms > 100.0:
                modes["latency_budget_exceeded"] = \
                    modes.get("latency_budget_exceeded", 0) + 1
            if r.fps < 20.0 and r.fps > 0:
                modes["fps_below_target"] = modes.get("fps_below_target", 0) + 1
        return modes

    # ------------------------------------------------------------------ #
    def beats_baseline(self, kind_a: str = "eval", kind_b: str = "human_baseline",
                       metric: str = "survival_s") -> Optional[bool]:
        """True only if mean_a > mean_b with non-overlapping 95% CIs."""
        a = summarize(self._col(kind_a, metric))
        b = summarize(self._col(kind_b, metric))
        if a.get("status") != "ok" or b.get("status") != "ok":
            return None
        lower_a = a["mean"] - a["ci95"]
        upper_b = b["mean"] + b["ci95"]
        return bool(lower_a > upper_b)

    def verdict(self) -> str:
        if not self.of_kind("eval"):
            return "NOT YET MEASURED: no evaluation episodes recorded."
        if not self.of_kind("human_baseline"):
            return ("Evaluation recorded, but no human baseline exists — "
                    "no comparative claim is made.")
        beats = self.beats_baseline()
        if beats is None:
            return "Insufficient data for a statistical comparison."
        if beats:
            return ("Evaluation mean survival statistically exceeds the recorded "
                    "baseline (non-overlapping 95% CIs). Still not 'superhuman' — "
                    "see per-metric details before any strong claim.")
        return ("Evaluation has NOT exceeded the recorded baseline. No "
                "performance claim is made.")

    # ------------------------------------------------------------------ #
    def to_markdown(self) -> str:
        lines = ["# Evaluation report", ""]
        for kind, title in (("train", "Training episodes"),
                            ("eval", "Evaluation episodes"),
                            ("human_baseline", "Human baseline")):
            lines.append(f"## {title}")
            recs = self.of_kind(kind)
            if not recs:
                lines.append(f"- {NOT_MEASURED}")
                lines.append("")
                continue
            for metric, human in (("survival_s", "Survival (s)"),
                                  ("total_reward", "Total reward"),
                                  ("score", "Score (OCR not validated: 0)"),
                                  ("fps", "Effective FPS"),
                                  ("action_latency_p95_ms", "Action latency p95 (ms)")):
                s = summarize(self._col(kind, metric))
                lines.append(
                    f"- {human}: mean={s['mean']:.2f} median={s['median']:.2f} "
                    f"std={s['std']:.2f} min={s['min']:.2f} max={s['max']:.2f} "
                    f"95%CI=±{s['ci95']:.2f} (n={s['n']})"
                )
            lines.append("")
        lines.append("## Failure modes")
        fm = self.failure_modes()
        if fm:
            for k, v in fm.items():
                lines.append(f"- {k}: {v}")
        else:
            lines.append("- none recorded")
        lines.append("")
        lines.append(f"## Verdict\n\n{self.verdict()}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "verdict": self.verdict(),
            "records": [asdict(r) for r in self.records],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md = p.with_suffix(".md")
        md.write_text(self.to_markdown(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationReport":
        rep = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for r in data.get("records", []):
            rep.add(EpisodeRecord(**r))
        return rep


def ablation_matrix() -> list[dict[str, str]]:
    """Ablation runs required by §15; each returns its own eval report."""
    return [
        {"name": "horizon_detector_off", "flag": "horizon.diff_threshold=1e9"},
        {"name": "behavior_cloning_off", "flag": "skip pretrain"},
        {"name": "per_off", "flag": "per.alpha=0 (uniform replay)"},
        {"name": "adaptive_timing_off", "flag": "scheduler fixed 4-frame cadence"},
        {"name": "color_death_detector_off", "flag": "death.confirm_frames=1e9"},
        {"name": "survival_reward_only", "flag": "reward.hazard_bonus=0"},
        {"name": "pixel_diff_reward_on", "flag": "reward.use_pixel_diff_reward=true"},
        {"name": "larger_model_profile", "flag": "rl.profile=balanced_cpu/quality_cpu"},
    ]
