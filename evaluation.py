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
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

NOT_MEASURED = "not yet measured"

#: Minimum evaluation episodes for a comparative claim (requirement §15:
#: "At least 20-50 evaluation episodes where practical").
MIN_EVAL_EPISODES = 20


def mann_whitney_u(xs: list[float], ys: list[float]) -> dict[str, float]:
    """Mann-Whitney U rank-sum test (no SciPy; normal approximation).

    Returns ``{"u": U, "p": two_sided_p, "n": len(xs)+len(ys)}``.  Ties get
    mid-ranks; the variance carries the tie correction.  Valid for n>=8-ish;
    below that the p-value is still returned but flagged low-power via ``n``.
    """
    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return {"u": float("nan"), "p": 1.0, "n": n1 + n2}
    pooled = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    pooled.sort(key=lambda t: t[0])
    ranks = [0.0] * len(pooled)
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        mid = 0.5 * (i + j) + 1.0  # average rank, 1-based
        for k in range(i, j + 1):
            ranks[k] = mid
        i = j + 1
    r1 = sum(r for r, (_v, g) in zip(ranks, pooled) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    mu = n1 * n2 / 2.0
    # tie-corrected variance
    tie_terms = 0.0
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        t = j - i + 1
        tie_terms += t ** 3 - t
        i = j + 1
    n = n1 + n2
    var = (n1 * n2 / 12.0) * ((n + 1) - tie_terms / (n * (n - 1))) if n > 1 else 1.0
    var = max(var, 1e-12)
    z = (u1 - mu) / math.sqrt(var)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return {"u": float(min(u1, u2)), "p": float(p), "n": n}


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0,
                 statistic: str = "mean") -> dict[str, float]:
    """Percentile bootstrap CI for the mean (or median) of ``values``.

    Deterministic for a given seed; used alongside the analytic CI so small-n
    results can be judged honestly.
    """
    if not values:
        return {"lo": float("nan"), "hi": float("nan"), "stat": float("nan")}
    fn = (lambda v: sum(v) / len(v)) if statistic == "mean" \
        else (lambda v: sorted(v)[len(v) // 2])
    point = fn(values)
    if len(values) == 1:
        return {"lo": point, "hi": point, "stat": point}
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(max(100, n_boot)):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        stats.append(fn(sample))
    stats.sort()
    lo = stats[int(0.025 * (len(stats) - 1))]
    hi = stats[int(0.975 * (len(stats) - 1))]
    return {"lo": lo, "hi": hi, "stat": point}


def define_target(baseline: dict[str, Any], margin_frac: float = 0.0) -> dict[str, Any]:
    """Adaptive success target derived from a measured baseline (§15 step 3).

    Target = baseline mean + 1 std (statistically "noticeably better than the
    baseline's typical spread"), optionally inflated by ``margin_frac``.  With
    fewer than MIN_EVAL_EPISODES baseline episodes the target is still produced
    but flagged as provisional.
    """
    if baseline.get("status") != "ok":
        return {"status": NOT_MEASURED,
                "reason": "no baseline episodes recorded"}
    target = baseline["mean"] + baseline["std"] * (1.0 + margin_frac)
    return {
        "status": "ok",
        "metric": "survival_s",
        "target": target,
        "rule": "baseline_mean + 1*baseline_std",
        "provisional": baseline.get("n", 0) < MIN_EVAL_EPISODES,
        "baseline_n": baseline.get("n", 0),
    }


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
                       metric: str = "survival_s") -> bool | None:
        """True only if mean_a > mean_b with non-overlapping 95% CIs."""
        a = summarize(self._col(kind_a, metric))
        b = summarize(self._col(kind_b, metric))
        if a.get("status") != "ok" or b.get("status") != "ok":
            return None
        lower_a = a["mean"] - a["ci95"]
        upper_b = b["mean"] + b["ci95"]
        return bool(lower_a > upper_b)

    def compare(self, kind_a: str = "eval", kind_b: str = "human_baseline",
                metric: str = "survival_s") -> dict[str, Any]:
        """Full statistical comparison of two episode sets on one metric.

        Includes both significance views (CI overlap + Mann-Whitney U p) and
        the adaptive target derived from the baseline set.
        """
        a = summarize(self._col(kind_a, metric))
        b = summarize(self._col(kind_b, metric))
        out: dict[str, Any] = {"metric": metric, "kind_a": kind_a,
                               "kind_b": kind_b, "a": a, "b": b}
        if a.get("status") != "ok" or b.get("status") != "ok":
            out["verdict"] = NOT_MEASURED
            return out
        out["delta_mean"] = a["mean"] - b["mean"]
        lower_a = a["mean"] - a["ci95"]
        upper_b = b["mean"] + b["ci95"]
        out["ci_overlap_clear"] = bool(lower_a > upper_b)
        mwu = mann_whitney_u(self._col(kind_a, metric), self._col(kind_b, metric))
        out["mann_whitney_p"] = mwu["p"]
        out["mann_whitney_u"] = mwu["u"]
        out["bootstrap"] = {
            "a": bootstrap_ci(self._col(kind_a, metric)),
            "b": bootstrap_ci(self._col(kind_b, metric)),
        }
        tgt = define_target(b)
        out["target"] = tgt
        out["meets_target"] = bool(a.get("status") == "ok" and tgt.get("status") == "ok"
                                   and a["mean"] >= tgt["target"])
        if out["ci_overlap_clear"] and mwu["p"] < 0.05:
            out["verdict"] = (f"{kind_a} beats {kind_b} on {metric} "
                              f"(CI-separated AND Mann-Whitney p={mwu['p']:.4f})")
        elif out["ci_overlap_clear"]:
            out["verdict"] = (f"{kind_a} ahead with non-overlapping CIs, but "
                              f"rank-sum p={mwu['p']:.4f} >= 0.05 — treat as "
                              f"suggestive, not proven")
        else:
            out["verdict"] = (f"no statistically-supported improvement of "
                              f"{kind_a} over {kind_b} on {metric}")
        return out

    def verdict(self) -> str:
        if not self.of_kind("eval"):
            return "NOT YET MEASURED: no evaluation episodes recorded."
        if not self.of_kind("human_baseline"):
            return ("Evaluation recorded, but no human baseline exists — "
                    "no comparative claim is made.")
        n_eval = len(self.of_kind("eval"))
        cmp = self.compare()
        lines = [cmp["verdict"] + "."]
        if n_eval < MIN_EVAL_EPISODES:
            lines.append(f"(only {n_eval} evaluation episodes; target protocol "
                         f"asks for >= {MIN_EVAL_EPISODES} before strong claims)")
        lines.append("Even when statistically ahead of the recorded baseline, "
                     "this is NOT a 'superhuman' claim — the baseline is the "
                     "human who recorded it, under these window conditions only.")
        return "\n".join(lines)

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
        if self.of_kind("eval") and self.of_kind("human_baseline"):
            cmp = self.compare()
            lines.append("## Statistical comparison (eval vs human baseline, "
                         "survival_s)")
            lines.append(f"- delta mean: {cmp['delta_mean']:.2f}s")
            lines.append(f"- 95% CIs separated: {cmp['ci_overlap_clear']}")
            lines.append(f"- Mann-Whitney U p-value: {cmp['mann_whitney_p']:.4f} "
                         f"(n={len(self.of_kind('eval')) + len(self.of_kind('human_baseline'))})")
            tgt = cmp.get("target", {})
            if tgt.get("status") == "ok":
                lines.append(f"- adaptive target (baseline mean + 1 std): "
                             f"{tgt['target']:.2f}s "
                             f"({'provisional' if tgt.get('provisional') else 'final'}, "
                             f"baseline n={tgt.get('baseline_n')})")
                lines.append(f"- eval mean meets target: {cmp['meets_target']}")
            lines.append(f"- verdict: {cmp['verdict']}")
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
    def load(cls, path: str | Path) -> EvaluationReport:
        rep = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for r in data.get("records", []):
            rep.add(EpisodeRecord(**r))
        return rep

    def merge_baseline(self, path: str | Path,
                       kind: str = "human_baseline") -> int:
        """Import records from a saved report as baseline episodes.

        Used by ``app.py --evaluate N --compare-baseline runs/human.json``:
        the comparison set and the training set must stay separate, so
        imported records are re-labelled ``kind`` (default human_baseline)
        instead of being mixed into the eval pool.
        """
        other = EvaluationReport.load(path)
        count = 0
        for rec in other.records:
            self.add(EpisodeRecord(
                episode_id=rec.episode_id, survival_s=rec.survival_s,
                total_reward=rec.total_reward, steps=rec.steps,
                env_frames=rec.env_frames, fps=rec.fps,
                action_latency_p95_ms=rec.action_latency_p95_ms,
                inference_p95_ms=rec.inference_p95_ms, score=rec.score,
                kind=kind, ts=rec.ts,
            ))
            count += 1
        return count


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
