"""Statistical evaluation upgrades: Mann-Whitney U, bootstrap CI, target rule.

Requirement §15: comparisons against a baseline must be statistical, and any
target must be DERIVED from a measured baseline instead of being invented.
"""

from __future__ import annotations

import math

import pytest

from evaluation import (
    MIN_EVAL_EPISODES,
    EpisodeRecord,
    EvaluationReport,
    bootstrap_ci,
    define_target,
    mann_whitney_u,
    summarize,
)


class TestMannWhitneyU:
    def test_separated_distributions_small_p(self) -> None:
        xs = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        ys = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        res = mann_whitney_u(xs, ys)
        assert res["p"] < 0.01
        assert res["u"] >= 0.0

    def test_identical_distributions_large_p(self) -> None:
        xs = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ys = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        res = mann_whitney_u(xs, ys)
        assert res["p"] > 0.5

    def test_all_ties_give_p_one(self) -> None:
        xs = [3.0] * 10
        ys = [3.0] * 10
        res = mann_whitney_u(xs, ys)
        assert res["p"] == pytest.approx(1.0)

    def test_empty_input_safe(self) -> None:
        res = mann_whitney_u([], [1.0])
        assert res["p"] >= 0.0 and math.isnan(res["u"])

    def test_asymmetric_sample_sizes(self) -> None:
        xs = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
        ys = [1.0, 2.0, 3.0]
        res = mann_whitney_u(xs, ys)
        assert res["p"] < 0.05
        assert res["n"] == 13


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self) -> None:
        values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ci = bootstrap_ci(values, n_boot=500, seed=1)
        assert ci["lo"] <= ci["stat"] <= ci["hi"]

    def test_deterministic_for_seed(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = bootstrap_ci(values, n_boot=300, seed=7)
        b = bootstrap_ci(values, n_boot=300, seed=7)
        assert a == b

    def test_single_value_degenerates(self) -> None:
        ci = bootstrap_ci([42.0])
        assert ci["lo"] == ci["hi"] == 42.0

    def test_empty_values_nan(self) -> None:
        ci = bootstrap_ci([])
        assert math.isnan(ci["lo"])


class TestDefineTarget:
    def test_target_is_baseline_mean_plus_std(self) -> None:
        base = {"status": "ok", "mean": 10.0, "std": 2.0, "n": 30}
        tgt = define_target(base)
        assert tgt["status"] == "ok"
        assert tgt["target"] == pytest.approx(12.0)
        assert tgt["provisional"] is False

    def test_provisional_with_few_episodes(self) -> None:
        base = {"status": "ok", "mean": 10.0, "std": 2.0, "n": 5}
        tgt = define_target(base)
        assert tgt["provisional"] is True
        assert MIN_EVAL_EPISODES > 5

    def test_no_baseline_gives_not_measured(self) -> None:
        tgt = define_target({"status": "not yet measured"})
        assert tgt["status"] == "not yet measured"

    def test_margin_inflates(self) -> None:
        base = {"status": "ok", "mean": 10.0, "std": 2.0, "n": 30}
        assert define_target(base, margin_frac=0.5)["target"] == pytest.approx(13.0)


def _rec(ep: int, survival: float, kind: str = "eval") -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=ep, survival_s=survival, total_reward=survival * 0.6,
        steps=int(survival * 30), env_frames=int(survival * 30), fps=30.0,
        action_latency_p95_ms=12.0, inference_p95_ms=2.0, kind=kind,
    )


class TestEvaluationReportCompare:
    def test_compare_full_path(self) -> None:
        rep = EvaluationReport()
        for i in range(25):
            rep.add(_rec(i, 30.0 + float(i % 5), kind="eval"))
            rep.add(_rec(1000 + i, 5.0 + float(i % 3), kind="human_baseline"))
        cmp = rep.compare()
        assert cmp["a"]["n"] == 25 and cmp["b"]["n"] == 25
        assert cmp["delta_mean"] > 0
        assert cmp["ci_overlap_clear"] is True
        assert cmp["mann_whitney_p"] < 0.001
        assert cmp["meets_target"] is True
        assert "beats" in cmp["verdict"]

    def test_compare_missing_baseline_not_measured(self) -> None:
        rep = EvaluationReport()
        rep.add(_rec(1, 10.0))
        cmp = rep.compare()
        assert cmp["verdict"] == "not yet measured"

    def test_compare_close_distributions_not_proven(self) -> None:
        rep = EvaluationReport()
        for i in range(25):
            rep.add(_rec(i, 10.0 + float(i % 4), kind="eval"))
            rep.add(_rec(1000 + i, 10.0 + float((i + 2) % 4),
                         kind="human_baseline"))
        cmp = rep.compare()
        assert cmp["ci_overlap_clear"] is False
        assert "no statistically-supported" in cmp["verdict"]

    def test_verdict_mentions_insufficient_n(self) -> None:
        rep = EvaluationReport()
        for i in range(5):
            rep.add(_rec(i, 20.0, kind="eval"))
            rep.add(_rec(1000 + i, 5.0, kind="human_baseline"))
        v = rep.verdict()
        assert "only 5 evaluation episodes" in v
        assert f">= {MIN_EVAL_EPISODES}" in v

    def test_merge_baseline_relabels(self, tmp_path) -> None:
        base = EvaluationReport()
        for i in range(4):
            base.add(_rec(i, 8.0, kind="eval"))
        p = tmp_path / "baseline.json"
        base.save(p)
        rep = EvaluationReport()
        rep.add(_rec(1, 30.0))
        assert rep.merge_baseline(p) == 4
        assert len(rep.of_kind("human_baseline")) == 4
        assert len(rep.of_kind("eval")) == 1

    def test_markdown_includes_comparison(self) -> None:
        rep = EvaluationReport()
        for i in range(25):
            rep.add(_rec(i, 30.0 + float(i % 5), kind="eval"))
            rep.add(_rec(1000 + i, 5.0, kind="human_baseline"))
        md = rep.to_markdown()
        assert "Statistical comparison" in md
        assert "Mann-Whitney" in md
        assert "adaptive target" in md

    def test_summarize_uses_n_minus_one_std(self) -> None:
        s = summarize([2.0, 4.0])
        assert s["std"] == pytest.approx(math.sqrt(2.0))
        assert s["median"] == pytest.approx(3.0)
        assert s["best_single_run"] == pytest.approx(4.0)
