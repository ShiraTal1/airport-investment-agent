"""Unit tests for the deterministic scoring layer (no LLM, no network)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import analytics as A  # noqa: E402
from agent.tools import run_tool  # noqa: E402


def test_percentile_bounds():
    ref = pd.Series([1, 2, 3, 4])
    p = A._percentile_vs(pd.Series([0, 2.5, 4, 10]), ref)
    assert list(p) == [0, 50, 100, 100]


def test_scores_in_range_and_weights_normalized():
    s = A.score_table()
    assert s["opportunity_score"].dropna().between(0, 100).all()
    w = A._normalize_weights({"demand_growth": 2, "capacity_pressure": 0, "congestion": 0, "scale": 0})
    assert abs(sum(w.values()) - 1) < 1e-9 and w["demand_growth"] == 1


def test_weights_change_ranking_deterministically():
    a = A.rank_airports(region="US", top_n=5, min_enplaned=1_000_000)
    b = A.rank_airports(region="US", top_n=5, min_enplaned=1_000_000)
    key = lambda res: [(r["code"], r["opportunity_score"]) for r in res["ranking"]]
    assert key(a) == key(b)  # same input -> same output
    c = A.rank_airports(region="US", top_n=5, min_enplaned=1_000_000,
                        weights={"demand_growth": 1, "capacity_pressure": 0, "congestion": 0, "scale": 0})
    # with growth-only weights, the score equals the demand-growth component
    assert all(abs(r["opportunity_score"] - r["score_demand_growth"]) < 0.11 for r in c["ranking"])


def test_region_filter_new_england():
    r = A.rank_airports(region="New England")
    assert r["ranking"] and all(x["state"] in A.states_for_region("New England")[1] for x in r["ranking"])


def test_spill_rate_monotonic():
    assert A._spill_rate(0.7) == 0 and A._spill_rate(0.8) == 0
    assert 0 < A._spill_rate(0.9) < A._spill_rate(0.96) == A._spill_rate(0.99) == 0.15


def test_route_mix_threshold_sensitivity():
    x = A.route_mix("ANC")
    s = x["sensitivity_by_threshold_pct"]
    assert s[">= 2000 mi"] >= s[">= 3000 mi"] >= s[">= 4000 mi"]


def test_alias_resolution_and_tool_assumptions():
    out = run_tool("compare_airports", {"codes": ["LA", "Santa Ana"], "focus": "congestion"})
    assert set(out["airports"]) == {"LAX", "SNA"}
    assert any("LAX" in a for a in out["assumptions_made"])


def test_unknown_tool_does_not_crash():
    assert "error" in run_tool("nope", {})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
