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


def test_unmet_flights_and_new_tools():
    d = A.demand_pressure("SFO")
    f = d["estimated_unmet_departing_flights_per_year"]
    assert 0 < f["low"] < f["central"] < f["high"]
    r = run_tool("rank_by_metric", {"metric": "load_factor", "states": ["MA", "ME"], "top_n": 3, "min_enplaned": 0})
    vals = [x["value"] for x in r["ranking"]]
    assert vals == sorted(vals, reverse=True) and "data_period" in r
    rl = run_tool("route_lookup", {"origin": "Denver", "dest": "LAS"})
    assert rl["directions"][0]["found"] and rl["total_passengers_both_directions"] > rl["directions"][0]["passengers"]
    assert "error" in run_tool("rank_by_metric", {"metric": "nope"})


def test_recommendation_never_overstates_partial_data():
    s = A.score_table()
    hi = s[s["recommendation"].str.startswith("Prioritize")]
    assert (hi["potential"] == "High potential").all() and (hi["confidence"] == "High").all()
    assert A._recommendation("High potential", "Low", []) == "Investigate further (data gaps)"
    assert A._recommendation("Low potential", "High", ["AIRSIDE-CONSTRAINED: x"]) == "Deprioritize"
    r = A.rank_airports(region="New England")
    assert r["data_required_before_investment_decision"] and all("drivers" in x for x in r["ranking"])


def test_capex_feasibility_ranges_and_labels():
    x = run_tool("capex_feasibility", {"code": "PWM"})
    c = x["capex_estimate_usd_millions"]
    assert 0 < c["low"] < c["central"] < c["high"]
    p = x["simple_payback_years_at_full_use"]
    assert p["optimistic (low cost, high margin)"] < p["central"] < p["pessimistic (high cost, low margin)"]
    assert x["overall_confidence"].startswith("Low")
    assert any("not final" in d["latest_status"].lower() for d in x["estimate_drift_in_benchmarks"])
    assert "Not the total cost" in x["federal_aip_context"]["what_this_is"]
    y = run_tool("capex_feasibility", {"code": "PWM", "added_annual_pax": 1_000_000})
    assert y["scenario"]["added_annual_passengers"] == 1_000_000


def test_unknown_tool_does_not_crash():
    assert "error" in run_tool("nope", {})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
