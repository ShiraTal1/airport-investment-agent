"""Deterministic checks for the sample questions (no API key needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.tools import run_tool  # noqa: E402

checks = []


def check(name, cond, detail=""):
    checks.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  -> {detail}" if detail else ""))


ne = run_tool("rank_airports", {"region": "New England"})
check("Q1 New England ranking non-empty", ne["ranking"], [(r["code"], r["opportunity_score"]) for r in ne["ranking"]])
cmp_ = run_tool("compare_airports", {"codes": ["LA", "Santa Ana"], "focus": "congestion"})
check("Q2 resolves LA->LAX and Santa Ana->SNA", set(cmp_["airports"]) == {"LAX", "SNA"}, cmp_.get("congestion_index_0_100"))
anc = run_tool("route_mix", {"code": "ANC"})
anc_c = run_tool("route_mix", {"code": "ANC", "include_cargo": True})
check("Q3 ANC long-haul share computed", anc["long_haul_share_pct"] is not None,
      f"pax {anc['long_haul_share_pct']}% / all incl. cargo {anc_c['long_haul_share_pct']}%")
check("Q3 ANC cargo share reported", (anc["cargo_only_departure_share_pct"] or 0) > 30, anc["cargo_only_departure_share_pct"])
sfo = run_tool("demand_pressure", {"code": "SFO"})
check("Q4 SFO unmet demand estimate with range", sfo["estimated_unmet_departing_pax_per_year"]["low"] <
      sfo["estimated_unmet_departing_pax_per_year"]["high"], sfo["estimated_unmet_departing_pax_per_year"])
live = run_tool("live_faa_status", {"codes": ["SFO"]})
check("Q8 live FAA call returns gracefully", "ok" in live, live.get("update_time") or live.get("error"))
print(f"\n{sum(c for _, c in checks)}/{len(checks)} checks passed")
