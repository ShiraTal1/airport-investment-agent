"""Deterministic metrics and scoring. No LLM involved here.

Every number the agent reports comes from this module, so results are reproducible,
testable and explainable. The LLM only chooses which function to call and explains
the output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import data
from .regions import METRO_GROUPS, ALIASES, states_for_region

LATEST_FULL_YEAR = 2025
YTD_YEAR = 2026           # partial year (Jan-Jul) used for momentum
BASELINE_YEAR = 2019      # pre-pandemic baseline
PEER_MIN_ENPLANED = 1_000_000  # "investable universe" used as the percentile reference

# ---------------------------------------------------------------------------
# Scoring configuration (documented in DESIGN.md). Weights are user-adjustable.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "demand_growth": 0.35,
    "capacity_pressure": 0.30,
    "congestion": 0.20,
    "scale": 0.15,
}

# metric -> (component, weight inside component, higher_is_more_attractive)
SUBMETRICS = {
    "pax_cagr_2023_2025":     ("demand_growth", 0.4, True),
    "ytd_pax_growth":         ("demand_growth", 0.4, True),
    "recovery_vs_2019":       ("demand_growth", 0.2, True),
    "load_factor":            ("capacity_pressure", 0.5, True),
    "high_lf_route_share":    ("capacity_pressure", 0.3, True),
    "demand_minus_supply_growth": ("capacity_pressure", 0.2, True),
    "dep_delay15_rate":       ("congestion", 0.3, True),
    "avg_taxi_out_min":       ("congestion", 0.3, True),
    "peak_hour_ops":          ("congestion", 0.2, True),
    "nas_delay_min_per_arrival": ("congestion", 0.2, True),
    "log_total_pax":          ("scale", 1.0, True),
}

METRIC_LABELS = {
    "pax_cagr_2023_2025": "Passenger CAGR 2023-2025",
    "ytd_pax_growth": "Passenger growth Jan-Jul 2026 vs 2025",
    "ytd_seat_growth": "Seat growth Jan-Jul 2026 vs 2025",
    "recovery_vs_2019": "2025 passengers as % of 2019",
    "load_factor": "Load factor (passengers / seats), 2025",
    "high_lf_route_share": "Share of seats on routes with load factor >= 85%",
    "demand_minus_supply_growth": "Passenger growth minus seat growth (YTD 2026)",
    "dep_delay15_rate": "Departures delayed 15+ min",
    "arr_delay15_rate": "Arrivals delayed 15+ min",
    "avg_taxi_out_min": "Average taxi-out time (min)",
    "avg_taxi_in_min": "Average taxi-in time (min)",
    "peak_hour_ops": "Avg. scheduled ops in the busiest hour of the day*",
    "nas_delay_min_per_arrival": "NAS (airspace/airport) delay minutes per arrival",
    "cancel_rate": "Cancellation rate",
    "log_total_pax": "Total passengers (log scale)",
}

LONG_HAUL_DEFAULT_MILES = 3000

# Turning a score into a decision-oriented label (deterministic, documented in DESIGN.md)
POTENTIAL_TIERS = [(65, "High potential"), (50, "Medium potential"), (0, "Low potential")]
COMPONENT_LABELS = {"demand_growth": "demand growth", "capacity_pressure": "capacity pressure",
                    "congestion": "congestion", "scale": "scale"}
DATA_REQUIRED_BEFORE_INVESTMENT = [
    "Terminal throughput and utilization (passengers per gate, peak-hour terminal capacity)",
    "Construction and operating cost of the proposed expansion",
    "Expected incremental passenger volume (airline commitments, official FAA forecasts)",
    "Revenue per incremental passenger (aeronautical fees, concessions, parking) and lease terms",
    "Regulatory, environmental and physical constraints (land, runways, airspace, community caps)",
]


# ---------------------------------------------------------------------------
# Metric table
# ---------------------------------------------------------------------------
def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


@lru_cache(maxsize=1)
def metrics_table() -> pd.DataFrame:
    """One row per airport with all KPIs (raw values, not scores)."""
    ay = data.airport_years()
    piv = {y: ay[ay.year == y].set_index("airport") for y in sorted(ay.year.unique())}
    cur = piv[LATEST_FULL_YEAR]
    m = pd.DataFrame(index=cur.index)
    m["state"] = cur["state"]
    m["pax_enplaned"] = cur["pax_out"]
    m["pax_total"] = cur["pax_out"] + cur["pax_in"]
    m["seats"] = cur["seats"]
    m["departures"] = cur["dep_all"]
    m["pax_departures"] = cur["dep_pax"]
    for y in (BASELINE_YEAR, 2023, 2024):
        if y in piv:
            m[f"pax_enplaned_{y}"] = piv[y]["pax_out"].reindex(m.index)
    m["recovery_vs_2019"] = _safe_div(m["pax_enplaned"], m.get(f"pax_enplaned_{BASELINE_YEAR}"))
    m["pax_cagr_2023_2025"] = _safe_div(m["pax_enplaned"], m.get("pax_enplaned_2023")) ** 0.5 - 1
    if YTD_YEAR in piv:
        ytd = piv[YTD_YEAR].reindex(m.index)
        m["pax_enplaned_ytd_2026"] = ytd["pax_out"]
        m["ytd_pax_growth"] = _safe_div(ytd["pax_out"], cur["pax_out_jan_jul"]) - 1
        m["ytd_seat_growth"] = _safe_div(ytd["seats"], cur["seats_jan_jul"]) - 1
        m["load_factor_ytd_2026"] = _safe_div(ytd["pax_out"], ytd["seats"])
    m["demand_minus_supply_growth"] = m["ytd_pax_growth"] - m["ytd_seat_growth"]
    m["load_factor"] = _safe_div(cur["pax_out"], cur["seats"])
    m["high_lf_route_share"] = _safe_div(cur["seats_on_hi_lf_routes"], cur["seats"])
    m["intl_seat_share"] = _safe_div(cur["intl_seats"], cur["seats"])
    m["cargo_only_departure_share"] = _safe_div(cur["dep_cargo_only"], cur["dep_all"])
    m["freight_tons_out"] = cur["freight_out_lbs"] / 2000
    m["log_total_pax"] = np.log10(m["pax_total"].clip(lower=1))

    ot = data.ontime().set_index("airport").reindex(m.index).astype(float)
    flown_dep = ot["dep"] - ot["dep_cxl"]
    flown_arr = ot["arr"] - ot["arr_cxl"]
    m["ontime_departures_sample"] = ot["dep"]
    m["dep_delay15_rate"] = _safe_div(ot["dep_del15"], flown_dep)
    m["arr_delay15_rate"] = _safe_div(ot["arr_del15"], flown_arr)
    m["cancel_rate"] = _safe_div(ot["dep_cxl"], ot["dep"])
    m["avg_taxi_out_min"] = _safe_div(ot["taxi_out"], ot["taxi_out_n"])
    m["avg_taxi_in_min"] = _safe_div(ot["taxi_in"], ot["taxi_in_n"])
    m["avg_dep_delay_min"] = _safe_div(ot["dep_delay_min"], flown_dep)
    m["nas_delay_min_per_arrival"] = _safe_div(ot["nas_min"], flown_arr)
    m["peak_hour_ops"] = _safe_div(ot["peak_sum"], ot["days"])
    # Too few reporting-carrier flights -> on-time metrics are noise
    # Fewer than ~8 reporting-carrier departures per day -> on-time metrics are noise
    thin = _safe_div(ot["dep"], ot["days"]).fillna(0) < 8
    for c in ["dep_delay15_rate", "arr_delay15_rate", "cancel_rate", "avg_taxi_out_min",
              "avg_taxi_in_min", "avg_dep_delay_min", "nas_delay_min_per_arrival", "peak_hour_ops"]:
        m.loc[thin, c] = np.nan

    # FAA-style hub size from share of national enplanements (approximation of the FAA rule)
    share = m["pax_enplaned"] / m["pax_enplaned"].sum()
    m["hub_size"] = np.select([share >= 0.01, share >= 0.0025, share >= 0.0005],
                              ["Large hub", "Medium hub", "Small hub"], default="Non-hub")

    names = data.airport_names().set_index("code")
    m["name"] = names["name"].reindex(m.index)
    m["city"] = names["city"].reindex(m.index)
    m.index.name = "code"
    return m


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _percentile_vs(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Percentile (0-100) of each value against a reference distribution."""
    ref = np.sort(reference.dropna().to_numpy())
    if len(ref) == 0:
        return pd.Series(np.nan, index=values.index)
    ranks = np.searchsorted(ref, values.to_numpy(), side="right") / len(ref) * 100
    return pd.Series(np.where(values.isna(), np.nan, ranks), index=values.index)


def _normalize_weights(weights: dict | None) -> dict:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in w and v is not None and v >= 0:
                w[k] = float(v)
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def score_table(weights: dict | None = None) -> pd.DataFrame:
    """Score every airport. Percentiles are computed against the peer universe
    (airports with >= 1M enplaned passengers) so scores are comparable nationally."""
    m = metrics_table()
    w = _normalize_weights(weights)
    peers = m[m["pax_enplaned"] >= PEER_MIN_ENPLANED]

    pct = pd.DataFrame(index=m.index)
    for metric, (_, _, higher_better) in SUBMETRICS.items():
        p = _percentile_vs(m[metric], peers[metric])
        pct[metric] = p if higher_better else 100 - p

    out = pd.DataFrame(index=m.index)
    missing_components = pd.Series(0, index=m.index)
    for comp in DEFAULT_WEIGHTS:
        subs = [(k, wt) for k, (c, wt, _) in SUBMETRICS.items() if c == comp]
        num = sum(pct[k].fillna(0) * wt for k, wt in subs)
        den = sum(pct[k].notna() * wt for k, wt in subs)
        out[comp] = (num / den.replace(0, np.nan)).round(1)
        missing_components += out[comp].isna().astype(int)

    # Overall: a component with no data at all is set to a neutral 50 (so missing data neither
    # helps nor hurts the rank) and the confidence label is lowered.
    out["opportunity_score"] = sum(out[c].fillna(50.0) * w[c] for c in w).round(1)

    # Confidence: data completeness + sample size
    conf = []
    for code in m.index:
        issues = int(missing_components[code])
        if m.at[code, "pax_enplaned"] < PEER_MIN_ENPLANED:
            issues += 1
        if pd.isna(m.at[code, "ytd_pax_growth"]) or pd.isna(m.at[code, "pax_cagr_2023_2025"]):
            issues += 1
        conf.append("High" if issues == 0 else "Medium" if issues == 1 else "Low")
    out["confidence"] = conf
    out["flags"] = [_flags(m.loc[c], peers) for c in m.index]
    out["potential"] = [_potential(v) for v in out["opportunity_score"]]
    out["recommendation"] = [_recommendation(out.at[c, "potential"], out.at[c, "confidence"], out.at[c, "flags"])
                             for c in m.index]
    out["drivers"] = [_drivers(out.loc[c]) for c in m.index]
    return out


def _potential(score) -> str:
    if pd.isna(score):
        return "Not scored"
    return next(label for cut, label in POTENTIAL_TIERS if score >= cut)


def _recommendation(potential: str, confidence: str, flags: list) -> str:
    """Score + confidence -> next step. A high score on partial data is never an automatic 'go'."""
    if potential == "High potential":
        rec = "Prioritize for due diligence" if confidence == "High" else "Investigate further (data gaps)"
    elif potential == "Medium potential":
        rec = "Investigate further" if confidence == "High" else "Monitor"
    else:
        rec = "Deprioritize"
    if any(f.startswith("AIRSIDE-CONSTRAINED") for f in flags) and rec != "Deprioritize":
        rec += "; check runway/airspace capacity first"
    return rec


def _drivers(row: pd.Series) -> str:
    """Plain-language strengths / weaknesses from component scores (deterministic)."""
    comps = {c: row[c] for c in DEFAULT_WEIGHTS if pd.notna(row[c])}
    strong = [f"{COMPONENT_LABELS[c]} ({v:.0f})" for c, v in sorted(comps.items(), key=lambda x: -x[1]) if v >= 70]
    weak = [f"{COMPONENT_LABELS[c]} ({v:.0f})" for c, v in sorted(comps.items(), key=lambda x: x[1]) if v < 40]
    missing = [COMPONENT_LABELS[c] for c in DEFAULT_WEIGHTS if pd.isna(row[c])]
    parts = []
    parts.append("Strengths: " + (", ".join(strong) if strong else "none above the 70th percentile"))
    parts.append("Weaknesses: " + (", ".join(weak) if weak else "none below the 40th percentile"))
    if missing:
        parts.append("Not scored (no data): " + ", ".join(missing))
    return ". ".join(parts) + "."


def _flags(row: pd.Series, peers: pd.DataFrame) -> list[str]:
    flags = []
    nas_p75 = peers["nas_delay_min_per_arrival"].quantile(0.75)
    taxi_p75 = peers["avg_taxi_out_min"].quantile(0.75)
    if (row["pax_enplaned"] >= PEER_MIN_ENPLANED and pd.notna(row["nas_delay_min_per_arrival"]) and row["nas_delay_min_per_arrival"] >= nas_p75
            and pd.notna(row["avg_taxi_out_min"]) and row["avg_taxi_out_min"] >= taxi_p75):
        flags.append("AIRSIDE-CONSTRAINED: high airspace/runway (NAS) delays and long taxi-out. "
                     "Terminal expansion alone may not unlock growth; runway/airspace limits may cap it.")
    if pd.notna(row["cargo_only_departure_share"]) and row["cargo_only_departure_share"] >= 0.3:
        flags.append(f"CARGO-HEAVY: {row['cargo_only_departure_share']:.0%} of departures are cargo-only; "
                     "passenger metrics understate total activity.")
    if row["pax_enplaned"] < PEER_MIN_ENPLANED:
        flags.append("BELOW PEER SIZE: < 1M enplaned passengers; percentiles are vs. larger airports.")
    if pd.notna(row.get("recovery_vs_2019")) and row["recovery_vs_2019"] < 0.9:
        flags.append("NOT RECOVERED: passengers still below 90% of 2019.")
    if pd.isna(row["dep_delay15_rate"]):
        flags.append("LIMITED DELAY DATA: few flights by on-time reporting carriers; congestion not scored.")
    return flags


# ---------------------------------------------------------------------------
# Airport resolution
# ---------------------------------------------------------------------------
def resolve_airport(query: str) -> dict:
    """Resolve a code, alias or name to an airport code. Returns {code, assumption}."""
    q = (query or "").strip()
    m = metrics_table()
    if q.upper() in m.index:
        return {"code": q.upper(), "assumption": None}
    if q.lower() in ALIASES:
        code = ALIASES[q.lower()]
        return {"code": code, "assumption": f"Interpreted '{q}' as {code} ({m.at[code, 'name'] if code in m.index else code})."}
    hits = search_airports(q, limit=1)
    if hits:
        h = hits[0]
        return {"code": h["code"], "assumption": f"Interpreted '{q}' as {h['code']} ({h['name']}, {h['city']})."}
    return {"code": None, "assumption": f"Could not match '{q}' to an airport in the dataset."}


def search_airports(query: str, limit: int = 8) -> list[dict]:
    m = metrics_table()
    q = (query or "").strip().lower()
    if not q:
        return []
    mask = (m.index.str.lower() == q) | m["name"].fillna("").str.lower().str.contains(q, regex=False) \
        | m["city"].fillna("").str.lower().str.contains(q, regex=False)
    res = m[mask].sort_values("pax_enplaned", ascending=False).head(limit)
    return [{"code": c, "name": r["name"], "city": r["city"], "state": r["state"],
             "hub_size": r["hub_size"], "pax_enplaned_2025": int(r["pax_enplaned"])}
            for c, r in res.iterrows()]


# ---------------------------------------------------------------------------
# Public analysis functions (called by agent tools)
# ---------------------------------------------------------------------------
def _fmt(v, kind):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if kind == "pct":
        return round(float(v) * 100, 1)
    if kind == "int":
        return int(round(float(v)))
    return round(float(v), 2)


PROFILE_FIELDS = [
    ("pax_enplaned", "int"), ("pax_total", "int"), ("seats", "int"), ("departures", "int"),
    (f"pax_enplaned_{BASELINE_YEAR}", "int"), ("pax_enplaned_ytd_2026", "int"),
    ("recovery_vs_2019", "pct"), ("pax_cagr_2023_2025", "pct"), ("ytd_pax_growth", "pct"),
    ("ytd_seat_growth", "pct"), ("demand_minus_supply_growth", "pct"),
    ("load_factor", "pct"), ("load_factor_ytd_2026", "pct"), ("high_lf_route_share", "pct"),
    ("intl_seat_share", "pct"), ("cargo_only_departure_share", "pct"),
    ("dep_delay15_rate", "pct"), ("arr_delay15_rate", "pct"), ("cancel_rate", "pct"),
    ("avg_taxi_out_min", "num"), ("avg_taxi_in_min", "num"), ("avg_dep_delay_min", "num"),
    ("nas_delay_min_per_arrival", "num"), ("peak_hour_ops", "num"),
]


def airport_profile(code: str, weights: dict | None = None) -> dict:
    m = metrics_table()
    if code not in m.index:
        return {"error": f"{code} not found in dataset"}
    r = m.loc[code]
    s = score_table(weights).loc[code]
    peers = m[m["pax_enplaned"] >= PEER_MIN_ENPLANED]
    metrics = {}
    for f, kind in PROFILE_FIELDS:
        val = _fmt(r.get(f), kind)
        entry = {"value": val, "unit": "%" if kind == "pct" else ""}
        if f in peers and pd.notna(r.get(f)) and f not in ("pax_enplaned_2019", "pax_enplaned_ytd_2026"):
            entry["peer_percentile"] = round(float(_percentile_vs(pd.Series([r[f]]), peers[f]).iloc[0]), 0)
        metrics[f] = entry
    return {
        "code": code, "name": r["name"], "city": r["city"], "state": r["state"], "hub_size": r["hub_size"],
        "opportunity_score": s["opportunity_score"], "potential": s["potential"],
        "recommendation": s["recommendation"], "drivers": s["drivers"],
        "component_scores": {c: s[c] for c in DEFAULT_WEIGHTS},
        "confidence": s["confidence"], "flags": s["flags"], "metrics": metrics,
        "data_required_before_investment_decision": DATA_REQUIRED_BEFORE_INVESTMENT,
    }


def rank_airports(region: str | None = None, states: list[str] | None = None,
                  codes: list[str] | None = None, top_n: int = 10,
                  min_enplaned: int = 250_000, weights: dict | None = None) -> dict:
    m = metrics_table()
    s = score_table(weights)
    scope_desc, notes = "US", []
    mask = pd.Series(True, index=m.index)
    if codes:
        mask &= m.index.isin([c.upper() for c in codes])
        scope_desc = ", ".join(codes)
    elif states:
        mask &= m["state"].isin([x.upper() for x in states])
        scope_desc = ", ".join(states)
    elif region:
        canon, st = states_for_region(region)
        metro = METRO_GROUPS.get(region.strip().lower())
        if metro:
            mask &= m.index.isin(metro)
            scope_desc = f"{region} ({', '.join(metro)})"
        elif canon is None:
            return {"error": f"Unknown region '{region}'. Pass a list of states instead."}
        elif canon == "Contiguous US":
            mask &= ~m["state"].isin(["AK", "HI"])
            scope_desc = canon
        elif st:
            mask &= m["state"].isin(st)
            scope_desc = f"{canon} ({', '.join(st)})"
    excluded = m[mask & (m["pax_enplaned"] < min_enplaned)]
    mask &= m["pax_enplaned"] >= min_enplaned
    if len(excluded):
        notes.append(f"Excluded {len(excluded)} airport(s) below {min_enplaned:,} enplaned passengers "
                     f"(e.g. {', '.join(excluded.sort_values('pax_enplaned', ascending=False).index[:5])}).")
    sub = s[mask].join(m[["name", "city", "state", "hub_size", "pax_enplaned", "load_factor",
                          "ytd_pax_growth", "pax_cagr_2023_2025", "dep_delay15_rate", "avg_taxi_out_min"]])
    sub = sub.sort_values("opportunity_score", ascending=False).head(top_n)
    rows = []
    for rank, (code, r) in enumerate(sub.iterrows(), 1):
        rows.append({
            "rank": rank, "code": code, "name": r["name"], "state": r["state"], "hub_size": r["hub_size"],
            "opportunity_score": r["opportunity_score"], "potential": r["potential"],
            "recommendation": r["recommendation"], "drivers": r["drivers"],
            **{f"score_{c}": r[c] for c in DEFAULT_WEIGHTS},
            "pax_enplaned_2025": int(r["pax_enplaned"]),
            "load_factor_pct": _fmt(r["load_factor"], "pct"),
            "pax_cagr_2023_2025_pct": _fmt(r["pax_cagr_2023_2025"], "pct"),
            "ytd_2026_pax_growth_pct": _fmt(r["ytd_pax_growth"], "pct"),
            "dep_delay15_pct": _fmt(r["dep_delay15_rate"], "pct"),
            "avg_taxi_out_min": _fmt(r["avg_taxi_out_min"], "num"),
            "confidence": r["confidence"], "flags": r["flags"],
        })
    return {"scope": scope_desc, "weights_used": _normalize_weights(weights), "ranking": rows,
            "tier_rules": "Potential: High >= 65, Medium 50-65, Low < 50. Recommendation combines potential with "
                          "confidence: only High potential + High confidence -> 'Prioritize for due diligence'.",
            "data_required_before_investment_decision": DATA_REQUIRED_BEFORE_INVESTMENT,
            "notes": notes + ["Scores are 0-100 percentiles vs. ~all US airports with >= 1M enplaned "
                              "passengers (national peer group), so they are comparable across regions."]}


COMPARE_FOCUS = {
    "congestion": ["dep_delay15_rate", "arr_delay15_rate", "avg_taxi_out_min", "avg_taxi_in_min",
                   "avg_dep_delay_min", "nas_delay_min_per_arrival", "peak_hour_ops", "cancel_rate",
                   "load_factor", "pax_enplaned"],
    "demand": ["pax_enplaned", "pax_cagr_2023_2025", "ytd_pax_growth", "ytd_seat_growth",
               "recovery_vs_2019", "load_factor", "high_lf_route_share", "demand_minus_supply_growth"],
}


def compare_airports(codes: list[str], focus: str = "all", weights: dict | None = None) -> dict:
    profiles = [airport_profile(c.upper(), weights) for c in codes]
    fields = COMPARE_FOCUS.get(focus)
    table = {}
    for p in profiles:
        if "error" in p:
            table[p.get("code", "?")] = p
            continue
        mets = p["metrics"] if not fields else {k: v for k, v in p["metrics"].items() if k in fields}
        table[p["code"]] = {"name": p["name"], "hub_size": p["hub_size"],
                            "opportunity_score": p["opportunity_score"], "potential": p["potential"],
                            "recommendation": p["recommendation"], "drivers": p["drivers"],
                            "component_scores": p["component_scores"], "confidence": p["confidence"],
                            "flags": p["flags"], "metrics": mets}
    result = {"focus": focus, "airports": table}
    if focus == "congestion":
        # Simple deterministic congestion index = mean percentile of congestion sub-metrics
        idx = {}
        for code, t in table.items():
            ps = [t["metrics"][k].get("peer_percentile") for k in
                  ("dep_delay15_rate", "avg_taxi_out_min", "peak_hour_ops", "nas_delay_min_per_arrival")
                  if k in t.get("metrics", {})]
            ps = [x for x in ps if x is not None]
            idx[code] = round(sum(ps) / len(ps), 1) if ps else None
        result["congestion_index_0_100"] = idx
        result["congestion_index_definition"] = ("Mean peer percentile of: % departures delayed 15+ min, "
                                                 "avg taxi-out, busiest-hour ops, NAS delay per arrival. "
                                                 "Higher = more congested.")
    return result


def route_mix(code: str, long_haul_miles: float = LONG_HAUL_DEFAULT_MILES,
              basis: str = "departures", include_cargo: bool = False) -> dict:
    code = code.upper()
    r = data.routes()
    r = r[(r["origin"] == code) & (r["dest"] != code)]
    if r.empty:
        return {"error": f"No route data for {code}"}
    if not include_cargo:
        # Freighters sometimes report a few seats; treat a route as passenger service only if it
        # averages >= 10 passengers per flight.
        r = r[r["passengers"] >= 10 * r["dep_pax_flights"].clip(lower=1)]
    dep_col = "departures" if include_cargo else "dep_pax_flights"
    val_col = {"departures": dep_col, "seats": "seats", "passengers": "passengers"}.get(basis, dep_col)
    total = r[val_col].sum()
    lh = r[r["distance"] >= long_haul_miles]
    share = lh[val_col].sum() / total if total else None
    top = (lh.groupby(["dest", "dest_country", "distance"])[[val_col, "passengers", "freight_lbs"]].sum()
           .sort_values(val_col, ascending=False).head(8).reset_index())
    ay = data.airport_years()
    cur = ay[(ay.airport == code) & (ay.year == LATEST_FULL_YEAR)]
    coverage = None
    if not cur.empty:
        if include_cargo:
            num, denom = r["departures"].sum(), cur["dep_all"].iloc[0]
        else:
            num, denom = r["passengers"].sum(), cur["pax_out"].iloc[0]
        coverage = round(num / denom * 100, 1) if denom else None
    cargo_share = None
    if not cur.empty and cur["dep_all"].iloc[0]:
        cargo_share = round(cur["dep_cargo_only"].iloc[0] / cur["dep_all"].iloc[0] * 100, 1)
    # Sensitivity: share at alternative thresholds
    sens = {}
    for t in (2000, 2500, 3000, 4000):
        sens[f">= {t} mi"] = round(r[r["distance"] >= t][val_col].sum() / total * 100, 1) if total else None
    return {
        "airport": code, "year": LATEST_FULL_YEAR,
        "definition": f"Long-haul = non-stop segment >= {long_haul_miles:,.0f} statute miles; measured by {basis}"
                      f" of {'all flights incl. cargo' if include_cargo else 'passenger flights (routes averaging >= 10 passengers/flight)'}.",
        "long_haul_share_pct": round(share * 100, 1) if share is not None else None,
        "long_haul_count": int(lh[val_col].sum()), "total_count": int(total),
        "sensitivity_by_threshold_pct": sens,
        "top_long_haul_destinations": top.rename(columns={val_col: basis}).to_dict(orient="records"),
        "cargo_only_departure_share_pct": cargo_share,
        "route_data_coverage_pct": coverage,
        "notes": ["Source: BTS T-100 Segment (all carriers, domestic + international) 2025.",
                  "Route file omits very small routes (<3,000 passengers and <1M lbs freight/yr); "
                  "coverage shows how much of the airport's traffic is included."],
    }


def _spill_rate(lf: float) -> float:
    """Heuristic share of demand turned away on a route at a given annual load factor.
    0 below 80% LF, rising linearly to 15% at >= 95% LF. Assumption, not a measurement."""
    if pd.isna(lf):
        return 0.0
    return float(np.clip((lf - 0.80) / 0.15, 0, 1) * 0.15)


def demand_pressure(code: str) -> dict:
    """Indicators of unmet ('spilled') demand. Unmet demand is not directly observable;
    this combines observable proxies and a transparent heuristic estimate."""
    code = code.upper()
    m = metrics_table()
    if code not in m.index:
        return {"error": f"{code} not found"}
    row = m.loc[code]
    r = data.routes()
    r = r[(r["origin"] == code) & (r["seats"] > 0) & (r["dest"] != code)].copy()
    r = r[r["passengers"] >= 10 * r["dep_pax_flights"].clip(lower=1)]  # passenger routes only
    r["load_factor"] = r["passengers"] / r["seats"]
    r["est_spilled_pax"] = r.apply(lambda x: x["passengers"] * _spill_rate(x["load_factor"]), axis=1)
    # Convert spilled passengers into flights at each route's own average passengers per flight
    r["pax_per_flight"] = r["passengers"] / r["dep_pax_flights"].clip(lower=1)
    r["est_spilled_flights"] = r["est_spilled_pax"] / r["pax_per_flight"].clip(lower=1)
    est = r["est_spilled_pax"].sum()
    est_fl = r["est_spilled_flights"].sum()
    pax_deps = r["dep_pax_flights"].sum()
    tight = r[(r["load_factor"] >= 0.85) & (r["seats"] >= 10000)].sort_values("passengers", ascending=False)
    prof = airport_profile(code)
    return {
        "airport": code, "name": row["name"],
        "indicators": {k: prof["metrics"][k] for k in
                       ("load_factor", "load_factor_ytd_2026", "high_lf_route_share", "ytd_pax_growth",
                        "ytd_seat_growth", "demand_minus_supply_growth", "recovery_vs_2019",
                        "dep_delay15_rate", "avg_taxi_out_min", "nas_delay_min_per_arrival", "peak_hour_ops")},
        "estimated_unmet_departing_pax_per_year": {
            "central": int(est), "low": int(est * 0.5), "high": int(est * 1.5),
            "as_pct_of_enplaned": round(est / row["pax_enplaned"] * 100, 1),
            "method": "Sum over routes of passengers x spill rate; spill rate is 0% at <=80% annual load "
                      "factor rising linearly to 15% at >=95%. Range is +/-50%. Heuristic assumption, low confidence.",
        },
        "estimated_unmet_departing_flights_per_year": {
            "central": int(est_fl), "low": int(est_fl * 0.5), "high": int(est_fl * 1.5),
            "per_day": round(est_fl / 365, 1),
            "as_pct_of_passenger_departures": round(est_fl / pax_deps * 100, 1) if pax_deps else None,
            "method": "Spilled passengers on each route divided by that route's average passengers per flight "
                      "(i.e. extra flights of the same size airlines would need). Same low confidence.",
        },
        "tightest_routes": [
            {"dest": t["dest"], "distance_mi": int(t["distance"]), "passengers": int(t["passengers"]),
             "seats": int(t["seats"]), "load_factor_pct": round(t["load_factor"] * 100, 1),
             "est_unmet_flights_per_year": int(t["est_spilled_flights"])}
            for _, t in tight.head(8).iterrows()],
        "flags": prof["flags"],
        "interpretation_hints": [
            "High load factor + passengers growing faster than seats = demand outpacing supplied capacity.",
            "High NAS delay + long taxi-out = runway/airspace limits; airlines may be unable to add flights "
            "even with demand (common at slot- or weather-constrained airports).",
        ],
    }


def methodology(weights: dict | None = None) -> dict:
    md = data.metadata()
    return {
        "opportunity_score": "Weighted average of 4 component scores (0-100). Each component averages "
                             "peer percentiles of its sub-metrics vs. US airports with >= 1M enplaned passengers.",
        "weights": _normalize_weights(weights),
        "submetrics": {k: {"component": c, "weight_in_component": w, "label": METRIC_LABELS.get(k, k)}
                       for k, (c, w, _) in SUBMETRICS.items()},
        "potential_tiers": "High potential >= 65, Medium 50-65, Low < 50",
        "recommendation_rule": "High potential + High confidence -> Prioritize for due diligence; High potential "
                               "with Medium/Low confidence -> Investigate further (data gaps); Medium + High "
                               "confidence -> Investigate further; Medium otherwise -> Monitor; Low -> Deprioritize. "
                               "Airside-constrained airports get 'check runway/airspace capacity first'.",
        "data_required_before_investment_decision": DATA_REQUIRED_BEFORE_INVESTMENT,
        "feasibility_method": "capex_feasibility: added capacity = 10 years of growth (CAGR bounded 1%-5%); "
                              "area = 0.05/0.07/0.09 sq ft per annual passenger; cost = area x $/sq ft from 4 "
                              "recent US terminals (min/median/max); added income = added enplanements x FAA CATS "
                              "revenue per enplanement x 25%/37%/45% margin; payback = cost / income. "
                              "Screening only, Low confidence.",
        "missing_data_rule": "Within a component, sub-metric weights are re-normalized over available data. "
                             "A component with no data is set to a neutral 50. Confidence is lowered either way.",
        "data_sources": md.get("sources", []),
        "data_vintage": md.get("vintage", {}),
        "key_assumptions": [
            "Enplaned passengers approximated by T-100 non-stop segment passengers departing the airport "
            "(includes connecting passengers, like FAA enplanement counts).",
            "Delay/taxi metrics come only from carriers that report on-time data (large US airlines and "
            "major regionals), not all traffic.",
            "*Busiest-hour ops counts scheduled departures + arrivals of reporting carriers only (an index, "
            "not total airport operations).",
            "Higher congestion and load factor are treated as signals of investment need (demand exceeds capacity), "
            "not as negatives.",
            "No data on gates, terminal square footage, construction cost, airline lease terms or regulatory caps; "
            "the score identifies where capacity pressure exists, not project-level profitability.",
        ],
    }


# Metrics that can be ranked on their own (rank_by_metric)
RANKABLE = {k: METRIC_LABELS.get(k, k) for k in [
    "pax_enplaned", "pax_total", "recovery_vs_2019", "pax_cagr_2023_2025", "ytd_pax_growth",
    "ytd_seat_growth", "demand_minus_supply_growth", "load_factor", "high_lf_route_share",
    "intl_seat_share", "cargo_only_departure_share", "dep_delay15_rate", "arr_delay15_rate",
    "cancel_rate", "avg_taxi_out_min", "avg_taxi_in_min", "avg_dep_delay_min",
    "nas_delay_min_per_arrival", "peak_hour_ops", "congestion_score", "demand_growth_score",
    "capacity_pressure_score", "opportunity_score"]}
PCT_METRICS = {"recovery_vs_2019", "pax_cagr_2023_2025", "ytd_pax_growth", "ytd_seat_growth",
               "demand_minus_supply_growth", "load_factor", "high_lf_route_share", "intl_seat_share",
               "cargo_only_departure_share", "dep_delay15_rate", "arr_delay15_rate", "cancel_rate"}


def _scope_mask(m: pd.DataFrame, region=None, states=None, codes=None):
    """Same scoping rules as rank_airports. Returns (mask, description) or (None, error)."""
    mask = pd.Series(True, index=m.index)
    if codes:
        return mask & m.index.isin([c.upper() for c in codes]), ", ".join(codes)
    if states:
        return mask & m["state"].isin([x.upper() for x in states]), ", ".join(states)
    if region:
        metro = METRO_GROUPS.get(region.strip().lower())
        if metro:
            return mask & m.index.isin(metro), f"{region} ({', '.join(metro)})"
        canon, st = states_for_region(region)
        if canon is None:
            return None, f"Unknown region '{region}'. Pass a list of states instead."
        if canon == "Contiguous US":
            return mask & ~m["state"].isin(["AK", "HI"]), canon
        if st:
            return mask & m["state"].isin(st), f"{canon} ({', '.join(st)})"
    return mask, "US"


def rank_by_metric(metric: str, region: str | None = None, states: list[str] | None = None,
                   codes: list[str] | None = None, hub_sizes: list[str] | None = None,
                   top_n: int = 10, ascending: bool = False, min_enplaned: int = 1_000_000) -> dict:
    """Rank airports on ONE metric (e.g. most congested, fastest growing, emptiest planes)."""
    if metric not in RANKABLE:
        return {"error": f"Unknown metric '{metric}'.", "available_metrics": RANKABLE}
    m = metrics_table().copy()
    s = score_table()
    m["congestion_score"] = s["congestion"]
    m["demand_growth_score"] = s["demand_growth"]
    m["capacity_pressure_score"] = s["capacity_pressure"]
    m["opportunity_score"] = s["opportunity_score"]
    mask, scope = _scope_mask(m, region, states, codes)
    if mask is None:
        return {"error": scope}
    if hub_sizes:
        mask &= m["hub_size"].str.lower().isin([h.lower() for h in hub_sizes])
    mask &= m["pax_enplaned"] >= min_enplaned
    sub = m[mask & m[metric].notna()].sort_values(metric, ascending=ascending).head(top_n)
    missing = int((mask & m[metric].isna()).sum())
    rows = []
    for rank, (code, r) in enumerate(sub.iterrows(), 1):
        val = r[metric]
        rows.append({"rank": rank, "code": code, "name": r["name"], "state": r["state"],
                     "hub_size": r["hub_size"],
                     "value": _fmt(val, "pct") if metric in PCT_METRICS else _fmt(val, "num"),
                     "unit": "%" if metric in PCT_METRICS else "",
                     "pax_enplaned_2025": int(r["pax_enplaned"]), "flags": s.at[code, "flags"]})
    notes = [f"Filtered to airports with >= {min_enplaned:,} enplaned passengers in 2025."]
    if missing:
        notes.append(f"{missing} airport(s) in scope have no value for this metric (e.g. too few on-time "
                     "reporting flights) and are not ranked.")
    return {"metric": metric, "metric_label": RANKABLE[metric],
            "order": "lowest first" if ascending else "highest first", "scope": scope,
            "hub_sizes": hub_sizes, "ranking": rows, "notes": notes}


def route_lookup(origin: str, dest: str) -> dict:
    """Non-stop traffic between two airports in 2025, both directions."""
    r = data.routes()
    out = {"origin": origin.upper(), "dest": dest.upper(), "year": LATEST_FULL_YEAR, "directions": []}
    for a, b in ((origin.upper(), dest.upper()), (dest.upper(), origin.upper())):
        x = r[(r["origin"] == a) & (r["dest"] == b)]
        if x.empty:
            out["directions"].append({"from": a, "to": b, "found": False,
                                      "note": "No non-stop segment in the dataset (no service, a very small "
                                              "route below the file threshold, or a non-US origin)."})
            continue
        t = x.sum(numeric_only=True)
        out["directions"].append({
            "from": a, "to": b, "found": True, "distance_mi": int(x["distance"].max()),
            "passengers": int(t["passengers"]), "seats": int(t["seats"]),
            "load_factor_pct": round(t["passengers"] / t["seats"] * 100, 1) if t["seats"] else None,
            "passenger_flights": int(t["dep_pax_flights"]), "cargo_only_flights": int(t["dep_cargo_only"]),
            "freight_tons": round(t["freight_lbs"] / 2000, 0)})
    found = [d for d in out["directions"] if d["found"]]
    if found:
        out["total_passengers_both_directions"] = sum(d["passengers"] for d in found)
    out["notes"] = ["BTS T-100 segment data: passengers on that non-stop flight leg, including connecting "
                    "passengers; not origin-destination demand."]
    return out


# ---------------------------------------------------------------------------
# Capex and return feasibility (screening level)
# ---------------------------------------------------------------------------
SQFT_PER_ANNUAL_PAX = (0.05, 0.07, 0.09)   # low / central / high, from benchmark terminals (see DESIGN.md)
MARGIN_RANGE = (0.25, 0.37, 0.45)          # incremental operating margin; central = FAA CATS FY2025 system margin
GROWTH_FLOOR, GROWTH_CAP = 0.01, 0.05       # bounds on the growth rate used to size the expansion

SOURCE_CONFIDENCE = [
    {"input": "Passenger growth (sizing)", "source": "BTS T-100 2023-2025", "nature": "Actual traffic; "
     "extrapolated forward", "confidence": "Medium"},
    {"input": "Terminal area per annual passenger", "source": "4 recent US terminal projects", "nature":
     "Derived from project size vs airport traffic", "confidence": "Low"},
    {"input": "Construction cost per sq ft", "source": "Same 4 projects (BUR, MCI, PIT, FLL)", "nature":
     "1 reported at completion (BUR); 3 are budgets or latest estimates, not final costs", "confidence": "Low"},
    {"input": "Revenue per enplanement", "source": "FAA CATS Form 127, FY2025", "nature": "Actual reported "
     "revenue, aggregated by hub tier (not airport-specific)", "confidence": "Medium"},
    {"input": "Incremental operating margin", "source": "FAA CATS FY2025 system average (37%)", "nature":
     "Average applied to incremental passengers; real incremental margin may differ", "confidence": "Low"},
    {"input": "Federal AIP grants FY2021-25", "source": "FAA AIP grant histories", "nature": "Grant amounts "
     "awarded (federal share at award), NOT total project cost and not final project cost", "confidence":
     "High for what it is"},
    {"input": "ACRP Report 120 (2014)", "source": "TRB cost-estimating guidebook + model", "nature": "Method "
     "and uncertainty guidance; its cost database covers airfield/ARFF/SRE projects in FY2014$, not terminals, "
     "so it is not used for terminal unit costs", "confidence": "n/a for terminals"},
]


def _rev_per_enplanement(hub_size: str) -> tuple[float, str]:
    fb = data.finance_benchmarks()
    m = metrics_table()
    total_rev = fb["fy2025_total_operating_revenue_usd"]
    lh_share = fb["large_hub_share_of_operating_revenue"]
    lh_enp = m.loc[m.hub_size == "Large hub", "pax_enplaned"].sum()
    other_enp = m.loc[m.hub_size != "Large hub", "pax_enplaned"].sum()
    if hub_size == "Large hub":
        return total_rev * lh_share / lh_enp, "Large-hub average (FAA CATS FY2025 / BTS 2025)"
    return total_rev * (1 - lh_share) / other_enp, "Average of non-large-hub airports (FAA CATS FY2025 / BTS 2025)"


def capex_feasibility(code: str, added_annual_pax: float | None = None, horizon_years: int = 10) -> dict:
    """Screening-level estimate of terminal expansion cost and the operating income it could add.
    Ranges, not point answers. Every input is labeled with its source and whether it is an estimate."""
    code = code.upper()
    m = metrics_table()
    if code not in m.index:
        return {"error": f"{code} not found"}
    r = m.loc[code]
    s = score_table().loc[code]

    # 1. Size the expansion
    g_raw = r["pax_cagr_2023_2025"]
    g = float(np.clip(0 if pd.isna(g_raw) else g_raw, GROWTH_FLOOR, GROWTH_CAP))
    if added_annual_pax is None:
        added = r["pax_total"] * ((1 + g) ** horizon_years - 1)
        sizing = (f"Capacity for {horizon_years} years of growth at {g:.1%}/yr (2023-25 CAGR "
                  f"{'n/a' if pd.isna(g_raw) else f'{g_raw:.1%}'}, bounded to 1%-5%)")
    else:
        added = float(added_annual_pax)
        sizing = "User-specified added annual passengers"
    area = [added * k for k in SQFT_PER_ANNUAL_PAX]

    # 2. Cost from benchmark $/sq ft
    b = data.cost_benchmarks().copy()
    b["usd_per_sqft"] = b["cost_usd"] / b["terminal_sqft"]
    lo, hi, mid = b["usd_per_sqft"].min(), b["usd_per_sqft"].max(), b["usd_per_sqft"].median()
    cost = {"low": area[0] * lo, "central": area[1] * mid, "high": area[2] * hi}

    # 3. Revenue and operating income at full use of the new capacity
    rpe, rpe_basis = _rev_per_enplanement(r["hub_size"])
    added_enp = added / 2
    revenue = added_enp * rpe
    op_inc = {k: revenue * v for k, v in zip(("low", "central", "high"), MARGIN_RANGE)}

    def payback(c, oi):
        return round(c / oi, 1) if oi else None

    # 4. Estimate drift seen in the benchmark projects
    drift = b.dropna(subset=["first_public_estimate_usd"]).copy()
    drift["growth_pct"] = (drift["cost_usd"] / drift["first_public_estimate_usd"] - 1) * 100

    aip = data.aip_grants()
    a = aip[aip["airport"] == code]
    aip_ctx = None
    if not a.empty:
        a = a.iloc[0]
        aip_ctx = {"aip_total_fy2021_2025_usd": int(a["aip_total_fy21_25"]),
                   "aip_terminal_related_fy2021_2025_usd": int(a["aip_terminal_fy21_25"]),
                   "terminal_grants_count": int(a["n_terminal_grants"]),
                   "largest_terminal_items": a["top_terminal_projects"],
                   "what_this_is": "Federal grant amounts awarded (federal share at award date). Not the total "
                                   "cost of those projects and not final costs. Large terminal programs are mostly "
                                   "funded by bonds, passenger facility charges and airline rates, so AIP shows "
                                   "only part of the picture."}

    M = 1e6
    return {
        "airport": code, "name": r["name"], "hub_size": r["hub_size"],
        "opportunity_score": s["opportunity_score"], "potential": s["potential"],
        "recommendation": s["recommendation"], "flags": s["flags"],
        "scenario": {"sizing_basis": sizing, "added_annual_passengers": int(added),
                     "added_annual_enplanements": int(added_enp),
                     "terminal_area_sqft_range": [int(x) for x in area]},
        "capex_estimate_usd_millions": {k: round(v / M) for k, v in cost.items()},
        "capex_basis": f"Area x ${lo:,.0f}-${hi:,.0f} per sq ft (median ${mid:,.0f}) from {len(b)} recent US terminals",
        "revenue_per_enplanement_usd": round(rpe, 2), "revenue_basis": rpe_basis,
        "added_operating_revenue_usd_millions_per_year": round(revenue / M, 1),
        "added_operating_income_usd_millions_per_year": {k: round(v / M, 1) for k, v in op_inc.items()},
        "simple_payback_years_at_full_use": {
            "optimistic (low cost, high margin)": payback(cost["low"], op_inc["high"]),
            "central": payback(cost["central"], op_inc["central"]),
            "pessimistic (high cost, low margin)": payback(cost["high"], op_inc["low"])},
        "estimate_drift_in_benchmarks": [
            {"airport": d.airport, "first_estimate_usd_m": round(d.first_public_estimate_usd / M),
             "first_estimate_year": int(d.first_estimate_year), "latest_usd_m": round(d.cost_usd / M),
             "latest_status": d.cost_status, "change_pct": round(d.growth_pct)} for d in drift.itertuples()],
        "federal_aip_context": aip_ctx,
        "source_confidence": SOURCE_CONFIDENCE,
        "overall_confidence": "Low (screening only, roughly order-of-magnitude). Not an investment-grade estimate.",
        "caveats": [
            "Payback assumes the new capacity is fully used; during a ramp-up over the sizing horizon, "
            "real payback is longer.",
            "Costs are terminal building only (no airfield, roadway, parking or financing costs) in current dollars.",
            "Airport revenue is regulated: much aeronautical revenue is set by airline agreements on a cost-recovery "
            "basis, so an investor's return depends on the deal structure (P3, concession, lease), not only on "
            "airport operating income.",
            "Early-stage terminal estimates in the benchmark set rose 55%-171% before completion or latest update; "
            "treat any early figure as likely to grow.",
        ],
    }


# ---------------------------------------------------------------------------
# Project-level investment report (one uniform format per project type)
# ---------------------------------------------------------------------------
PROJECT_TYPES = {
    "terminal_expansion": "Expand the existing terminal (more gates, gate-area and processing space)",
    "new_terminal": "Build a new terminal / replacement terminal",
    "terminal_renovation": "Modernize the existing terminal without adding capacity",
    "parking_expansion": "Add public parking capacity",
    "runway_airfield": "Add or extend a runway / airfield capacity",
}
FIN = {"construction_years": 3, "operating_years": 30, "discount_rate": 0.07,
       "opex_to_revenue_base": round(21.8 / 34.5, 2)}  # FAA CATS FY2025 cash opex / operating revenue = 0.63


def project_needs(code: str) -> dict:
    """Rule-based screen of WHICH project type an airport most likely needs (not all airports need all)."""
    m = metrics_table(); s = score_table()
    if code not in m.index:
        return {"error": f"{code} not found"}
    r, sc = m.loc[code], s.loc[code]
    needs = []
    if any(f.startswith("AIRSIDE-CONSTRAINED") for f in sc["flags"]):
        needs.append({"project": "runway_airfield", "why": "Airside-constrained flag: NAS delays and taxi-out in "
                      "the top quartile; runway/airspace capacity may cap growth before the terminal does."})
    if pd.notna(sc["capacity_pressure"]) and sc["capacity_pressure"] >= 60 and pd.notna(sc["demand_growth"]) \
            and sc["demand_growth"] >= 50:
        needs.append({"project": "terminal_expansion", "why": f"Capacity pressure {sc['capacity_pressure']:.0f} and "
                      f"demand growth {sc['demand_growth']:.0f} (percentiles): full planes and growing demand."})
    elif pd.notna(sc["demand_growth"]) and sc["demand_growth"] >= 70:
        needs.append({"project": "terminal_expansion", "why": f"Strong demand growth ({sc['demand_growth']:.0f}) "
                      "suggests capacity will be needed even though pressure is moderate today."})
    if not needs:
        needs.append({"project": "terminal_renovation", "why": "No strong capacity signal: if anything, "
                      "modernization rather than added capacity; revenue case must come from the airport."})
    return {"airport": code, "suggested_projects": needs,
            "note": "Rule-based screen from traffic and delay data. Parking needs cannot be screened (no public "
                    "parking utilization data)."}


def _irr(cfs: list[float]) -> float | None:
    def npv(rate):
        return sum(cf / (1 + rate) ** t for t, cf in enumerate(cfs))
    lo, hi = -0.99, 1.0
    if npv(lo) * npv(hi) > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return round(mid, 4)


def _cash_flows(capex: float, full_revenue: float, opex_ratio: float, ramp_years: int) -> dict:
    cy, oy, r = FIN["construction_years"], FIN["operating_years"], FIN["discount_rate"]
    cfs, rows = [], []
    for t in range(cy + oy):
        if t < cy:
            rev, opx, cap = 0.0, 0.0, capex / cy
        else:
            k = t - cy + 1
            rev = full_revenue * min(1.0, k / ramp_years)
            opx, cap = rev * opex_ratio, 0.0
        cf = rev - opx - cap
        cfs.append(cf); rows.append((t, rev, opx, cap, cf))
    npv = sum(cf / (1 + r) ** t for t, cf in enumerate(cfs))
    cum, payback = 0.0, None
    for t, cf in enumerate(cfs):
        cum += cf
        if payback is None and cum >= 0 and t >= cy:
            payback = t + 1
    M = 1e6
    sample = [{"year": t + 1, "revenue_m": round(a / M, 1), "opex_m": round(b / M, 1), "capex_m": round(c / M, 1),
               "net_cash_flow_m": round(d / M, 1)} for t, a, b, c, d in rows if t in (0, 3, 7, 12, 20, 32)]
    return {"npv_m": round(npv / M), "irr": _irr(cfs), "payback_years_from_start": payback,
            "cash_flow_sample": sample}


def project_report(code: str, project: str = "terminal_expansion", added_annual_pax: float | None = None) -> dict:
    """Uniform short investment report for one project at one airport."""
    code = code.upper()
    if project not in PROJECT_TYPES:
        return {"error": f"Unknown project '{project}'", "project_types": PROJECT_TYPES}
    m = metrics_table()
    if code not in m.index:
        return {"error": f"{code} not found"}
    r = m.loc[code]; s = score_table().loc[code]
    needs = project_needs(code)
    fit = any(n["project"] == project for n in needs["suggested_projects"])

    base = {"airport": code, "name": r["name"], "project": project, "project_description": PROJECT_TYPES[project],
            "potential": s["potential"], "confidence": s["confidence"], "recommendation": s["recommendation"],
            "flags": [f.split(":")[0] for f in s["flags"]],
            "project_fit": ("Matches the airport's screened needs" if fit else
                            "Not among the screened needs for this airport") + f": {needs['suggested_projects']}",
            "financial_assumptions": {"construction_years": FIN["construction_years"],
                                      "operating_years": FIN["operating_years"],
                                      "discount_rate": FIN["discount_rate"],
                                      "ramp": "Added traffic ramps up linearly over 10 operating years"}}

    if project in ("terminal_expansion", "new_terminal"):
        f = capex_feasibility(code, added_annual_pax)
        capex = f["capex_estimate_usd_millions"]
        full_rev = f["added_operating_revenue_usd_millions_per_year"] * 1e6
        scen = {
            "conservative": _cash_flows(capex["high"] * 1e6, full_rev, 1 - MARGIN_RANGE[0], 15),
            "base": _cash_flows(capex["central"] * 1e6, full_rev, FIN["opex_to_revenue_base"], 10),
            "optimistic": _cash_flows(capex["low"] * 1e6, full_rev, 1 - MARGIN_RANGE[2], 7),
        }
        b = scen["base"]
        verdict = ("Positive NPV at 7% in the base case" if b["npv_m"] > 0 else
                   "Negative NPV at 7% in the base case: does not pay back on airport operating income alone")
        base.update({
            "executive_summary": {"investment_usd_m_base": capex["central"],
                                  "investment_range_usd_m": [capex["low"], capex["high"]],
                                  "added_passengers_per_year": f["scenario"]["added_annual_passengers"],
                                  "initial_conclusion": verdict},
            "financial_case_base": {"capex_usd_m": capex["central"],
                                    "added_revenue_usd_m_per_year_at_full_use": round(full_rev / 1e6, 1),
                                    "added_opex_usd_m_per_year_at_full_use":
                                        round(full_rev * FIN["opex_to_revenue_base"] / 1e6, 1),
                                    "npv_usd_m_at_7pct": b["npv_m"], "irr": b["irr"],
                                    "payback_years_from_construction_start": b["payback_years_from_start"],
                                    "cash_flow_sample_usd_m": b["cash_flow_sample"]},
            "scenarios": {k: {"capex_usd_m": c, "npv_usd_m": v["npv_m"], "irr": v["irr"],
                              "payback_years": v["payback_years_from_start"]}
                          for (k, v), c in zip(scen.items(), (capex["high"], capex["central"], capex["low"]))},
            "scenario_definitions": "Conservative: high capex, 75% opex ratio, 15-yr ramp. Base: central capex, "
                                    "63% opex ratio (CATS FY2025), 10-yr ramp. Optimistic: low capex, 55% opex, 7-yr ramp.",
            "key_assumptions": [
                {"assumption": "Capex per sq ft $1,500-$3,662 (4 US terminals)", "impact": "Largest driver: high vs "
                 "low cost changes NPV by more than the whole revenue case"},
                {"assumption": f"Revenue per enplanement ${f['revenue_per_enplanement_usd']} (CATS tier average)",
                 "impact": "Each ±10% moves base NPV roughly ±10% of the revenue side"},
                {"assumption": f"Opex = {FIN['opex_to_revenue_base']:.0%} of added revenue (CATS system ratio)",
                 "impact": "Incremental opex may be lower; base case is conservative here"},
                {"assumption": "Growth-based sizing and 10-year ramp", "impact": "Slower ramp delays payback"},
                {"assumption": "7% discount rate, 30 operating years, no terminal value", "impact": "Lower rate or "
                 "residual value improve NPV"}],
            "data_quality": {
                "official": ["Traffic and growth (BTS T-100)", "Delays (BTS On-Time)",
                             "Federal AIP grants awarded (FAA)", "Airport revenue and opex totals (FAA CATS FY2025)"],
                "estimated": ["Construction cost per sq ft (1 final cost, 3 budgets/estimates)",
                              "Terminal area per passenger", "Revenue and opex per passenger (tier averages)",
                              "Ramp-up, discount rate, construction period"],
                "missing": ["Airport-specific construction estimate", "Airport-specific revenue per passenger",
                            "Current terminal utilization (gates, peak-hour throughput)", "Financing structure"]},
            "estimate_drift_warning": "Early terminal estimates in the benchmark set rose 55%-171%.",
        })
    else:
        why = {"terminal_renovation": "no added capacity, so the revenue case depends on airport-specific "
                                      "non-aeronautical uplift and asset-condition data",
               "parking_expansion": "no public data on parking utilization, spaces or price per space",
               "runway_airfield": "ACRP Report 120 has a runway cost relationship but in FY2014 dollars, as a "
                                  "proof-of-concept, and runway benefits are delay savings and slots, not "
                                  "directly airport revenue"}[project]
        base.update({
            "executive_summary": {"investment_usd_m_base": None, "initial_conclusion":
                                  f"Financial case not estimable from public data: {why}."},
            "financial_case_base": None, "scenarios": None,
            "key_assumptions": [{"assumption": "Not modeled", "impact": why}],
            "data_quality": {"official": ["Traffic, growth and delay data for context"],
                             "estimated": [], "missing": ["Project cost", "Project revenue/benefit",
                                                          "Utilization of the asset"]},
        })
    base["validation_required"] = DATA_REQUIRED_BEFORE_INVESTMENT[:3] + (
        ["Runway/airspace capacity study (airside-constrained)"] if "AIRSIDE-CONSTRAINED" in base["flags"] else []) \
        + ["Deal structure (P3/concession/lease) and regulated rate base"]
    base["overall_confidence"] = "Low (screening only)"
    return base
