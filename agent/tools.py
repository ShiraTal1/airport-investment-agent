"""Tool definitions exposed to the LLM, and the dispatcher that runs them.

Each tool is a thin wrapper over a deterministic function in analytics.py / data.py.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import analytics as A
from . import data

WEIGHTS_SCHEMA = {
    "type": "object",
    "description": "Optional custom weights for the opportunity score (re-normalized to sum to 1). "
                   "Use only if the user asks to change priorities.",
    "properties": {k: {"type": "number"} for k in A.DEFAULT_WEIGHTS},
}

TOOLS = [
    {
        "name": "search_airports",
        "description": "Find airport codes by code, city or airport name. Use to resolve ambiguous names "
                       "(e.g. 'Santa Ana' -> SNA) before calling other tools.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "rank_airports",
        "description": "Rank airports by the deterministic Terminal Expansion Opportunity Score (0-100) within a "
                       "region (Census divisions like 'New England', 'Pacific', 'Mountain', or 'Northeast', "
                       "'Midwest', 'South', 'West', 'West Coast', 'Contiguous US', 'US'; metro groups like "
                       "'los angeles area', 'new york area'), a list of states, or a list of airport codes. "
                       "Returns component scores, key metrics, confidence and flags.",
        "input_schema": {"type": "object", "properties": {
            "region": {"type": "string"},
            "states": {"type": "array", "items": {"type": "string"}, "description": "2-letter state codes"},
            "codes": {"type": "array", "items": {"type": "string"}},
            "top_n": {"type": "integer", "default": 10},
            "min_enplaned": {"type": "integer", "default": 250000,
                             "description": "Exclude airports with fewer annual enplaned passengers"},
            "weights": WEIGHTS_SCHEMA,
        }},
    },
    {
        "name": "airport_profile",
        "description": "Full KPI profile for one airport: traffic, growth, load factor, delays, taxi times, "
                       "peer percentiles, opportunity score with components, confidence and flags.",
        "input_schema": {"type": "object", "properties": {"code": {"type": "string"}, "weights": WEIGHTS_SCHEMA},
                         "required": ["code"]},
    },
    {
        "name": "compare_airports",
        "description": "Side-by-side comparison of 2-6 airports. focus='congestion' adds a deterministic "
                       "congestion index; focus='demand' shows growth/capacity metrics; 'all' shows everything.",
        "input_schema": {"type": "object", "properties": {
            "codes": {"type": "array", "items": {"type": "string"}},
            "focus": {"type": "string", "enum": ["congestion", "demand", "all"], "default": "all"},
            "weights": WEIGHTS_SCHEMA}, "required": ["codes"]},
    },
    {
        "name": "route_mix",
        "description": "Share of long-haul flights from an airport, computed from 2025 route-level data, with "
                       "sensitivity to the threshold and top long-haul destinations. Also reports cargo share.",
        "input_schema": {"type": "object", "properties": {
            "code": {"type": "string"},
            "long_haul_miles": {"type": "number", "default": 3000},
            "basis": {"type": "string", "enum": ["departures", "seats", "passengers"], "default": "departures"},
            "include_cargo": {"type": "boolean", "default": False}}, "required": ["code"]},
    },
    {
        "name": "demand_pressure",
        "description": "Unmet-demand analysis for one airport: proxies (load factor, demand vs seat growth, delays), "
                       "a heuristic estimate of spilled passengers with a range, and the tightest routes.",
        "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
    {
        "name": "live_faa_status",
        "description": "Live FAA NAS status right now (ground stops, delay programs, closures). Optionally filter "
                       "to airport codes. Use for 'right now/today' questions or to add current context.",
        "input_schema": {"type": "object", "properties": {
            "codes": {"type": "array", "items": {"type": "string"}}}},
    },
    {
        "name": "methodology",
        "description": "Scoring methodology, weights, data sources, vintages and key assumptions.",
        "input_schema": {"type": "object", "properties": {"weights": WEIGHTS_SCHEMA}},
    },
]


def _clean(obj):
    """Make results JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) else round(float(obj), 4)
    if obj is pd.NA:
        return None
    return obj


def _resolve(code: str, assumptions: list) -> str | None:
    res = A.resolve_airport(code)
    if res["assumption"]:
        assumptions.append(res["assumption"])
    return res["code"]


def run_tool(name: str, args: dict) -> dict:
    assumptions: list[str] = []
    try:
        if name == "search_airports":
            out = {"matches": A.search_airports(args["query"])}
        elif name == "rank_airports":
            codes = [c for c in (_resolve(c, assumptions) for c in args.get("codes") or []) if c] or None
            out = A.rank_airports(region=args.get("region"), states=args.get("states"), codes=codes,
                                  top_n=int(args.get("top_n", 10)),
                                  min_enplaned=int(args.get("min_enplaned", 250_000)),
                                  weights=args.get("weights"))
        elif name == "airport_profile":
            code = _resolve(args["code"], assumptions)
            out = A.airport_profile(code, args.get("weights")) if code else {"error": "airport not found"}
        elif name == "compare_airports":
            codes = [c for c in (_resolve(c, assumptions) for c in args["codes"]) if c]
            out = A.compare_airports(codes, args.get("focus", "all"), args.get("weights"))
        elif name == "route_mix":
            code = _resolve(args["code"], assumptions)
            out = A.route_mix(code, float(args.get("long_haul_miles", 3000)), args.get("basis", "departures"),
                              bool(args.get("include_cargo", False))) if code else {"error": "airport not found"}
        elif name == "demand_pressure":
            code = _resolve(args["code"], assumptions)
            out = A.demand_pressure(code) if code else {"error": "airport not found"}
        elif name == "live_faa_status":
            live = data.live_faa_status()
            codes = {c.upper() for c in (args.get("codes") or [])}
            if codes and live.get("ok"):
                live["events"] = [e for e in live["events"] if e["airport"] in codes]
                live["note"] = ("No active FAA delay programs, ground stops or closures listed for "
                                f"{', '.join(sorted(codes))} right now.") if not live["events"] else None
            out = live
        elif name == "methodology":
            out = A.methodology(args.get("weights"))
        else:
            out = {"error": f"unknown tool {name}"}
    except Exception as exc:  # surface errors to the LLM rather than crashing the chat
        out = {"error": f"{type(exc).__name__}: {exc}"}
    if assumptions:
        out = {**out, "assumptions_made": assumptions}
    return _clean(out)


def run_tool_json(name: str, args: dict) -> str:
    return json.dumps(run_tool(name, args), ensure_ascii=False)
