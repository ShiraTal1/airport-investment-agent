"""Data access layer.

Two kinds of data:
1. Historical snapshot (data/*.csv) built by the ETL scripts in etl/ from public
   BTS TranStats downloads (T-100 Segment, Airline On-Time Performance, airport lookup).
   Snapshotting makes answers fast, reproducible and testable.
2. Live data from the FAA NAS Status public API, fetched at question time.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FAA_NAS_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"


@lru_cache(maxsize=1)
def airport_years() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "t100_airport_year.csv")


@lru_cache(maxsize=1)
def routes() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "t100_routes_2025.csv")


@lru_cache(maxsize=1)
def ontime() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ontime_sample_months.csv")


@lru_cache(maxsize=1)
def airport_names() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "airport_names.csv", keep_default_na=False)


@lru_cache(maxsize=1)
def metadata() -> dict:
    path = DATA_DIR / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def live_faa_status(timeout: float = 8.0) -> dict:
    """Fetch current FAA airport status (delays, ground stops, closures).

    Returns {"ok": bool, "update_time": str, "events": [ {airport, type, detail} ]}.
    Never raises: on failure returns ok=False with the error, so the agent can say
    "live data unavailable" instead of crashing.
    """
    try:
        resp = requests.get(FAA_NAS_STATUS_URL, timeout=timeout)
        resp.raise_for_status()
        return parse_faa_status(resp.content)
    except Exception as exc:  # network, parse, HTTP errors
        return {"ok": False, "error": f"Live FAA data unavailable: {exc}", "events": []}


def parse_faa_status(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)
    events = []
    for delay_type in root.findall("Delay_type"):
        type_name = (delay_type.findtext("Name") or "").strip()
        # Each delay type nests a list element with repeated children; walk generically.
        for node in delay_type.iter():
            arpt = node.findtext("ARPT")
            if not arpt or node.tag in ("Delay_type",):
                continue
            detail = {child.tag: (child.text or "").strip()
                      for child in node if child.tag != "ARPT" and (child.text or "").strip()}
            # Arrival/departure delay blocks nest one more level
            for sub in node:
                if len(sub):
                    for leaf in sub.iter():
                        if leaf is not sub and (leaf.text or "").strip():
                            detail[f"{sub.tag}.{leaf.tag}"] = leaf.text.strip()
            events.append({"airport": arpt.strip(), "type": type_name, "detail": detail})
    # De-duplicate (iter() can visit the same airport node via nested lists)
    seen, unique = set(), []
    for ev in events:
        key = (ev["airport"], ev["type"], json.dumps(ev["detail"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(ev)
    return {"ok": True, "update_time": root.findtext("Update_Time"), "events": unique}
