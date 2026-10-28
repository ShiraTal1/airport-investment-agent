# ETL: refreshing the data snapshot

The agent answers from a snapshot in `data/` built from public BTS TranStats downloads:

| File | Source | Script |
|---|---|---|
| `t100_airport_year.csv` | BTS T-100 Segment (All Carriers), years 2019, 2023, 2024, 2025, 2026 (Jan-Jul) | `fetch_t100.py` |
| `t100_routes_2025.csv` | same, route level, 2025 | `fetch_t100.py` |
| `ontime_sample_months.csv` | BTS Airline On-Time Performance (PREZIP monthly files), sample months | `fetch_ontime.py` |
| `airport_names.csv` | BTS L_AIRPORT lookup table | `fetch_t100.py` |

Run: `python -m etl.fetch_t100` then `python -m etl.fetch_ontime` (needs internet; ~10-40 min,
BTS servers are slow). The snapshot shipped in the repo was produced with the same
aggregation logic (first run in a browser session because the build sandbox had no direct
internet access; the Python scripts are the reproducible version).

Live data (FAA NAS Status API) is fetched at question time by `agent/data.py`.
