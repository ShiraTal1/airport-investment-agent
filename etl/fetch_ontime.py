"""Download BTS Airline On-Time Performance monthly files and aggregate per airport.

Produces additive totals (so months can be combined) plus the average daily busiest-hour
count of scheduled departures + arrivals.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "ontime_sample_months.csv"
COLS = ["FlightDate", "Origin", "Dest", "CRSDepTime", "CRSArrTime", "DepDelay", "DepDel15", "TaxiOut",
        "TaxiIn", "ArrDelay", "ArrDel15", "Cancelled", "NASDelay", "WeatherDelay"]
DEFAULT_MONTHS = [(2025, 9), (2025, 12), (2026, 3), (2026, 6)]


def load_month(y, m):
    r = requests.get(URL.format(y=y, m=m), timeout=1800)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    name = next(n for n in zf.namelist() if n.endswith(".csv"))
    return pd.read_csv(zf.open(name), usecols=COLS, low_memory=False)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.Cancelled != 1]
    dep = df.groupby("Origin").agg(dep=("Origin", "size"), dep_cxl=("Cancelled", "sum"))
    dep = dep.join(ok.groupby("Origin").agg(
        dep_del15=("DepDel15", "sum"), dep_delay_min=("DepDelay", lambda s: s.clip(lower=0).sum()),
        taxi_out=("TaxiOut", "sum"), taxi_out_n=("TaxiOut", "count")))
    arr = df.groupby("Dest").agg(arr=("Dest", "size"), arr_cxl=("Cancelled", "sum"))
    arr = arr.join(ok.groupby("Dest").agg(
        arr_del15=("ArrDel15", "sum"), arr_delay_min=("ArrDelay", lambda s: s.clip(lower=0).sum()),
        taxi_in=("TaxiIn", "sum"), taxi_in_n=("TaxiIn", "count"),
        nas_min=("NASDelay", lambda s: s[s > 0].sum()), nas_n=("NASDelay", lambda s: (s > 0).sum()),
        wx_min=("WeatherDelay", lambda s: s[s > 0].sum())))
    # Busiest scheduled hour per airport-day (departures by CRS dep hour + arrivals by CRS arr hour)
    d = pd.DataFrame({"a": df.Origin, "date": df.FlightDate, "h": df.CRSDepTime // 100})
    a_ = pd.DataFrame({"a": df.Dest, "date": df.FlightDate, "h": df.CRSArrTime // 100})
    hourly = pd.concat([d, a_]).groupby(["a", "date", "h"]).size()
    daily_peak = hourly.groupby(level=["a", "date"]).max()
    peak = daily_peak.groupby(level="a").agg(peak_sum="sum", days="size")
    out = dep.join(arr, how="outer").join(peak, how="outer").fillna(0)
    out.index.name = "airport"
    return out


def main(months=DEFAULT_MONTHS):
    total = None
    for y, m in months:
        print(f"On-time {y}-{m:02d} ...", file=sys.stderr)
        agg = aggregate(load_month(y, m))
        total = agg if total is None else total.add(agg, fill_value=0)
    total.reset_index().to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
