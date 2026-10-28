"""Download BTS T-100 Segment (All Carriers) and aggregate to airport-year and route level.

BTS has no REST API for T-100; the public download form (an ASP.NET page) is posted
programmatically, which is the same thing the website's Download button does.
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://www.transtats.bts.gov"
FORM = BASE + "/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4%20Pn44vr45"
LOOKUP = BASE + "/Download_Lookup.asp?Y11x72=Y_NVecbeg"
FIELDS = ["DEPARTURES_PERFORMED", "SEATS", "PASSENGERS", "FREIGHT", "DISTANCE", "ORIGIN",
          "ORIGIN_STATE_ABR", "ORIGIN_COUNTRY", "DEST", "DEST_COUNTRY", "MONTH", "CLASS"]
OUT = Path(__file__).resolve().parent.parent / "data"
BINS = [0, 500, 1000, 2000, 3000, 4000, np.inf]
BIN_NAMES = ["lt500", "500_999", "1000_1999", "2000_2999", "3000_3999", "4000p"]


def download_year(year: int, session: requests.Session) -> pd.DataFrame:
    html = session.get(FORM, timeout=120).text
    form = {}
    for name, value in re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', html):
        form[name] = value
    for f in FIELDS:
        form[f] = "on"
    form.update({"cboGeography": "All", "cboYear": str(year), "cboPeriod": "All",
                 "chkDownloadZip": "on", "btnDownload": "Download"})
    resp = session.post(FORM, data=form, timeout=1800)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    name = next(n for n in zf.namelist() if n.startswith("T_T100"))
    return pd.read_csv(zf.open(name))


def aggregate(df: pd.DataFrame, year: int):
    us_out = df[df.ORIGIN_COUNTRY == "US"].copy()
    pax_fl = us_out[us_out.SEATS > 0]
    g = us_out.groupby("ORIGIN")
    a = pd.DataFrame({
        "state": g.ORIGIN_STATE_ABR.last(), "months": g.MONTH.nunique(),
        "dep_all": g.DEPARTURES_PERFORMED.sum(), "freight_out_lbs": g.FREIGHT.sum(),
        "pax_out": g.PASSENGERS.sum(),
        "pax_out_jan_jul": us_out[us_out.MONTH <= 7].groupby("ORIGIN").PASSENGERS.sum(),
    })
    gp = pax_fl.groupby("ORIGIN")
    a["dep_pax"] = gp.DEPARTURES_PERFORMED.sum()
    a["seats"] = gp.SEATS.sum()
    a["seats_jan_jul"] = pax_fl[pax_fl.MONTH <= 7].groupby("ORIGIN").SEATS.sum()
    a["dep_cargo_only"] = us_out[(us_out.SEATS == 0)].groupby("ORIGIN").DEPARTURES_PERFORMED.sum()
    intl = pax_fl[pax_fl.DEST_COUNTRY != "US"].groupby("ORIGIN")
    a["intl_dep_pax"], a["intl_seats"], a["intl_pax_out"] = (intl.DEPARTURES_PERFORMED.sum(),
                                                             intl.SEATS.sum(), intl.PASSENGERS.sum())
    pax_fl = pax_fl.assign(band=pd.cut(pax_fl.DISTANCE, BINS, right=False, labels=BIN_NAMES))
    for col, src in (("dep", "DEPARTURES_PERFORMED"), ("seats", "SEATS")):
        piv = pax_fl.pivot_table(index="ORIGIN", columns="band", values=src, aggfunc="sum", observed=False)
        for b in BIN_NAMES:
            a[f"{col}_{b}"] = piv[b] if b in piv else 0
    arr = df[df.DEST_COUNTRY == "US"].groupby("DEST")
    a["pax_in"], a["freight_in_lbs"] = arr.PASSENGERS.sum(), arr.FREIGHT.sum()
    r = us_out.groupby(["ORIGIN", "DEST", "DEST_COUNTRY"]).agg(
        distance=("DISTANCE", "max"), departures=("DEPARTURES_PERFORMED", "sum"),
        seats=("SEATS", "sum"), passengers=("PASSENGERS", "sum"), freight_lbs=("FREIGHT", "sum")).reset_index()
    r["dep_pax_flights"] = us_out[us_out.SEATS > 0].groupby(["ORIGIN", "DEST", "DEST_COUNTRY"]) \
        .DEPARTURES_PERFORMED.sum().reindex(pd.MultiIndex.from_frame(r[["ORIGIN", "DEST", "DEST_COUNTRY"]])).fillna(0).values
    r["dep_cargo_only"] = r.departures - r.dep_pax_flights
    hi = r[(r.seats >= 10000) & (r.passengers / r.seats.replace(0, np.nan) >= 0.85)]
    a["seats_on_hi_lf_routes"] = hi.groupby("ORIGIN").seats.sum()
    a["pax_routes"] = r[r.seats > 0].groupby("ORIGIN").size()
    a = a.fillna(0)
    a = a[a.state.astype(str).str.len() > 0]
    a = a[((a.pax_out + a.pax_in) >= 20000) | (a.freight_out_lbs >= 5e6)]
    a.insert(0, "year", year)
    a.index.name = "airport"
    r = r[r.ORIGIN.isin(a.index) & ((r.passengers >= 3000) | (r.freight_lbs >= 1e6))]
    r = r.rename(columns={"ORIGIN": "origin", "DEST": "dest", "DEST_COUNTRY": "dest_country"})
    r.insert(4, "year", year)
    return a.reset_index(), r


def main(years=(2019, 2023, 2024, 2025, 2026)):
    s = requests.Session()
    frames, routes = [], None
    for y in years:
        print(f"Downloading T-100 {y} ...", file=sys.stderr)
        a, r = aggregate(download_year(y, s), y)
        frames.append(a)
        if y == 2025:
            routes = r
    pd.concat(frames).to_csv(OUT / "t100_airport_year.csv", index=False)
    if routes is not None:
        routes.to_csv(OUT / "t100_routes_2025.csv", index=False)
    lk = pd.read_csv(io.StringIO(s.get(LOOKUP, timeout=120).text))
    keep = set(pd.concat(frames).airport)
    lk = lk[lk.Code.isin(keep)]
    parts = lk.Description.str.extract(r"^(?P<city>[^,]+), (?P<st>[A-Z]{2}): (?P<name>.+)$")
    pd.DataFrame({"code": lk.Code, "city": parts.city, "name": parts.name}).to_csv(
        OUT / "airport_names.csv", index=False)


if __name__ == "__main__":
    main()
