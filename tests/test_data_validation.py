"""Data validation: is the snapshot complete, internally consistent, and close to published figures?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import analytics as A  # noqa: E402
from agent import data  # noqa: E402

# Airport-reported 2025 total passengers (enplaned + deplaned), as published in public airport
# rankings (see DESIGN.md, Section 3b). T-100 counts only commercial air-carrier segments, so it runs
# slightly below airport-reported totals; we accept up to 6% difference.
PUBLISHED_2025_TOTAL_PAX = {
    "ATL": 106_302_208, "DFW": 85_660_127, "ORD": 84_856_018, "DEN": 82_427_962, "LAX": 73_709_594,
    "JFK": 62_629_455, "MCO": 57_675_573, "MIA": 55_314_661, "LAS": 55_030_375, "SFO": 54_532_613,
    "BOS": 42_260_667,
}


def test_all_years_present_and_complete():
    ay = data.airport_years()
    assert set(ay.year) == {2019, 2023, 2024, 2025, 2026}
    full = ay[(ay.year == 2025) & (ay.pax_out > 5_000_000)]
    assert (full.months == 12).all()                 # full-year data for big airports
    assert (ay[ay.year == 2026].months <= 7).all()   # partial year as documented


def test_values_are_physically_plausible():
    m = A.metrics_table()
    big = m[m.pax_enplaned >= 1_000_000]
    assert len(big) >= 80
    assert big.load_factor.between(0.5, 1.0).all()
    assert big.dep_delay15_rate.dropna().between(0.05, 0.6).all()
    assert big.avg_taxi_out_min.dropna().between(5, 45).all()
    assert (m.pax_enplaned >= 0).all()


def test_enplaned_roughly_equals_deplaned():
    m = A.metrics_table()
    big = m[m.pax_enplaned >= 1_000_000]
    ratio = big.pax_enplaned / (big.pax_total - big.pax_enplaned)
    assert ratio.between(0.9, 1.1).all()


def test_matches_published_airport_totals():
    m = A.metrics_table()
    for code, published in PUBLISHED_2025_TOTAL_PAX.items():
        diff = m.at[code, "pax_total"] / published - 1
        assert abs(diff) < 0.06, (code, diff)


def test_route_file_covers_airport_traffic():
    for code in ["SFO", "BOS", "ANC", "DEN"]:
        assert A.route_mix(code)["route_data_coverage_pct"] > 90


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
