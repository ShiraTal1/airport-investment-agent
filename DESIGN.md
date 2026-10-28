# Airport Investment Intelligence Agent — Design Note

## 1. Problem framing
The firm invests in airport modernization and wants to know **where added terminal/flight capacity will be
used**. "Most profitable" is not directly measurable with public data (no construction costs, lease terms,
concession revenue or financing). So I reframed the question into what *can* be measured:

> **Where is passenger demand growing, and where is existing capacity already under pressure?**

An airport with strong growth, full planes and congested operations is where new capacity is most likely to
fill quickly, which is the main driver of returns on a capacity project. The agent is explicit that its score
ranks *capacity need*, not project IRR.

**Approach: Define → Measure → Rank → Explain → Validate**

| Step | What it means here |
|---|---|
| **Define** | An attractive opportunity = growing demand + capacity already under pressure (Section 4) |
| **Measure** | Public BTS/FAA data turned into KPIs for demand, activity and capacity pressure (Section 3) |
| **Rank** | A pre-defined, deterministic formula; same data → same ranking (Section 4) |
| **Explain** | Why an airport ranks high, which data supports it, and what is still unknown (Section 5) |
| **Validate** | Automated checks against the source data now; analysts and real financials in a pilot (Section 9) |

The goal is not to build an agent for its own sake, but to make an investment screening decision faster,
more consistent and easier to audit.

## 2. Architecture

```
 Analyst ──► Streamlit chat (text / optional voice)
                 │
                 ▼
        Claude (tool-use loop)  ── plans, resolves names, explains, handles follow-ups
                 │  calls tools (JSON)
                 ▼
        Deterministic tools (Python)  ── rank / rank_by_metric / profile / compare / route_mix / route_lookup / demand_pressure / live status
                 │
        ┌────────┴──────────────┐
        ▼                       ▼
  Snapshot data (CSV)     FAA NAS Status API (live)
  built by etl/ from BTS
```

| Layer | File | Responsibility |
|---|---|---|
| Data | `etl/`, `agent/data.py` | Pull public BTS data, aggregate to small CSVs; live FAA client |
| Analytics | `agent/analytics.py` | All metrics, percentiles, scores, flags, confidence. **No LLM.** |
| Tools | `agent/tools.py` | JSON schemas the LLM can call; name resolution; error handling |
| Agent | `agent/llm_agent.py` | System prompt with guardrails, tool loop, conversation memory |
| UI | `app.py` | Chat + "How I got this" trace, no-LLM leaderboard, methodology, weight sliders |

## 3. Data (all public)
| Source | Used for |
|---|---|
| **BTS T-100 Segment, all carriers** (US + foreign, domestic + international), 2019, 2023-2025, Jan-Jul 2026 | Passengers, seats, departures, load factor, growth, route distances, cargo share |
| **BTS Airline On-Time Performance** (reporting carriers), sample months Dec 2025 + Jun 2026 | Delay rate, taxi-out/in, NAS (airspace/airport) delay, busiest-hour scheduled ops |
| **BTS airport lookup** | Names/cities |
| **FAA NAS Status API** (live) | Current ground stops, delay programs, closures |
| **FAA AIP grant histories** FY2021-2025 (Excel, 5 files) | Federal grants awarded per airport, incl. terminal-related items |
| **FAA CATS (Form 5100-127)** FY2025 aggregates | Airport operating revenue and margin, for revenue per enplanement |
| **4 recent US terminal projects** (BUR, MCI, PIT, FLL; public reporting) | Construction cost per sq ft and terminal size per passenger |
| **ACRP Report 120** (TRB, 2014) | Cost-estimating method and uncertainty guidance (see 4b) |

## 3b. Data pipeline and validation
**How data is retrieved and processed** (`etl/`): (1) T-100 Segment is pulled per year from the BTS TranStats
download endpoint (the same form the website's Download button posts); (2) On-Time Performance monthly files
come from the BTS PREZIP directory; (3) rows are aggregated to one row per airport-year (passengers, seats,
departures by distance band, cargo-only flights, international share) and one row per route (2025); (4) the
small CSVs in `data/` are what the app reads. Live FAA status is fetched at question time and parsed from XML.

**How the data was checked** (`tests/test_data_validation.py`, runs automatically):

| Check | Result |
|---|---|
| Completeness | All 5 years present; every airport with >5M passengers has 12 months for 2025; 2026 = Jan-Jul |
| Plausibility | Load factor 50-100%, delay rate 5-60%, taxi-out 5-45 min for all ~90 peer airports |
| Internal consistency | Enplaned ≈ deplaned (within ±10%) at every peer airport; route file covers >90% of traffic |
| **External cross-check** | 2025 total passengers vs airport-published figures for the 11 busiest US airports: within **0.2%-4%** (e.g. BOS 42.34M vs 42.26M published, LAX 73.40M vs 73.71M). T-100 runs slightly low because it counts only commercial carrier segments. The order matches for the top 7; places 8-10 (MIA, LAS, SFO) are within 0.6% of each other. |

## 4. Scoring methodology (deterministic)
**Terminal Expansion Opportunity Score (0-100)** = weighted average of four components:

| Component (weight) | Sub-metrics (weight within component) | Why it matters |
|---|---|---|
| **Demand growth (35%)** | Passenger CAGR 2023→2025 (40%), Jan-Jul 2026 vs 2025 growth (40%), 2025 vs 2019 recovery (20%) | Growth is what fills new capacity; mixes medium-term trend and current momentum |
| **Capacity pressure (30%)** | Load factor (50%), share of seats on routes with ≥85% load factor (30%), passenger growth minus seat growth (20%) | Full planes + demand outpacing supply = existing capacity is binding |
| **Congestion (20%)** | % departures delayed 15+ min (30%), avg taxi-out (30%), busiest-hour ops (20%), NAS delay minutes per arrival (20%) | Operational strain on the facility |
| **Scale (15%)** | Total passengers (log) | Larger base = larger absolute return on a given % improvement |

**Explicit formula:**

```
Opportunity Score = 0.35·G + 0.30·C + 0.20·K + 0.15·S            (0-100)

G (demand growth)     = 0.4·P(CAGR 2023-25) + 0.4·P(growth Jan-Jul 2026 vs 2025) + 0.2·P(2025 / 2019)
C (capacity pressure) = 0.5·P(load factor) + 0.3·P(share of seats on routes with LF ≥ 85%)
                        + 0.2·P(passenger growth − seat growth)
K (congestion)        = 0.3·P(% departures delayed 15+ min) + 0.3·P(avg taxi-out)
                        + 0.2·P(busiest-hour ops) + 0.2·P(NAS delay min per arrival)
S (scale)             = P(log10 total passengers)

P(x) = percentile of x among US airports with ≥ 1M enplaned passengers (0-100)
```

**Normalization:** each sub-metric is converted to a **percentile vs. a national peer group** (all US airports
with ≥1M enplaned passengers, ~90 airports). Using a national reference (rather than ranking within the
region) keeps scores comparable: a "70" in New England means the same as a "70" in Texas.

**Missing data:** within a component, weights are re-normalized over available sub-metrics; a component with
no data at all (e.g. congestion for airports with few on-time-reporting flights) is set to a **neutral 50**, so
missing data neither helps nor hurts the rank, and the **confidence** label drops (High / Medium / Low).

**Flags (rule-based):**
- *Airside-constrained*: NAS delay and taxi-out both in the top quartile → growth may be capped by
  runways/airspace, so terminal capacity alone may not unlock it.
- *Cargo-heavy*: ≥30% of departures are cargo-only (e.g. ANC) → passenger metrics understate activity.
- *Below peer size*, *Not recovered vs 2019*, *Limited delay data*.

**Unmet demand** (not directly observable) is reported as proxies plus a transparent heuristic: route-level
"spill" rises from 0% at ≤80% annual load factor to 15% at ≥95%, summed over routes, shown with a ±50% range
and labeled low confidence. Spilled passengers are converted to **flights** using each route's average
passengers per flight (the extra same-size flights airlines would need).

Weights are adjustable (sidebar sliders or by asking the agent, e.g. "weight growth twice as much").

**From score to recommendation.** The score measures *initial potential*, not proven profitability, so it is
translated into a decision-oriented label, again with fixed rules:

| Potential tier | Rule | Combined with confidence → recommendation |
|---|---|---|
| **High potential** | score ≥ 65 | High confidence → *Prioritize for due diligence*; otherwise → *Investigate further (data gaps)* |
| **Medium potential** | 50-65 | High confidence → *Investigate further*; otherwise → *Monitor* |
| **Low potential** | < 50 | *Deprioritize* |

A high score on partial data never becomes an automatic "go". Airside-constrained airports also get
*"check runway/airspace capacity first"*. Each airport carries a deterministic **drivers** line (components
above the 70th / below the 40th percentile), so the explanation is grounded in the actual metrics:
e.g. *BOS: strengths scale (86), capacity pressure (78), congestion (76); flagged airside-constrained*.

**Data required before an investment decision** (shown after every ranking; not available publicly):

1. Terminal throughput and utilization (passengers per gate, peak-hour terminal capacity)
2. Construction and operating cost of the proposed expansion
3. Expected incremental passenger volume (airline commitments, official FAA forecasts)
4. Revenue per incremental passenger (aeronautical fees, concessions, parking) and lease terms
5. Regulatory, environmental and physical constraints (land, runways, airspace, community caps)

## 4b. Renovation cost and return (screening level)
The score says *where* capacity is needed. A separate tool, `capex_feasibility`, gives a first view of *what it
might cost and earn*, always as ranges with labeled sources:

```
Added capacity  X  = total passengers × ((1 + g)^10 − 1),  g = 2023-25 CAGR bounded to 1%-5%  (or user input)
Terminal area      = X × {0.05, 0.07, 0.09} sq ft per annual passenger
Capex              = area × {$1,500 min, $2,850 median, $3,662 max} per sq ft
Added revenue/yr   = (X / 2 enplanements) × revenue per enplanement (FAA CATS FY2025, by hub tier)
Added op. income   = added revenue × {25%, 37%, 45%} margin  (37% = CATS FY2025 system margin)
Simple payback     = capex / added operating income   (optimistic / central / pessimistic)
```

**What the requested sources do and do not provide, and how confident each figure is:**

| Source | What it is | Final cost or estimate? | How it is used |
|---|---|---|---|
| **ACRP Report 120** (2014) + Web-Only Doc. 18 | Guidebook and spreadsheet model for airport capital cost estimating | Model built on ~183 historical projects (after outliers), normalized to FY2014 dollars; the authors call it a proof-of-concept. Its data covers taxiways, aprons, runways, fencing, PAPI, weather equipment and ARFF/SRE buildings, **not terminals** | Method only: ranges instead of points, explicit uncertainty, escalation. Its unit costs are not used for terminals |
| **FAA AIP grant histories** | Every AIP grant FY2021-2025 by airport (e.g. SFO $608M total, $134M terminal-related) | **Grant amounts awarded** (federal share at award date). Neither the total project cost nor the final cost. Example: BUR shows ~$81M of terminal-related AIP grants vs a ~$1.3B terminal | Context on federal funding already received; not a cost estimate |
| **Terminal benchmarks** | BUR, MCI, PIT, FLL | Only **BUR** was reported at completion (on time and on budget). MCI = budget cap, PIT and FLL = latest estimates | $/sq ft range and area per passenger |
| **FAA CATS FY2025** | Reported airport financials (417 airports) | Actual reported revenue, but used here as tier averages | Revenue per enplanement: ~$34.7 large hubs, ~$36.6 others |

**Estimates tend to grow.** First public estimates in the benchmark set rose a lot: MCI $964M (2016) → $1.5B (+56%),
PIT $1.1B (2019) → $1.7B (+55%), FLL Terminal 5 $306M (2021) → ~$829M (+171%, completion slipped to ~2030). The
tool reports this drift with every answer.

**Result and limitation.** Example: PWM, sized for 10 years of 5% growth (+1.6M passengers/yr), comes out at
$121M-$531M (central $322M) for ~$11M/yr of added operating income, a payback of ~9-72 years (central ~30). Because
unit cost and revenue per passenger are national benchmarks, **payback barely differs between airports**; the
differences come from scale, growth and risk flags. Overall confidence: **Low, screening only**. The first
upgrades are airport-specific CATS revenue and cost estimates from the airport's own capital plan.

## 5. Where and how AI is used
| Task | LLM? | Notes |
|---|---|---|
| Understand the question, pick tools & parameters | Yes | Claude tool use |
| Resolve names ("LA", "Santa Ana") | Partly | Deterministic alias table + search tool; LLM states the assumption |
| Compute any metric, score or ranking | No | Python only; reproducible and unit-tested |
| Explain results, connect metrics into a narrative | Yes | Must cite tool numbers; system prompt forbids invented figures |
| Domain context (e.g. SNA's community noise/access caps, SFO's closely spaced runways and fog) | Yes | Must be labeled "general knowledge, not from the dataset" |
| Follow-ups, re-weighting | Yes | Conversation memory + re-calling tools with new weights |
| Investment takeaway in every answer | Yes | Ties each result back to "is this a promising opportunity?": tier, confidence, recommendation, cost/return range for top candidates, next step. Never a final invest/don't-invest call |

Every answer shows a **"How I got this"** panel with the exact tool calls and raw outputs, so an analyst can
audit the reasoning.

## 5b. What the model says about the sample questions (deterministic outputs)
| Question | Key result | Why it is interesting |
|---|---|---|
| New England candidates | BOS 67.2, BGR 61.7 (Medium conf.), PWM 56.6, BTV 53.6, BDL 46.3, PVD 43.6 | BOS leads on pressure/congestion but is flagged airside-constrained; PWM is a mid-size airport with 82% load factor and 7.5%/yr growth |
| LAX vs SNA congestion | Congestion index 65.5 vs 47.0; delay rate ~equal (24.5% vs 25.1%) but LAX has 4x busiest-hour ops (91 vs 22) and longer taxi-out (18.5 vs 15.9 min) | Same delay rate, very different physical congestion; SNA's limits are largely regulatory (labeled as general knowledge) |
| ANC long-haul share | 4.9% of passenger departures ≥3,000 mi; 32.4% including cargo; 54% of departures are cargo-only | The answer depends on definitions; the agent shows the sensitivity instead of one number |
| SFO unmet demand | ~0.95M spilled passengers/yr (0.5-1.4M) ≈ ~6,000 flights/yr (~16 per day), low confidence; load factor 82% (90th pct), NAS delay 14.2 min/arrival (highest among peers), flat seat growth | Demand exists but airside limits cap supply, so terminal capacity alone may not unlock it |

## 6. Key tradeoffs
1. **Snapshot vs. live data.** Historical data is pre-aggregated into small CSVs (fast, reproducible,
   testable, works offline; BTS downloads take minutes). Live FAA status is fetched per question. Cost: the
   snapshot must be refreshed with `etl/`.
2. **Percentiles vs. raw values / z-scores.** Percentiles are robust to outliers (e.g. JFK's size) and easy to
   explain, but lose magnitude information. Raw values are always shown next to scores.
3. **Proxies vs. ground truth.** No gate counts, terminal square footage or cost data publicly at scale. The
   score finds capacity *pressure*; a real investment case needs a project-level study.
4. **Sampled delay data.** Two months (winter + summer) instead of 12 to fit the one-day build; seasonal bias
   is reduced but not removed. The ETL script supports any number of months.
5. **LLM flexibility vs. control.** The LLM can only act through 11 typed tools, which limits creativity but
   prevents hallucinated numbers and keeps answers consistent between runs.
6. **Simple, explainable weights vs. fitted model.** Weights are expert judgment, not learned (no labeled
   "successful expansion" data). With the firm's past deal outcomes, weights could be calibrated.

## 7. Assumptions & limitations
- Enplanements approximated by T-100 departing segment passengers (includes connections, like FAA counts).
- Delay metrics cover only on-time reporting carriers (large US airlines + major regionals).
- "Long-haul" defaults to ≥3,000 statute miles; the tool reports sensitivity at 2,000/2,500/3,000/4,000.
- 2026 is partial (Jan-Jul); YTD growth compares like-for-like months.
- Renovation cost and return are screening-level ranges from national benchmarks, not airport-specific estimates.
- No regulatory, environmental, community or funding constraints are modeled (the LLM may mention them as
  labeled context).

## 8. Success metrics for a pilot
| Metric | What it checks | How it is measured |
|---|---|---|
| **Data accuracy** | Do the numbers match the source? | Spot-check tool outputs against BTS/FAA source tables; unit tests on every metric |
| **Ranking consistency** | Does the same data always produce the same ranking? | Deterministic scoring + automated test (same input → identical ranking) |
| **Analyst usefulness** | Can analysts understand the ranking and use it for the next step? | Analyst review of answers against the eval rubric; share of shortlisted airports that move to deep-dive; time saved per screening |

These are deliberately stricter than "does the model give a convincing answer".

## 9. With more time / path to production
- Add FAA Terminal Area Forecast (official 20-year projections) and FAA ACAIS enplanements.
- Add facility data (gates, terminal sq ft) from airport master plans → passengers per gate as a true
  terminal-capacity metric.
- Calibrate weights with the firm's analysts and historical deals; add backtesting (did high-score airports
  in 2015 expand and grow?).
- Scheduled refresh of the snapshot; caching and observability (log every tool call and answer).
- Evaluation harness: `eval/questions.md` rubric run automatically with an LLM grader.
