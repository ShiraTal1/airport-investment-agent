# Airport Investment Intelligence Agent — Design Note

**Live demo (chat interface):** [airport-agent-shira.streamlit.app](https://airport-agent-shira.streamlit.app)
· **Source code:** [github.com/ShiraTal1/airport-investment-agent](https://github.com/ShiraTal1/airport-investment-agent)

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
| **Validate** | Automated tests check the data (2025 passenger totals match airport-published figures within 0.2%-4%); every answer shows the exact calculations behind it, so it can be checked in seconds (Section 5) |

The goal is not to build an agent for its own sake, but to make an investment screening decision faster,
more consistent and easier to audit.

## 2. Architecture

```
 (1) Analyst asks a question in the Streamlit chat
                │
                ▼
 (2) Claude (LLM) understands the question and chooses which tool to call
                │
                ▼
 (3) Scoring engine (Python, no LLM)            ◄── reads the data:
     computes metrics → percentiles →                BTS snapshot (CSV files)
     component scores → Opportunity Score →          + live FAA NAS Status API
     tier, flags, recommendation
                │  returns the numbers
                ▼
 (4) Claude turns the numbers into a written answer
     (it can only use the numbers it received)
                │
                ▼
 (5) Streamlit shows the answer, with a "How I got this" panel
     listing the exact tool calls and numbers
```

**Where the score is calculated:** only in step (3), in `agent/analytics.py`. The LLM never calculates or
changes a score: it chooses what to ask for (step 2) and turns the result into words (step 4).

| Layer | File | Responsibility |
|---|---|---|
| Data | `etl/`, `agent/data.py` | Pull public BTS data, aggregate to small CSVs; live FAA client |
| Analytics | `agent/analytics.py` | All metrics, percentiles, scores, flags, confidence. **No LLM.** |
| Tools | `agent/tools.py` | JSON schemas the LLM can call; name resolution; error handling |
| Agent | `agent/llm_agent.py` | System prompt with guardrails, tool loop, conversation memory |
| UI | `app.py` | Chat + "How I got this" trace, no-LLM leaderboard, what-if tab, methodology, weight sliders |

## 3. Data (all public)
| Source | Used for |
|---|---|
| **BTS T-100 Segment, all carriers**: the monthly traffic report every airline (US + foreign) files with the US Department of Transportation for each flight segment; 2019, 2023-2025, Jan-Jul 2026 | Passengers, seats, departures, load factor, growth, route distances, cargo share |
| **BTS Airline On-Time Performance** (reporting carriers), sample months Dec 2025 + Jun 2026 | Delay rate, taxi-out/in, NAS (airspace/airport) delay, busiest-hour scheduled ops |
| **BTS airport lookup** | Names/cities |
| **FAA NAS Status API** (live) | Current ground stops, delay programs, closures |

## 4. Scoring methodology (deterministic)
**Terminal Expansion Opportunity Score (0-100)** = weighted average of four components:

| Component (weight) | Sub-metrics (weight within component) | Why it matters |
|---|---|---|
| **Demand growth (35%)** | Passenger CAGR 2023→2025 (40%), Jan-Jul 2026 vs 2025 growth (40%), 2025 vs 2019 recovery (20%) | Growth is what fills new capacity; mixes medium-term trend and current momentum |
| **Capacity pressure (30%)** | Load factor (50%), share of seats on routes with ≥85% load factor (30%), passenger growth minus seat growth (20%) | Full planes + demand outpacing supply = existing capacity is binding |
| **Congestion (20%)** | % departures delayed 15+ min (30%), avg taxi-out (30%), busiest-hour ops (40%) | Strain that a terminal investment can relieve; peak hour drives terminal sizing. NAS delay is **not** scored (see Flags) |
| **Scale (15%)** | Total passengers (log) | Larger base = larger absolute return on a given % improvement |

**Explicit formula:**

```
Opportunity Score = 0.35·G + 0.30·C + 0.20·K + 0.15·S            (0-100)

G (demand growth)     = 0.4·P(CAGR 2023-25) + 0.4·P(growth Jan-Jul 2026 vs 2025) + 0.2·P(2025 / 2019)
C (capacity pressure) = 0.5·P(load factor) + 0.3·P(share of seats on routes with LF ≥ 85%)
                        + 0.2·P(passenger growth − seat growth)
K (congestion)        = 0.3·P(% departures delayed 15+ min) + 0.3·P(avg taxi-out)
                        + 0.4·P(busiest-hour ops)
S (scale)             = P(log10 total passengers)

P(x) = percentile of x among US airports with ≥ 1M enplaned passengers (0-100)
```

**Normalization:** each sub-metric is converted to a **percentile vs. a national peer group** (all US airports
with ≥1M enplaned passengers, 105 airports). Using a national reference (rather than ranking within the
region) keeps scores comparable: a "70" in New England means the same as a "70" in Texas.

**Missing data:** within a component, weights are re-normalized over available sub-metrics; a component with
no data at all (e.g. congestion for airports with few on-time-reporting flights) is set to a **neutral 50**, so
missing data neither helps nor hurts the rank, and the **confidence** label drops (High / Medium / Low).

**Flags (rule-based):**

- *Airside-constrained*: NAS delay and taxi-out both in the top quartile → growth may be capped by
  runways/airspace, so terminal capacity alone may not unlock it. This is the **only** place NAS delay is used:
  NAS delay measures airspace and runway congestion, which a terminal investment does not relieve, so it
  warns about the problem instead of raising the score.
- *Cargo-heavy*: ≥30% of departures are cargo-only (e.g. ANC) → passenger metrics understate activity.
- *Below peer size*, *Not recovered vs 2019*, *Limited delay data*.

Weights are adjustable (sidebar sliders or by asking the agent, e.g. "weight growth twice as much").

**What-if analysis.** An analyst can change one or more of an airport's input values (e.g. *"what if PWM's
load factor were 90%?"*) in the chat or in the **What-if** tab. The airport is re-scored with the same formula
while all other airports and the peer group stay unchanged, and the tool shows before vs after: score,
components, potential, confidence, recommendation, flags and national rank. Example: PWM's load factor
82.4% → 90% raises the score only 54.8 → 56.1, because PWM is already at the 91st percentile on that metric.

**From score to recommendation.** The score measures *initial potential*, not proven profitability, so it is
translated into a decision-oriented label, again with fixed rules:

| Potential tier | Rule | Combined with confidence → recommendation |
|---|---|---|
| **High potential** | score ≥ 65 | High confidence → *Prioritize for due diligence*; otherwise → *Investigate further (data gaps)* |
| **Medium potential** | 50-65 | High confidence → *Investigate further*; otherwise → *Monitor* |
| **Low potential** | < 50 | *Deprioritize* |

A high score on partial data never becomes an automatic "go".

**Text notes shown next to every score.** Besides the number, each airport gets three short notes, all
produced by fixed rules (not written by the LLM):

1. **Recommendation:** from the table above, e.g. *"Prioritize for due diligence"*.
2. **Warning:** if the airport has the airside-constrained flag, the recommendation adds *"check
   runway/airspace capacity first"*, because a new terminal will not help if the runways are the bottleneck.
3. **Strengths and weaknesses:** every component score of 70 or more is listed as a strength and every
   component score below 40 as a weakness, so the explanation comes directly from the numbers.

Example, Boston (BOS), score 67.3: *Prioritize for due diligence, check runway/airspace capacity first.
Strengths: scale 86, capacity pressure 78, congestion 77. Flag: airside-constrained.*

**Data required before an investment decision** (shown after every ranking; not available publicly):

1. Terminal throughput and utilization (passengers per gate, peak-hour terminal capacity)
2. Construction and operating cost of the proposed expansion
3. Expected incremental passenger volume (airline commitments, official FAA forecasts)
4. Revenue per incremental passenger (aeronautical fees, concessions, parking) and lease terms
5. Regulatory, environmental and physical constraints (land, runways, airspace, community caps)

## 5. Where and how AI is used
| Task | LLM? | Notes |
|---|---|---|
| Understand the question, pick tools & parameters | Yes | Claude tool use |
| Resolve names ("LA", "Santa Ana") | Partly | Deterministic alias table + search tool; LLM states the assumption |
| Compute any metric, score or ranking | No | Python only; reproducible and unit-tested |
| Explain results, connect metrics into a narrative | Yes | Must cite tool numbers; system prompt forbids invented figures |
| Domain context (e.g. SNA's community noise/access caps, SFO's closely spaced runways and fog) | Yes | Must be labeled "general knowledge, not from the dataset" |
| Follow-ups, re-weighting | Yes | Conversation memory + re-calling tools with new weights |

Every answer shows a **"How I got this"** panel with the exact tool calls and raw outputs, so an analyst can
audit the reasoning.

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
