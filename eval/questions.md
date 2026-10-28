# Evaluation set

Each question lists the tool(s) the agent should call and what a good answer must contain.
`run_eval.py` checks the deterministic part automatically; the LLM part is a manual rubric.

| # | Question | Expected tool(s) | A good answer must... |
|---|---|---|---|
| 1 | Which airports in New England are strong candidates for terminal expansion? | rank_airports(region="New England") | Give a ranked list with scores and component drivers; note excluded small airports; state confidence; mention score ≠ profitability |
| 2 | Compare LA and Santa Ana airport congestion levels. | compare_airports([LAX, SNA], focus=congestion) | State LA→LAX assumption (other LA airports exist); compare delay %, taxi-out, busiest-hour ops, NAS delay; give congestion index; label any regulatory context (SNA's access plan caps) as general knowledge |
| 3 | What is the percentage of long haul flights out of Anchorage airport? | route_mix(ANC) and ideally route_mix(ANC, include_cargo=true) | State the threshold definition; show threshold sensitivity; flag that ANC is cargo-dominated and give the with-cargo figure |
| 4 | What is the unmet flight demand in SFO airport and why? | demand_pressure(SFO) | Explain that unmet demand is not directly observable; give proxies + heuristic range; explain "why" with data (load factor, tight routes, delays) and label weather/runway context as general knowledge |
| 5 | Follow-up after #1: "What if growth matters twice as much?" | rank_airports with custom weights | Re-run with weights, say what changed in the order |
| 6 | "Which airport will give us the best IRR?" | methodology / none | Refuse to give IRR; explain scope (no cost/revenue data); offer what it can rank |
| 7 | "How many gates does BOS have?" | none / profile | Say the dataset has no gate data; do not invent a number |
| 8 | "Any delays at SFO right now?" | live_faa_status([SFO]) | Use live API; report time of update; say if no events |
| 9 | "What are the 5 most congested large airports?" | rank_by_metric(congestion_score, hub_sizes=[Large hub]) | Ranked list with values; note congestion ≠ attractiveness |
| 10 | "How many passengers fly DEN–LAS?" | route_lookup(DEN, LAS) | Both directions; segment (not O&D) caveat |
| 11 | "Is a terminal expansion at PWM worth it? What would it cost?" | capex_feasibility(PWM) | Ranges only; Low confidence; estimate-vs-final labels; AIP = grants not cost; ACRP 120 scope caveat |
