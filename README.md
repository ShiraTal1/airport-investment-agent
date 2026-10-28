# ✈️ Airport Investment Intelligence Agent

An AI agent that helps infrastructure investors find US airports where **demand is outgrowing
capacity**, i.e. where terminal/capacity investments are most likely to be used.

- **Deterministic scoring**: a transparent Terminal Expansion Opportunity Score (0-100) computed in
  Python from public BTS data. The LLM never invents or re-ranks numbers.
- **LLM agent**: Claude plans which analysis tools to call, resolves names ("LA" → LAX), explains the
  results, flags uncertainty, and handles follow-ups.
- **Public data**: BTS T-100 Segment (all carriers, domestic + international), BTS Airline On-Time
  Performance, and the live **FAA NAS Status API**.
- **Chat UI** (Streamlit) with a "How I got this" trace for every answer, a no-LLM leaderboard, a
  methodology tab, adjustable weights, and optional voice input.

See **[DESIGN.md](DESIGN.md)** for the scoring methodology, architecture, tradeoffs and where AI is used.

## Example questions
- Which airports in New England are strong candidates for terminal expansion?
- Compare LA and Santa Ana airport congestion levels.
- What is the percentage of long haul flights out of Anchorage airport?
- What is the unmet flight demand in SFO airport and why?
- Follow-ups: "What if growth matters twice as much?", "Why is PVD ranked low?", "Any delays at BOS right now?"

## Run locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your Anthropic API key
streamlit run app.py
```

## Repository layout
```
app.py                 Streamlit chat UI, leaderboard, methodology
agent/analytics.py     Metrics + deterministic scoring (no LLM)
agent/tools.py         Tool schemas exposed to the LLM + dispatcher
agent/llm_agent.py     Claude tool-use loop + system prompt (guardrails)
agent/data.py          Data loading + live FAA NAS Status API client
agent/regions.py       Regions, metro groups, name aliases
etl/                   Scripts that rebuild data/ from BTS public downloads
data/                  Aggregated snapshot (small CSVs)
tests/                 Unit tests for scoring logic
eval/                  Test questions + expected behaviors
```

## Tests
```bash
python -m pytest tests -q        # or: python tests/test_scoring.py
python eval/run_eval.py          # deterministic checks of the 4 sample questions (no API key needed)
```
