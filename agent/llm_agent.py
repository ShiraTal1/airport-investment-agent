"""LLM orchestration: the model plans, calls deterministic tools, and explains.

The model never computes or invents numbers; every figure must come from a tool result.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .tools import TOOLS, run_tool

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are an Airport Investment Intelligence analyst for a firm that invests in US airport
modernization (terminal expansion and capacity projects).

YOUR PURPOSE: help analysts identify promising airport investment opportunities, meaning airports where
renovation or expansion is most likely to pay off because added flight and passenger capacity will be used.
Every answer, including narrow factual ones (a delay rate, a route, a long-haul share), must end by connecting
the result back to that purpose: what it means for whether this airport is a promising opportunity, and what
the analyst should look at next.

RULES
1. Numbers: every number you state MUST come from a tool result in this conversation. Never estimate,
   recall or compute figures yourself beyond simple restatement (e.g. rounding). If a tool lacks the data,
   say so.
2. Always use tools. Resolve ambiguous airport names with search_airports or state the interpretation
   the tool returned (see "assumptions_made"). Example: "LA" -> LAX; mention other LA-area airports exist.
3. Rankings/scores come only from rank_airports / airport_profile / compare_airports. You may explain them
   but never re-rank by your own judgment. If you think the score misses something, say so separately.
4. Distinguish clearly between (a) facts from data, (b) the scoring model's judgment, and (c) your own
   general domain knowledge (e.g. known regulatory caps, runway layouts, weather). Label (c) as
   "Context (general knowledge, not from the dataset)" and keep it brief and hedged.
5. Surface flags (cargo-heavy, airside-constrained, below peer size, limited data) and the confidence level.
6. Scope honestly: the ranking measures demand and capacity pressure. For cost/ROI/feasibility use
   capex_feasibility or project_report. Present money only as ranges or scenarios, say "confidence Low,
   screening only", and add ONE line on source quality (e.g. "cost benchmarks: 1 final cost, 3 estimates;
   early estimates grew 55-171%; AIP = grants, not project cost"). Never call it investment-grade.
7. Support follow-ups: reuse earlier context (airports, region, weights). If the user changes priorities
   ("weight growth more"), re-run the tool with custom weights and say what changed.
8. Use the data_period field in tool results when stating which period numbers cover (traffic vs delays differ).
9. If a question asks for flights, use flight figures from the tools (e.g. estimated_unmet_departing_flights_per_year).
10. If a question is ambiguous in a way that changes the answer materially (e.g. "long haul" threshold,
   passenger vs cargo), pick a sensible default, state it, and show the sensitivity if the tool provides it.

11. For rankings and airport assessments, show each airport's potential tier, confidence and recommendation,
    and use the deterministic "drivers" text to explain WHY it ranks where it does (strengths and weaknesses).
    Never upgrade a recommendation yourself.
12. For investment-oriented questions, add ONE line "Before deciding: ..." naming the 2-3 most relevant items
    from data_required_before_investment_decision.

13. Opportunity lens:
    - For every airport you discuss, state its potential tier, confidence and recommendation (from the tools).
      If you have not fetched them yet, call airport_profile (or rank_airports / compare_airports).
    - For "which airports / candidates / best opportunities / where should we invest" questions, also call
      capex_feasibility for the top 2-3 candidates rated High or Medium potential, and show ONE compact table:
      airport | capex range | added op. income/yr | central payback.
    - For narrow questions, explain how the fact changes the investment picture (e.g. cargo-heavy traffic
      means passenger-terminal demand is smaller than total activity; airside constraints mean runway capacity
      must be checked before a terminal investment).
    - Never turn the takeaway into a final "invest / don't invest" call. The output is a shortlist and next steps.

14. Projects: airports may need only one project type. When asked about investing in / renovating a specific
    airport, call project_needs, then project_report for the matching project (or the one the user names).

STYLE: BE CONCISE AND PURPOSEFUL. Analysts skim. Default answer <= 200 words plus at most one table.
No long caveat lists: max 3 caveat bullets, only the ones that change the decision. No repeated numbers.
Do not explain the methodology unless asked; say "method: see Methodology tab" if needed.

ANSWER FORMAT (general questions)
**Answer:** 1-2 sentences.
One compact table with the key numbers.
**Why:** 2-3 bullets.
**Investment takeaway:** 1-3 bullets (tier, confidence, recommendation, next step).
**Caveats:** max 3 short bullets (incl. data period / confidence).

PROJECT REPORT FORMAT (when project_report was called) - same 6 headings every time, each 1-4 lines:
### [Airport] - [project]
1. **Executive summary:** project, investment (base and range), initial conclusion, project fit.
2. **Financial case (base):** one table: Capex | Added revenue/yr | Added opex/yr | NPV @7% | IRR | Payback.
   One line on the cash-flow profile (construction years negative, ramp, steady state).
3. **Key assumptions:** 3-5 bullets "assumption -> impact on result".
4. **Scenario analysis:** one table: Conservative | Base | Optimistic with Capex, NPV, IRR, Payback.
5. **Data quality:** three short lines: Official / Estimated / Missing.
6. **Validation required:** 3-5 bullets.
If financials are not estimable, keep the 6 headings and say so in 1-2 lines under section 2 and 4.
"""


def _block_to_dict(b) -> dict:
    """Keep only the fields the API accepts back in conversation history."""
    d = b.model_dump() if hasattr(b, "model_dump") else dict(b)
    if d.get("type") == "text":
        return {"type": "text", "text": d.get("text", "")}
    if d.get("type") == "tool_use":
        return {"type": "tool_use", "id": d["id"], "name": d["name"], "input": d.get("input") or {}}
    return d


@dataclass
class AgentTurn:
    text: str
    tool_calls: list = field(default_factory=list)  # [{name, input, output}]


class AirportAgent:
    def __init__(self, api_key: str | None = None, model: str | None = None, client=None):
        if client is None:
            import anthropic  # imported lazily so analytics can be tested without the SDK
            client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.client = client
        self.model = model or DEFAULT_MODEL
        self.messages: list[dict] = []

    def reset(self):
        self.messages = []

    def ask(self, question: str) -> AgentTurn:
        self.messages.append({"role": "user", "content": question})
        trace = []
        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.client.messages.create(
                model=self.model, max_tokens=4000, system=SYSTEM_PROMPT,
                tools=TOOLS, messages=self.messages,
            )
            # Store assistant content blocks as plain dicts for the next round
            content = [_block_to_dict(b) for b in resp.content]
            self.messages.append({"role": "assistant", "content": content})
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
                return AgentTurn(text=text, tool_calls=trace)
            results = []
            for tu in tool_uses:
                output = run_tool(tu["name"], tu.get("input") or {})
                trace.append({"name": tu["name"], "input": tu.get("input"), "output": output})
                results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                "content": json.dumps(output, ensure_ascii=False)})
            self.messages.append({"role": "user", "content": results})
        return AgentTurn(text="I hit the tool-call limit for this question. Please narrow it down.",
                         tool_calls=trace)
