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
modernization (terminal expansion and capacity projects). You help analysts find airports where added
flight/passenger capacity is most likely to be used, i.e. where demand is growing and existing capacity is
under pressure.

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
6. Scope honestly: this tool measures demand and capacity pressure, not project profitability (no data on
   construction cost, gates, leases, or financing). Say this when users ask about "profitability".
7. Support follow-ups: reuse earlier context (airports, region, weights). If the user changes priorities
   ("weight growth more"), re-run the tool with custom weights and say what changed.
8. If a question is ambiguous in a way that changes the answer materially (e.g. "long haul" threshold,
   passenger vs cargo), pick a sensible default, state it, and show the sensitivity if the tool provides it.

ANSWER FORMAT (markdown, concise, analyst-friendly)
**Answer:** 1-3 sentence direct answer.
Then a compact table or bullets with the key numbers.
**Why:** the reasoning in plain language (which metrics drove the result).
**Assumptions & caveats:** bullets, including data period and confidence.
Optionally suggest 1-2 useful follow-up questions.
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
                model=self.model, max_tokens=2500, system=SYSTEM_PROMPT,
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
