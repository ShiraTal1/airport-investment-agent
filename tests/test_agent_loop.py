"""Tests the agent's tool-use loop with a fake LLM client (no API key or network needed)."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.llm_agent import AirportAgent  # noqa: E402


class Block(NS):
    def model_dump(self):
        return dict(self.__dict__)


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:  # first round: ask for a tool
            return NS(stop_reason="tool_use", content=[
                Block(type="text", text="Let me check.", citations=None),
                Block(type="tool_use", id="t1", name="route_mix", input={"code": "Anchorage"})])
        # second round: must see the tool result in history, then answer
        last = kw["messages"][-1]["content"][0]
        assert last["type"] == "tool_result" and last["tool_use_id"] == "t1"
        return NS(stop_reason="end_turn", content=[Block(type="text", text="**Answer:** done", citations=None)])


def test_loop():
    fake = NS(messages=FakeMessages())
    agent = AirportAgent(client=fake)
    turn = agent.ask("What % of long haul flights out of Anchorage?")
    assert turn.text.startswith("**Answer:**")
    assert turn.tool_calls[0]["name"] == "route_mix"
    assert "ANC" in str(turn.tool_calls[0]["output"]["assumptions_made"])
    # history holds only API-accepted fields
    assert agent.messages[1]["content"][0] == {"type": "text", "text": "Let me check."}
    # follow-up keeps context
    fake.messages.calls.clear()
    agent.ask("and with cargo?")
    assert len(agent.messages) == 8 and agent.messages[4]["content"] == "and with cargo?"


if __name__ == "__main__":
    test_loop()
    print("PASS test_loop")
