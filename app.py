"""Streamlit chat UI for the Airport Investment Intelligence Agent.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from agent import analytics as A
from agent import data
from agent.llm_agent import AirportAgent

st.set_page_config(page_title="Airport Investment Intelligence", page_icon="✈️", layout="wide")

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]


def get_secret(name: str) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("✈️ Airport Investment Agent")
    st.caption("Finds US airports where demand is outgrowing capacity, using public BTS/FAA data, "
               "a deterministic scoring model, and an LLM that explains the results.")
    api_key = get_secret("ANTHROPIC_API_KEY") or st.text_input("Anthropic API key", type="password")

    st.subheader("Score weights")
    st.caption("The agent uses these unless you ask it to change them in chat.")
    w = {}
    for comp, default in A.DEFAULT_WEIGHTS.items():
        w[comp] = st.slider(comp.replace("_", " ").title(), 0.0, 1.0, float(default), 0.05)
    st.session_state["ui_weights"] = w

    if st.button("New conversation"):
        st.session_state.pop("agent", None)
        st.session_state["history"] = []
        st.rerun()

    md = data.metadata()
    with st.expander("Data sources & vintage"):
        for s in md.get("sources", []):
            st.markdown(f"- {s}")
        for k, v in md.get("vintage", {}).items():
            st.markdown(f"- **{k}:** {v}")

tab_chat, tab_board, tab_method = st.tabs(["💬 Chat", "📊 Leaderboard", "📐 Methodology"])

# ---------------------------------------------------------------- chat
with tab_chat:
    st.session_state.setdefault("history", [])
    if not api_key:
        st.info("Add an Anthropic API key in the sidebar (or in Streamlit secrets) to chat. "
                "The Leaderboard and Methodology tabs work without a key.")

    # Voice input (bonus): browser microphone -> OpenAI Whisper if a key is configured
    voice_text = None
    openai_key = get_secret("OPENAI_API_KEY")
    if openai_key and hasattr(st, "audio_input"):
        audio = st.audio_input("🎤 Ask by voice (optional)")
        if audio is not None and st.session_state.get("last_audio_id") != audio.file_id:
            st.session_state["last_audio_id"] = audio.file_id
            import requests
            r = requests.post("https://api.openai.com/v1/audio/transcriptions",
                              headers={"Authorization": f"Bearer {openai_key}"},
                              files={"file": ("q.wav", audio.getvalue(), "audio/wav")},
                              data={"model": "whisper-1"}, timeout=60)
            if r.ok:
                voice_text = r.json().get("text")
            else:
                st.warning(f"Transcription failed: {r.status_code}")

    if not st.session_state["history"]:
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
                st.session_state["pending"] = ex
                st.rerun()

    # All messages go into this container, which sits ABOVE the input box,
    # so the "ask another question" box always stays below the latest answer.
    chat_area = st.container()
    with chat_area:
        for msg in st.session_state["history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_calls"):
                    with st.expander(f"🔎 How I got this ({len(msg['tool_calls'])} tool calls)"):
                        for tc in msg["tool_calls"]:
                            st.markdown(f"**{tc['name']}** `{tc['input']}`")
                            st.json(tc["output"], expanded=False)

    prompt = st.chat_input("Ask about US airports…") or voice_text or st.session_state.pop("pending", None)
    if prompt and api_key:
        if "agent" not in st.session_state:
            st.session_state["agent"] = AirportAgent(api_key=api_key)
        agent = st.session_state["agent"]
        st.session_state["history"].append({"role": "user", "content": prompt})
        # Pass slider weights as context if the user changed them
        ui_w = st.session_state.get("ui_weights", {})
        weight_note = ""
        if any(abs(ui_w.get(k, v) - v) > 1e-9 for k, v in A.DEFAULT_WEIGHTS.items()):
            weight_note = f"\n\n[UI setting: use custom score weights {ui_w} in scoring tools]"
        with chat_area:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing…"):
                    try:
                        turn = agent.ask(prompt + weight_note)
                        answer, calls = turn.text, turn.tool_calls
                    except Exception as exc:
                        answer, calls = f"⚠️ Error calling the model: {exc}", []
                st.markdown(answer)
                if calls:
                    with st.expander(f"🔎 How I got this ({len(calls)} tool calls)"):
                        for tc in calls:
                            st.markdown(f"**{tc['name']}** `{tc['input']}`")
                            st.json(tc["output"], expanded=False)
        st.session_state["history"].append({"role": "assistant", "content": answer, "tool_calls": calls})
    elif prompt and not api_key:
        st.warning("Please add an API key first.")

# ---------------------------------------------------------------- leaderboard (no LLM)
with tab_board:
    st.subheader("Terminal Expansion Opportunity Score — deterministic, no LLM")
    c1, c2, c3 = st.columns(3)
    region = c1.selectbox("Region", ["US", "Contiguous US", "New England", "Mid-Atlantic", "South Atlantic",
                                     "East North Central", "West North Central", "East South Central",
                                     "West South Central", "Mountain", "Pacific", "West Coast"])
    min_pax = c2.number_input("Min. enplaned passengers (2025)", value=1_000_000, step=250_000)
    top_n = c3.slider("Show top", 5, 50, 20)
    res = A.rank_airports(region=region, top_n=top_n, min_enplaned=int(min_pax),
                          weights=st.session_state.get("ui_weights"))
    df = pd.DataFrame(res["ranking"])
    if not df.empty:
        df["flags"] = df["flags"].apply(lambda f: " | ".join(x.split(":")[0] for x in f))
        first = ["rank", "code", "name", "state", "potential", "confidence", "recommendation",
                 "opportunity_score", "drivers"]
        df = df[first + [c for c in df.columns if c not in first]]
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(res["tier_rules"])
    for n in res["notes"]:
        st.caption(n)
    st.markdown("**Data required before an investment decision** (not in public data):")
    for d in res["data_required_before_investment_decision"]:
        st.markdown(f"- {d}")

# ---------------------------------------------------------------- methodology
with tab_method:
    meth = A.methodology(st.session_state.get("ui_weights"))
    st.markdown(f"**Opportunity score.** {meth['opportunity_score']}")
    st.markdown("**Component weights (current):**")
    st.table(pd.DataFrame([meth["weights"]]).T.rename(columns={0: "weight"}))
    st.markdown("**Sub-metrics:**")
    st.table(pd.DataFrame(meth["submetrics"]).T)
    st.markdown(f"**Potential tiers:** {meth['potential_tiers']}")
    st.markdown(f"**Recommendation rule:** {meth['recommendation_rule']}")
    st.markdown(f"**Missing data:** {meth['missing_data_rule']}")
    st.markdown("**Key assumptions:**")
    for a in meth["key_assumptions"]:
        st.markdown(f"- {a}")
