"""
Ask GridWatch — conversational NL query interface.

Operator types a question in English; the backend turns it into grounded SQL,
runs it read-only, and explains the result. Shows the answer, the SQL it ran,
the result table, and a heuristic chart.
"""
import os

import httpx
import pandas as pd
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")

st.set_page_config(page_title="Ask GridWatch", page_icon="💬", layout="wide")
st.title("💬 Ask GridWatch")
st.caption("Ask about carbon intensity, generation mix, regions, anomalies or forecasts — in plain English.")
with st.expander("How this works", expanded=False):
    st.markdown(
        "Your question is turned into a SQL query against 90+ days of live grid data, "
        "**executed read-only** (it can never modify anything), and the result is explained "
        "back in plain English. Similar past anomalies are retrieved for context.\n\n"
        "Try: *“Why did carbon intensity spike yesterday evening?”* · "
        "*“Which region is cleanest right now?”* · *“When was renewable generation highest this week?”*"
    )

EXAMPLES = [
    "Why did carbon intensity change this evening?",
    "Which region is currently cleanest?",
    "How does today's wind generation compare to last week?",
    "What's the best time to run a high-energy workload today?",
    "Has there been any unusual grid behaviour in the last 24 hours?",
]


def render_result(res: dict) -> None:
    if res.get("error"):
        st.error(res["error"])
        if res.get("sql"):
            st.code(res["sql"], language="sql")
        return

    st.markdown(res.get("answer", ""))

    rows = res.get("rows", [])
    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Heuristic chart: a time-like column + numeric columns -> line chart.
    if not df.empty:
        time_col = next((c for c in df.columns
                         if any(t in c.lower() for t in ("time", "hour", "date", "bucket"))), None)
        num_cols = [c for c in df.columns if c != time_col
                    and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
        if time_col and num_cols and len(df) > 1:
            chart = df.copy()
            chart[time_col] = pd.to_datetime(chart[time_col], errors="coerce")
            for c in num_cols:
                chart[c] = pd.to_numeric(chart[c], errors="coerce")
            st.line_chart(chart.set_index(time_col)[num_cols])

    with st.expander(f"🔎 SQL and data ({res.get('row_count', 0)} rows)"):
        if res.get("sql"):
            st.code(res["sql"], language="sql")
        if not df.empty:
            st.dataframe(df, hide_index=True, use_container_width=True)
        if res.get("similar"):
            st.markdown("**Similar past anomalies (retrieved):**")
            for s in res["similar"]:
                st.markdown(f"- {s}")


def ask(question: str) -> None:
    st.session_state.history.append(("user", question, None))
    try:
        r = httpx.post(f"{API_BASE}/api/nl/query", json={"question": question}, timeout=60)
        r.raise_for_status()
        res = r.json()
    except Exception as e:  # noqa: BLE001
        res = {"error": f"Request failed: {e}"}
    st.session_state.history.append(("assistant", None, res))


if "history" not in st.session_state:
    st.session_state.history = []

# Example-question buttons (only before the first question, to keep it clean).
if not st.session_state.history:
    st.markdown("**Try one:**")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            ask(ex)
            st.rerun()

# Render conversation.
for role, text, res in st.session_state.history:
    with st.chat_message(role):
        if role == "user":
            st.markdown(text)
        else:
            render_result(res)

# Chat input.
if q := st.chat_input("Ask about the grid…"):
    ask(q)
    st.rerun()
