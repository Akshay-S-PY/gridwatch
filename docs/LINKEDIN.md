# LinkedIn post — draft

Built around a real anomaly the system caught: **29 June 2026, 23:30 UTC — carbon
intensity spiked to 239 gCO₂/kWh (flagged *critical*)**. Context at that moment:
**wind was only 10.6%** of generation and **gas had climbed to 56%** — a low-wind
night where the grid fell back on gas. The Isolation Forest flagged it; the NL layer
explains *why* in one sentence.

*(Numbers below are from the running system — swap in whatever your live data shows
on demo day.)*

---

## Draft (before/after framing)

> **I built a UK grid intelligence platform that you can talk to. Here's the moment it clicked.**
>
> On the night of June 29th, carbon intensity on the GB grid jumped to 239 gCO₂/kWh —
> and my system flagged it as a *critical* anomaly automatically.
>
> **Before:** to understand *why*, you'd open a dashboard, find the right chart,
> cross-reference generation mix against weather, and reason it out yourself.
>
> **After:** I typed one sentence — *"Why did carbon intensity spike on the evening
> of June 29th?"* — and got:
>
> *"The spike was primarily due to a significant increase in gas generation, which
> rose from 43.9% at 17:00 to 52.9% by 21:30. Solar dropped sharply from 11.7% to 0%
> as the evening progressed, and wind didn't compensate for the loss of that
> low-carbon source."*
>
> It wrote the SQL, ran it read-only against 90 days of live grid data, and explained
> the result — catching the gas rise, the evening solar drop, and the wind shortfall
> on its own.
>
> **GridWatch** ingests live National Grid + weather data every 30 minutes into
> TimescaleDB, runs Isolation Forest anomaly detection and Prophet forecasting, serves
> it through FastAPI + a Streamlit dashboard, and adds a conversational layer:
> English → SQL → grounded answer, with RAG over past anomalies for context.
>
> The pattern — live operational data → anomaly detection → an LLM reasoning layer
> over a conversational interface — is exactly what operational-intelligence teams are
> building. I wanted to build the whole thing end-to-end, safely (the generated SQL is
> validated and executed read-only — no writes, ever).
>
> Stack: Python · TimescaleDB · Qdrant · scikit-learn · Prophet · FastAPI · Streamlit
> · OpenAI. All on free public data.
>
> Repo + 90-second demo 👇
> #DataEngineering #MachineLearning #LLM #EnergyTech #Python

---

## Notes for posting

- **You** post this — it publishes to your network. Update the numbers to whatever the
  live system shows the day you record.
- Attach the 90-second demo video (see `DEMO_SCRIPT.md`) — video outperforms links.
- First comment: drop the GitHub link (LinkedIn suppresses posts with outbound links
  in the body).
- The "before/after" + a concrete real anomaly is the hook — lead with the moment, not
  the tech stack.
