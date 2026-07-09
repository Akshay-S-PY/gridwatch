"""
GridWatch — UK National Grid operational intelligence dashboard.

Streamlit front end over the FastAPI query layer. Sections:
  - Live KPI cards (intensity, index, renewable %, fossil %)
  - Generation mix donut
  - 24h intensity time series with anomaly overlays
  - 2-hour Prophet forecast with confidence band
  - Regional carbon-intensity bubble map (14 GB DNO regions)
  - Clean-windows panel (best times to shift flexible load)
  - Alert panel (recent anomaly flags)

Auto-refreshes every 30 minutes (aligned with the ingestion cadence).
"""
import os
from datetime import datetime, timezone

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
REFRESH_MS = 30 * 60 * 1000  # 30 minutes

SEVERITY_COLORS = {
    "critical": "#d62728",
    "high": "#ff7f0e",
    "medium": "#e6b800",
    "low": "#1f77b4",
}

# Approximate centroids for the 14 GB DNO regions (region_id 1-14). Regions 15-18
# (England / Scotland / Wales / GB aggregates) are excluded — they aren't points
# on a map. A true choropleth would need DNO polygon GeoJSON we don't bundle, so
# this is a bubble map coloured + labelled by forecast intensity.
REGION_CENTROIDS = {
    1: (57.5, -4.2), 2: (55.6, -3.8), 3: (53.8, -2.6), 4: (54.9, -1.7),
    5: (53.8, -1.1), 6: (53.2, -3.4), 7: (51.7, -3.4), 8: (52.5, -2.0),
    9: (52.9, -1.0), 10: (52.2, 0.5), 11: (50.9, -3.5), 12: (51.0, -1.4),
    13: (51.5, -0.12), 14: (51.1, 0.4),
}

st.set_page_config(page_title="GridWatch", page_icon="⚡", layout="wide")


# ── data access ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def api_get(path: str) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        st.session_state.setdefault("_errors", []).append(f"{path}: {e}")
        return None


def index_emoji(index: str) -> str:
    return {
        "very low": "🟢", "low": "🟢", "moderate": "🟡",
        "high": "🟠", "very high": "🔴",
    }.get((index or "").lower(), "⚪")


# ── header ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 1])
with left:
    st.title("⚡ GridWatch")
    st.caption("UK National Grid — live carbon intensity, generation mix, anomalies & forecasts")
with right:
    if st.button("🔄 Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Loaded {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")

latest = api_get("/api/grid/latest")
stats = (api_get("/api/grid/stats?days=30") or {}).get("stats", {})

if not latest:
    st.error(f"Could not reach the GridWatch API at {API_BASE}. Is the api service up?")
    for err in st.session_state.get("_errors", []):
        st.caption(err)
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────
mean_30d = stats.get("mean")
carbon = latest.get("intensity_actual") or latest.get("intensity_forecast")
delta_txt = None
if mean_30d and carbon is not None:
    diff = carbon - mean_30d
    delta_txt = f"{diff:+.0f} vs 30-day avg"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Carbon intensity", f"{carbon} gCO₂/kWh", delta_txt, delta_color="inverse")
k2.metric("Intensity index", f"{index_emoji(latest.get('intensity_index'))} {latest.get('intensity_index','—').title()}")
k3.metric("Renewable", f"{latest.get('renewable_perc', 0):.0f}%")
k4.metric("Fossil", f"{latest.get('fossil_perc', 0):.0f}%")

st.divider()

# ── generation mix donut + 24h time series ───────────────────────────────────
col_mix, col_ts = st.columns([1, 2])

with col_mix:
    st.subheader("Generation mix")
    fuels = ["gas", "coal", "nuclear", "wind", "solar", "hydro", "biomass", "imports", "other"]
    mix = {f: latest.get(f"{f}_perc") or 0 for f in fuels}
    mix = {f: v for f, v in mix.items() if v > 0}
    donut = go.Figure(go.Pie(
        labels=[f.title() for f in mix], values=list(mix.values()),
        hole=0.55, sort=False,
    ))
    donut.update_traces(textposition="inside", textinfo="label+percent")
    donut.update_layout(showlegend=False, height=340, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(donut, use_container_width=True)

with col_ts:
    st.subheader("Carbon intensity — last 24h")
    ts = api_get("/api/grid/timeseries?hours=24") or {"data": []}
    tsdf = pd.DataFrame(ts["data"])
    fig = go.Figure()
    if not tsdf.empty:
        tsdf["timestamp"] = pd.to_datetime(tsdf["timestamp"])
        fig.add_trace(go.Scatter(x=tsdf["timestamp"], y=tsdf["intensity_actual"],
                                 name="Actual", line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=tsdf["timestamp"], y=tsdf["intensity_forecast"],
                                 name="Forecast", line=dict(color="#999", width=1, dash="dot")))
        # anomaly overlay: intensity_actual anomalies within the window
        anoms = (api_get("/api/anomalies?limit=200") or {}).get("anomalies", [])
        adf = pd.DataFrame([a for a in anoms if a["signal"] == "intensity_actual"])
        if not adf.empty:
            adf["timestamp"] = pd.to_datetime(adf["timestamp"])
            adf = adf[adf["timestamp"] >= tsdf["timestamp"].min()]
            if not adf.empty:
                fig.add_trace(go.Scatter(
                    x=adf["timestamp"], y=adf["value"], mode="markers",
                    name="Anomaly",
                    marker=dict(symbol="x", size=11,
                                color=[SEVERITY_COLORS.get(s, "#d62728") for s in adf["severity"]]),
                ))
    fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                      yaxis_title="gCO₂/kWh", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── 2h forecast + regional map ───────────────────────────────────────────────
col_fc, col_map = st.columns([1, 1])

with col_fc:
    st.subheader("Carbon intensity — 2h forecast")
    fc = (api_get("/api/grid/forecast?signal=intensity_actual") or {}).get("data", [])
    fcdf = pd.DataFrame(fc)
    figf = go.Figure()
    # recent actuals for context (last ~6h)
    if not tsdf.empty:
        recent = tsdf.tail(12)
        figf.add_trace(go.Scatter(x=recent["timestamp"], y=recent["intensity_actual"],
                                  name="Actual", line=dict(color="#1f77b4", width=2)))
    if not fcdf.empty:
        fcdf["timestamp"] = pd.to_datetime(fcdf["timestamp"])
        # confidence band
        figf.add_trace(go.Scatter(
            x=list(fcdf["timestamp"]) + list(fcdf["timestamp"][::-1]),
            y=list(fcdf["upper_bound"]) + list(fcdf["lower_bound"][::-1]),
            fill="toself", fillcolor="rgba(255,127,14,0.2)",
            line=dict(color="rgba(255,255,255,0)"), name="80% interval", hoverinfo="skip",
        ))
        figf.add_trace(go.Scatter(x=fcdf["timestamp"], y=fcdf["forecast_value"],
                                  name="Prophet forecast",
                                  line=dict(color="#ff7f0e", width=2, dash="dash")))
    figf.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                       yaxis_title="gCO₂/kWh", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(figf, use_container_width=True)

with col_map:
    st.subheader("Regional carbon intensity")
    regions = (api_get("/api/regional/snapshot") or {}).get("regions", [])
    rows = []
    for r in regions:
        c = REGION_CENTROIDS.get(r["region_id"])
        if not c:
            continue
        rows.append({
            "name": r["region_name"], "lat": c[0], "lon": c[1],
            "intensity": r["intensity_forecast"], "index": r["intensity_index"],
        })
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        # Plotly Scattergeo: token-free, tile-free, no WebGL dependency (unlike
        # pydeck/deck.gl) — renders reliably for the demo. Bubbles over a UK-scoped
        # natural-earth base, coloured by forecast intensity.
        geo = go.Figure(go.Scattergeo(
            lon=rdf["lon"], lat=rdf["lat"],
            text=rdf["name"] + ": " + rdf["intensity"].astype(str) + " gCO₂/kWh (" + rdf["index"] + ")",
            hoverinfo="text",
            marker=dict(
                size=20,
                color=rdf["intensity"],
                colorscale=[[0, "#2ecc71"], [0.5, "#f1c40f"], [1, "#e74c3c"]],
                cmin=0, cmax=300,
                colorbar=dict(title="gCO₂/kWh"),
                line=dict(width=1, color="white"),
            ),
        ))
        geo.update_geos(
            projection_type="mercator", resolution=50,
            lataxis_range=[49.8, 59.0], lonaxis_range=[-8.5, 2.5],
            showcountries=True, countrycolor="#bbb",
            showland=True, landcolor="#f5f5f5", showocean=True, oceancolor="#eaf3fb",
        )
        geo.update_layout(height=380, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(geo, use_container_width=True)
    st.caption("Bubble colour = forecast intensity (🟢 clean → 🔴 dirty). 14 GB DNO regions.")

st.divider()

# ── clean windows + alerts ───────────────────────────────────────────────────
col_cw, col_al = st.columns([1, 1])

with col_cw:
    st.subheader("🔋 Clean windows — best times to shift load")
    windows = (api_get("/api/grid/clean-windows?hours_ahead=24") or {}).get("windows", [])
    if windows:
        wdf = pd.DataFrame(windows)
        wdf["timestamp"] = pd.to_datetime(wdf["timestamp"])
        wdf = wdf[["timestamp", "intensity_forecast", "opportunity_score"]].rename(columns={
            "timestamp": "When (UTC)", "intensity_forecast": "gCO₂/kWh",
            "opportunity_score": "Score",
        })
        st.dataframe(wdf, hide_index=True, use_container_width=True,
                     column_config={"When (UTC)": st.column_config.DatetimeColumn(format="MMM D, HH:mm")})
    else:
        st.info("No forecast windows available yet.")

with col_al:
    st.subheader("🚨 Recent anomalies")
    anoms = (api_get("/api/anomalies?limit=15") or {}).get("anomalies", [])
    if anoms:
        adf = pd.DataFrame(anoms)
        adf["timestamp"] = pd.to_datetime(adf["timestamp"])
        show = adf[["timestamp", "signal", "value", "severity"]].rename(columns={
            "timestamp": "When (UTC)", "signal": "Signal",
            "value": "Value", "severity": "Severity",
        })
        st.dataframe(
            show, hide_index=True, use_container_width=True,
            column_config={"When (UTC)": st.column_config.DatetimeColumn(format="MMM D, HH:mm")},
        )
        counts = adf["severity"].value_counts().to_dict()
        st.caption("  ·  ".join(f"{index_emoji('')}{s}: {counts[s]}" for s in
                                ["critical", "high", "medium", "low"] if s in counts))
    else:
        st.success("No anomalies flagged.")

# ── auto-refresh every 30 min ────────────────────────────────────────────────
st.markdown(
    f"<script>setTimeout(function(){{window.parent.location.reload();}}, {REFRESH_MS});</script>",
    unsafe_allow_html=True,
)
