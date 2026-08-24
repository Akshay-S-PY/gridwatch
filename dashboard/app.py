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

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1320px;}
[data-testid="stMetric"] {
    background: #f8fafc; border: 1px solid #e6eaf0; border-radius: 12px; padding: 14px 18px;
}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #64748b;}
.gw-sub {color:#475569; font-size:1.0rem; margin-top:-8px;}
.gw-banner {border-radius:12px; padding:14px 18px; margin:10px 0 6px 0; font-size:1.03rem;}
.gw-hint {color:#64748b; font-size:0.88rem; margin:2px 0 6px 0;}
h2, h3 {letter-spacing:-0.01em;}
</style>
""", unsafe_allow_html=True)


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


# ── gridwatch-style dial gauges ───────────────────────────────────────────────
# Coloured arcs matching the carbon-intensity index bands (a nod to the classic
# gridwatch.co.uk analog meters, modernised).
INTENSITY_STEPS = [
    {"range": [0, 45],    "color": "#a7f3d0"},   # very low
    {"range": [45, 130],  "color": "#d9f99d"},   # low
    {"range": [130, 210], "color": "#fde68a"},   # moderate
    {"range": [210, 270], "color": "#fdba74"},   # high
    {"range": [270, 350], "color": "#fca5a5"},   # very high
]
RENEWABLE_STEPS = [
    {"range": [0, 40],   "color": "#fee2e2"},
    {"range": [40, 70],  "color": "#fef9c3"},
    {"range": [70, 100], "color": "#dcfce7"},
]
FOSSIL_STEPS = [
    {"range": [0, 25],   "color": "#dcfce7"},
    {"range": [25, 50],  "color": "#fef9c3"},
    {"range": [50, 100], "color": "#fee2e2"},
]


def _gauge_layout(fig):
    fig.update_layout(height=250, margin=dict(t=54, b=8, l=28, r=28),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def gauge_intensity(value, ref):
    """Carbon-intensity dial with coloured zones + a needle, and a delta vs the 30-day average."""
    has_ref = ref is not None
    ind = dict(
        mode="gauge+number+delta" if has_ref else "gauge+number",
        value=value,
        number={"suffix": " gCO₂"},
        title={"text": "Carbon intensity", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 350], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": "rgba(15,23,42,0.85)", "thickness": 0.22},
            "steps": INTENSITY_STEPS,
            "threshold": {"line": {"color": "#0f172a", "width": 4}, "thickness": 0.8, "value": value},
        },
    )
    if has_ref:
        ind["delta"] = {"reference": round(ref),
                        "increasing": {"color": "#dc2626"},   # dirtier than avg = bad
                        "decreasing": {"color": "#16a34a"}}   # cleaner than avg = good
    return _gauge_layout(go.Figure(go.Indicator(**ind)))


def gauge_pct(title, value, steps, barcolor):
    return _gauge_layout(go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%"},
        title={"text": title, "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": barcolor, "thickness": 0.22},
            "steps": steps,
        },
    )))


def gauge_frequency(value):
    """The classic gridwatch dial — grid frequency, nominal 50 Hz ± 0.2."""
    return _gauge_layout(go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": " Hz", "valueformat": ".3f"},
        title={"text": "Grid frequency", "font": {"size": 15}},
        gauge={
            "axis": {"range": [49.5, 50.5], "tickcolor": "#94a3b8"},
            "bar": {"color": "rgba(15,23,42,0.85)", "thickness": 0.22},
            "steps": [
                {"range": [49.5, 49.8], "color": "#fca5a5"},
                {"range": [49.8, 50.2], "color": "#a7f3d0"},   # healthy band
                {"range": [50.2, 50.5], "color": "#fca5a5"},
            ],
            "threshold": {"line": {"color": "#0f172a", "width": 4}, "thickness": 0.8, "value": value},
        },
    )))


def gauge_demand(value_mw):
    """National demand in GW (GB runs ~18 GW overnight to ~45 GW winter peak)."""
    return _gauge_layout(go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_mw / 1000.0,
        number={"suffix": " GW", "valueformat": ".1f"},
        title={"text": "National demand", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 50], "tickcolor": "#94a3b8"},
            "bar": {"color": "#334155", "thickness": 0.22},
            "steps": [
                {"range": [0, 25],  "color": "#dcfce7"},
                {"range": [25, 40], "color": "#fef9c3"},
                {"range": [40, 50], "color": "#fee2e2"},
            ],
        },
    )))


# ── sidebar: what this is & how it works ─────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ GridWatch")
    st.markdown(
        "Live UK electricity-grid intelligence. Every **30 minutes** it ingests "
        "carbon-intensity, generation-mix and weather data, then layers ML on top."
    )
    st.markdown(
        "**How it works**\n\n"
        "1. Ingest grid + weather data (30-min cadence)\n"
        "2. Detect anomalies (Isolation Forest) & forecast 2h ahead (Prophet)\n"
        "3. Serve it here — and via natural-language chat"
    )
    st.info("💬 Ask questions in plain English on the **Ask GridWatch** page →")
    st.caption("Sources: carbonintensity.org.uk · open-meteo.com")

# ── header ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 1])
with left:
    st.title("⚡ GridWatch")
    st.markdown(
        "<div class='gw-sub'>Live UK National Grid intelligence — carbon intensity, "
        "generation mix, ML anomaly detection &amp; 2-hour forecasts, refreshed every 30 minutes.</div>",
        unsafe_allow_html=True,
    )
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

carbon = latest.get("intensity_actual") or latest.get("intensity_forecast")
mean_30d = stats.get("mean")
index = (latest.get("intensity_index") or "").lower()
ren = latest.get("renewable_perc", 0) or 0
foss = latest.get("fossil_perc", 0) or 0

# ── plain-English status banner (colour shows how clean the grid is now) ──────
BANNER = {
    "very low":  ("#dcfce7", "#166534", "🟢", "The grid is very clean right now"),
    "low":       ("#dcfce7", "#166534", "🟢", "The grid is clean right now"),
    "moderate":  ("#fef9c3", "#854d0e", "🟡", "The grid is moderately clean right now"),
    "high":      ("#ffedd5", "#9a3412", "🟠", "The grid is fairly carbon-intensive right now"),
    "very high": ("#fee2e2", "#991b1b", "🔴", "The grid is very carbon-intensive right now"),
}
bg, fg, dot, phrase = BANNER.get(index, ("#f1f5f9", "#334155", "⚪", "Live grid status"))
st.markdown(
    f"<div class='gw-banner' style='background:{bg};color:{fg};'>"
    f"{dot} <b>{phrase}</b> — {carbon} gCO₂/kWh ({index or 'n/a'}). "
    f"Renewables are supplying <b>{ren:.0f}%</b> of generation, fossil fuels <b>{foss:.0f}%</b>."
    f"</div>",
    unsafe_allow_html=True,
)

# ── live dial gauges (gridwatch-style) ───────────────────────────────────────
g1, g2, g3 = st.columns(3)
g1.plotly_chart(gauge_intensity(carbon, mean_30d if mean_30d else None), use_container_width=True)
g2.plotly_chart(gauge_pct("Renewable share", ren, RENEWABLE_STEPS, "#059669"), use_container_width=True)
g3.plotly_chart(gauge_pct("Fossil share", foss, FOSSIL_STEPS, "#334155"), use_container_width=True)
st.markdown(
    "<div class='gw-hint'>Live dials — carbon intensity in gCO₂/kWh (lower is cleaner; coloured arc = index "
    "band, needle = now, delta = vs the 30-day average) and the renewable / fossil share of generation.</div>",
    unsafe_allow_html=True,
)

# ── grid vitals: frequency + demand (Elexon/BMRS) ────────────────────────────
freq = api_get("/api/grid/frequency") or {}
demand_latest = (api_get("/api/grid/demand?hours=1") or {}).get("latest") or {}
if freq.get("frequency_hz") is not None or demand_latest.get("demand_mw") is not None:
    v1, v2, _v3 = st.columns(3)
    if freq.get("frequency_hz") is not None:
        v1.plotly_chart(gauge_frequency(freq["frequency_hz"]), use_container_width=True)
    if demand_latest.get("demand_mw") is not None:
        v2.plotly_chart(gauge_demand(demand_latest["demand_mw"]), use_container_width=True)
    st.markdown(
        "<div class='gw-hint'>Grid vitals from Elexon/BMRS — frequency must stay near 50 Hz "
        "(the live balance of supply and demand), and total national demand in GW.</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── generation mix donut + 24h time series ───────────────────────────────────
col_mix, col_ts = st.columns([1, 2])

with col_mix:
    st.subheader("Generation mix")
    st.markdown("<div class='gw-hint'>What's generating GB electricity this half-hour. "
                "Wind, solar, hydro &amp; biomass are low-carbon; gas is the main fossil source.</div>",
                unsafe_allow_html=True)
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
    st.markdown("<div class='gw-hint'>Actual vs forecast intensity over 24 hours. "
                "✕ marks anomalies the ML flagged — it usually dips midday (solar) and climbs into the evening peak.</div>",
                unsafe_allow_html=True)
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
    st.markdown("<div class='gw-hint'>Two forward projections: the official National Grid "
                "forecast (green) and our own Prophet model (orange), which uses forecasted "
                "wind &amp; solar as inputs. The shaded band is Prophet's 80% confidence "
                "interval.</div>", unsafe_allow_html=True)
    fc = (api_get("/api/grid/forecast?signal=intensity_actual&model=prophet") or {}).get("data", [])
    api_fc = (api_get("/api/grid/forecast?signal=carbon_intensity&model=carbon_api") or {}).get("data", [])
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
            line=dict(color="rgba(255,255,255,0)"), name="Prophet 80% interval", hoverinfo="skip",
        ))
        figf.add_trace(go.Scatter(x=fcdf["timestamp"], y=fcdf["forecast_value"],
                                  name="Prophet (ours)",
                                  line=dict(color="#ff7f0e", width=2, dash="dash")))
    # official National Grid forward forecast — the accuracy benchmark
    apidf = pd.DataFrame(api_fc)
    if not apidf.empty:
        apidf["timestamp"] = pd.to_datetime(apidf["timestamp"])
        figf.add_trace(go.Scatter(x=apidf["timestamp"], y=apidf["forecast_value"],
                                  name="Official (National Grid)",
                                  line=dict(color="#10b981", width=2)))
    figf.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                       yaxis_title="gCO₂/kWh", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(figf, use_container_width=True)

with col_map:
    st.subheader("Regional carbon intensity")
    st.markdown("<div class='gw-hint'>Forecast intensity across the 14 GB distribution regions. "
                "Scotland is usually cleanest (wind); southern regions can lean on gas.</div>",
                unsafe_allow_html=True)
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

# ── interconnector flows (Elexon/BMRS) ───────────────────────────────────────
st.subheader("🔌 Interconnector flows")
st.markdown("<div class='gw-hint'>Live power across GB's subsea cables — 🟢 importing into GB, "
            "🔴 exporting. From Elexon/BMRS.</div>", unsafe_allow_html=True)
inter = api_get("/api/grid/interconnectors") or {}
idata = inter.get("data", [])
if idata:
    idf = pd.DataFrame(idata).sort_values("flow_mw")
    colors = ["#059669" if (v or 0) >= 0 else "#dc2626" for v in idf["flow_mw"]]
    fig_i = go.Figure(go.Bar(
        x=idf["flow_mw"], y=idf["country"], orientation="h",
        marker_color=colors,
        text=[f"{(v or 0):+,.0f} MW" for v in idf["flow_mw"]], textposition="outside",
    ))
    fig_i.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=60),
                        xaxis_title="MW  (+ importing · − exporting)")
    st.plotly_chart(fig_i, use_container_width=True)
    net = inter.get("net_mw", 0) / 1000
    st.caption(f"Net position: GB is **{'importing' if net >= 0 else 'exporting'} {abs(net):.1f} GW** right now.")
else:
    st.info("Interconnector data not available yet.")

st.divider()

# ── clean windows + alerts ───────────────────────────────────────────────────
col_cw, col_al = st.columns([1, 1])

with col_cw:
    st.subheader("🔋 Clean windows — best times to shift load")
    st.markdown("<div class='gw-hint'>The lowest-carbon upcoming half-hours — ideal for shifting "
                "flexible load like EV charging or batch compute.</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='gw-hint'>Unusual readings flagged by per-signal Isolation Forest "
                "models, graded low → critical.</div>", unsafe_allow_html=True)
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
