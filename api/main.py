"""
GridWatch FastAPI application.
Exposes the KPI query layer as typed REST endpoints.
The dashboard and NL query layer both consume these.
"""
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.queries import (
    get_latest_grid_state,
    get_time_series,
    get_forecast_accuracy,
    get_generation_mix_trend,
    get_regional_snapshot,
    get_carbon_intensity_stats,
    get_renewable_windows,
    get_forecast,
    get_demand,
    get_frequency,
    get_anomaly_history,
    check_data_quality,
)

app = FastAPI(
    title="GridWatch API",
    description="UK National Grid operational intelligence platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "GridWatch API",
        "version": "0.1.0",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    """Health check — also validates DB connectivity."""
    try:
        state = get_latest_grid_state()
        db_ok = bool(state)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")
    return {
        "status": "healthy" if db_ok else "degraded",
        "db": "connected" if db_ok else "empty",
        "latest_data": state.get("timestamp"),
    }


# ─── Core grid endpoints ──────────────────────────────────────────────────────

@app.get("/api/grid/latest")
def latest_grid_state():
    """
    Current half-hour grid snapshot.
    Includes intensity, generation mix, renewable %, fossil %, weather context.
    This is the primary endpoint for dashboard KPI cards.
    """
    state = get_latest_grid_state()
    if not state:
        raise HTTPException(status_code=404, detail="No grid data found. Has ingestion run?")
    return state


@app.get("/api/grid/timeseries")
def time_series(
    hours: int = Query(default=24, ge=1, le=168, description="Hours of history (1–168)")
):
    """
    Rolling time series for the last N hours.
    Default: 24hrs. Max: 7 days (168hrs).
    Used for the main trend chart.
    """
    data = get_time_series(hours=hours)
    return {"hours": hours, "count": len(data), "data": data}


@app.get("/api/grid/generation")
def generation_mix(
    hours: int = Query(default=48, ge=1, le=168)
):
    """
    Hourly bucketed generation mix trend.
    Returns wind, solar, gas, nuclear, etc. aggregated per hour.
    """
    data = get_generation_mix_trend(hours=hours)
    return {"hours": hours, "count": len(data), "data": data}


@app.get("/api/grid/forecast-accuracy")
def forecast_accuracy(
    days: int = Query(default=7, ge=1, le=30)
):
    """
    Forecast vs actual intensity delta over last N days.
    High error = unexpected demand or generation swing.
    Used for the forecast quality panel.
    """
    data = get_forecast_accuracy(days=days)
    return {"days": days, "count": len(data), "data": data}


# ─── Regional endpoints ───────────────────────────────────────────────────────

@app.get("/api/regional/snapshot")
def regional_snapshot():
    """
    Latest carbon intensity reading for the 17 GB DNO regions plus the GB
    national aggregate (18 rows total).
    Used for the regional heatmap.
    """
    data = get_regional_snapshot()
    return {"count": len(data), "regions": data}


# ─── Intelligence endpoints ───────────────────────────────────────────────────

@app.get("/api/grid/stats")
def carbon_stats(
    days: int = Query(default=30, ge=7, le=90)
):
    """
    Statistical summary of carbon intensity over N days.
    Mean, min, max, percentiles, stddev.
    Used for the 'context' panel — "today vs 30-day average".
    """
    stats = get_carbon_intensity_stats(days=days)
    return {"days": days, "stats": stats}


@app.get("/api/grid/clean-windows")
def clean_windows(
    hours_ahead: int = Query(default=24, ge=1, le=48)
):
    """
    Best upcoming windows for flexible load shifting.
    Ranked by opportunity score (low carbon + high renewable).
    Use case: when to charge EVs, run data centre batch jobs.
    """
    windows = get_renewable_windows(hours_ahead=hours_ahead)
    return {"hours_ahead": hours_ahead, "windows": windows}


# ─── Forecast endpoint ────────────────────────────────────────────────────────

@app.get("/api/grid/forecast")
def grid_forecast(
    signal: str = Query(default="intensity_actual",
                        description="Signal to forecast (Prophet output)"),
):
    """
    Upcoming 2-hour Prophet forecast for a signal, with confidence bounds.
    Used for the dashboard forecast chart.
    """
    data = get_forecast(signal=signal, model_version="prophet")
    return {"signal": signal, "count": len(data), "data": data}


# ─── Anomaly endpoints ────────────────────────────────────────────────────────

@app.get("/api/anomalies")
def anomaly_history(
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Recent anomaly flags from the ML layer (Phase 3).
    Returns empty list until anomaly detection is wired up.
    """
    data = get_anomaly_history(limit=limit)
    return {"count": len(data), "anomalies": data}


# ─── Elexon/BMRS: demand + frequency ──────────────────────────────────────────

@app.get("/api/grid/demand")
def grid_demand(hours: int = Query(default=24, ge=1, le=168)):
    """Latest national demand (MW) + recent trend."""
    return get_demand(hours=hours)


@app.get("/api/grid/frequency")
def grid_frequency():
    """Most recent grid frequency (Hz)."""
    return get_frequency()


# ─── Natural-language query layer ─────────────────────────────────────────────

class NLQuery(BaseModel):
    question: str


@app.post("/api/nl/query")
def nl_query(body: NLQuery):
    """
    Conversational query: English question -> grounded SQL -> executed read-only ->
    plain-English answer. Returns {answer, sql, rows, similar} (or {error}).
    """
    from nl.engine import answer
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
    return answer(body.question.strip())


# ─── Data quality endpoint ────────────────────────────────────────────────────

@app.get("/api/admin/data-quality")
def data_quality():
    """
    Data quality report.
    Shows completeness, gaps, missing values.
    Run this after backfill to validate your data before ML training.
    """
    report = check_data_quality()
    return report
