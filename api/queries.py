"""
KPI query layer — the operational data model.
These functions are the single source of truth for every metric
consumed by the dashboard, ML layer, and NL query engine.
"""
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import text
from db.config import SessionLocal


def get_latest_grid_state() -> dict:
    """
    Most recent half-hour grid snapshot.
    This is the 'current state' card on the dashboard.
    """
    sql = text("""
        SELECT
            g.timestamp,
            g.intensity_forecast,
            g.intensity_actual,
            g.intensity_index,
            g.gas_perc,
            g.coal_perc,
            g.nuclear_perc,
            g.wind_perc,
            g.solar_perc,
            g.hydro_perc,
            g.biomass_perc,
            g.imports_perc,
            g.other_perc,
            -- Derived: renewable percentage
            COALESCE(g.wind_perc, 0) + COALESCE(g.solar_perc, 0) +
            COALESCE(g.hydro_perc, 0) + COALESCE(g.biomass_perc, 0) AS renewable_perc,
            -- Derived: fossil percentage
            COALESCE(g.gas_perc, 0) + COALESCE(g.coal_perc, 0) AS fossil_perc,
            -- Forecast accuracy for this period
            g.intensity_actual - g.intensity_forecast AS forecast_error,
            -- Weather context
            w.wind_speed_10m,
            w.shortwave_radiation,
            w.temperature_2m
        FROM grid_events g
        LEFT JOIN weather_readings w
            ON DATE_TRUNC('hour', g.timestamp) = DATE_TRUNC('hour', w.timestamp)
        ORDER BY g.timestamp DESC
        LIMIT 1
    """)

    with SessionLocal() as session:
        row = session.execute(sql).mappings().fetchone()
        return dict(row) if row else {}


def get_time_series(hours: int = 24) -> list[dict]:
    """
    Rolling time series for the last N hours.
    Used for the main trend chart on the dashboard.
    """
    sql = text("""
        SELECT
            g.timestamp,
            g.intensity_actual,
            g.intensity_forecast,
            g.intensity_index,
            g.wind_perc,
            g.solar_perc,
            g.gas_perc,
            g.nuclear_perc,
            COALESCE(g.wind_perc, 0) + COALESCE(g.solar_perc, 0) +
            COALESCE(g.hydro_perc, 0) + COALESCE(g.biomass_perc, 0) AS renewable_perc,
            w.wind_speed_10m,
            w.temperature_2m
        FROM grid_events g
        LEFT JOIN weather_readings w
            ON DATE_TRUNC('hour', g.timestamp) = DATE_TRUNC('hour', w.timestamp)
        WHERE g.timestamp >= NOW() - INTERVAL ':hours hours'
        ORDER BY g.timestamp ASC
    """)

    with SessionLocal() as session:
        rows = session.execute(sql, {"hours": hours}).mappings().fetchall()
        return [dict(r) for r in rows]


def get_forecast_accuracy(days: int = 7) -> list[dict]:
    """
    Forecast vs actual delta over last N days.
    Surfaces how well the grid operator's demand forecasts are performing.
    High errors = grid stress, unexpected demand or generation swings.
    """
    sql = text("""
        SELECT
            DATE_TRUNC('hour', timestamp) AS hour,
            AVG(intensity_actual)                           AS avg_actual,
            AVG(intensity_forecast)                         AS avg_forecast,
            AVG(intensity_actual - intensity_forecast)      AS avg_error,
            MAX(ABS(intensity_actual - intensity_forecast)) AS max_abs_error,
            STDDEV(intensity_actual - intensity_forecast)   AS error_stddev
        FROM grid_events
        WHERE
            timestamp >= NOW() - INTERVAL ':days days'
            AND intensity_actual IS NOT NULL
            AND intensity_forecast IS NOT NULL
        GROUP BY DATE_TRUNC('hour', timestamp)
        ORDER BY hour ASC
    """)

    with SessionLocal() as session:
        rows = session.execute(sql, {"days": days}).mappings().fetchall()
        return [dict(r) for r in rows]


def get_generation_mix_trend(hours: int = 48) -> list[dict]:
    """
    Hourly bucketed generation mix trend.
    TimescaleDB time_bucket gives us clean hourly aggregates
    even if underlying data is 30-min resolution.
    """
    sql = text("""
        SELECT
            time_bucket('1 hour', timestamp) AS hour,
            AVG(wind_perc)    AS wind,
            AVG(solar_perc)   AS solar,
            AVG(gas_perc)     AS gas,
            AVG(nuclear_perc) AS nuclear,
            AVG(coal_perc)    AS coal,
            AVG(biomass_perc) AS biomass,
            AVG(hydro_perc)   AS hydro,
            AVG(imports_perc) AS imports,
            AVG(COALESCE(wind_perc,0) + COALESCE(solar_perc,0) +
                COALESCE(hydro_perc,0) + COALESCE(biomass_perc,0)) AS renewable_total
        FROM grid_events
        WHERE timestamp >= NOW() - INTERVAL ':hours hours'
        GROUP BY hour
        ORDER BY hour ASC
    """)

    with SessionLocal() as session:
        rows = session.execute(sql, {"hours": hours}).mappings().fetchall()
        return [dict(r) for r in rows]


def get_regional_snapshot() -> list[dict]:
    """
    Latest reading for each region: the 17 GB DNO regions plus the GB national
    aggregate (region_id 18) — 18 rows total. The Phase 4 choropleth should
    exclude region_id 18 since it isn't a mappable polygon.
    Used for the regional carbon intensity map.
    """
    sql = text("""
        SELECT DISTINCT ON (region_id)
            region_id,
            region_name,
            intensity_forecast,
            intensity_index,
            wind_perc,
            solar_perc,
            gas_perc,
            timestamp
        FROM regional_readings
        ORDER BY region_id, timestamp DESC
    """)

    with SessionLocal() as session:
        rows = session.execute(sql).mappings().fetchall()
        return [dict(r) for r in rows]


def get_carbon_intensity_stats(days: int = 30) -> dict:
    """
    Summary statistics for carbon intensity over N days.
    Used for the 'context' panel — e.g. "Today is 20% cleaner than the 30-day average."
    """
    sql = text("""
        SELECT
            AVG(intensity_actual)                   AS mean,
            MIN(intensity_actual)                   AS min,
            MAX(intensity_actual)                   AS max,
            PERCENTILE_CONT(0.25) WITHIN GROUP
                (ORDER BY intensity_actual)         AS p25,
            PERCENTILE_CONT(0.75) WITHIN GROUP
                (ORDER BY intensity_actual)         AS p75,
            STDDEV(intensity_actual)                AS stddev,
            COUNT(*)                                AS n_periods
        FROM grid_events
        WHERE
            timestamp >= NOW() - INTERVAL ':days days'
            AND intensity_actual IS NOT NULL
    """)

    with SessionLocal() as session:
        row = session.execute(sql, {"days": days}).mappings().fetchone()
        return dict(row) if row else {}


def get_renewable_windows(hours_ahead: int = 24) -> list[dict]:
    """
    Identify the cleanest upcoming windows for flexible load shifting.
    Reads the forward carbon-intensity forecast (ingested from the Carbon
    Intensity API's fw48h endpoint into the forecasts table) and joins the
    weather forecast for solar/wind context.
    This is the 'when to charge your EV / run the data centre job' layer.

    Ranked by lowest forecast carbon intensity (cleanest first). opportunity_score
    is a monotonic convenience: higher = cleaner + better renewable weather.
    """
    sql = text("""
        SELECT
            f.timestamp,
            f.forecast_value                    AS intensity_forecast,
            w.wind_speed_10m,
            w.shortwave_radiation,
            w.temperature_2m,
            -- Lower forecast carbon is the primary signal; add a small bonus for
            -- strong forecast wind/solar. Coefficients are illustrative, not
            -- calibrated — the ORDER BY on forecast_value is the real ranking.
            ROUND(
                (300 - f.forecast_value)
                + COALESCE(w.wind_speed_10m, 0) * 2
                + COALESCE(w.shortwave_radiation, 0) / 50.0
            )                                   AS opportunity_score
        FROM forecasts f
        LEFT JOIN weather_readings w
            ON DATE_TRUNC('hour', f.timestamp) = DATE_TRUNC('hour', w.timestamp)
        WHERE
            f.signal = :signal
            AND f.model_version = :model
            AND f.timestamp BETWEEN NOW() AND NOW() + INTERVAL ':hours hours'
        ORDER BY f.forecast_value ASC
        LIMIT 10
    """)

    with SessionLocal() as session:
        rows = session.execute(
            sql,
            {"hours": hours_ahead, "signal": "carbon_intensity", "model": "carbon_api"},
        ).mappings().fetchall()
        return [dict(r) for r in rows]


def get_forecast(signal: str = "intensity_actual", model_version: str = "prophet") -> list[dict]:
    """
    Upcoming forecast for a signal (Prophet 2h-ahead by default), with confidence
    bounds. Used for the dashboard forecast chart.
    """
    sql = text("""
        SELECT timestamp, forecast_value, lower_bound, upper_bound, model_version
        FROM forecasts
        WHERE signal = :signal
          AND model_version = :model
          AND timestamp >= NOW() - INTERVAL '30 minutes'
        ORDER BY timestamp ASC
    """)

    with SessionLocal() as session:
        rows = session.execute(
            sql, {"signal": signal, "model": model_version}
        ).mappings().fetchall()
        return [dict(r) for r in rows]


def get_demand(hours: int = 24) -> dict:
    """Latest national demand (INDO, MW) plus the recent trend — from Elexon/BMRS."""
    sql = text("""
        SELECT timestamp, demand_mw
        FROM demand_readings
        WHERE timestamp >= NOW() - INTERVAL ':hours hours'
          AND demand_mw IS NOT NULL
        ORDER BY timestamp ASC
    """)
    with SessionLocal() as session:
        rows = [dict(r) for r in session.execute(sql, {"hours": hours}).mappings().fetchall()]
    return {"latest": rows[-1] if rows else None, "count": len(rows), "data": rows}


def get_frequency() -> dict:
    """Most recent grid-frequency reading (Hz) — from Elexon/BMRS."""
    sql = text("""
        SELECT timestamp, frequency_hz
        FROM frequency_readings
        WHERE frequency_hz IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    with SessionLocal() as session:
        row = session.execute(sql).mappings().fetchone()
    return dict(row) if row else {}


def get_interconnectors() -> dict:
    """
    Latest interconnector flows aggregated by country (+ = importing into GB,
    - = exporting). From Elexon/BMRS.
    """
    with SessionLocal() as session:
        ts = session.execute(
            text("SELECT MAX(timestamp) FROM interconnector_flows")
        ).scalar()
        rows = [dict(r) for r in session.execute(text("""
            SELECT country, SUM(flow_mw) AS flow_mw
            FROM interconnector_flows
            WHERE timestamp = (SELECT MAX(timestamp) FROM interconnector_flows)
            GROUP BY country
            ORDER BY SUM(flow_mw) DESC
        """)).mappings().fetchall()]
    return {"timestamp": ts, "net_mw": sum(r["flow_mw"] or 0 for r in rows), "data": rows}


def get_anomaly_history(limit: int = 20) -> list[dict]:
    """Recent anomaly flags — used by dashboard alert panel and NL query layer."""
    sql = text("""
        SELECT
            id, timestamp, signal, value,
            anomaly_score, severity, llm_explanation, acknowledged
        FROM anomaly_flags
        ORDER BY timestamp DESC
        LIMIT :limit
    """)

    with SessionLocal() as session:
        rows = session.execute(sql, {"limit": limit}).mappings().fetchall()
        return [dict(r) for r in rows]


def check_data_quality() -> dict:
    """
    Data quality report — run this to validate your backfill
    and catch issues before they reach the ML layer.
    """
    sql = text("""
        SELECT
            COUNT(*)                                        AS total_records,
            COUNT(intensity_actual)                         AS records_with_actual,
            COUNT(*) - COUNT(intensity_actual)              AS missing_actual,
            COUNT(*) - COUNT(wind_perc)                     AS missing_wind,
            MIN(timestamp)                                  AS earliest,
            MAX(timestamp)                                  AS latest,
            -- Gap detection: periods where we expect data but have none
            (
                EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 1800
            )::INTEGER                                      AS expected_periods,
            COUNT(*)                                        AS actual_periods
        FROM grid_events
    """)

    with SessionLocal() as session:
        row = session.execute(sql).mappings().fetchone()
        result = dict(row)

    # Derived quality metrics
    if result.get("expected_periods") and result["expected_periods"] > 0:
        result["completeness_pct"] = round(
            result["actual_periods"] / result["expected_periods"] * 100, 1
        )
        result["gap_count"] = max(
            0, result["expected_periods"] - result["actual_periods"]
        )
    return result
