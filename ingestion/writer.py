"""
DB writer — inserts ingested records into TimescaleDB.
Uses INSERT ... ON CONFLICT DO NOTHING to handle duplicate poll cycles safely.
"""
import logging
from sqlalchemy import text
from db.config import SessionLocal

logger = logging.getLogger(__name__)


def write_grid_events(records: list[dict]) -> int:
    """
    Upsert grid event records.
    Returns number of rows inserted.
    """
    if not records:
        return 0

    sql = text("""
        INSERT INTO grid_events (
            timestamp, intensity_forecast, intensity_actual, intensity_index,
            gas_perc, coal_perc, nuclear_perc, wind_perc, solar_perc,
            hydro_perc, biomass_perc, imports_perc, other_perc
        ) VALUES (
            :timestamp, :intensity_forecast, :intensity_actual, :intensity_index,
            :gas_perc, :coal_perc, :nuclear_perc, :wind_perc, :solar_perc,
            :hydro_perc, :biomass_perc, :imports_perc, :other_perc
        )
        ON CONFLICT DO NOTHING
    """)

    with SessionLocal() as session:
        result = session.execute(sql, records)
        session.commit()
        inserted = result.rowcount
        logger.info(f"Grid events: inserted {inserted}/{len(records)} records")
        return inserted


def write_weather(records: list[dict]) -> int:
    if not records:
        return 0

    sql = text("""
        INSERT INTO weather_readings (
            timestamp, wind_speed_10m, shortwave_radiation, temperature_2m
        ) VALUES (
            :timestamp, :wind_speed_10m, :shortwave_radiation, :temperature_2m
        )
        ON CONFLICT DO NOTHING
    """)

    with SessionLocal() as session:
        result = session.execute(sql, records)
        session.commit()
        logger.info(f"Weather: inserted {result.rowcount}/{len(records)} records")
        return result.rowcount


def write_demand(records: list[dict]) -> int:
    """National demand rows (Elexon INDO). Idempotent on timestamp."""
    if not records:
        return 0
    sql = text("""
        INSERT INTO demand_readings (timestamp, demand_mw)
        VALUES (:timestamp, :demand_mw)
        ON CONFLICT (timestamp) DO NOTHING
    """)
    with SessionLocal() as session:
        result = session.execute(sql, records)
        session.commit()
        logger.info(f"Demand: inserted {result.rowcount}/{len(records)} rows")
        return result.rowcount


def write_interconnectors(records: list[dict]) -> int:
    """Interconnector flow rows (Elexon). Idempotent on (timestamp, name)."""
    if not records:
        return 0
    sql = text("""
        INSERT INTO interconnector_flows (timestamp, name, country, flow_mw)
        VALUES (:timestamp, :name, :country, :flow_mw)
        ON CONFLICT (timestamp, name) DO NOTHING
    """)
    with SessionLocal() as session:
        result = session.execute(sql, records)
        session.commit()
        logger.info(f"Interconnectors: inserted {result.rowcount}/{len(records)} rows")
        return result.rowcount


def write_frequency(record: dict) -> int:
    """A single grid-frequency snapshot. Idempotent on timestamp."""
    if not record:
        return 0
    sql = text("""
        INSERT INTO frequency_readings (timestamp, frequency_hz)
        VALUES (:timestamp, :frequency_hz)
        ON CONFLICT (timestamp) DO NOTHING
    """)
    with SessionLocal() as session:
        result = session.execute(sql, record)
        session.commit()
        return result.rowcount


def write_anomaly_flags(records: list[dict]) -> int:
    """
    Insert anomaly flags. Idempotent on (timestamp, signal) so re-scoring the
    same recent window each poll doesn't create duplicates.
    """
    if not records:
        return 0

    sql = text("""
        INSERT INTO anomaly_flags (
            timestamp, signal, value, anomaly_score, severity, llm_explanation
        ) VALUES (
            :timestamp, :signal, :value, :anomaly_score, :severity, :llm_explanation
        )
        ON CONFLICT (timestamp, signal) DO NOTHING
    """)

    with SessionLocal() as session:
        result = session.execute(sql, records)
        session.commit()
        logger.info(f"Anomaly flags: inserted {result.rowcount}/{len(records)} rows")
        return result.rowcount


def write_forecasts(records: list[dict]) -> int:
    """
    Upsert forecast rows (e.g. the Carbon Intensity forward 48h forecast).
    Unlike the append-only tables, forecasts for a future period get refined on
    every poll, so we DO UPDATE the value rather than DO NOTHING.
    """
    if not records:
        return 0

    sql = text("""
        INSERT INTO forecasts (
            timestamp, signal, forecast_value, lower_bound, upper_bound, model_version
        ) VALUES (
            :timestamp, :signal, :forecast_value, :lower_bound, :upper_bound, :model_version
        )
        ON CONFLICT (signal, timestamp, model_version)
        DO UPDATE SET
            forecast_value = EXCLUDED.forecast_value,
            lower_bound    = EXCLUDED.lower_bound,
            upper_bound    = EXCLUDED.upper_bound
    """)

    with SessionLocal() as session:
        result = session.execute(sql, records)
        session.commit()
        logger.info(f"Forecasts: upserted {result.rowcount}/{len(records)} rows")
        return result.rowcount


def write_regional(records: list[dict]) -> int:
    if not records:
        return 0

    sql = text("""
        INSERT INTO regional_readings (
            timestamp, region_id, region_name,
            intensity_forecast, intensity_index,
            wind_perc, solar_perc, gas_perc
        ) VALUES (
            :timestamp, :region_id, :region_name,
            :intensity_forecast, :intensity_index,
            :wind_perc, :solar_perc, :gas_perc
        )
        ON CONFLICT DO NOTHING
    """)

    with SessionLocal() as session:
        result = session.execute(sql, records)
        session.commit()
        logger.info(f"Regional: inserted {result.rowcount}/{len(records)} records")
        return result.rowcount
