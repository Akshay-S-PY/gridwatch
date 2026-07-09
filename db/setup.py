"""
Database initialisation.
Creates Postgres tables and converts time-series tables to TimescaleDB hypertables.
Run once on startup: python -m db.setup
"""
import logging
from sqlalchemy import create_engine, text
from db.config import DATABASE_URL

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- ─── Core time-series tables ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS grid_events (
    timestamp           TIMESTAMPTZ     NOT NULL,
    intensity_forecast  INTEGER,
    intensity_actual    INTEGER,
    intensity_index     TEXT,
    gas_perc            FLOAT,
    coal_perc           FLOAT,
    nuclear_perc        FLOAT,
    wind_perc           FLOAT,
    solar_perc          FLOAT,
    hydro_perc          FLOAT,
    biomass_perc        FLOAT,
    imports_perc        FLOAT,
    other_perc          FLOAT,
    created_at          TIMESTAMPTZ     DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS weather_readings (
    timestamp           TIMESTAMPTZ     NOT NULL,
    wind_speed_10m      FLOAT,
    shortwave_radiation FLOAT,
    temperature_2m      FLOAT,
    created_at          TIMESTAMPTZ     DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS regional_readings (
    timestamp           TIMESTAMPTZ     NOT NULL,
    region_id           INTEGER         NOT NULL,
    region_name         TEXT            NOT NULL,
    intensity_forecast  INTEGER,
    intensity_index     TEXT,
    wind_perc           FLOAT,
    solar_perc          FLOAT,
    gas_perc            FLOAT,
    created_at          TIMESTAMPTZ     DEFAULT NOW()
);

-- ─── Anomaly flags (written by ML layer in Phase 3) ────────────────────────

CREATE TABLE IF NOT EXISTS anomaly_flags (
    id                  SERIAL,
    timestamp           TIMESTAMPTZ     NOT NULL,
    signal              TEXT            NOT NULL,  -- e.g. 'intensity_actual', 'wind_perc'
    value               FLOAT           NOT NULL,
    anomaly_score       FLOAT           NOT NULL,  -- Isolation Forest score
    severity            TEXT            NOT NULL,  -- low | medium | high | critical
    llm_explanation     TEXT,                      -- filled by LLM layer
    acknowledged        BOOLEAN         DEFAULT FALSE,
    created_at          TIMESTAMPTZ     DEFAULT NOW()
);

-- ─── Forecast store (written by Prophet in Phase 3) ────────────────────────

CREATE TABLE IF NOT EXISTS forecasts (
    timestamp           TIMESTAMPTZ     NOT NULL,
    signal              TEXT            NOT NULL,
    forecast_value      FLOAT           NOT NULL,
    lower_bound         FLOAT,
    upper_bound         FLOAT,
    model_version       TEXT,
    created_at          TIMESTAMPTZ     DEFAULT NOW()
);

-- ─── Indexes ────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_grid_events_timestamp
    ON grid_events (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_weather_timestamp
    ON weather_readings (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_regional_timestamp_region
    ON regional_readings (timestamp DESC, region_id);

CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp
    ON anomaly_flags (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_forecast_signal_timestamp
    ON forecasts (signal, timestamp DESC);
"""

HYPERTABLE_SQL = """
-- Convert to TimescaleDB hypertables (idempotent)
SELECT create_hypertable('grid_events',    'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('weather_readings','timestamp', if_not_exists => TRUE);
SELECT create_hypertable('regional_readings','timestamp', if_not_exists => TRUE);
SELECT create_hypertable('anomaly_flags',  'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('forecasts',      'timestamp', if_not_exists => TRUE);
"""

# Remove any pre-existing duplicates (from before unique indexes existed) so the
# unique index creation below can succeed. Keeps one row per key. Duplicates
# share a timestamp and therefore live in the same hypertable chunk, so ctid
# comparison is valid.
DEDUP_SQL = """
DELETE FROM grid_events a USING grid_events b
    WHERE a.ctid < b.ctid AND a.timestamp = b.timestamp;
DELETE FROM weather_readings a USING weather_readings b
    WHERE a.ctid < b.ctid AND a.timestamp = b.timestamp;
DELETE FROM regional_readings a USING regional_readings b
    WHERE a.ctid < b.ctid
      AND a.timestamp = b.timestamp
      AND a.region_id = b.region_id;
"""

# Unique indexes give ON CONFLICT DO NOTHING something to conflict on, making
# writes idempotent. TimescaleDB requires the partitioning column (timestamp)
# to be part of every unique index — which it is here.
UNIQUE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS uq_grid_events_timestamp
    ON grid_events (timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS uq_weather_timestamp
    ON weather_readings (timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS uq_regional_timestamp_region
    ON regional_readings (timestamp, region_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_forecasts_signal_timestamp_model
    ON forecasts (signal, timestamp, model_version);
CREATE UNIQUE INDEX IF NOT EXISTS uq_anomaly_timestamp_signal
    ON anomaly_flags (timestamp, signal);
"""

def init_db() -> None:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        logger.info("Creating tables...")
        conn.execute(text(SCHEMA_SQL))
        conn.commit()

        logger.info("Converting to TimescaleDB hypertables...")
        try:
            conn.execute(text(HYPERTABLE_SQL))
            conn.commit()
            logger.info("Hypertables ready.")
        except Exception as e:
            # TimescaleDB extension may not be loaded in dev — warn but continue
            logger.warning(f"Hypertable creation skipped (TimescaleDB not available?): {e}")
            conn.rollback()

        logger.info("Deduplicating and adding unique indexes...")
        conn.execute(text(DEDUP_SQL))
        conn.execute(text(UNIQUE_INDEX_SQL))
        conn.commit()
        logger.info("Unique indexes ready.")

    logger.info("Database initialisation complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
