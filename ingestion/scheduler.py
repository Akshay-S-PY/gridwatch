"""
Ingestion scheduler.
- On startup: runs a 90-day backfill if the DB is empty
- Every 30 minutes: polls live data from all three sources
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import text

from db.config import SessionLocal
from db.setup import init_db
from ingestion.carbon_client import CarbonIntensityClient
from ingestion.weather_client import WeatherClient
from ingestion.writer import (
    write_grid_events, write_weather, write_regional, write_forecasts
)
from ml.anomaly import AnomalyDetector
from ml.detect import run_detection, seed_history_if_empty
from ml.train import train_all

# Signal + provider tags for the forward carbon-intensity forecast we ingest.
FORECAST_SIGNAL = "carbon_intensity"
FORECAST_MODEL = "carbon_api"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def poll_live() -> None:
    """Fetch current half-hour data from all APIs and write to DB."""
    logger.info("─── Live poll started ───")
    carbon = CarbonIntensityClient()
    weather = WeatherClient()

    try:
        # 1. National intensity + generation
        intensity = carbon.get_current_intensity()
        generation = carbon.get_current_generation()

        gen_map = {}
        if generation.data:
            gen_map = generation.data[0].as_dict()

        grid_records = []
        for period in intensity.data:
            grid_records.append({
                "timestamp":           period.from_time,
                "intensity_forecast":  period.intensity.forecast,
                "intensity_actual":    period.intensity.actual,
                "intensity_index":     period.intensity.index,
                "gas_perc":            gen_map.get("gas"),
                "coal_perc":           gen_map.get("coal"),
                "nuclear_perc":        gen_map.get("nuclear"),
                "wind_perc":           gen_map.get("wind"),
                "solar_perc":          gen_map.get("solar"),
                "hydro_perc":          gen_map.get("hydro"),
                "biomass_perc":        gen_map.get("biomass"),
                "imports_perc":        gen_map.get("imports"),
                "other_perc":          gen_map.get("other"),
            })
        write_grid_events(grid_records)

        # 2. Regional intensity
        regional_resp = carbon.get_current_regional()
        regional_records = []
        if regional_resp.data:
            period = regional_resp.data[0]
            for region in period.regions:
                gen = {f.fuel: f.perc for f in region.generationmix}
                regional_records.append({
                    "timestamp":           period.from_time,
                    "region_id":           region.regionid,
                    "region_name":         region.shortname,
                    "intensity_forecast":  region.intensity.forecast,
                    "intensity_index":     region.intensity.index,
                    "wind_perc":           gen.get("wind"),
                    "solar_perc":          gen.get("solar"),
                    "gas_perc":            gen.get("gas"),
                })
        write_regional(regional_records)

        # 3. Weather forecast (next 48hrs)
        weather_records = weather.get_forecast(hours_ahead=48)
        write_weather(weather_records)

        # 4. Forward 48h carbon-intensity forecast (powers clean-windows /
        #    load-shifting). Stored in the forecasts table, not grid_events, so
        #    future rows never shadow the current actual state.
        forward = carbon.get_forward_forecast()
        forecast_records = [
            {
                "timestamp":      period.from_time,
                "signal":         FORECAST_SIGNAL,
                "forecast_value": period.intensity.forecast,
                "lower_bound":    None,
                "upper_bound":    None,
                "model_version":  FORECAST_MODEL,
            }
            for period in forward.data
            if period.intensity.forecast is not None
        ]
        write_forecasts(forecast_records)

    except Exception as e:
        logger.error(f"Live poll failed: {e}", exc_info=True)
    finally:
        carbon.close()
        weather.close()

    # 5. ML inference on the freshly-ingested data (anomaly flags + 2h forecast).
    #    Isolated so a model/inference failure never breaks ingestion.
    try:
        run_detection()
    except Exception as e:
        logger.error(f"ML detection failed: {e}", exc_info=True)

    logger.info("─── Live poll complete ───")


BACKFILL_DAYS = 90
# Consider history "sufficient" if the earliest grid event predates this cutoff.
# Leaves slack below BACKFILL_DAYS so we don't re-backfill on every restart, but
# still self-heals if a prior backfill was stubbed/partial (e.g. only live-poll rows).
HISTORY_SUFFICIENT_DAYS = 85


def _has_sufficient_history() -> bool:
    """True if grid_events already spans roughly the full backfill window."""
    with SessionLocal() as session:
        earliest = session.execute(
            text("SELECT MIN(timestamp) FROM grid_events")
        ).scalar()

    if earliest is None:
        return False

    cutoff = datetime.now(timezone.utc) - timedelta(days=HISTORY_SUFFICIENT_DAYS)
    return earliest <= cutoff


def run_backfill() -> None:
    """Run a 90-day backfill unless the DB already holds enough history.

    Writes are idempotent (unique indexes + ON CONFLICT DO NOTHING), so a partial
    or previously-failed backfill self-heals: on restart we detect the missing
    history and re-run without creating duplicates.
    """
    if _has_sufficient_history():
        logger.info(
            f"grid_events already spans ≥{HISTORY_SUFFICIENT_DAYS} days — skipping backfill."
        )
        return

    logger.info(f"Insufficient history — starting {BACKFILL_DAYS}-day backfill...")
    carbon  = CarbonIntensityClient()
    weather = WeatherClient()

    try:
        # Carbon backfill first, and write it immediately, so a later weather
        # failure can't discard the grid history we just fetched.
        grid_records = carbon.backfill(days=BACKFILL_DAYS)
        write_grid_events(grid_records)
        logger.info(f"Backfill: wrote {len(grid_records)} grid events.")

        try:
            weather_records = weather.get_historical(days=BACKFILL_DAYS)
            write_weather(weather_records)
            logger.info(f"Backfill: wrote {len(weather_records)} weather records.")
        except Exception as e:
            logger.error(f"Weather backfill failed (grid history preserved): {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
    finally:
        carbon.close()
        weather.close()


def main():
    # 1. Ensure tables exist
    init_db()

    # 2. Backfill if needed
    run_backfill()

    # 3. Train ML models on first run so the first poll can already detect /
    #    forecast. Nightly retraining (below) keeps them fresh thereafter.
    if AnomalyDetector.load() is None:
        logger.info("No trained models found — training on startup...")
        try:
            train_all()
        except Exception as e:
            logger.error(f"Startup training failed: {e}", exc_info=True)

    # 3b. Seed historical anomaly flags once so the dashboard/API have data.
    try:
        seed_history_if_empty()
    except Exception as e:
        logger.error(f"Anomaly seeding failed: {e}", exc_info=True)

    # 3c. Seed the Qdrant RAG index from those anomalies (idempotent; needs an LLM
    #     key for embeddings). Non-fatal — the NL layer degrades without RAG.
    try:
        from nl import rag
        rag.seed()
    except Exception as e:
        logger.error(f"RAG seeding failed (NL layer still works without it): {e}", exc_info=True)

    # 4. First immediate live poll (also runs ML inference)
    poll_live()

    # 5. Schedule live polling (30 min) + nightly model retraining (02:00 UTC)
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        poll_live,
        trigger=IntervalTrigger(minutes=30),
        id="live_poll",
        name="Live grid data poll",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        train_all,
        trigger=CronTrigger(hour=2, minute=0),
        id="nightly_retrain",
        name="Nightly model retraining",
        max_instances=1,
        coalesce=True,
    )

    logger.info("Scheduler started — polling every 30 min, retraining nightly at 02:00 UTC.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
