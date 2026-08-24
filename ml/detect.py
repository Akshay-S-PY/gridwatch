"""
Per-poll inference: score the newest data for anomalies and refresh the 2-hour
Prophet forecast. Models are trained separately (nightly); this only loads and
applies them, so it's cheap enough to run on every 30-minute poll.
"""
import logging

import pandas as pd

from ml.alerts import alert_if_severe
from ml.anomaly import AnomalyDetector
from ml.features import load_feature_frame
from ml.forecast import Forecaster
from db.config import SessionLocal
from ingestion.writer import write_anomaly_flags, write_forecasts
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Context window loaded for scoring (features are per-row, so a short window is
# enough); only rows newer than SCORE_RECENT_HOURS are actually flagged.
CONTEXT_DAYS = 2
SCORE_RECENT_HOURS = 6
SEED_DAYS = 90


def seed_history_if_empty() -> None:
    """
    One-time backfill of historical anomaly flags so the dashboard/API have data
    immediately (the anomalies the system 'caught' across the 90-day history).
    Skipped if anomaly_flags already holds rows. Does NOT alert — alerting is for
    live/recent anomalies only, not months-old ones.
    """
    with SessionLocal() as session:
        count = session.execute(text("SELECT COUNT(*) FROM anomaly_flags")).scalar()
    if count and count > 0:
        return

    detector = AnomalyDetector.load()
    if detector is None:
        return
    df = load_feature_frame(days=SEED_DAYS)
    if df.empty:
        return
    flags = detector.score(df)
    write_anomaly_flags(flags)
    logger.info(f"detect: seeded {len(flags)} historical anomaly flags")


def run_detection() -> None:
    df = load_feature_frame(days=CONTEXT_DAYS)
    if df.empty:
        logger.info("detect: no data to score")
        return

    latest = df["timestamp"].max()

    # ── anomaly detection ────────────────────────────────────────────────────
    detector = AnomalyDetector.load()
    if detector is None:
        logger.info("detect: no anomaly model yet — train first")
    else:
        cutoff = latest - pd.Timedelta(hours=SCORE_RECENT_HOURS)
        recent = df[df["timestamp"] >= cutoff]
        flags = detector.score(recent)
        alert_if_severe(flags)          # attaches llm_explanation to high/critical
        write_anomaly_flags(flags)
        if flags:
            by_sev = {}
            for f in flags:
                by_sev[f["severity"]] = by_sev.get(f["severity"], 0) + 1
            logger.info(f"detect: {len(flags)} anomalies flagged {by_sev}")

    # ── forecasting ──────────────────────────────────────────────────────────
    forecaster = Forecaster.load()
    if forecaster is None:
        logger.info("detect: no forecast model yet — train first")
    else:
        # last actuals, so the forecast is persistence-anchored to 'now'
        latest_row = df[df["timestamp"] == latest].iloc[0]
        latest_vals = {s: (float(latest_row[s]) if pd.notna(latest_row.get(s)) else None)
                       for s in ("intensity_actual", "wind_perc", "solar_perc", "renewable_perc")
                       if s in latest_row}
        rows = forecaster.predict(latest, latest_vals)
        write_forecasts(rows)
        logger.info(f"detect: wrote {len(rows)} forecast rows (2h ahead, anchored)")


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO)
    run_detection()
