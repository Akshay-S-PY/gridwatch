"""
Model training entrypoint — run nightly (and once on startup if no model yet).

Trains the Isolation Forest anomaly models and the Prophet forecasters on the
last 90 days of grid_events, then persists them to MODEL_DIR.

    python -m ml.train
"""
import logging

from ml.anomaly import AnomalyDetector
from ml.features import load_feature_frame
from ml.forecast import Forecaster

logger = logging.getLogger(__name__)


def train_all(days: int = 90) -> bool:
    """Retrain and persist all models. Returns True if training ran."""
    df = load_feature_frame(days=days)
    if df.empty:
        logger.warning("train: no grid_events data — skipping training")
        return False

    logger.info(f"train: loaded {len(df)} rows for training")
    AnomalyDetector().train(df).save()
    Forecaster().train(df).save()
    logger.info("train: all models trained and saved")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_all()
