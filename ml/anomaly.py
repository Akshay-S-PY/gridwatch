"""
Isolation Forest anomaly detection — one model per signal.

Each model trains on [value, hour_sin, hour_cos, dow], so it flags both globally
extreme values and values unusual for the time of day. Severity is derived from
where a point's isolation score falls in the training-set score distribution.
"""
import logging
import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from ml.features import ANOMALY_SIGNALS, signal_matrix

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "ml/artifacts")
ANOMALY_PATH = os.path.join(MODEL_DIR, "anomaly_forest.joblib")

CONTAMINATION = 0.02   # ~2% of points flagged as anomalies
MIN_TRAIN_ROWS = 100


class AnomalyDetector:
    def __init__(self):
        self.models: dict = {}       # signal -> IsolationForest
        self.thresholds: dict = {}   # signal -> {p90, p95, p99} of raw anomaly score

    # ── training ────────────────────────────────────────────────────────────
    def train(self, df) -> "AnomalyDetector":
        for sig in ANOMALY_SIGNALS:
            _, X = signal_matrix(df, sig)
            if len(X) < MIN_TRAIN_ROWS:
                logger.warning(f"anomaly: skipping {sig} — only {len(X)} rows")
                continue
            model = IsolationForest(
                n_estimators=200, contamination=CONTAMINATION, random_state=42
            )
            model.fit(X)
            raw = -model.score_samples(X)  # higher = more anomalous
            self.models[sig] = model
            # Grade severity WITHIN the anomalous population (points the model
            # flags as outliers), not the full distribution — otherwise every
            # flagged point (already the top ~2%) reads as high/critical and
            # low/medium never occur. Percentiles of the flagged scores spread
            # anomalies across all four severities, keeping critical rare.
            flagged = raw[model.predict(X) == -1]
            ref = flagged if len(flagged) >= 20 else raw
            self.thresholds[sig] = {
                "medium": float(np.percentile(ref, 40)),
                "high": float(np.percentile(ref, 70)),
                "critical": float(np.percentile(ref, 90)),
            }
            logger.info(f"anomaly: trained {sig} on {len(X)} rows")
        return self

    def _severity(self, sig: str, raw: float) -> str:
        t = self.thresholds[sig]
        if raw >= t["critical"]:
            return "critical"
        if raw >= t["high"]:
            return "high"
        if raw >= t["medium"]:
            return "medium"
        return "low"

    # ── scoring ─────────────────────────────────────────────────────────────
    def score(self, df) -> list[dict]:
        """Return anomaly-flag dicts for rows Isolation Forest marks as outliers."""
        flags: list[dict] = []
        for sig, model in self.models.items():
            sub, X = signal_matrix(df, sig)
            if len(X) == 0:
                continue
            preds = model.predict(X)          # -1 = anomaly
            raw = -model.score_samples(X)
            values = sub[sig].to_numpy()
            timestamps = sub["timestamp"].tolist()
            for ts, val, pred, r in zip(timestamps, values, preds, raw):
                if pred != -1:
                    continue
                flags.append({
                    "timestamp": ts.to_pydatetime(),
                    "signal": sig,
                    "value": float(val),
                    "anomaly_score": float(r),
                    "severity": self._severity(sig, float(r)),
                    "llm_explanation": None,
                })
        return flags

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: str = ANOMALY_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"models": self.models, "thresholds": self.thresholds}, path)
        logger.info(f"anomaly: saved {len(self.models)} models -> {path}")

    @classmethod
    def load(cls, path: str = ANOMALY_PATH):
        if not os.path.exists(path):
            return None
        state = joblib.load(path)
        det = cls()
        det.models = state["models"]
        det.thresholds = state["thresholds"]
        return det
