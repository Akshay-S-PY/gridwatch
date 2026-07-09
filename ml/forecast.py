"""
Prophet 2-hour-ahead forecasting for key grid signals.

Models train nightly on 90 days of history; predictions are anchored to the
latest actual settlement period so the forecast is always the next 2 hours
(4 x 30-min periods) from *now*, not from when the model was trained.

Written to the forecasts table with model_version='prophet', which keeps them
distinct from the Carbon Intensity API's own forward forecast
(signal='carbon_intensity', model_version='carbon_api').
"""
import logging
import os

import joblib
import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "ml/artifacts")
FORECAST_PATH = os.path.join(MODEL_DIR, "prophet_models.joblib")

FORECAST_SIGNALS = ["intensity_actual", "wind_perc", "solar_perc", "renewable_perc"]
HORIZON_PERIODS = 4          # 4 x 30 min = 2 hours ahead
FREQ = "30min"
MODEL_VERSION = "prophet"
MIN_TRAIN_ROWS = 200


class Forecaster:
    def __init__(self):
        self.models: dict = {}

    def train(self, df) -> "Forecaster":
        for sig in FORECAST_SIGNALS:
            sub = (
                df[["timestamp", sig]]
                .dropna()
                .rename(columns={"timestamp": "ds", sig: "y"})
                .copy()
            )
            if len(sub) < MIN_TRAIN_ROWS:
                logger.warning(f"forecast: skipping {sig} — only {len(sub)} rows")
                continue
            # Prophet needs tz-naive timestamps.
            sub["ds"] = sub["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                interval_width=0.8,
            )
            model.fit(sub)
            self.models[sig] = model
            logger.info(f"forecast: trained {sig} on {len(sub)} rows")
        return self

    def predict(self, anchor_ts) -> list[dict]:
        """Forecast the next 2 hours starting just after `anchor_ts` (tz-aware UTC)."""
        anchor_naive = pd.Timestamp(anchor_ts).tz_convert("UTC").tz_localize(None)
        future = pd.DataFrame({
            "ds": pd.date_range(
                start=anchor_naive + pd.Timedelta(FREQ),
                periods=HORIZON_PERIODS,
                freq=FREQ,
            )
        })

        rows: list[dict] = []
        for sig, model in self.models.items():
            fc = model.predict(future)
            for _, r in fc.iterrows():
                rows.append({
                    "timestamp": r["ds"].tz_localize("UTC").to_pydatetime(),
                    "signal": sig,
                    "forecast_value": float(r["yhat"]),
                    "lower_bound": float(r["yhat_lower"]),
                    "upper_bound": float(r["yhat_upper"]),
                    "model_version": MODEL_VERSION,
                })
        return rows

    def save(self, path: str = FORECAST_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.models, path)
        logger.info(f"forecast: saved {len(self.models)} models -> {path}")

    @classmethod
    def load(cls, path: str = FORECAST_PATH):
        if not os.path.exists(path):
            return None
        fc = cls()
        fc.models = joblib.load(path)
        return fc
