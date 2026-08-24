"""
Prophet 2-hour-ahead forecasting for key grid signals.

Models train nightly on 90 days of history; predictions are anchored to the
latest actual settlement period so the forecast is always the next 2 hours
(4 x 30-min periods) from *now*, not from when the model was trained.

Carbon intensity is driven by the generation mix, so a univariate model of
intensity's own history forecasts it poorly (it can't see the wind that moves
it). The intensity model therefore uses forecasted wind % and solar % as
*regressors* — we forecast those well, and feed them in. wind/solar/renewable
stay univariate.

Written to the forecasts table with model_version='prophet', distinct from the
Carbon Intensity API's forward forecast (signal='carbon_intensity', 'carbon_api').
"""
import logging
import os

import joblib
import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "ml/artifacts")
FORECAST_PATH = os.path.join(MODEL_DIR, "prophet_models.joblib")

UNIVARIATE_SIGNALS = ["wind_perc", "solar_perc", "renewable_perc"]
INTENSITY_REGRESSORS = ["wind_perc", "solar_perc"]
HORIZON_PERIODS = 4          # 4 x 30 min = 2 hours ahead
FREQ = "30min"
MODEL_VERSION = "prophet"
MIN_TRAIN_ROWS = 200


def _prophet() -> Prophet:
    return Prophet(daily_seasonality=True, weekly_seasonality=True, interval_width=0.8)


class Forecaster:
    def __init__(self):
        self.models: dict = {}            # univariate: wind_perc, solar_perc, renewable_perc
        self.intensity_model = None       # intensity_actual with wind/solar regressors

    def train(self, df) -> "Forecaster":
        # ── univariate signals ────────────────────────────────────────────────
        for sig in UNIVARIATE_SIGNALS:
            sub = (df[["timestamp", sig]].dropna()
                   .rename(columns={"timestamp": "ds", sig: "y"}).copy())
            if len(sub) < MIN_TRAIN_ROWS:
                logger.warning(f"forecast: skipping {sig} — only {len(sub)} rows")
                continue
            sub["ds"] = sub["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
            m = _prophet()
            m.fit(sub)
            self.models[sig] = m
            logger.info(f"forecast: trained {sig} on {len(sub)} rows")

        # ── intensity with renewable regressors ──────────────────────────────
        cols = ["timestamp", "intensity_actual"] + INTENSITY_REGRESSORS
        sub = (df[cols].dropna()
               .rename(columns={"timestamp": "ds", "intensity_actual": "y"}).copy())
        if len(sub) >= MIN_TRAIN_ROWS:
            sub["ds"] = sub["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
            m = _prophet()
            for reg in INTENSITY_REGRESSORS:
                m.add_regressor(reg)
            m.fit(sub)
            self.intensity_model = m
            logger.info(f"forecast: trained intensity_actual (regressors {INTENSITY_REGRESSORS}) on {len(sub)} rows")
        return self

    def _future_frame(self, anchor_ts) -> pd.DataFrame:
        anchor_naive = pd.Timestamp(anchor_ts).tz_convert("UTC").tz_localize(None)
        return pd.DataFrame({"ds": pd.date_range(
            start=anchor_naive + pd.Timedelta(FREQ), periods=HORIZON_PERIODS, freq=FREQ)})

    def _anchor_row(self, anchor_ts) -> pd.DataFrame:
        anchor_naive = pd.Timestamp(anchor_ts).tz_convert("UTC").tz_localize(None)
        return pd.DataFrame({"ds": [anchor_naive]})

    @staticmethod
    def _shift(fc: pd.DataFrame, offset: float) -> pd.DataFrame:
        """Persistence-anchor: shift a seasonal forecast so it continues from the
        last actual instead of snapping to the time-of-day mean. At a 2h horizon the
        current value carries most of the signal; Prophet supplies the *shape*."""
        for col in ("yhat", "yhat_lower", "yhat_upper"):
            fc[col] = fc[col] + offset
        return fc

    def _rows(self, fc, signal) -> list[dict]:
        out = []
        for _, r in fc.iterrows():
            out.append({
                "timestamp": r["ds"].tz_localize("UTC").to_pydatetime(),
                "signal": signal,
                "forecast_value": float(r["yhat"]),
                "lower_bound": float(r["yhat_lower"]),
                "upper_bound": float(r["yhat_upper"]),
                "model_version": MODEL_VERSION,
            })
        return out

    def predict(self, anchor_ts, latest: dict | None = None) -> list[dict]:
        """Forecast the next 2 hours starting just after `anchor_ts` (tz-aware UTC).

        `latest` maps each signal to its most recent actual value. When supplied,
        every signal's forecast is persistence-anchored to it, so the 2h line
        continues smoothly from 'now' rather than jumping to the seasonal mean.
        """
        latest = latest or {}
        future = self._future_frame(anchor_ts)
        anchor = self._anchor_row(anchor_ts)
        rows: list[dict] = []

        # forecast the univariate signals first (their yhat feeds the intensity model)
        reg_forecasts = {}
        for sig, model in self.models.items():
            fc = model.predict(future)
            if latest.get(sig) is not None:
                offset = float(latest[sig]) - float(model.predict(anchor)["yhat"].iloc[0])
                fc = self._shift(fc, offset)
            reg_forecasts[sig] = fc["yhat"].values
            rows += self._rows(fc, sig)

        # intensity, using the (anchored) wind/solar forecasts as regressors
        if self.intensity_model is not None and all(r in reg_forecasts for r in INTENSITY_REGRESSORS):
            ifuture = future.copy()
            for reg in INTENSITY_REGRESSORS:
                ifuture[reg] = reg_forecasts[reg]
            ifc = self.intensity_model.predict(ifuture)
            if latest.get("intensity_actual") is not None:
                arow = anchor.copy()
                for reg in INTENSITY_REGRESSORS:            # anchor uses the actual regressors
                    arow[reg] = float(latest.get(reg, reg_forecasts[reg][0]))
                offset = float(latest["intensity_actual"]) - float(self.intensity_model.predict(arow)["yhat"].iloc[0])
                ifc = self._shift(ifc, offset)
            rows += self._rows(ifc, "intensity_actual")
        return rows

    def save(self, path: str = FORECAST_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"models": self.models, "intensity_model": self.intensity_model}, path)
        logger.info(f"forecast: saved {len(self.models)} univariate + intensity model -> {path}")

    @classmethod
    def load(cls, path: str = FORECAST_PATH):
        if not os.path.exists(path):
            return None
        state = joblib.load(path)
        fc = cls()
        if isinstance(state, dict) and "models" in state:      # new format
            fc.models = state["models"]
            fc.intensity_model = state.get("intensity_model")
        else:                                                   # legacy: plain dict of models
            fc.models = {k: v for k, v in state.items() if k in UNIVARIATE_SIGNALS}
            fc.intensity_model = state.get("intensity_actual")
        return fc
