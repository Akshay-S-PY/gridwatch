"""
Feature engineering for the ML layer.

Loads grid_events into a pandas DataFrame with the four target signals plus
cyclical time-of-day features. Both anomaly detection and forecasting build on
this — the time features let Isolation Forest catch values that are unusual
*for the time of day*, not just globally extreme.
"""
import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.config import engine

logger = logging.getLogger(__name__)

# Signals anomaly detection runs on. forecast_error and renewable_perc are
# derived below; the other two are raw columns.
ANOMALY_SIGNALS = ["intensity_actual", "wind_perc", "forecast_error", "renewable_perc"]

# Columns appended to each signal's value to give Isolation Forest time context.
TIME_FEATURES = ["hour_sin", "hour_cos", "dow"]


def load_feature_frame(days: int = 90) -> pd.DataFrame:
    """Load the last N days of grid_events with derived signals + time features."""
    sql = text("""
        SELECT
            timestamp,
            intensity_actual,
            intensity_forecast,
            wind_perc, solar_perc, hydro_perc, biomass_perc,
            gas_perc, coal_perc
        FROM grid_events
        WHERE timestamp >= NOW() - make_interval(days => :days)
          AND intensity_actual IS NOT NULL
        ORDER BY timestamp ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"days": days}, parse_dates=["timestamp"])

    if df.empty:
        return df

    df["forecast_error"] = df["intensity_actual"] - df["intensity_forecast"]
    df["renewable_perc"] = (
        df[["wind_perc", "solar_perc", "hydro_perc", "biomass_perc"]]
        .fillna(0)
        .sum(axis=1)
    )

    hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dow"] = df["timestamp"].dt.dayofweek

    return df


def signal_matrix(df: pd.DataFrame, signal: str):
    """
    Return (sub, X) for a signal: `sub` keeps timestamp + value for labelling,
    `X` is the [value, hour_sin, hour_cos, dow] matrix Isolation Forest trains on.
    """
    cols = ["timestamp", signal] + TIME_FEATURES
    sub = df[cols].dropna()
    X = sub[[signal] + TIME_FEATURES].to_numpy()
    return sub, X
