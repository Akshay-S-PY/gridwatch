"""
Client for api.open-meteo.com
Fetches wind, solar radiation, and temperature for UK grid centre.
No API key required. Free tier: 10,000 req/day.
"""
import logging
import os
from datetime import datetime, timedelta
import httpx
from models.schemas import WeatherResponse

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1"
# Historical data lives on a separate host, not the forecast host.
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Default to London — good proxy for national demand centre
DEFAULT_LAT = float(os.getenv("WEATHER_LAT", "51.5074"))
DEFAULT_LON = float(os.getenv("WEATHER_LON", "-0.1278"))


class WeatherClient:

    def __init__(self, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON):
        self.lat = lat
        self.lon = lon
        self.client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    def get_forecast(self, hours_ahead: int = 48) -> list[dict]:
        """
        Get hourly weather forecast for next N hours.
        Returns flat list of dicts ready for DB insertion.
        """
        resp = self.client.get("/forecast", params={
            "latitude":  self.lat,
            "longitude": self.lon,
            "hourly":    "wind_speed_10m,shortwave_radiation,temperature_2m",
            "wind_speed_unit": "ms",   # default is km/h; we store/label m/s
            "forecast_days": max(1, hours_ahead // 24 + 1),
            "timezone":  "UTC",
        })
        resp.raise_for_status()
        weather = WeatherResponse.model_validate(resp.json())
        return weather.to_records()

    def get_historical(self, days: int = 90) -> list[dict]:
        """
        Fetch historical hourly weather for backfill.
        Open-Meteo historical endpoint goes back years.
        """
        end_date   = datetime.utcnow().date()
        start_date = (datetime.utcnow() - timedelta(days=days)).date()

        resp = self.client.get(ARCHIVE_URL, params={
            "latitude":   self.lat,
            "longitude":  self.lon,
            "start_date": start_date.isoformat(),
            "end_date":   end_date.isoformat(),
            "hourly":     "wind_speed_10m,shortwave_radiation,temperature_2m",
            "wind_speed_unit": "ms",   # default is km/h; we store/label m/s
            "timezone":   "UTC",
        })
        resp.raise_for_status()
        weather = WeatherResponse.model_validate(resp.json())
        records = weather.to_records()
        logger.info(f"Weather backfill: {len(records)} hourly records fetched")
        return records

    def close(self):
        self.client.close()
