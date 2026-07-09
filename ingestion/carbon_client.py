"""
Client for api.carbonintensity.org.uk
Covers: national intensity, generation mix, regional intensity.
No API key required.
"""
import logging
from datetime import datetime, timedelta
import httpx
from models.schemas import (
    IntensityResponse, GenerationResponse, RegionalResponse
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.carbonintensity.org.uk"
TIMEOUT = 30.0


class CarbonIntensityClient:

    def __init__(self):
        self.client = httpx.Client(
            base_url=BASE_URL,
            timeout=TIMEOUT,
            headers={"Accept": "application/json"}
        )

    def get_current_intensity(self) -> IntensityResponse:
        """Current half-hour national carbon intensity."""
        resp = self.client.get("/intensity")
        resp.raise_for_status()
        return IntensityResponse.model_validate(resp.json())

    def get_intensity_range(
        self, from_dt: datetime, to_dt: datetime
    ) -> IntensityResponse:
        """
        Historical intensity between two datetimes.
        Max range: 14 days per call.
        """
        from_str = from_dt.strftime("%Y-%m-%dT%H:%MZ")
        to_str   = to_dt.strftime("%Y-%m-%dT%H:%MZ")
        resp = self.client.get(f"/intensity/{from_str}/{to_str}")
        resp.raise_for_status()
        return IntensityResponse.model_validate(resp.json())

    def get_forward_forecast(self) -> IntensityResponse:
        """
        Forward-looking national carbon intensity forecast for the next 48 hours.
        Every period has actual=None (future); forecast is populated.
        Used to power clean-windows / load-shifting recommendations.
        """
        from_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
        resp = self.client.get(f"/intensity/{from_str}/fw48h")
        resp.raise_for_status()
        return IntensityResponse.model_validate(resp.json())

    def get_current_generation(self) -> GenerationResponse:
        """Current half-hour generation mix by fuel type."""
        resp = self.client.get("/generation")
        resp.raise_for_status()
        return GenerationResponse.model_validate(resp.json())

    def get_generation_range(
        self, from_dt: datetime, to_dt: datetime
    ) -> GenerationResponse:
        """Historical generation mix between two datetimes."""
        from_str = from_dt.strftime("%Y-%m-%dT%H:%MZ")
        to_str   = to_dt.strftime("%Y-%m-%dT%H:%MZ")
        resp = self.client.get(f"/generation/{from_str}/{to_str}")
        resp.raise_for_status()
        return GenerationResponse.model_validate(resp.json())

    def get_current_regional(self) -> RegionalResponse:
        """Current half-hour carbon intensity for all GB regions."""
        resp = self.client.get("/regional")
        resp.raise_for_status()
        return RegionalResponse.model_validate(resp.json())

    def backfill(self, days: int = 90) -> list[dict]:
        """
        Pull historical intensity + generation for the last N days.
        Chunks into 14-day windows to respect API limits.
        Returns merged list of records ready for DB insertion.
        """
        records = []
        now = datetime.utcnow()
        chunk_days = 14

        for chunk_start in range(0, days, chunk_days):
            to_dt   = now - timedelta(days=chunk_start)
            from_dt = now - timedelta(days=min(chunk_start + chunk_days, days))

            try:
                intensity = self.get_intensity_range(from_dt, to_dt)
                generation = self.get_generation_range(from_dt, to_dt)

                # Index generation by timestamp for joining
                gen_map = {
                    p.from_time: p.as_dict()
                    for p in generation.data
                }

                for period in intensity.data:
                    gen = gen_map.get(period.from_time, {})
                    records.append({
                        "timestamp":           period.from_time,
                        "intensity_forecast":  period.intensity.forecast,
                        "intensity_actual":    period.intensity.actual,
                        "intensity_index":     period.intensity.index,
                        "gas_perc":            gen.get("gas"),
                        "coal_perc":           gen.get("coal"),
                        "nuclear_perc":        gen.get("nuclear"),
                        "wind_perc":           gen.get("wind"),
                        "solar_perc":          gen.get("solar"),
                        "hydro_perc":          gen.get("hydro"),
                        "biomass_perc":        gen.get("biomass"),
                        "imports_perc":        gen.get("imports"),
                        "other_perc":          gen.get("other"),
                    })

                logger.info(
                    f"Backfilled {from_dt.date()} → {to_dt.date()}: "
                    f"{len(intensity.data)} records"
                )

            except Exception as e:
                logger.error(f"Backfill chunk failed ({from_dt} → {to_dt}): {e}")

        return records

    def close(self):
        self.client.close()
