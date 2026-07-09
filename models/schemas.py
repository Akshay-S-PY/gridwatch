"""
Pydantic schemas for all external API responses.
These act as a contract — if the API changes shape, we catch it here immediately.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─── Carbon Intensity API ────────────────────────────────────────────────────

class IntensityValue(BaseModel):
    forecast: int
    actual: Optional[int] = None
    index: str  # very low | low | moderate | high | very high

class IntensityPeriod(BaseModel):
    from_time: datetime = Field(alias="from")
    to_time: datetime = Field(alias="to")
    intensity: IntensityValue

    model_config = {"populate_by_name": True}

class IntensityResponse(BaseModel):
    data: list[IntensityPeriod]


# ─── Generation Mix ──────────────────────────────────────────────────────────

class FuelMix(BaseModel):
    fuel: str   # gas | coal | nuclear | wind | solar | hydro | biomass | imports | other
    perc: float

    @field_validator("perc")
    @classmethod
    def clamp_perc(cls, v: float) -> float:
        return round(max(0.0, min(100.0, v)), 2)

class GenerationPeriod(BaseModel):
    from_time: datetime = Field(alias="from")
    to_time: datetime = Field(alias="to")
    generationmix: list[FuelMix]

    model_config = {"populate_by_name": True}

    def as_dict(self) -> dict[str, float]:
        """Flatten fuel mix into {fuel: perc} dict for easy DB insertion."""
        return {f.fuel: f.perc for f in self.generationmix}

class GenerationResponse(BaseModel):
    data: list[GenerationPeriod]

    @field_validator("data", mode="before")
    @classmethod
    def wrap_single_period(cls, v):
        """
        /generation (current) returns `data` as a single object, while
        /generation/{from}/{to} (range) returns `data` as a list. Normalise
        the single-object form into a one-element list so both parse.
        """
        if isinstance(v, dict):
            return [v]
        return v


# ─── Regional Intensity ──────────────────────────────────────────────────────

class RegionalIntensity(BaseModel):
    forecast: int
    index: str

class Region(BaseModel):
    regionid: int
    dnoregion: str
    shortname: str
    intensity: RegionalIntensity
    generationmix: list[FuelMix]

class RegionalPeriod(BaseModel):
    from_time: datetime = Field(alias="from")
    to_time: datetime = Field(alias="to")
    regions: list[Region]

    model_config = {"populate_by_name": True}

class RegionalResponse(BaseModel):
    data: list[RegionalPeriod]


# ─── Open-Meteo Weather ──────────────────────────────────────────────────────

class HourlyWeather(BaseModel):
    time: list[str]
    wind_speed_10m: list[Optional[float]]
    shortwave_radiation: list[Optional[float]]
    temperature_2m: list[Optional[float]]

class WeatherResponse(BaseModel):
    latitude: float
    longitude: float
    hourly: HourlyWeather

    def to_records(self) -> list[dict]:
        """Zip time + values into a flat list of dicts."""
        records = []
        for i, t in enumerate(self.hourly.time):
            records.append({
                "timestamp": datetime.fromisoformat(t),
                "wind_speed_10m": self.hourly.wind_speed_10m[i],
                "shortwave_radiation": self.hourly.shortwave_radiation[i],
                "temperature_2m": self.hourly.temperature_2m[i],
            })
        return records


# ─── Internal DB schemas (what we write to Postgres) ─────────────────────────

class GridEventDB(BaseModel):
    """Normalised row written to grid_events table."""
    timestamp: datetime
    intensity_forecast: Optional[int]
    intensity_actual: Optional[int]
    intensity_index: Optional[str]
    gas_perc: Optional[float]
    coal_perc: Optional[float]
    nuclear_perc: Optional[float]
    wind_perc: Optional[float]
    solar_perc: Optional[float]
    hydro_perc: Optional[float]
    biomass_perc: Optional[float]
    imports_perc: Optional[float]
    other_perc: Optional[float]

class WeatherDB(BaseModel):
    """Row written to weather_readings table."""
    timestamp: datetime
    wind_speed_10m: Optional[float]
    shortwave_radiation: Optional[float]
    temperature_2m: Optional[float]

class RegionalReadingDB(BaseModel):
    """Row written to regional_readings table."""
    timestamp: datetime
    region_id: int
    region_name: str
    intensity_forecast: int
    intensity_index: str
    wind_perc: Optional[float]
    solar_perc: Optional[float]
    gas_perc: Optional[float]
