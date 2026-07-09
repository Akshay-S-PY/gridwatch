"""
Phase 1 smoke test — no DB needed.
Run this BEFORE Docker to confirm APIs are live and schemas parse correctly.

Usage:
    pip install httpx pydantic
    python tests/test_apis.py
"""
import sys
import json
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from ingestion.carbon_client import CarbonIntensityClient
from ingestion.weather_client import WeatherClient


def test_carbon_intensity():
    print("\n── Carbon Intensity API ─────────────────────────────")
    client = CarbonIntensityClient()

    result = client.get_current_intensity()
    period = result.data[0]
    print(f"  Period:    {period.from_time} → {period.to_time}")
    print(f"  Forecast:  {period.intensity.forecast} gCO2/kWh")
    print(f"  Actual:    {period.intensity.actual} gCO2/kWh")
    print(f"  Index:     {period.intensity.index}")
    print("  ✓ National intensity OK")

    client.close()


def test_generation_mix():
    print("\n── Generation Mix API ───────────────────────────────")
    client = CarbonIntensityClient()

    result = client.get_current_generation()
    period = result.data[0]
    mix = period.as_dict()
    print(f"  Period: {period.from_time} → {period.to_time}")
    for fuel, perc in sorted(mix.items(), key=lambda x: -x[1]):
        bar = "█" * int(perc / 2)
        print(f"  {fuel:<12} {perc:5.1f}%  {bar}")
    print("  ✓ Generation mix OK")

    client.close()


def test_regional():
    print("\n── Regional Carbon Intensity ─────────────────────────")
    client = CarbonIntensityClient()

    result = client.get_current_regional()
    period = result.data[0]
    print(f"  Period: {period.from_time}")
    for region in sorted(period.regions, key=lambda r: r.intensity.forecast):
        print(f"  {region.shortname:<30} {region.intensity.forecast:>4} gCO2  [{region.intensity.index}]")
    print("  ✓ Regional data OK")

    client.close()


def test_weather():
    print("\n── Open-Meteo Weather API ────────────────────────────")
    client = WeatherClient()

    records = client.get_forecast(hours_ahead=6)
    print(f"  Fetched {len(records)} hourly records")
    for r in records[:3]:
        print(
            f"  {r['timestamp']}  "
            f"wind={r['wind_speed_10m']}m/s  "
            f"solar={r['shortwave_radiation']}W/m²  "
            f"temp={r['temperature_2m']}°C"
        )
    print("  ✓ Weather OK")

    client.close()


def test_historical_sample():
    print("\n── Historical range (last 2 days) ────────────────────")
    client = CarbonIntensityClient()

    to_dt   = datetime.utcnow()
    from_dt = to_dt - timedelta(days=2)
    result  = client.get_intensity_range(from_dt, to_dt)
    print(f"  Fetched {len(result.data)} half-hour periods")
    print(f"  First: {result.data[0].from_time}  intensity={result.data[0].intensity.actual}")
    print(f"  Last:  {result.data[-1].from_time}  intensity={result.data[-1].intensity.actual}")
    print("  ✓ Historical range OK")

    client.close()


if __name__ == "__main__":
    print("=" * 55)
    print("  GridWatch — Phase 1 API smoke test")
    print("=" * 55)

    tests = [
        test_carbon_intensity,
        test_generation_mix,
        test_regional,
        test_weather,
        test_historical_sample,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print(f"\n{'=' * 55}")
    print(f"  {passed}/{len(tests)} tests passed")
    print(f"{'=' * 55}\n")

    if passed < len(tests):
        sys.exit(1)
