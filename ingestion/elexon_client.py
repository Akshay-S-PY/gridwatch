"""
Client for the Elexon Insights (BMRS) API — data.elexon.co.uk.

Free, no API key. Provides GB grid signals the Carbon Intensity API doesn't:
  - system frequency (Hz)
  - national demand (INDO, MW)
"""
import logging
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
TIMEOUT = 30.0


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%MZ")


class ElexonClient:
    def __init__(self):
        self.client = httpx.Client(
            base_url=BASE_URL, timeout=TIMEOUT, headers={"Accept": "application/json"}
        )

    def get_latest_frequency(self) -> dict | None:
        """Most recent system frequency reading (Hz). Frequency updates ~every 15s."""
        now = datetime.utcnow()
        resp = self.client.get("/system/frequency", params={
            "from": _iso(now - timedelta(minutes=15)),
            "to": _iso(now),
        })
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None
        last = data[-1]
        return {"timestamp": last["measurementTime"], "frequency_hz": last["frequency"]}

    def get_demand(self) -> list[dict]:
        """
        National demand (INDO, MW) per 30-min settlement period. The endpoint
        returns a fixed recent window (~30 days), which naturally backfills the
        demand history — each poll re-fetches it and idempotent writes dedupe.
        """
        resp = self.client.get("/demand/outturn")
        resp.raise_for_status()
        rows = []
        for d in resp.json().get("data", []):
            if d.get("initialDemandOutturn") is None:
                continue
            rows.append({
                "timestamp": d["startTime"],
                "demand_mw": d["initialDemandOutturn"],
            })
        return rows

    def close(self):
        self.client.close()
