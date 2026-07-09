"""
Phase 2 validation script.
Run this after backfill to confirm data is clean before ML training.

Usage (inside Docker):
    docker compose exec ingestion python tests/validate_data.py

Usage (local, with DB port-forwarded):
    python tests/validate_data.py
"""
import sys
sys.path.insert(0, ".")

from api.queries import check_data_quality, get_latest_grid_state, get_regional_snapshot
from db.config import SessionLocal
from sqlalchemy import text


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "✓" if condition else "✗"
    print(f"  {status}  {label}" + (f"  — {detail}" if detail else ""))
    return condition


def run_validation():
    print("\n" + "=" * 58)
    print("  GridWatch — Phase 2 Data Validation")
    print("=" * 58)

    passed = 0
    total  = 0

    # ── 1. Grid events ──────────────────────────────────────────
    print("\n── Grid Events ─────────────────────────────────────────")
    report = check_data_quality()

    total += 1
    ok = report.get("total_records", 0) > 100
    if check("Sufficient records", ok, f"{report.get('total_records', 0)} rows"):
        passed += 1

    total += 1
    completeness = report.get("completeness_pct", 0)
    ok = completeness >= 85
    if check("Completeness ≥ 85%", ok, f"{completeness}%"):
        passed += 1

    total += 1
    missing = report.get("missing_actual", 0)
    ok = missing < report.get("total_records", 1) * 0.1
    if check("Missing actuals < 10%", ok, f"{missing} missing"):
        passed += 1

    earliest = report.get("earliest")
    latest   = report.get("latest")
    total += 1
    ok = earliest is not None and latest is not None
    if check("Has timestamp range", ok, f"{earliest} → {latest}"):
        passed += 1

    # ── 2. Latest state ─────────────────────────────────────────
    print("\n── Latest Grid State ───────────────────────────────────")
    state = get_latest_grid_state()

    total += 1
    ok = bool(state)
    if check("Latest state exists", ok):
        passed += 1

    if state:
        total += 1
        ok = state.get("intensity_actual") is not None or state.get("intensity_forecast") is not None
        if check("Has intensity value", ok,
                 f"forecast={state.get('intensity_forecast')} actual={state.get('intensity_actual')}"):
            passed += 1

        total += 1
        ok = state.get("wind_perc") is not None
        if check("Has generation mix", ok,
                 f"wind={state.get('wind_perc')}% gas={state.get('gas_perc')}%"):
            passed += 1

        total += 1
        renewable = state.get("renewable_perc", 0) or 0
        ok = 0 <= renewable <= 100
        if check("Renewable % in valid range", ok, f"{round(renewable, 1)}%"):
            passed += 1

    # ── 3. Regional data ─────────────────────────────────────────
    print("\n── Regional Data ───────────────────────────────────────")
    regions = get_regional_snapshot()

    total += 1
    ok = len(regions) >= 10
    if check("Has regional data", ok, f"{len(regions)} regions"):
        passed += 1

    # ── 4. Weather data ──────────────────────────────────────────
    print("\n── Weather Data ────────────────────────────────────────")
    with SessionLocal() as session:
        weather_count = session.execute(
            text("SELECT COUNT(*) FROM weather_readings")
        ).scalar()

        weather_nulls = session.execute(
            text("SELECT COUNT(*) FROM weather_readings WHERE wind_speed_10m IS NULL")
        ).scalar()

    total += 1
    ok = (weather_count or 0) > 50
    if check("Has weather records", ok, f"{weather_count} rows"):
        passed += 1

    total += 1
    null_pct = (weather_nulls / weather_count * 100) if weather_count else 100
    ok = null_pct < 20
    if check("Weather nulls < 20%", ok, f"{round(null_pct, 1)}% null"):
        passed += 1

    # ── 5. DB connectivity ───────────────────────────────────────
    print("\n── Database ────────────────────────────────────────────")
    with SessionLocal() as session:
        tables = session.execute(text("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
        """)).fetchall()
        table_names = [t[0] for t in tables]

    expected = ["grid_events", "weather_readings", "regional_readings", "anomaly_flags", "forecasts"]
    for table in expected:
        total += 1
        ok = table in table_names
        if check(f"Table exists: {table}", ok):
            passed += 1

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 58}")
    print(f"  {passed}/{total} checks passed")
    if passed == total:
        print("  🟢 Data is clean — ready for Phase 3 (ML layer)")
    elif passed >= total * 0.8:
        print("  🟡 Mostly clean — check failures above before ML training")
    else:
        print("  🔴 Significant issues — re-run backfill before continuing")
    print(f"{'=' * 58}\n")

    return passed == total


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
