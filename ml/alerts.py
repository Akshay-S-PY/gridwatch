"""
Anomaly alerting: a Claude-generated one-line explanation plus a Slack push.

Both paths degrade gracefully so the pipeline runs without credentials:
  - no valid ANTHROPIC_API_KEY  -> templated explanation instead of Claude
  - no SLACK_WEBHOOK_URL        -> log the alert instead of posting

They light up automatically once real credentials are added to .env.
"""
import logging
import os

import httpx

from llm import client as llm

logger = logging.getLogger(__name__)

ALERT_SEVERITIES = {"high", "critical"}

_SYSTEM = "You are a UK National Grid operations assistant."


def _template(flag: dict) -> str:
    return (
        f"{flag['severity'].upper()} anomaly on {flag['signal']} = "
        f"{flag['value']:.1f} at {flag['timestamp']} "
        f"(isolation score {flag['anomaly_score']:.3f})."
    )


def explain_anomaly(flag: dict) -> str:
    """One-sentence operational explanation via the LLM, template fallback."""
    if not llm.available():
        return _template(flag)
    try:
        prompt = (
            "In ONE concise sentence, explain what the following grid anomaly likely "
            "indicates operationally and why it matters. Do not restate the numbers.\n\n"
            f"Signal: {flag['signal']}\n"
            f"Value: {flag['value']:.1f}\n"
            f"Severity: {flag['severity']}\n"
            f"Time (UTC): {flag['timestamp']}"
        )
        text = llm.chat(_SYSTEM, prompt, max_tokens=150)
        return text or _template(flag)
    except Exception as e:  # noqa: BLE001 — never let alerting break ingestion
        logger.warning(f"LLM explanation failed, using template: {e}")
        return _template(flag)


def send_slack(message: str) -> bool:
    url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not url:
        logger.info(f"[alert:log-only] {message}")
        return False
    try:
        httpx.post(url, json={"text": message}, timeout=10).raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001
        logger.error(f"Slack send failed: {e}")
        return False


def alert_if_severe(flags: list[dict]) -> list[dict]:
    """
    For high/critical flags: attach an explanation (used as llm_explanation on
    the DB row) and push to Slack. Mutates flags in place and returns them.
    """
    for flag in flags:
        if flag["severity"] in ALERT_SEVERITIES:
            explanation = explain_anomaly(flag)
            flag["llm_explanation"] = explanation
            send_slack(f":warning: GridWatch alert — {explanation}")
    return flags
