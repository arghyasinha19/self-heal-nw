"""
Agent 1 — Upstream Alert Processing (PLACEHOLDER)
──────────────────────────────────────────────────
This is a placeholder for your existing Agent 1 implementation.

Agent 1 receives the raw DNAC alert event and performs upstream
processing (e.g., enrichment, filtering, deduplication, severity
assessment). It then signals whether the alert should proceed
to Agent 2 (classification) by setting `agent1_passed = True`.

Replace the placeholder logic below with your actual Agent 1
implementation. The contract is:

    INPUT  →  state["alert_event"]   (raw DNAC alert dict)
    OUTPUT →  state["agent1_output"] (your processed output)
              state["agent1_passed"] (bool: proceed to Agent 2?)
"""

import logging
import time
from typing import Any, Dict

from app.agents.state import AlertState

logger = logging.getLogger(__name__)


def agent1_node(state: AlertState) -> Dict[str, Any]:
    """
    LangGraph node for Agent 1.

    This placeholder implementation:
    1. Passes the alert through unchanged
    2. Sets agent1_passed = True (always forwards to Agent 2)

    ─────────────────────────────────────────────────────────────────
    TODO: Replace with your actual Agent 1 logic.

    Example integrations:
    - LLM-based alert triage (e.g., severity classification)
    - Rule-based filtering (e.g., ignore known false positives)
    - Alert enrichment (e.g., fetch device context from DNAC)
    - Deduplication (e.g., suppress duplicate alerts within a window)
    ─────────────────────────────────────────────────────────────────
    """
    start_time = time.time()
    alert_event = state.get("alert_event", {})
    errors = list(state.get("errors", []))

    logger.info(
        f"Agent 1 processing alert | "
        f"eventId={alert_event.get('eventId', 'N/A')}"
    )

    try:
        # ┌─────────────────────────────────────────────────────────────┐
        # │  PLACEHOLDER: Replace this block with your Agent 1 logic   │
        # └─────────────────────────────────────────────────────────────┘

        agent1_output = {
            "status": "processed",
            "event_id": alert_event.get("eventId", "N/A"),
            "description": alert_event.get("description", ""),
            "severity": alert_event.get("severity", "unknown"),
            "agent": "agent1",
            "action": "pass_through",  # Replace with your actual action
        }

        # Gate: should Agent 2 run?
        # Set to False to skip classification for this alert.
        agent1_passed = True

        # ┌─────────────────────────────────────────────────────────────┐
        # │  END PLACEHOLDER                                           │
        # └─────────────────────────────────────────────────────────────┘

        elapsed = time.time() - start_time
        logger.info(
            f"Agent 1 completed in {elapsed:.3f}s | "
            f"passed={agent1_passed}"
        )

        return {
            "agent1_output": agent1_output,
            "agent1_passed": agent1_passed,
            "errors": errors,
            "workflow_metadata": {
                **state.get("workflow_metadata", {}),
                "agent1_elapsed_seconds": round(elapsed, 4),
            },
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Agent 1 failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)

        return {
            "agent1_output": None,
            "agent1_passed": False,
            "errors": errors,
            "workflow_metadata": {
                **state.get("workflow_metadata", {}),
                "agent1_elapsed_seconds": round(elapsed, 4),
                "agent1_error": str(e),
            },
        }
