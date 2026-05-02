"""
Agent 2 — DistilBERT Alert Classification
──────────────────────────────────────────
Classifies DNAC alerts as "Auto resolving" or "Non-Auto Resolving"
using the fine-tuned DistilBERT model.

This agent is only invoked if Agent 1 sets `agent1_passed = True`.

INPUT  →  state["alert_event"]   (raw event — for description extraction)
          state["agent1_output"] (Agent 1's processed output)
OUTPUT →  state["agent2_output"] (classification result)
          state["enriched_event"](final enriched alert event)
"""

import logging
import time
from typing import Any, Dict, Optional

from app.agents.state import AlertState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level classifier singleton (loaded once, reused across invocations)
# ─────────────────────────────────────────────────────────────────────────────
_classifier = None
_classifier_load_attempted = False


def _get_classifier():
    """
    Lazy-load the AlertClassifier singleton.
    Returns None if the model is not available (not yet trained).
    """
    global _classifier, _classifier_load_attempted

    if _classifier is not None:
        return _classifier

    if _classifier_load_attempted:
        return None  # Already tried and failed; don't retry every call

    _classifier_load_attempted = True

    try:
        import yaml
        import os

        # Load model path from config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config.yaml",
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model_path = config.get("classifier", {}).get("model_path", "models")

        from app.classifier import AlertClassifier

        _classifier = AlertClassifier.load(model_path)
        logger.info(f"Agent 2: Classifier loaded from {model_path}")
        return _classifier

    except FileNotFoundError:
        logger.warning(
            "Agent 2: No trained model found. "
            "Run: python train_model.py --data data/training_data.csv"
        )
        return None
    except Exception as e:
        logger.error(f"Agent 2: Failed to load classifier: {e}")
        return None


def _extract_description(state: AlertState) -> str:
    """
    Extract the alert description from the state.
    Checks multiple possible locations in the event payload.
    """
    alert_event = state.get("alert_event", {})
    agent1_output = state.get("agent1_output", {})

    # Priority: agent1_output.description > alert_event.description > alert_event.details.description
    description = (
        (agent1_output or {}).get("description")
        or alert_event.get("description")
        or alert_event.get("details", {}).get("description", "")
        or alert_event.get("name", "")
    )

    return str(description).strip()


def agent2_node(state: AlertState) -> Dict[str, Any]:
    """
    LangGraph node for Agent 2 — DistilBERT classification.

    Classifies the alert description and produces an enriched event
    with the prediction injected.
    """
    start_time = time.time()
    alert_event = state.get("alert_event", {})
    errors = list(state.get("errors", []))

    logger.info(
        f"Agent 2 classifying alert | "
        f"eventId={alert_event.get('eventId', 'N/A')}"
    )

    # ── Load classifier ──
    classifier = _get_classifier()

    if classifier is None:
        elapsed = time.time() - start_time
        error_msg = "Agent 2: Classifier not available — passing through unclassified"
        logger.warning(error_msg)
        errors.append(error_msg)

        # Build enriched event without classification
        enriched = {**alert_event, "_source": "dnac-webhook"}
        enriched["_classification"] = {
            "status": "skipped",
            "reason": "classifier_not_loaded",
        }

        return {
            "agent2_output": {"status": "skipped", "reason": "classifier_not_loaded"},
            "enriched_event": enriched,
            "errors": errors,
            "workflow_metadata": {
                **state.get("workflow_metadata", {}),
                "agent2_elapsed_seconds": round(elapsed, 4),
            },
        }

    # ── Extract description ──
    description = _extract_description(state)

    if not description:
        elapsed = time.time() - start_time
        enriched = {**alert_event, "_source": "dnac-webhook"}
        enriched["_classification"] = {
            "status": "skipped",
            "reason": "no_description_found",
        }

        return {
            "agent2_output": {"status": "skipped", "reason": "no_description_found"},
            "enriched_event": enriched,
            "errors": errors,
            "workflow_metadata": {
                **state.get("workflow_metadata", {}),
                "agent2_elapsed_seconds": round(elapsed, 4),
            },
        }

    # ── Classify ──
    try:
        import yaml, os

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config.yaml",
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        threshold = config.get("classifier", {}).get("confidence_threshold", 0.6)

        result = classifier.predict(description)
        elapsed = time.time() - start_time

        # Apply confidence threshold
        if result["confidence"] >= threshold:
            final_category = result["category"]
        else:
            final_category = "uncertain"

        classification_output = {
            "status": "success",
            "predicted_category": final_category,
            "raw_prediction": result["category"],
            "confidence": result["confidence"],
            "label_id": result["label_id"],
            "threshold_applied": threshold,
        }

        # Build enriched event
        enriched = {**alert_event, "_source": "dnac-webhook"}
        enriched["predicted_category"] = final_category
        enriched["prediction_confidence"] = result["confidence"]
        enriched["_classification"] = classification_output

        # Merge Agent 1 output if present
        agent1_output = state.get("agent1_output")
        if agent1_output:
            enriched["_agent1"] = agent1_output

        logger.info(
            f"Agent 2 classified alert | "
            f"prediction={final_category} | "
            f"confidence={result['confidence']:.4f} | "
            f"elapsed={elapsed:.3f}s"
        )

        return {
            "agent2_output": classification_output,
            "enriched_event": enriched,
            "errors": errors,
            "workflow_metadata": {
                **state.get("workflow_metadata", {}),
                "agent2_elapsed_seconds": round(elapsed, 4),
            },
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Agent 2 classification failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)

        # Enriched event without classification on error
        enriched = {**alert_event, "_source": "dnac-webhook"}
        enriched["_classification"] = {"status": "error", "reason": str(e)}

        return {
            "agent2_output": {"status": "error", "reason": str(e)},
            "enriched_event": enriched,
            "errors": errors,
            "workflow_metadata": {
                **state.get("workflow_metadata", {}),
                "agent2_elapsed_seconds": round(elapsed, 4),
                "agent2_error": str(e),
            },
        }


def agent2_skip_node(state: AlertState) -> Dict[str, Any]:
    """
    LangGraph node invoked when Agent 1 does NOT pass the alert forward.
    Builds the enriched event without classification.
    """
    alert_event = state.get("alert_event", {})

    logger.info(
        f"Agent 2 SKIPPED (Agent 1 did not pass) | "
        f"eventId={alert_event.get('eventId', 'N/A')}"
    )

    enriched = {**alert_event, "_source": "dnac-webhook"}
    enriched["_classification"] = {
        "status": "skipped",
        "reason": "agent1_did_not_pass",
    }

    # Merge Agent 1 output
    agent1_output = state.get("agent1_output")
    if agent1_output:
        enriched["_agent1"] = agent1_output

    return {
        "agent2_output": {"status": "skipped", "reason": "agent1_did_not_pass"},
        "enriched_event": enriched,
        "workflow_metadata": {
            **state.get("workflow_metadata", {}),
            "agent2_elapsed_seconds": 0,
        },
    }
