"""
LangGraph Workflow — DNAC Alert Processing Pipeline
────────────────────────────────────────────────────
Defines the directed graph that orchestrates Agent 1 → Agent 2.

Graph structure:

    START → agent1 → should_continue? ──→ agent2 → END
                         │
                         └──→ agent2_skip → END

The conditional edge checks `agent1_passed`:
  - True  → invoke Agent 2 (classification)
  - False → skip Agent 2, pass event through unclassified
"""

import logging
import time
import uuid
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from app.agents.state import AlertState
from app.agents.agent1 import agent1_node
from app.agents.agent2 import agent2_node, agent2_skip_node

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Edge: Agent 1 Gate
# ─────────────────────────────────────────────────────────────────────────────
def should_continue_to_agent2(state: AlertState) -> str:
    """
    Routing function for the conditional edge after Agent 1.

    Returns:
        "agent2"      — if Agent 1 passed the alert forward
        "agent2_skip" — if Agent 1 blocked the alert
    """
    passed = state.get("agent1_passed", False)

    if passed:
        logger.info("Gate: Agent 1 PASSED → routing to Agent 2")
        return "agent2"
    else:
        logger.info("Gate: Agent 1 BLOCKED → skipping Agent 2")
        return "agent2_skip"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_alert_graph() -> StateGraph:
    """
    Build and compile the LangGraph state graph for alert processing.

    Returns a compiled graph that can be invoked with:
        graph.invoke({"alert_event": {...}})
    """
    # Create the graph with our state schema
    graph = StateGraph(AlertState)

    # ── Add nodes ──
    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)
    graph.add_node("agent2_skip", agent2_skip_node)

    # ── Define edges ──
    # START → Agent 1
    graph.set_entry_point("agent1")

    # Agent 1 → conditional → Agent 2 or Skip
    graph.add_conditional_edges(
        "agent1",
        should_continue_to_agent2,
        {
            "agent2": "agent2",
            "agent2_skip": "agent2_skip",
        },
    )

    # Agent 2 → END
    graph.add_edge("agent2", END)

    # Agent 2 Skip → END
    graph.add_edge("agent2_skip", END)

    # ── Compile ──
    compiled = graph.compile()
    logger.info("Alert processing graph compiled successfully.")

    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# Module-level compiled graph singleton
# ─────────────────────────────────────────────────────────────────────────────
_compiled_graph = None


def get_graph():
    """Get or build the compiled graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_alert_graph()
    return _compiled_graph


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_alert_pipeline(alert_event: dict) -> Dict[str, Any]:
    """
    Execute the full agent pipeline on a single DNAC alert event.

    This is the primary entry point for the workflow. It:
    1. Initializes the state with the raw alert event
    2. Runs Agent 1 → (conditional) → Agent 2
    3. Returns the final state including the enriched event

    Args:
        alert_event: Raw DNAC alert payload dict.

    Returns:
        Final AlertState dict with all fields populated.
    """
    run_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(
        f"Pipeline [{run_id}] starting | "
        f"eventId={alert_event.get('eventId', 'N/A')}"
    )

    # Initialize state
    initial_state: AlertState = {
        "alert_event": alert_event,
        "agent1_output": None,
        "agent1_passed": False,
        "agent2_output": None,
        "enriched_event": None,
        "errors": [],
        "workflow_metadata": {
            "run_id": run_id,
            "start_time": time.time(),
        },
    }

    # Execute graph
    graph = get_graph()
    final_state = graph.invoke(initial_state)

    # Add final timing metadata
    elapsed = time.time() - start_time
    if "workflow_metadata" not in final_state:
        final_state["workflow_metadata"] = {}
    final_state["workflow_metadata"]["total_elapsed_seconds"] = round(elapsed, 4)
    final_state["workflow_metadata"]["run_id"] = run_id

    # Log summary
    enriched = final_state.get("enriched_event", {})
    prediction = enriched.get("predicted_category", "N/A")
    confidence = enriched.get("prediction_confidence", "N/A")
    error_count = len(final_state.get("errors", []))

    logger.info(
        f"Pipeline [{run_id}] complete in {elapsed:.3f}s | "
        f"prediction={prediction} | "
        f"confidence={confidence} | "
        f"errors={error_count}"
    )

    return final_state
