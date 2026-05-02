"""
Shared State Schema for the DNAC Alert Processing Pipeline
───────────────────────────────────────────────────────────
Defines the TypedDict that flows through every node in the LangGraph.
All agents read from and write to this shared state.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AlertState(TypedDict, total=False):
    """
    The single source of truth passed through every node in the graph.

    Fields
    ──────
    alert_event : dict
        The raw DNAC alert payload received via webhook. This is the
        initial input and is never mutated by any agent.

    agent1_output : dict | None
        Output produced by Agent 1 (upstream processing).
        If this is None or empty after Agent 1 runs, Agent 2 is skipped.

    agent1_passed : bool
        Explicit gate flag — True if Agent 1 decided the alert should
        proceed to Agent 2. Set by Agent 1 node.

    agent2_output : dict | None
        Output produced by Agent 2 (DistilBERT classification).
        Contains predicted_category, confidence, and classification metadata.

    enriched_event : dict | None
        The final enriched alert event after all agents have run.
        This is what gets published to RabbitMQ.

    errors : list[str]
        Error messages accumulated during the pipeline.
        Non-fatal errors are appended here; the pipeline continues.

    workflow_metadata : dict
        Metadata about the pipeline run: timestamps, agent versions,
        execution times, etc.
    """

    # ── Inputs ──
    alert_event: dict[str, Any]

    # ── Agent 1 ──
    agent1_output: Optional[dict[str, Any]]
    agent1_passed: bool

    # ── Agent 2 ──
    agent2_output: Optional[dict[str, Any]]

    # ── Final ──
    enriched_event: Optional[dict[str, Any]]

    # ── Observability ──
    errors: list[str]
    workflow_metadata: dict[str, Any]
