"""
DNAC Alert Processing — Agentic Workflow Package
─────────────────────────────────────────────────
LangGraph-based multi-agent pipeline for DNAC alert processing.

Agent 1 (placeholder): User-defined upstream processing
Agent 2 (classifier) : DistilBERT auto-resolve classification

The graph executes Agent 2 only if Agent 1 produces valid output.
"""

from app.agents.graph import build_alert_graph, run_alert_pipeline  # noqa: F401

__all__ = ["build_alert_graph", "run_alert_pipeline"]
