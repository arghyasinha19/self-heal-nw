import yaml
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# find_dotenv() walks UP from the CWD until it finds a .env file — works
# regardless of which directory uvicorn is launched from.
load_dotenv(find_dotenv(), override=True)

from app.dnac_client import DNACClient
from app.mq_publisher import RabbitMQPublisher

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

def load_config() -> dict:
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# ─────────────────────────────────────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────────────────────────────────────
dnac_client = DNACClient(config['dnac'])
mq_publisher = RabbitMQPublisher(config['rabbitmq'])

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models for API
# ─────────────────────────────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    description: str

class ClassifyBatchRequest(BaseModel):
    descriptions: list[str]

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: startup / shutdown hooks
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    logger.info("Service starting up...")

    # Confirm credentials were loaded from .env
    _required_vars = ["DNAC_USERNAME", "DNAC_PASSWORD", "RABBITMQ_USERNAME", "RABBITMQ_PASSWORD"]
    for var in _required_vars:
        val = os.environ.get(var)
        if val:
            logger.info(f"  ✔ {var} is set ({len(val)} chars)")
        else:
            logger.warning(f"  ✘ {var} is NOT set — check your .env file!")

    # Only authenticate to warm up the token.
    # Webhook registration is a deliberate one-time action — use:
    #   POST /api/v1/subscriptions/register  (via Swagger at /docs)
    try:
        dnac_client.authenticate()
        logger.info("DNAC token obtained. Service is ready.")
        logger.info("➜ To register this service as a DNAC webhook receiver, call:")
        logger.info("  POST /api/v1/subscriptions/register  (Swagger: http://<host>:8000/docs)")
    except Exception as e:
        logger.error(f"Startup DNAC authentication failed: {e}")

    # ── Warm up the LangGraph agent pipeline ──
    classifier_config = config.get("classifier", {})
    if classifier_config.get("enabled", True):
        try:
            from app.agents.graph import get_graph
            get_graph()  # Compiles the graph; Agent 2 lazy-loads the model on first call
            logger.info("✔ LangGraph agent pipeline compiled and ready.")
        except Exception as e:
            logger.warning(f"⚠ Agent pipeline initialization warning: {e}")
            logger.warning("  Pipeline will attempt to initialize on first request.")
    else:
        logger.info("ℹ Agent pipeline is disabled in config.yaml (classifier.enabled: false)")

    yield

    # ── SHUTDOWN ──
    logger.info("Service shutting down...")
    mq_publisher.close()

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DNAC Webhook → Agentic Processing → RabbitMQ Service",
    description=(
        "Receives push-based alert events from Cisco DNA Center via webhook, "
        "processes them through a LangGraph multi-agent pipeline "
        "(Agent 1: upstream processing → Agent 2: DistilBERT classification), "
        "and publishes enriched events to RabbitMQ."
    ),
    version="3.0.0",
    lifespan=lifespan
)

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Operations"])
def health_check():
    """Liveness probe for load balancers / k8s."""
    return {
        "status": "healthy",
        "service": "dnac-webhook-ingestion",
    }


@app.post("/api/v1/webhook", tags=["Webhook"])
async def dnac_webhook_receiver(request: Request):
    """
    The HTTP endpoint that Cisco DNAC pushes alert events to.
    DNAC sends a JSON payload for each event matching our subscription filter.

    Each event is processed through the LangGraph agent pipeline:
      Agent 1 (upstream) → conditional gate → Agent 2 (classification)

    The enriched event is then published to RabbitMQ.
    """
    try:
        payload: Any = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload received.")

    # DNAC can send a single object or a list of events
    events = payload if isinstance(payload, list) else [payload]

    classifier_enabled = config.get("classifier", {}).get("enabled", True)
    published = 0

    for event in events:
        try:
            if classifier_enabled:
                # ── Run through agent pipeline ──
                from app.agents.graph import run_alert_pipeline

                final_state = run_alert_pipeline(event)
                enriched_event = final_state.get("enriched_event", event)
                enriched_event["_workflow_metadata"] = final_state.get("workflow_metadata", {})

                if final_state.get("errors"):
                    enriched_event["_workflow_errors"] = final_state["errors"]
            else:
                # Pipeline disabled — pass through raw
                enriched_event = event
                enriched_event["_source"] = "dnac-webhook"

            mq_publisher.publish(enriched_event)
            published += 1

            logger.info(
                f"Published event to RabbitMQ | "
                f"eventId={event.get('eventId', 'N/A')} | "
                f"severity={event.get('severity', 'N/A')} | "
                f"predicted={enriched_event.get('predicted_category', 'N/A')} | "
                f"confidence={enriched_event.get('prediction_confidence', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"Failed to process/publish event {event.get('eventId', 'N/A')}: {e}")

    return JSONResponse(
        status_code=200,
        content={"status": "received", "events_published": published}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Pipeline Endpoints (for testing & debugging)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/pipeline/run", tags=["Agent Pipeline"])
async def run_pipeline(request: Request):
    """
    Run a raw event through the full LangGraph agent pipeline.
    Returns the complete final state including all agent outputs.
    Useful for debugging the pipeline without publishing to RabbitMQ.
    """
    try:
        event: Any = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    from app.agents.graph import run_alert_pipeline

    final_state = run_alert_pipeline(event)

    return {
        "status": "completed",
        "enriched_event": final_state.get("enriched_event"),
        "agent1_output": final_state.get("agent1_output"),
        "agent1_passed": final_state.get("agent1_passed"),
        "agent2_output": final_state.get("agent2_output"),
        "errors": final_state.get("errors", []),
        "workflow_metadata": final_state.get("workflow_metadata", {}),
    }


@app.post("/api/v1/classify", tags=["Agent Pipeline"])
def classify_description(req: ClassifyRequest):
    """
    Classify a single alert description through Agent 2 only.
    Bypasses Agent 1 — useful for testing the classifier in isolation.
    """
    # Build a minimal event and run just the classification
    from app.agents.graph import run_alert_pipeline

    event = {"description": req.description, "eventId": "manual-test"}
    final_state = run_alert_pipeline(event)

    enriched = final_state.get("enriched_event", {})
    classification = enriched.get("_classification", {})

    return {
        "description": req.description,
        "predicted_category": enriched.get("predicted_category", "N/A"),
        "confidence": enriched.get("prediction_confidence", "N/A"),
        "classification_detail": classification,
        "workflow_metadata": final_state.get("workflow_metadata", {}),
    }


@app.post("/api/v1/classify/batch", tags=["Agent Pipeline"])
def classify_batch(req: ClassifyBatchRequest):
    """
    Classify multiple alert descriptions through the full pipeline.
    Each description is run independently through Agent 1 → Agent 2.
    """
    from app.agents.graph import run_alert_pipeline

    results = []
    for desc in req.descriptions:
        event = {"description": desc, "eventId": "batch-test"}
        final_state = run_alert_pipeline(event)
        enriched = final_state.get("enriched_event", {})

        results.append({
            "description": desc,
            "predicted_category": enriched.get("predicted_category", "N/A"),
            "confidence": enriched.get("prediction_confidence", "N/A"),
        })

    return {"results": results, "total": len(results)}


@app.get("/api/v1/pipeline/info", tags=["Agent Pipeline"])
def pipeline_info():
    """
    Return metadata about the agent pipeline and loaded classifier model.
    """
    info = {
        "pipeline_enabled": config.get("classifier", {}).get("enabled", True),
        "confidence_threshold": config.get("classifier", {}).get("confidence_threshold", 0.6),
        "agents": {
            "agent1": {
                "name": "Upstream Processing",
                "status": "placeholder",
                "description": "Replace with your Agent 1 implementation",
            },
            "agent2": {
                "name": "DistilBERT Classification",
                "status": "unknown",
            },
        },
    }

    # Try to get Agent 2 model info
    try:
        from app.agents.agent2 import _get_classifier
        classifier = _get_classifier()
        if classifier:
            info["agents"]["agent2"]["status"] = "loaded"
            info["agents"]["agent2"]["model_info"] = classifier.get_info()
        else:
            info["agents"]["agent2"]["status"] = "not_loaded"
            info["agents"]["agent2"]["detail"] = (
                "Train a model first: python train_model.py --data data/training_data.csv"
            )
    except Exception as e:
        info["agents"]["agent2"]["status"] = "error"
        info["agents"]["agent2"]["error"] = str(e)

    return info


# ─────────────────────────────────────────────────────────────────────────────
# Webhook Management Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/subscriptions", tags=["Webhook Management"])
def list_subscriptions():
    """
    List all webhook/event subscriptions currently registered in DNAC.
    Use this to confirm that this service is correctly registered as a receiver.
    """
    try:
        subs = dnac_client.list_event_subscriptions()
        return {
            "total": len(subs),
            "subscriptions": subs
        }
    except Exception as e:
        logger.error(f"Failed to list subscriptions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/subscriptions/register", tags=["Webhook Management"])
def register_webhook():
    """
    Register this service as a REST/Webhook subscriber in DNAC.
    DNAC will then push event payloads to our FastAPI receiver_url.
    This is always a manual, on-demand action — never called automatically.

    The operation is idempotent: if a subscription with the same name already
    exists in DNAC, it will be returned as-is without creating a duplicate.
    """
    try:
        # Re-authenticate to ensure the token is fresh before registering
        dnac_client.authenticate()
        result = dnac_client.register_webhook()
        return {
            "status": "registered",
            "subscription": result
        }
    except Exception as e:
        logger.error(f"Failed to register webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/subscriptions/deregister", tags=["Webhook Management"])
def deregister_webhook():
    """
    Manually de-register this service's webhook subscription from DNAC.

    After calling this, DNAC will stop pushing events to this service.
    You can re-register at any time using POST /api/v1/subscriptions/register.
    """
    try:
        if not dnac_client._subscription_id:
            # Try to find it from the live list first
            subs = dnac_client.list_event_subscriptions()
            target_name = dnac_client.webhook_config.get('name', 'FalseAlertDetection')
            for sub in subs:
                if sub.get('name') == target_name:
                    dnac_client._subscription_id = sub.get('subscriptionId')
                    break

        if not dnac_client._subscription_id:
            return {
                "status": "not_found",
                "detail": "No active subscription found with the configured name. Nothing to deregister."
            }

        dnac_client.deregister_webhook()
        return {
            "status": "deregistered",
            "detail": "Webhook subscription removed from DNAC. DNAC will no longer push events to this service."
        }
    except Exception as e:
        logger.error(f"Failed to deregister webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))
