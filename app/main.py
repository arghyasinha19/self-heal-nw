import yaml
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Load .env file so ${VAR} placeholders in config.yaml resolve correctly
load_dotenv()

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

def _interpolate(value):
    """Recursively resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(value, str):
        return re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(i) for i in value]
    return value

def load_config() -> dict:
    with open(CONFIG_PATH, 'r') as f:
        raw = yaml.safe_load(f)
    return _interpolate(raw)

config = load_config()

# ─────────────────────────────────────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────────────────────────────────────
dnac_client = DNACClient(config['dnac'])
mq_publisher = RabbitMQPublisher(config['rabbitmq'])

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: startup / shutdown hooks
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    logger.info("Service starting up...")
    try:
        dnac_client.authenticate()
        dnac_client.register_webhook()
    except Exception as e:
        logger.error(f"Startup error during DNAC registration: {e}")
    yield
    # ── SHUTDOWN ──
    logger.info("Service shutting down...")
    dnac_client.deregister_webhook()
    mq_publisher.close()

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DNAC Webhook → RabbitMQ Ingestion Service",
    description=(
        "Receives push-based alert events from Cisco DNA Center via webhook "
        "and publishes them to RabbitMQ for the false-alert ML pipeline."
    ),
    version="2.0.0",
    lifespan=lifespan
)

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Operations"])
def health_check():
    """Liveness probe for load balancers / k8s."""
    return {"status": "healthy", "service": "dnac-webhook-ingestion"}


@app.post("/api/v1/webhook", tags=["Webhook"])
async def dnac_webhook_receiver(request: Request):
    """
    The HTTP endpoint that Cisco DNAC pushes alert events to.
    DNAC sends a JSON payload for each event matching our subscription filter.
    We parse it and publish it directly to RabbitMQ.
    """
    try:
        payload: Any = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload received.")

    # DNAC can send a single object or a list of events
    events = payload if isinstance(payload, list) else [payload]

    published = 0
    for event in events:
        try:
            # Enrich the event with ingestion metadata
            event['_source'] = "dnac-webhook"
            mq_publisher.publish(event)
            published += 1
            logger.info(
                f"Published event to RabbitMQ | "
                f"eventId={event.get('eventId', 'N/A')} | "
                f"severity={event.get('severity', 'N/A')} | "
                f"category={event.get('category', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"Failed to publish event {event.get('eventId', 'N/A')}: {e}")

    return JSONResponse(
        status_code=200,
        content={"status": "received", "events_published": published}
    )


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
    Manually register this service as a webhook receiver in DNAC.

    This is called automatically on service startup, but you can also trigger
    it manually here — for example, after changing the receiver_url in config,
    or if the subscription was deleted from the DNAC side.

    The operation is idempotent: if the subscription already exists (matched by
    name), DNAC will not create a duplicate.
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
