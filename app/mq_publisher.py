import pika
import pika.exceptions
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum reconnection attempts before giving up
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 2


class RabbitMQPublisher:
    """
    A production-grade RabbitMQ publisher with automatic reconnection and retry logic.

    Design decisions:
    - Uses a lazy connection that is established on first publish.
    - Automatically re-connects if the broker drops the connection (e.g. restart).
    - Retries publishing up to MAX_RETRIES times with exponential backoff.
    - Declares the queue as durable so it survives broker restarts.
    - Publishes messages as persistent (delivery_mode=2) so they survive broker restarts.
    """

    def __init__(self, config: dict):
        self.host = config['host']
        self.port = config.get('port', 5672)
        # Read credentials from env vars first, fall back to config dict
        self.username = os.environ.get('RABBITMQ_USERNAME') or config.get('username', 'guest')
        self.password = os.environ.get('RABBITMQ_PASSWORD') or config.get('password', 'guest')
        self.queue_name = config['queue']
        self.exchange = config.get('exchange', '')
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None

    # -------------------------------------------------------------------------
    # Internal: Connection Management
    # -------------------------------------------------------------------------
    def _build_parameters(self) -> pika.ConnectionParameters:
        credentials = pika.PlainCredentials(self.username, self.password)
        return pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            heartbeat=600,                    # Keep connection alive under load
            blocked_connection_timeout=300    # Raise error if broker blocks for too long
        )

    def _connect(self) -> None:
        """Establish a fresh connection and channel to RabbitMQ."""
        logger.info(f"Connecting to RabbitMQ at {self.host}:{self.port}...")
        self._connection = pika.BlockingConnection(self._build_parameters())
        self._channel = self._connection.channel()

        # Ensure queue exists and is durable so it survives broker restarts
        self._channel.queue_declare(queue=self.queue_name, durable=True)

        # Limit unacknowledged messages to 1, protecting the broker under burst load
        self._channel.basic_qos(prefetch_count=1)

        logger.info(f"RabbitMQ connected. Queue '{self.queue_name}' ready.")

    def _is_connected(self) -> bool:
        return (
            self._connection is not None
            and not self._connection.is_closed
            and self._channel is not None
            and self._channel.is_open
        )

    def _ensure_connected(self) -> None:
        """Reconnect if the connection has dropped."""
        if not self._is_connected():
            self._connect()

    # -------------------------------------------------------------------------
    # Public: Publish
    # -------------------------------------------------------------------------
    def publish(self, message: dict) -> None:
        """
        Publish a single JSON message to the configured queue.
        Retries up to MAX_RETRIES times with exponential backoff on connection failure.
        """
        body = json.dumps(message, default=str)  # default=str handles datetime objects
        properties = pika.BasicProperties(
            delivery_mode=2,                 # Persistent: survives broker restart
            content_type='application/json'
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._ensure_connected()
                self._channel.basic_publish(
                    exchange=self.exchange,
                    routing_key=self.queue_name,
                    body=body,
                    properties=properties
                )
                return  # Success — return immediately

            except (
                pika.exceptions.AMQPConnectionError,
                pika.exceptions.AMQPChannelError,
                pika.exceptions.StreamLostError,
                ConnectionResetError
            ) as e:
                delay = RETRY_DELAY_SECONDS * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(
                    f"RabbitMQ publish failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {delay}s..."
                )
                self._connection = None
                self._channel = None
                time.sleep(delay)

        raise RuntimeError(
            f"Failed to publish message to RabbitMQ after {MAX_RETRIES} attempts."
        )

    # -------------------------------------------------------------------------
    # Public: Lifecycle
    # -------------------------------------------------------------------------
    def close(self) -> None:
        """Gracefully close the connection."""
        if self._connection and not self._connection.is_closed:
            try:
                self._connection.close()
                logger.info("RabbitMQ connection closed.")
            except Exception as e:
                logger.warning(f"Error closing RabbitMQ connection: {e}")
