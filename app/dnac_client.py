import requests
from requests.auth import HTTPBasicAuth
import urllib3
import logging
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger(__name__)


class DNACClient:
    def __init__(self, config: dict):
        self.base_url = config['base_url'].rstrip('/')
        # Read credentials from env vars first, fall back to config dict
        self.username = os.environ.get('DNAC_USERNAME') or config.get('username')
        self.password = os.environ.get('DNAC_PASSWORD') or config.get('password')
        self.verify_ssl = config.get('verify_ssl', False)
        self.webhook_config = config.get('webhook_registration', {})
        self.token = None
        self._subscription_id = None

        if not self.username or not self.password:
            raise ValueError(
                "DNAC credentials missing. Set DNAC_USERNAME and DNAC_PASSWORD in your .env file."
            )

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------
    def authenticate(self) -> str:
        """Fetch and cache a DNAC auth token."""
        url = f"{self.base_url}/dna/system/api/v1/auth/token"
        logger.info("Authenticating with DNAC...")
        response = requests.post(
            url,
            auth=HTTPBasicAuth(self.username, self.password),
            verify=self.verify_ssl
        )
        response.raise_for_status()
        self.token = response.json()['Token']
        logger.info("DNAC authentication successful.")
        return self.token

    def _get_headers(self) -> dict:
        if not self.token:
            self.authenticate()
        return {
            "x-auth-token": self.token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    # -------------------------------------------------------------------------
    # Webhook Subscription Management
    # -------------------------------------------------------------------------
    def list_event_subscriptions(self) -> list:
        """
        List all current webhook subscriptions registered in DNAC.
        Returns an empty list if DNAC responds with 204 No Content
        (which means no subscriptions exist yet).
        """
        url = f"{self.base_url}/dna/intent/api/v1/event/subscription"
        response = requests.get(url, headers=self._get_headers(), verify=self.verify_ssl)

        # 204 No Content = no subscriptions registered yet — not an error
        if response.status_code == 204:
            logger.info("DNAC returned 204 — no webhook subscriptions registered yet.")
            return []

        response.raise_for_status()
        return response.json()

    def register_webhook(self) -> dict:
        """
        Register this service as a REST/Webhook subscriber in DNAC.
        DNAC will push event payloads to our FastAPI receiver_url.
        This is always a manual, on-demand operation — never auto-called on startup.
        """
        receiver_url = self.webhook_config['receiver_url']
        name = self.webhook_config.get('name', 'FalseAlertDetection')
        description = self.webhook_config.get('description', '')
        event_categories = self.webhook_config.get('event_categories', [])

        # Build the subscription filter by category
        filter_payload = {}
        if event_categories:
            filter_payload["eventIds"] = []  # Can be specific event IDs OR use category filter below
            filter_payload["categories"] = event_categories

        payload = [
            {
                "name": name,
                "description": description,
                "subscriptionEndpoints": [
                    {
                        "instanceId": f"{name}-endpoint",
                        "subscriptionDetails": {
                            "connectorType": "REST",
                            "url": receiver_url,
                            "method": "POST",
                            "headers": [
                                {"string": "Content-Type: application/json"}
                            ]
                        }
                    }
                ],
                "filter": filter_payload
            }
        ]

        url = f"{self.base_url}/dna/intent/api/v1/event/subscription"
        logger.info(f"Registering webhook with DNAC. Receiver URL: {receiver_url}")

        # Check if already registered to avoid duplicates
        existing = self.list_event_subscriptions()
        for sub in existing:
            if sub.get('name') == name:
                self._subscription_id = sub.get('subscriptionId')
                logger.info(f"Webhook '{name}' already registered (ID: {self._subscription_id}). Skipping.")
                return sub

        response = requests.post(url, headers=self._get_headers(), json=payload, verify=self.verify_ssl)
        response.raise_for_status()
        result = response.json()
        self._subscription_id = result[0].get('subscriptionId') if isinstance(result, list) else result.get('subscriptionId')
        logger.info(f"Webhook registered successfully. Subscription ID: {self._subscription_id}")
        return result

    def deregister_webhook(self) -> None:
        """Remove the webhook subscription from DNAC on service shutdown."""
        if not self._subscription_id:
            return
        url = f"{self.base_url}/dna/intent/api/v1/event/subscription"
        params = {"subscriptionIds": self._subscription_id}
        logger.info(f"De-registering webhook (ID: {self._subscription_id})...")
        response = requests.delete(url, headers=self._get_headers(), params=params, verify=self.verify_ssl)
        if response.ok:
            logger.info("Webhook de-registered successfully.")
        else:
            logger.warning(f"Failed to de-register webhook: {response.status_code} {response.text}")
