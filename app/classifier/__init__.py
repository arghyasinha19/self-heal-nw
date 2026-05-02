"""
DNAC Alert Classifier Package
──────────────────────────────
Fine-tuned DistilBERT binary classifier that predicts whether a DNAC alert
is "Auto resolving" or "Non-Auto Resolving".
"""

from app.classifier.model import AlertClassifier  # noqa: F401

__all__ = ["AlertClassifier"]
