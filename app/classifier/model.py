"""
Production Inference Engine for DNAC Alert Classifier
─────────────────────────────────────────────────────
Loads the fine-tuned DistilBERT model (ONNX or PyTorch fallback)
and exposes a simple predict() / predict_batch() API.

Design:
- ONNX Runtime is the primary inference backend (~3x faster than PyTorch on CPU)
- Falls back to PyTorch if ONNX model is not available
- Thread-safe: uses a single inference session, no mutable state
- Lazy loading: model is loaded on first call to load()
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AlertClassifier:
    """
    Production-grade classifier for DNAC alert descriptions.

    Usage:
        classifier = AlertClassifier.load("models")
        result = classifier.predict("AP went offline and recovered")
        # {"category": "Auto resolving", "confidence": 0.94}
    """

    def __init__(
        self,
        tokenizer,
        onnx_session=None,
        pytorch_model=None,
        label_mapping: Optional[Dict[int, str]] = None,
        max_length: int = 128,
        model_metadata: Optional[Dict] = None,
    ):
        self.tokenizer = tokenizer
        self.onnx_session = onnx_session
        self.pytorch_model = pytorch_model
        self.max_length = max_length
        self.model_metadata = model_metadata or {}

        # Default label mapping
        self.id2label = label_mapping or {0: "Auto resolving", 1: "Non-Auto Resolving"}

        # Determine backend
        if self.onnx_session is not None:
            self._backend = "onnx"
        elif self.pytorch_model is not None:
            self._backend = "pytorch"
        else:
            raise ValueError("Either onnx_session or pytorch_model must be provided.")

        logger.info(f"AlertClassifier initialized. Backend: {self._backend}")

    # ─────────────────────────────────────────────────────────────────────
    # Factory: Load from Disk
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, model_dir: str) -> "AlertClassifier":
        """
        Load a trained classifier from disk.

        Tries ONNX first (faster), falls back to PyTorch model.

        Args:
            model_dir: Directory containing the model artifacts.
                        Expected contents:
                        - model.onnx (optional, preferred)
                        - distilbert_model/ (PyTorch fallback)
                        - evaluation_report.json (optional metadata)
        """
        from transformers import AutoTokenizer

        onnx_path = os.path.join(model_dir, "model.onnx")
        pytorch_dir = os.path.join(model_dir, "distilbert_model")
        report_path = os.path.join(model_dir, "evaluation_report.json")

        # Load metadata
        metadata = {}
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                metadata = json.load(f)

        # Extract label mapping from metadata or use default
        label_mapping = None
        if "label_mapping" in metadata:
            # Convert string keys to int keys
            label_mapping = {v: k for k, v in metadata["label_mapping"].items()}

        max_length = metadata.get("max_length", 128)

        # Try ONNX first
        onnx_session = None
        pytorch_model = None

        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort

                # Optimize for CPU inference
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = os.cpu_count() or 4
                sess_options.inter_op_num_threads = 1

                onnx_session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                logger.info(f"Loaded ONNX model from {onnx_path}")
            except ImportError:
                logger.warning(
                    "onnxruntime not installed. Install with: pip install onnxruntime. "
                    "Falling back to PyTorch."
                )
            except Exception as e:
                logger.warning(f"Failed to load ONNX model: {e}. Falling back to PyTorch.")

        # Fallback to PyTorch
        if onnx_session is None:
            if not os.path.exists(pytorch_dir):
                raise FileNotFoundError(
                    f"No model found in {model_dir}. "
                    f"Expected model.onnx or distilbert_model/ directory. "
                    f"Run train_model.py first."
                )

            import torch
            from transformers import DistilBertForSequenceClassification

            pytorch_model = DistilBertForSequenceClassification.from_pretrained(pytorch_dir)
            pytorch_model.eval()

            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pytorch_model.to(device)

            logger.info(f"Loaded PyTorch model from {pytorch_dir} (device: {device})")

        # Load tokenizer — from PyTorch dir (it's saved alongside the model)
        tokenizer_dir = pytorch_dir if os.path.exists(pytorch_dir) else model_dir
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        logger.info(f"Loaded tokenizer from {tokenizer_dir}")

        return cls(
            tokenizer=tokenizer,
            onnx_session=onnx_session,
            pytorch_model=pytorch_model,
            label_mapping=label_mapping,
            max_length=max_length,
            model_metadata=metadata,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────
    def predict(self, description: str) -> Dict:
        """
        Classify a single alert description.

        Args:
            description: Raw alert description text.

        Returns:
            {
                "category": "Auto resolving" | "Non-Auto Resolving",
                "confidence": 0.0 - 1.0,
                "label_id": 0 | 1
            }
        """
        results = self.predict_batch([description])
        return results[0]

    def predict_batch(self, descriptions: List[str]) -> List[Dict]:
        """
        Classify a batch of alert descriptions.

        Args:
            descriptions: List of raw alert description texts.

        Returns:
            List of dicts, each with "category", "confidence", and "label_id".
        """
        from app.classifier.preprocessor import clean_text

        # Clean texts
        cleaned = [clean_text(d) for d in descriptions]

        # Tokenize
        encoded = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np" if self._backend == "onnx" else "pt",
        )

        # Run inference
        if self._backend == "onnx":
            logits = self._predict_onnx(encoded)
        else:
            logits = self._predict_pytorch(encoded)

        # Convert logits to predictions
        probabilities = self._softmax(logits)
        predicted_ids = np.argmax(probabilities, axis=-1)
        confidences = np.max(probabilities, axis=-1)

        results = []
        for pred_id, conf in zip(predicted_ids, confidences):
            results.append({
                "category": self.id2label.get(int(pred_id), f"Unknown({pred_id})"),
                "confidence": round(float(conf), 4),
                "label_id": int(pred_id),
            })

        return results

    def _predict_onnx(self, encoded) -> np.ndarray:
        """Run inference using ONNX Runtime."""
        ort_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        ort_outputs = self.onnx_session.run(None, ort_inputs)
        return ort_outputs[0]  # logits

    def _predict_pytorch(self, encoded) -> np.ndarray:
        """Run inference using PyTorch."""
        import torch

        device = next(self.pytorch_model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.pytorch_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.logits.cpu().numpy()

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # ─────────────────────────────────────────────────────────────────────
    # Model Info
    # ─────────────────────────────────────────────────────────────────────
    def get_info(self) -> Dict:
        """Return metadata about the loaded model."""
        info = {
            "backend": self._backend,
            "max_length": self.max_length,
            "labels": self.id2label,
        }

        if self.model_metadata:
            info["model_name"] = self.model_metadata.get("model_name", "unknown")
            info["training_time_seconds"] = self.model_metadata.get("training_time_seconds")
            info["training_samples"] = self.model_metadata.get("training_samples")
            info["device_trained_on"] = self.model_metadata.get("device")

            test_metrics = self.model_metadata.get("test_metrics", {})
            if test_metrics:
                info["test_accuracy"] = test_metrics.get("accuracy")
                info["test_f1"] = test_metrics.get("f1")
                info["test_precision"] = test_metrics.get("precision")
                info["test_recall"] = test_metrics.get("recall")

        return info
