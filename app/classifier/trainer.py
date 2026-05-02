"""
DistilBERT Fine-Tuning Trainer for DNAC Alert Classification
─────────────────────────────────────────────────────────────
Orchestrates the full training pipeline:
  1. Load & preprocess CSV data
  2. Tokenize with DistilBERT tokenizer
  3. Compute class weights for imbalanced datasets
  4. Fine-tune DistilBertForSequenceClassification
  5. Evaluate on held-out test set
  6. Export to ONNX for fast production inference
  7. Save all artifacts (model, tokenizer, metrics) to disk
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from app.classifier.preprocessor import (
    clean_text,
    normalize_label,
    label_to_id,
    LABEL2ID,
    ID2LABEL,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters (sensible defaults for ~2,500 samples)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 128,          # Max token length (alert descriptions are short)
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "early_stopping_patience": 2,
    "fp16": torch.cuda.is_available(),  # Mixed precision only on GPU
    "seed": 42,
    "test_size": 0.1,
    "val_size": 0.1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Custom Trainer with Class-Weighted Loss
# ─────────────────────────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    """
    Extends HuggingFace Trainer to use class-weighted CrossEntropyLoss.
    This handles imbalanced datasets (e.g., 70/30 auto-resolving vs not).
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
# Main Trainer Class
# ─────────────────────────────────────────────────────────────────────────────
class DistilBertTrainer:
    """
    End-to-end fine-tuning pipeline for DNAC alert classification.

    Usage:
        trainer = DistilBertTrainer(data_path="data/training_data.csv")
        metrics = trainer.run(output_dir="models")
    """

    def __init__(
        self,
        data_path: str,
        config: Optional[Dict] = None,
    ):
        self.data_path = data_path
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Training device: {self.device}")
        logger.info(f"Config: {json.dumps(self.config, indent=2, default=str)}")

        # Will be populated during training
        self.tokenizer = None
        self.model = None
        self.class_weights = None

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Load & Preprocess Data
    # ─────────────────────────────────────────────────────────────────────
    def _load_data(self) -> pd.DataFrame:
        """Load CSV, clean text, normalize labels."""
        logger.info(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        if "description" not in df.columns or "category" not in df.columns:
            raise ValueError(
                f"CSV must have 'description' and 'category' columns. "
                f"Found: {list(df.columns)}"
            )

        original_len = len(df)

        # Drop rows with missing values
        df = df.dropna(subset=["description", "category"]).reset_index(drop=True)
        if len(df) < original_len:
            logger.warning(f"Dropped {original_len - len(df)} rows with missing values.")

        # Clean text
        df["description"] = df["description"].apply(clean_text)

        # Normalize labels
        df["category"] = df["category"].apply(normalize_label)
        df["label"] = df["category"].apply(label_to_id)

        # Log class distribution
        dist = df["category"].value_counts()
        logger.info(f"Class distribution:\n{dist.to_string()}")

        return df

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Split Data
    # ─────────────────────────────────────────────────────────────────────
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified split into train/val/test."""
        test_size = self.config["test_size"]
        val_size = self.config["val_size"]

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=self.config["seed"],
        )

        # Second split: train vs val
        relative_val_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            stratify=train_val_df["label"],
            random_state=self.config["seed"],
        )

        logger.info(
            f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        return train_df, val_df, test_df

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Tokenize
    # ─────────────────────────────────────────────────────────────────────
    def _tokenize(self, df: pd.DataFrame) -> Dataset:
        """Convert a DataFrame to a tokenized HuggingFace Dataset."""
        dataset = Dataset.from_pandas(
            df[["description", "label"]].rename(columns={"description": "text"}),
            preserve_index=False,
        )

        def tokenize_fn(batch):
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config["max_length"],
            )

        dataset = dataset.map(tokenize_fn, batched=True, batch_size=64)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        return dataset

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Compute Class Weights
    # ─────────────────────────────────────────────────────────────────────
    def _compute_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced data."""
        label_counts = train_df["label"].value_counts().sort_index()
        total = len(train_df)
        n_classes = len(label_counts)
        weights = total / (n_classes * label_counts.values)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Class weights: {dict(zip(LABEL2ID.keys(), weights_tensor.tolist()))}")
        return weights_tensor

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Evaluation Metrics
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_metrics(eval_pred):
        """Compute metrics for HuggingFace Trainer."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )
        acc = accuracy_score(labels, predictions)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Export to ONNX
    # ─────────────────────────────────────────────────────────────────────
    def _export_onnx(self, output_dir: str) -> str:
        """Export model to ONNX format for fast CPU inference."""
        onnx_path = os.path.join(output_dir, "model.onnx")

        logger.info(f"Exporting model to ONNX: {onnx_path}")

        self.model.eval()
        self.model.to("cpu")

        # Create dummy input
        dummy_input = self.tokenizer(
            "dummy alert text for onnx export",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config["max_length"],
        )

        torch.onnx.export(
            self.model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            onnx_path,
            opset_version=14,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"},
            },
        )

        logger.info(f"ONNX model exported: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
        return onnx_path

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: Full Evaluation on Test Set
    # ─────────────────────────────────────────────────────────────────────
    def _evaluate_test(self, test_dataset: Dataset) -> Dict:
        """Run full evaluation on the held-out test set."""
        logger.info("Evaluating on test set...")

        self.model.eval()
        self.model.to(self.device)

        all_preds = []
        all_labels = []

        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted"
        )
        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(
            all_labels,
            all_preds,
            target_names=list(LABEL2ID.keys()),
            output_dict=True,
        )

        metrics = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        logger.info(f"Test Accuracy:  {acc:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall:    {recall:.4f}")
        logger.info(f"Test F1:        {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")

        # Pretty-print the classification report
        report_str = classification_report(
            all_labels, all_preds, target_names=list(LABEL2ID.keys())
        )
        logger.info(f"\nClassification Report:\n{report_str}")

        return metrics

    # ─────────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────────────────────────────────
    def run(self, output_dir: str = "models") -> Dict:
        """
        Execute the full training pipeline.

        Returns:
            dict: Evaluation metrics from the test set.
        """
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        # ── Load & split ──
        df = self._load_data()
        train_df, val_df, test_df = self._split_data(df)

        # ── Initialize tokenizer & model ──
        logger.info(f"Loading pre-trained model: {self.config['model_name']}")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.config["model_name"])
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config["model_name"],
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        # ── Tokenize datasets ──
        logger.info("Tokenizing datasets...")
        train_dataset = self._tokenize(train_df)
        val_dataset = self._tokenize(val_df)
        test_dataset = self._tokenize(test_df)

        # ── Class weights ──
        class_weights = self._compute_class_weights(train_df)

        # ── Training arguments ──
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=self.config["epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"] * 2,
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            warmup_ratio=self.config["warmup_ratio"],
            eval_strategy=self.config["eval_strategy"],
            save_strategy=self.config["save_strategy"],
            load_best_model_at_end=self.config["load_best_model_at_end"],
            metric_for_best_model=self.config["metric_for_best_model"],
            greater_is_better=True,
            fp16=self.config["fp16"],
            seed=self.config["seed"],
            logging_dir=os.path.join(checkpoint_dir, "logs"),
            logging_steps=10,
            report_to="none",       # Disable W&B / MLflow
            save_total_limit=2,     # Keep only 2 best checkpoints
            disable_tqdm=False,
        )

        # ── Train ──
        logger.info("Starting fine-tuning...")
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config["early_stopping_patience"]
                )
            ],
        )

        trainer.train()

        # ── Evaluate on test set ──
        test_metrics = self._evaluate_test(test_dataset)

        # ── Save model & tokenizer ──
        model_save_path = os.path.join(output_dir, "distilbert_model")
        logger.info(f"Saving model to {model_save_path}")
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)

        # ── Export ONNX ──
        try:
            self._export_onnx(output_dir)
        except Exception as e:
            logger.warning(f"ONNX export failed (non-fatal): {e}")
            logger.warning("The PyTorch model is still available for inference.")

        # ── Save evaluation report ──
        elapsed = time.time() - start_time
        report = {
            "model_name": self.config["model_name"],
            "training_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "epochs": self.config["epochs"],
            "learning_rate": self.config["learning_rate"],
            "max_length": self.config["max_length"],
            "training_time_seconds": round(elapsed, 2),
            "device": self.device,
            "test_metrics": test_metrics,
            "label_mapping": LABEL2ID,
        }
        report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Evaluation report saved to {report_path}")

        # ── Clean up checkpoints to save disk space ──
        import shutil
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logger.info("Cleaned up training checkpoints.")

        logger.info(f"Training complete in {elapsed:.1f}s. All artifacts saved to {output_dir}/")

        return test_metrics
