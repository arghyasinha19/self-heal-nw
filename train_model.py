#!/usr/bin/env python3
"""
DNAC Alert Classifier — Training CLI
─────────────────────────────────────
Fine-tune DistilBERT on labelled DNAC alert data and export
a production-ready model.

Usage:
    python train_model.py --data data/training_data.csv
    python train_model.py --data data/training_data.csv --output-dir models --epochs 8

The script will:
  1. Load & preprocess the CSV dataset
  2. Fine-tune DistilBERT (distilbert-base-uncased)
  3. Evaluate on a held-out test set
  4. Export the model to ONNX format for fast inference
  5. Save all artifacts to the output directory
"""

import argparse
import logging
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("train_model")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for DNAC alert classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py --data data/training_data.csv
  python train_model.py --data data/training_data.csv --epochs 8 --batch-size 32
  python train_model.py --data data/training_data.csv --output-dir models/v2
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the labelled CSV file (columns: description, category).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model artifacts. Default: models/",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs. Default: 5",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size. Default: 16",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate. Default: 2e-5",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token length. Default: 128",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )

    args = parser.parse_args()

    # Validate data file exists
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        logger.error("Please provide a valid CSV file with 'description' and 'category' columns.")
        sys.exit(1)

    # Build config overrides from CLI args
    config_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
    }

    logger.info("=" * 70)
    logger.info("  DNAC Alert Classifier — DistilBERT Fine-Tuning")
    logger.info("=" * 70)
    logger.info(f"  Data file:     {args.data}")
    logger.info(f"  Output dir:    {args.output_dir}")
    logger.info(f"  Epochs:        {args.epochs}")
    logger.info(f"  Batch size:    {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Max length:    {args.max_length}")
    logger.info(f"  Seed:          {args.seed}")
    logger.info("=" * 70)

    # Import and run trainer
    from app.classifier.trainer import DistilBertTrainer

    trainer = DistilBertTrainer(
        data_path=args.data,
        config=config_overrides,
    )

    test_metrics = trainer.run(output_dir=args.output_dir)

    # ── Summary ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("  TRAINING COMPLETE — Results Summary")
    logger.info("=" * 70)
    logger.info(f"  Accuracy:  {test_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Precision: {test_metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall:    {test_metrics.get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {test_metrics.get('f1', 0):.4f}")
    logger.info("=" * 70)
    logger.info(f"  Model saved to: {args.output_dir}/")
    logger.info(f"  Evaluation report: {args.output_dir}/evaluation_report.json")
    logger.info("")
    logger.info("  To use the model in the FastAPI service:")
    logger.info("    1. Ensure config.yaml has: classifier.model_path: \"models\"")
    logger.info("    2. Restart the service: uvicorn app.main:app --reload")
    logger.info("    3. Test: POST /api/v1/classify {\"description\": \"your alert text\"}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
