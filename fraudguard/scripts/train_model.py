#!/usr/bin/env python3
"""Train a fraud model end to end.

    python scripts/train_model.py                          # synthetic data
    python scripts/train_model.py --source csv --path my_transactions.csv
    python scripts/train_model.py --compare                # benchmark algorithms

Writes artifacts/model.joblib and prints its SHA-256, which you then pin as
FG_MODEL_SHA256 in the serving environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudguard.data.generate import generate
from fraudguard.data.loaders import (
    DatasetError,
    describe,
    load_transactions_csv,
)
from fraudguard.features.engineering import frame_from_transactions
from fraudguard.models.train import compare_algorithms, train

KEY_METRICS = (
    "pr_auc", "roc_auc", "brier_score", "expected_calibration_error",
    "chosen_threshold", "precision_at_threshold", "recall_at_threshold",
    "false_positive_rate", "alert_rate",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the FraudGuard model")
    parser.add_argument("--source", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--path", type=Path, help="CSV path when --source csv")
    parser.add_argument("--column-map", type=str,
                        help='JSON mapping, e.g. \'{"amount": "txn_amount"}\'')
    parser.add_argument("--cards", type=int, default=1200)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--model", choices=["gbdt", "logreg", "rf"], default="gbdt")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="benchmark all algorithms instead of training one")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        if args.source == "csv":
            if args.path is None:
                parser.error("--path is required with --source csv")
            mapping = json.loads(args.column_map) if args.column_map else None
            rows = load_transactions_csv(args.path, mapping)
        else:
            print(f"Generating transactions for {args.cards} cards over {args.days} days...")
            rows = generate(n_cards=args.cards, days=args.days, seed=args.seed)
    except DatasetError as exc:
        print(f"\nCould not load data: {exc}\n")
        return 1

    summary = describe(rows)
    print(f"  {summary['transactions']:,} transactions, "
          f"{summary['fraudulent']:,} fraudulent "
          f"({summary['fraud_rate'] * 100:.2f}%), "
          f"{summary['distinct_cards']:,} cards")

    print("Building features (chronological replay, no leakage)...")
    frame = frame_from_transactions(rows)

    if args.compare:
        print("\nBenchmarking algorithms on identical splits...\n")
        header = f"{'algorithm':<10}{'PR-AUC':>9}{'ROC-AUC':>9}{'ECE':>8}{'recall':>9}{'saving/1k':>12}"
        print(header)
        print("-" * len(header))
        for row in compare_algorithms(frame):
            print(f"{row['algorithm']:<10}{row['pr_auc']:>9.4f}{row['roc_auc']:>9.4f}"
                  f"{row['ece']:>8.4f}{row['recall']:>9.4f}{row['saving_vs_baseline']:>12.2f}")
        print("\nRerun without --compare to train and save one of these.")
        return 0

    print(f"Training {args.model}...")
    report = train(frame, args.artifacts, kind=args.model,
                   calibrate=not args.no_calibration)

    print("\n" + "=" * 64)
    print(f"TRAINING REPORT  ({report.algorithm})")
    print("=" * 64)
    payload = asdict(report)
    for key in KEY_METRICS:
        print(f"  {key.replace('_', ' '):<28} {payload[key]:.4f}")

    print("-" * 64)
    saving = report.baseline_cost_per_1k - report.expected_cost_per_1k
    print(f"  {'cost per 1k transactions':<28} {report.expected_cost_per_1k:.2f}")
    print(f"  {'vs approving everything':<28} {report.baseline_cost_per_1k:.2f}")
    print(f"  {'net saving per 1k':<28} {saving:.2f}")

    if report.expected_calibration_error > 0.05:
        print("\n  ! Calibration error above 0.05 — the cost threshold is being")
        print("    set on probabilities that do not mean what they claim.")

    print("\n  Top features by permutation importance:")
    for name, value in list(report.feature_importance.items())[:5]:
        print(f"    {name:<26} {value:+.5f}")

    print("=" * 64)
    print(f"\nPin this in your serving env:\n  FG_MODEL_SHA256={report.model_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
