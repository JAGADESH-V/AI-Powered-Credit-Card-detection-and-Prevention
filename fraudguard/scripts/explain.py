#!/usr/bin/env python3
"""Offline SHAP explanations for case review.

    python scripts/explain.py --transactions 200

The API returns fast leave-one-out attributions on the request path. This
script computes exact Shapley values instead: slower, but the right tool when
an analyst is building a case file or a decision is being challenged.

Falls back to permutation importance if SHAP is not installed, so the script
still tells you something useful rather than failing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd

from fraudguard.data.generate import generate
from fraudguard.features.engineering import FEATURE_ORDER, frame_from_transactions
from fraudguard.models.train import permutation_importance_scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain model decisions")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--transactions", type=int, default=200)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"No model at {args.model}. Run scripts/train_model.py first.")
        return 1

    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]

    frame = frame_from_transactions(generate(n_cards=800, days=60, seed=99))

    # Keep every fraud case and fill the rest with legitimate traffic. A plain
    # tail slice can contain no fraud at all, which makes importance scores
    # meaningless — every feature looks useless when there is nothing to find.
    fraud = frame[frame["is_fraud"] == 1]
    legit = frame[frame["is_fraud"] == 0].sample(
        n=min(max(args.transactions - len(fraud), 1), (frame["is_fraud"] == 0).sum()),
        random_state=0,
    )
    sample = pd.concat([fraud, legit]).sort_index()

    if fraud.empty:
        print("No fraud in the sample — importance scores would be meaningless.")
        return 1

    X = sample[FEATURE_ORDER].to_numpy()
    y = sample["is_fraud"].to_numpy()
    print(f"Sample: {len(sample)} transactions, {len(fraud)} fraudulent")

    print(f"Model {bundle['version']} ({bundle.get('algorithm', 'unknown')})")
    print(f"Explaining {len(sample)} transactions\n")

    try:
        import shap
    except ImportError:
        print("SHAP is not installed (pip install shap).")
        print("Falling back to permutation importance.\n")
        print(f"{'feature':<28}{'PR-AUC drop when shuffled':>26}")
        print("-" * 54)
        for name, value in list(
            permutation_importance_scores(pipeline, X, y).items()
        )[: args.top]:
            print(f"{name:<28}{value:>26.5f}")
        return 0

    # Model-agnostic explainer: works for the calibrated wrapper, which the
    # tree-specific explainers cannot see through.
    background = shap.sample(X, min(100, len(X)), random_state=0)
    explainer = shap.KernelExplainer(lambda data: pipeline.predict_proba(data)[:, 1], background)
    values = explainer.shap_values(X[: min(50, len(X))], silent=True)

    mean_absolute = np.abs(values).mean(axis=0)
    ranked = sorted(zip(FEATURE_ORDER, mean_absolute, strict=True), key=lambda p: p[1], reverse=True)

    print(f"{'feature':<28}{'mean |SHAP|':>14}")
    print("-" * 42)
    for name, value in ranked[: args.top]:
        print(f"{name:<28}{value:>14.5f}")

    print("\nHighest-risk transaction in the sample:")
    scores = pipeline.predict_proba(X)[:, 1]
    index = int(np.argmax(scores[: len(values)]))
    print(f"  probability {scores[index]:.4f}\n")
    contributions = sorted(
        zip(FEATURE_ORDER, values[index], strict=True), key=lambda p: abs(p[1]), reverse=True
    )
    for name, value in contributions[:6]:
        direction = "raises" if value > 0 else "lowers"
        print(f"  {name:<26} {value:+.5f}  ({direction} risk)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
