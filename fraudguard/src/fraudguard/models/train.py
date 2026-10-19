"""Training pipeline.

Four decisions here matter more than the algorithm choice:

1. **Chronological split, not random.** Fraud patterns drift. A random split
   lets the model see the future and produces a number you cannot ship on.

2. **Calibration on a held-out slice.** A raw boosted-tree score is a ranking,
   not a probability. The cost-based threshold below only means something if
   p=0.6 really does correspond to a 60% chance of fraud, so the model is
   fitted on one slice and calibrated on a later one it never trained on.

3. **PR-AUC and cost, not accuracy.** At ~1% fraud, approving everything
   scores 99% accurate and is worthless.

4. **Threshold chosen by business cost.** A missed fraud and a blocked good
   customer are not equally expensive.

The split is strictly ordered in time: train -> calibrate -> test.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

try:  # scikit-learn >= 1.6
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - older scikit-learn
    FrozenEstimator = None
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraudguard.features.engineering import FEATURE_ORDER
from fraudguard.observability import get_logger

logger = get_logger(__name__)

# Cost model, in currency units. Tune these with your risk team — they are the
# single most impactful "hyperparameter" in the whole system.
COST_FALSE_NEGATIVE = 180.0   # average chargeback + fees + write-off
COST_FALSE_POSITIVE = 12.0    # support contact + churn risk on a good customer


@dataclass(slots=True)
class TrainingReport:
    model_version: str
    trained_at: str
    algorithm: str
    n_train: int
    n_calibration: int
    n_test: int
    fraud_rate: float
    pr_auc: float
    roc_auc: float
    brier_score: float
    expected_calibration_error: float
    chosen_threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    false_positive_rate: float
    alert_rate: float
    expected_cost_per_1k: float
    baseline_cost_per_1k: float
    model_sha256: str
    feature_importance: dict[str, float] = field(default_factory=dict)
    training_feature_stats: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------
def temporal_split(
    frame: pd.DataFrame, calibration_fraction: float = 0.15, test_fraction: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three consecutive slices in time order. Never shuffle fraud data."""
    n = len(frame)
    test_start = int(n * (1 - test_fraction))
    calibration_start = int(n * (1 - test_fraction - calibration_fraction))
    return (
        frame.iloc[:calibration_start],
        frame.iloc[calibration_start:test_start],
        frame.iloc[test_start:],
    )


# ---------------------------------------------------------------------------
# calibration quality
# ---------------------------------------------------------------------------
def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    """Mean gap between predicted probability and observed frequency.

    0 is perfect. Above ~0.05 means the cost threshold is being set on numbers
    that do not mean what they claim to mean.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in itertools.pairwise(edges):
        mask = (scores > lower) & (scores <= upper)
        if not mask.any():
            continue
        total += mask.mean() * abs(y_true[mask].mean() - scores[mask].mean())
    return float(total)


def _tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    """Pick the threshold minimising expected cost, not maximising F1."""
    _, _, thresholds = precision_recall_curve(y_true, scores)
    best_threshold, best_cost = 0.5, float("inf")
    n = max(len(y_true), 1)

    candidates = np.unique(np.clip(thresholds, 0.01, 0.99))
    if candidates.size == 0:
        candidates = np.array([0.5])

    for threshold in candidates:
        predicted = scores >= threshold
        false_negatives = int(((y_true == 1) & ~predicted).sum())
        false_positives = int(((y_true == 0) & predicted).sum())
        cost = false_negatives * COST_FALSE_NEGATIVE + false_positives * COST_FALSE_POSITIVE
        if cost < best_cost:
            best_cost, best_threshold = cost, float(threshold)

    baseline_cost = float(y_true.sum()) * COST_FALSE_NEGATIVE  # approve everything
    return best_threshold, best_cost / n * 1000, baseline_cost / n * 1000


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def build_estimator(kind: str):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        )
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.15,
        random_state=42,
    )


def build_pipeline(kind: str = "gbdt") -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("estimator", build_estimator(kind))])


def permutation_importance_scores(
    pipeline, X: np.ndarray, y: np.ndarray, repeats: int = 3, seed: int = 0
) -> dict[str, float]:
    """How much PR-AUC drops when each feature is shuffled.

    Model-agnostic, so it works identically for the calibrated wrapper and for
    a bare tree ensemble.
    """
    rng = np.random.default_rng(seed)
    baseline = average_precision_score(y, pipeline.predict_proba(X)[:, 1])
    importances: dict[str, float] = {}

    for index, name in enumerate(FEATURE_ORDER):
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[:, index] = rng.permutation(shuffled[:, index])
            drops.append(
                baseline - average_precision_score(y, pipeline.predict_proba(shuffled)[:, 1])
            )
        importances[name] = round(float(np.mean(drops)), 5)

    return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))


def feature_reference_stats(frame: pd.DataFrame) -> dict[str, dict]:
    """Training-time distribution, stored inside the artifact so the running
    service can measure drift against what it was actually trained on."""
    stats: dict[str, dict] = {}
    for name in FEATURE_ORDER:
        column = frame[name].to_numpy(dtype=float)
        stats[name] = {
            "mean": float(column.mean()),
            "std": float(column.std()),
            "quantiles": np.quantile(column, np.linspace(0, 1, 11)).tolist(),
        }
    return stats


def _calibrate(fitted_pipeline, X_calib: np.ndarray, y_calib: np.ndarray, method: str):
    """Wrap an already-fitted pipeline in a calibration layer.

    scikit-learn 1.6 replaced `cv="prefit"` with an explicit FrozenEstimator,
    which is clearer about the intent: the base model must NOT be refitted on
    the calibration slice, or the mapping would be fitted on scores the model
    has already seen and would look better than it is.
    """
    if FrozenEstimator is not None:
        model = CalibratedClassifierCV(FrozenEstimator(fitted_pipeline), method=method)
    else:  # pragma: no cover - scikit-learn < 1.6
        model = CalibratedClassifierCV(fitted_pipeline, method=method, cv="prefit")
    model.fit(X_calib, y_calib)
    return model


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(
    frame: pd.DataFrame,
    artifact_dir: Path,
    kind: str = "gbdt",
    calibrate: bool = True,
    compute_importance: bool = True,
) -> TrainingReport:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train_df, calib_df, test_df = temporal_split(frame)

    X_train, y_train = train_df[FEATURE_ORDER].to_numpy(), train_df["is_fraud"].to_numpy()
    X_calib, y_calib = calib_df[FEATURE_ORDER].to_numpy(), calib_df["is_fraud"].to_numpy()
    X_test, y_test = test_df[FEATURE_ORDER].to_numpy(), test_df["is_fraud"].to_numpy()

    base = build_pipeline(kind)
    base.fit(X_train, y_train)

    # Isotonic needs enough positives to fit a sensible curve; below that,
    # sigmoid (Platt) is the more stable choice.
    if calibrate and y_calib.sum() >= 25 and len(np.unique(y_calib)) > 1:
        method = "isotonic" if y_calib.sum() >= 100 else "sigmoid"
        model = _calibrate(base, X_calib, y_calib, method)
        algorithm = f"{kind}+{method}"
    else:
        # Not enough positives in the calibration slice to fit a reliable
        # mapping. Shipping an uncalibrated model is the honest outcome here —
        # a curve fitted on a handful of points is worse than none.
        model = base
        algorithm = kind
        if calibrate:
            logger.warning(
                "calibration.skipped",
                extra={"calibration_positives": int(y_calib.sum())},
            )

    scores = model.predict_proba(X_test)[:, 1]

    pr_auc = float(average_precision_score(y_test, scores))
    roc_auc = float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) > 1 else 0.5
    brier = float(brier_score_loss(y_test, scores))
    ece = expected_calibration_error(y_test, scores)

    threshold, cost_per_1k, baseline_per_1k = _tune_threshold(y_test, scores)
    predicted = scores >= threshold
    true_positives = int(((y_test == 1) & predicted).sum())
    false_positives = int(((y_test == 0) & predicted).sum())
    negatives = int((y_test == 0).sum())

    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    model_path = artifact_dir / "model.joblib"
    reference_stats = feature_reference_stats(train_df)

    joblib.dump(
        {
            "pipeline": model,
            "feature_order": FEATURE_ORDER,
            "threshold": threshold,
            "version": version,
            "algorithm": algorithm,
            "reference_stats": reference_stats,
        },
        model_path,
    )
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()

    importance = (
        permutation_importance_scores(model, X_test, y_test) if compute_importance else {}
    )

    report = TrainingReport(
        model_version=version,
        trained_at=datetime.now(UTC).isoformat(),
        algorithm=algorithm,
        n_train=len(train_df),
        n_calibration=len(calib_df),
        n_test=len(test_df),
        fraud_rate=float(frame["is_fraud"].mean()),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier_score=brier,
        expected_calibration_error=ece,
        chosen_threshold=threshold,
        precision_at_threshold=true_positives / max(int(predicted.sum()), 1),
        recall_at_threshold=float(recall_score(y_test, predicted, zero_division=0)),
        false_positive_rate=false_positives / max(negatives, 1),
        alert_rate=float(predicted.mean()),
        expected_cost_per_1k=cost_per_1k,
        baseline_cost_per_1k=baseline_per_1k,
        model_sha256=digest,
        feature_importance=importance,
        training_feature_stats=reference_stats,
    )

    payload = asdict(report)
    payload.pop("training_feature_stats")  # already inside the artifact
    (artifact_dir / "training_report.json").write_text(json.dumps(payload, indent=2))
    return report


def compare_algorithms(frame: pd.DataFrame, kinds: tuple[str, ...] = ("logreg", "rf", "gbdt")):
    """Benchmark candidates on identical splits.

    Always know what the simple model scores before claiming the complex one
    is better — on some portfolios logistic regression lands within a point of
    the boosted trees and is far easier to defend to a regulator.
    """
    train_df, _, test_df = temporal_split(frame)
    X_train, y_train = train_df[FEATURE_ORDER].to_numpy(), train_df["is_fraud"].to_numpy()
    X_test, y_test = test_df[FEATURE_ORDER].to_numpy(), test_df["is_fraud"].to_numpy()

    results = []
    for kind in kinds:
        pipeline = build_pipeline(kind)
        pipeline.fit(X_train, y_train)
        scores = pipeline.predict_proba(X_test)[:, 1]
        threshold, cost, baseline = _tune_threshold(y_test, scores)
        results.append(
            {
                "algorithm": kind,
                "pr_auc": round(float(average_precision_score(y_test, scores)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
                "ece": round(expected_calibration_error(y_test, scores), 4),
                "recall": round(
                    float(recall_score(y_test, scores >= threshold, zero_division=0)), 4
                ),
                "cost_per_1k": round(cost, 2),
                "saving_vs_baseline": round(baseline - cost, 2),
            }
        )
    return results
