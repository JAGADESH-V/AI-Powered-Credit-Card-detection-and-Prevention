"""Model loading and scoring.

Security note: a pickled model is executable code. Loading one from an
untrusted path is remote code execution. So the loader verifies the file
against a hash pinned in configuration and refuses to start if it does not
match. Treat model artifacts with the same care as application binaries.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from fraudguard.features.engineering import FEATURE_ORDER, CardHistory, build_features
from fraudguard.observability import get_logger
from fraudguard.schemas import Decision, ReasonCode

logger = get_logger(__name__)

REASON_TEXT = {
    "txn_count_1h": "unusual number of transactions in the last hour",
    "txn_count_24h": "elevated transaction count in 24 hours",
    "amount_zscore_vs_card": "amount far from this card's normal spend",
    "implied_speed_kmh": "location change too fast to be physically possible",
    "distance_from_last_km": "large distance from the previous transaction",
    "cross_border": "merchant country differs from cardholder country",
    "new_device": "first time this device has been seen on the card",
    "is_night": "transaction occurred at an unusual hour",
    "high_risk_mcc": "merchant category associated with elevated fraud",
    "is_3ds_authenticated": "no 3-D Secure authentication",
    "amount_log": "transaction amount",
    "amount_sum_24h": "high cumulative spend in 24 hours",
    "channel_ecommerce": "card-not-present channel",
    "distinct_merchants_24h": "many different merchants in a short period",
    "seconds_since_last_txn": "very short gap since the previous transaction",
}


@dataclass(frozen=True, slots=True)
class ScoringResult:
    probability: float
    decision: Decision
    reason_codes: list[ReasonCode]
    model_version: str
    feature_vector: np.ndarray


def verify_artifact(path: Path, expected_sha256: str | None) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(
            "model artifact hash mismatch — refusing to load a model that does "
            "not match the pinned hash"
        )
    return digest


class FraudScorer:
    def __init__(self, model_path: Path, expected_sha256: str | None = None,
                 review_threshold: float = 0.30) -> None:
        digest = verify_artifact(model_path, expected_sha256)
        bundle = joblib.load(model_path)

        if bundle["feature_order"] != FEATURE_ORDER:
            raise RuntimeError("feature order in artifact does not match running code")

        self.pipeline = bundle["pipeline"]
        self.decline_threshold: float = bundle["threshold"]
        self.review_threshold = min(review_threshold, self.decline_threshold)
        self.version: str = bundle["version"]
        self.algorithm: str = bundle.get("algorithm", "unknown")
        self.reference_stats: dict = bundle.get("reference_stats", {})
        self.pipeline_features: list[str] = list(FEATURE_ORDER)
        self._baseline: np.ndarray | None = None
        logger.info(
            "model.loaded",
            extra={"version": self.version, "sha256": digest[:16], "threshold": self.decline_threshold},
        )

    # ---- explainability -------------------------------------------------
    def _contributions(self, vector: np.ndarray) -> list[ReasonCode]:
        """Leave-one-out attribution: re-score with each feature replaced by a
        neutral value and measure how much the probability drops.

        Model-agnostic and fast enough for the request path. For offline case
        review, run SHAP over the same pipeline for exact Shapley values.
        """
        if self._baseline is None:
            self._baseline = np.zeros((1, len(FEATURE_ORDER)))

        base_probability = float(self.pipeline.predict_proba(vector)[0, 1])
        perturbed = np.repeat(vector, len(FEATURE_ORDER), axis=0)
        for i in range(len(FEATURE_ORDER)):
            perturbed[i, i] = self._baseline[0, i]

        probabilities = self.pipeline.predict_proba(perturbed)[:, 1]
        deltas = base_probability - probabilities

        ranked = sorted(
            zip(FEATURE_ORDER, deltas, strict=True), key=lambda pair: abs(pair[1]), reverse=True
        )[:4]
        return [
            ReasonCode(
                feature=name,
                contribution=round(float(delta), 4),
                description=REASON_TEXT.get(name, name.replace("_", " ")),
            )
            for name, delta in ranked
            if abs(delta) > 1e-4
        ]

    def score(self, txn: dict, history: CardHistory) -> ScoringResult:
        features = build_features(txn, history)
        vector = np.array([[features[name] for name in FEATURE_ORDER]], dtype=float)

        if not np.isfinite(vector).all():
            raise ValueError("feature vector contains non-finite values")

        probability = float(self.pipeline.predict_proba(vector)[0, 1])

        if probability >= self.decline_threshold:
            decision = Decision.DECLINE
        elif probability >= self.review_threshold:
            decision = Decision.REVIEW
        else:
            decision = Decision.APPROVE

        reasons = self._contributions(vector) if decision is not Decision.APPROVE else []
        return ScoringResult(probability, decision, reasons, self.version, vector)
