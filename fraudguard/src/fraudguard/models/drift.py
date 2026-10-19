"""Drift detection.

Fraud models decay in two distinct ways, and they need different alarms:

  * **Feature drift** — the traffic changes (a new merchant, a new country, a
    new device mix). Measured with Population Stability Index against the
    distribution the model was actually trained on.

  * **Score drift** — the model's output distribution shifts. A sudden fall in
    flagged volume is the one to watch: it looks like good news on a
    dashboard, and it is often an attacker who has found a gap.

Both matter because chargeback labels arrive 30-90 days late. Drift is the
early warning you get in the meantime — it is a proxy for performance decay,
not a measurement of it, and it should trigger a review rather than a retrain.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock

import numpy as np

# Conventional PSI reading, widely used in credit risk:
PSI_MINOR = 0.10    # worth noting
PSI_MAJOR = 0.25    # investigate; model likely needs retraining


@dataclass(frozen=True, slots=True)
class DriftSignal:
    feature: str
    psi: float
    severity: str

    @property
    def actionable(self) -> bool:
        return self.severity != "stable"


def population_stability_index(
    reference_quantiles: list[float], observed: np.ndarray, epsilon: float = 1e-4
) -> float:
    """PSI between a reference distribution (as deciles) and observed values.

    Bins come from the training quantiles, so the comparison is against what
    the model actually learned rather than against a fresh binning of the new
    data, which would hide exactly the shift being looked for.
    """
    edges = np.unique(np.asarray(reference_quantiles, dtype=float))
    if edges.size < 2 or observed.size == 0:
        return 0.0

    interior = edges[1:-1] if edges.size > 2 else edges[:1]
    bins = np.concatenate([[-np.inf], interior, [np.inf]])

    expected_counts = np.full(len(bins) - 1, 1.0 / (len(bins) - 1))
    observed_counts = np.histogram(observed, bins=bins)[0].astype(float)
    observed_counts /= max(observed_counts.sum(), 1.0)

    expected_counts = np.clip(expected_counts, epsilon, None)
    observed_counts = np.clip(observed_counts, epsilon, None)

    return float(np.sum((observed_counts - expected_counts) * np.log(observed_counts / expected_counts)))


def classify(psi: float) -> str:
    if psi >= PSI_MAJOR:
        return "major"
    if psi >= PSI_MINOR:
        return "minor"
    return "stable"


class DriftMonitor:
    """Keeps a bounded window of recent feature vectors and scores.

    Bounded on purpose: an unbounded monitor is a memory leak that takes the
    scoring service down with it, which is a worse outcome than missing drift.
    """

    def __init__(
        self,
        reference_stats: dict[str, dict],
        feature_order: list[str],
        window: int = 5000,
        min_samples: int = 500,
    ) -> None:
        self.reference_stats = reference_stats
        self.feature_order = feature_order
        self.min_samples = min_samples
        self._vectors: deque[np.ndarray] = deque(maxlen=window)
        self._scores: deque[float] = deque(maxlen=window)
        self._decisions: deque[str] = deque(maxlen=window)
        self._lock = Lock()

    def observe(self, vector: np.ndarray, score: float, decision: str) -> None:
        with self._lock:
            self._vectors.append(np.asarray(vector, dtype=float).ravel())
            self._scores.append(float(score))
            self._decisions.append(decision)

    @property
    def samples(self) -> int:
        return len(self._vectors)

    def feature_drift(self) -> list[DriftSignal]:
        with self._lock:
            if len(self._vectors) < self.min_samples:
                return []
            matrix = np.vstack(self._vectors)

        signals = []
        for index, name in enumerate(self.feature_order):
            reference = self.reference_stats.get(name, {}).get("quantiles")
            if not reference:
                continue
            psi = population_stability_index(reference, matrix[:, index])
            signals.append(DriftSignal(name, round(psi, 4), classify(psi)))

        return sorted(signals, key=lambda s: s.psi, reverse=True)

    def score_summary(self) -> dict[str, float]:
        with self._lock:
            if not self._scores:
                return {"samples": 0}
            scores = np.asarray(self._scores)
            decisions = list(self._decisions)

        return {
            "samples": len(scores),
            "mean_score": round(float(scores.mean()), 5),
            "p95_score": round(float(np.quantile(scores, 0.95)), 5),
            "alert_rate": round(
                sum(1 for d in decisions if d != "approve") / len(decisions), 5
            ),
            "decline_rate": round(
                sum(1 for d in decisions if d == "decline") / len(decisions), 5
            ),
        }

    def report(self) -> dict:
        signals = self.feature_drift()
        return {
            "samples_observed": self.samples,
            "sufficient_data": self.samples >= self.min_samples,
            "scores": self.score_summary(),
            "features_drifting": [
                {"feature": s.feature, "psi": s.psi, "severity": s.severity}
                for s in signals
                if s.actionable
            ],
            "worst_psi": signals[0].psi if signals else 0.0,
        }
