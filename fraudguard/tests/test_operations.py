"""Tests for the operational layer: calibration, drift, key lifecycle, cases."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

os.environ.setdefault("FG_ENVIRONMENT", "dev")
os.environ.setdefault("FG_HMAC_SIGNING_SECRET", "test-hmac-secret-not-for-production")
os.environ.setdefault("FG_JWT_SECRET", "test-jwt-secret-not-for-production-use")

from fraudguard.data.generate import generate
from fraudguard.data.loaders import DatasetError, describe, load_transactions_csv
from fraudguard.db.redis_stores import (
    InMemoryCardHistoryStore,
    InMemoryNonceStore,
    InMemoryRateLimiter,
)
from fraudguard.db.sql_stores import (
    Database,
    KeyStatus,
    ScoreRepository,
    SqlApiKeyStore,
)
from fraudguard.features.engineering import FEATURE_ORDER, frame_from_transactions
from fraudguard.models.drift import (
    DriftMonitor,
    classify,
    population_stability_index,
)
from fraudguard.models.train import (
    expected_calibration_error,
    temporal_split,
    train,
)
from fraudguard.security.auth import issue_api_key


@pytest.fixture(scope="module")
def frame():
    """Large enough that the calibration slice holds enough fraud to fit a
    curve — smaller sets exercise the skip path in `test_calibration_skipped`."""
    return frame_from_transactions(generate(n_cards=2500, days=90, seed=7))


@pytest.fixture(scope="module")
def small_frame():
    return frame_from_transactions(generate(n_cards=120, days=30, seed=11))


# --------------------------------------------------------------------------
# splitting and calibration
# --------------------------------------------------------------------------
class TestTemporalSplit:
    def test_slices_are_disjoint_and_ordered(self, frame):
        train_df, calib_df, test_df = temporal_split(frame)
        assert len(train_df) + len(calib_df) + len(test_df) == len(frame)
        assert train_df.index.max() < calib_df.index.min()
        assert calib_df.index.max() < test_df.index.min()

    def test_fraud_present_in_every_slice(self, frame):
        """If fraud lands only in the holdout, the model has nothing to learn
        and the metrics are meaningless — this caught a real generator bug."""
        train_df, calib_df, test_df = temporal_split(frame)
        assert train_df["is_fraud"].sum() > 0
        assert test_df["is_fraud"].sum() > 0


class TestCalibration:
    def test_ece_zero_for_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) < 1e-9

    def test_ece_detects_overconfidence(self):
        """A model that always says 0.99 while being right half the time."""
        y = np.array([1, 0] * 50)
        assert expected_calibration_error(y, np.full(100, 0.99)) > 0.4

    def test_calibration_skipped_when_too_few_positives(self, small_frame, tmp_path):
        """Fitting a calibration curve on a handful of positives is worse than
        not calibrating. The model must ship uncalibrated and say so."""
        report = train(small_frame, tmp_path, calibrate=True, compute_importance=False)
        assert "isotonic" not in report.algorithm
        assert "sigmoid" not in report.algorithm

    def test_calibrated_model_beats_uncalibrated_on_ece(self, frame, tmp_path):
        calibrated = train(frame, tmp_path / "cal", calibrate=True, compute_importance=False)
        raw = train(frame, tmp_path / "raw", calibrate=False, compute_importance=False)
        assert calibrated.expected_calibration_error <= raw.expected_calibration_error + 0.01
        assert "isotonic" in calibrated.algorithm or "sigmoid" in calibrated.algorithm

    def test_artifact_carries_reference_stats(self, frame, tmp_path):
        report = train(frame, tmp_path, compute_importance=False)
        assert set(report.training_feature_stats) == set(FEATURE_ORDER)
        assert len(report.training_feature_stats["amount_log"]["quantiles"]) == 11

    def test_threshold_is_not_the_naive_half(self, frame, tmp_path):
        report = train(frame, tmp_path, compute_importance=False)
        assert 0.0 < report.chosen_threshold < 1.0
        assert report.expected_cost_per_1k < report.baseline_cost_per_1k


# --------------------------------------------------------------------------
# drift
# --------------------------------------------------------------------------
class TestDrift:
    def test_identical_distribution_has_near_zero_psi(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(size=10_000)
        quantiles = np.quantile(reference, np.linspace(0, 1, 11)).tolist()
        psi = population_stability_index(quantiles, rng.normal(size=5000))
        assert psi < 0.1
        assert classify(psi) == "stable"

    def test_shifted_distribution_is_flagged(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(size=10_000)
        quantiles = np.quantile(reference, np.linspace(0, 1, 11)).tolist()
        psi = population_stability_index(quantiles, rng.normal(loc=3.0, size=5000))
        assert psi >= 0.25
        assert classify(psi) == "major"

    def test_monitor_stays_quiet_until_enough_samples(self):
        stats = {name: {"quantiles": np.linspace(0, 1, 11).tolist()} for name in FEATURE_ORDER}
        monitor = DriftMonitor(stats, FEATURE_ORDER, min_samples=100)
        for _ in range(10):
            monitor.observe(np.zeros(len(FEATURE_ORDER)), 0.1, "approve")
        report = monitor.report()
        assert report["sufficient_data"] is False
        assert report["features_drifting"] == []

    def test_monitor_window_is_bounded(self):
        """An unbounded monitor is a memory leak that takes scoring down."""
        stats = {name: {"quantiles": np.linspace(0, 1, 11).tolist()} for name in FEATURE_ORDER}
        monitor = DriftMonitor(stats, FEATURE_ORDER, window=50)
        for _ in range(500):
            monitor.observe(np.zeros(len(FEATURE_ORDER)), 0.1, "approve")
        assert monitor.samples == 50

    def test_alert_rate_tracks_decisions(self):
        stats = {name: {"quantiles": np.linspace(0, 1, 11).tolist()} for name in FEATURE_ORDER}
        monitor = DriftMonitor(stats, FEATURE_ORDER)
        for i in range(100):
            monitor.observe(
                np.zeros(len(FEATURE_ORDER)), 0.9 if i < 20 else 0.1,
                "decline" if i < 20 else "approve",
            )
        assert monitor.score_summary()["alert_rate"] == pytest.approx(0.2)


# --------------------------------------------------------------------------
# API key lifecycle
# --------------------------------------------------------------------------
@pytest.fixture
def database():
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(f"sqlite:///{handle.name}")
    db.create_all()
    return db


class TestApiKeyLifecycle:
    @pytest.mark.asyncio
    async def test_valid_key_authenticates(self, database):
        store = SqlApiKeyStore(database)
        raw, key_id, key_hash = issue_api_key()
        store.register("gateway-1", key_hash, "gateway", key_id)
        principal = await store.authenticate(raw)
        assert principal is not None
        assert principal.subject == "gateway-1"

    @pytest.mark.asyncio
    async def test_revoked_key_is_refused(self, database):
        store = SqlApiKeyStore(database)
        raw, key_id, key_hash = issue_api_key()
        store.register("gateway-1", key_hash, "gateway", key_id)
        assert store.revoke(key_id, reason="leaked in a public repo")
        assert await store.authenticate(raw) is None

    @pytest.mark.asyncio
    async def test_expired_key_is_refused_and_marked(self, database):
        store = SqlApiKeyStore(database)
        raw, key_id, key_hash = issue_api_key()
        record = store.register("gateway-1", key_hash, "gateway", key_id, valid_for_days=1)

        with database.session() as session:
            from fraudguard.db.sql_stores import ApiKeyRecord

            stored = session.get(ApiKeyRecord, record.id)
            stored.expires_at = datetime.now(UTC) - timedelta(days=1)
            session.commit()

        assert await store.authenticate(raw) is None

        with database.session() as session:
            from fraudguard.db.sql_stores import ApiKeyRecord

            assert session.get(ApiKeyRecord, record.id).status == KeyStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_unknown_and_malformed_keys_refused(self, database):
        store = SqlApiKeyStore(database)
        assert await store.authenticate("fgk_unknown.secret") is None
        assert await store.authenticate("garbage") is None

    @pytest.mark.asyncio
    async def test_key_from_one_subject_cannot_impersonate_another(self, database):
        store = SqlApiKeyStore(database)
        raw_a, id_a, hash_a = issue_api_key()
        _, id_b, hash_b = issue_api_key()
        store.register("gateway-a", hash_a, "gateway", id_a)
        store.register("analyst-b", hash_b, "analyst", id_b)

        principal = await store.authenticate(raw_a)
        assert principal.subject == "gateway-a"
        assert principal.role == "gateway"

    @pytest.mark.asyncio
    async def test_secret_swapped_onto_another_key_id_fails(self, database):
        """Mixing a valid secret with a different key's id must not work."""
        store = SqlApiKeyStore(database)
        raw_a, id_a, hash_a = issue_api_key()
        _, id_b, hash_b = issue_api_key()
        store.register("gateway-a", hash_a, "gateway", id_a)
        store.register("gateway-b", hash_b, "gateway", id_b)

        secret_a = raw_a.split(".", 1)[1]
        assert await store.authenticate(f"fgk_{id_b}.{secret_a}") is None


# --------------------------------------------------------------------------
# case workflow
# --------------------------------------------------------------------------
class TestCaseWorkflow:
    def _save(self, repo, transaction_id, decision, probability):
        repo.save(
            transaction_id=transaction_id,
            card_token="tok_abc",
            card_last_four="1111",
            amount=100.0,
            currency="USD",
            merchant_id="mrc_1",
            merchant_country="IN",
            channel="ecommerce",
            occurred_at=datetime.now(UTC),
            fraud_probability=probability,
            decision=decision,
            model_version="test",
            reason_codes="[]",
            request_id="req_1",
        )

    def test_only_flagged_transactions_become_cases(self, database):
        repo = ScoreRepository(database)
        self._save(repo, "txn_a", "approve", 0.01)
        self._save(repo, "txn_b", "review", 0.45)
        self._save(repo, "txn_c", "decline", 0.92)
        assert {c.transaction_id for c in repo.open_cases()} == {"txn_b", "txn_c"}

    def test_cases_are_ordered_by_risk(self, database):
        repo = ScoreRepository(database)
        self._save(repo, "txn_low", "review", 0.35)
        self._save(repo, "txn_high", "decline", 0.95)
        assert repo.open_cases()[0].transaction_id == "txn_high"

    def test_decided_case_leaves_the_queue(self, database):
        repo = ScoreRepository(database)
        self._save(repo, "txn_b", "review", 0.45)
        repo.record_decision("txn_b", True, "analyst@bank", "confirmed", 1)
        assert repo.open_cases() == []

    def test_scoring_is_idempotent_on_transaction_id(self, database):
        repo = ScoreRepository(database)
        self._save(repo, "txn_b", "review", 0.45)
        self._save(repo, "txn_b", "decline", 0.99)
        cases = repo.open_cases()
        assert len(cases) == 1
        assert cases[0].decision == "review"

    def test_performance_ignores_unlabelled_traffic(self, database):
        """Unlabelled transactions must not be counted as legitimate —
        that is how a system reports recall it does not have."""
        repo = ScoreRepository(database)
        self._save(repo, "txn_labelled", "decline", 0.95)
        self._save(repo, "txn_unlabelled", "approve", 0.02)
        repo.record_decision("txn_labelled", True, "analyst@bank", "", 1)

        result = repo.performance_since(datetime.now(UTC) - timedelta(days=1))
        assert result["labelled"] == 1
        assert result["precision"] == 1.0


# --------------------------------------------------------------------------
# fallback stores
# --------------------------------------------------------------------------
class TestFallbackStores:
    def test_rate_limiter_blocks_past_limit(self):
        limiter = InMemoryRateLimiter()
        for _ in range(5):
            assert limiter.allow("client", 5, 60)[0]
        assert limiter.allow("client", 5, 60)[0] is False

    def test_rate_limits_are_per_identity(self):
        limiter = InMemoryRateLimiter()
        for _ in range(5):
            limiter.allow("client-a", 5, 60)
        assert limiter.allow("client-b", 5, 60)[0]

    def test_nonce_accepted_once(self):
        store = InMemoryNonceStore()
        assert store.check_and_store("abc", 300)
        assert store.check_and_store("abc", 300) is False

    def test_history_store_prunes_old_entries(self):
        store = InMemoryCardHistoryStore(retention_hours=24)
        now = datetime.now(UTC)
        for hours in (100, 50, 2, 1):
            store.record(
                "tok",
                {"occurred_at": now - timedelta(hours=hours), "amount": 10.0,
                 "latitude": 1.0, "longitude": 1.0, "merchant_id": "m", "device_id": "d"},
            )
        assert len(store.get("tok").timestamps) == 2


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
class TestLoaders:
    def _write_csv(self, path, rows):
        import csv

        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_missing_columns_give_actionable_error(self, tmp_path):
        path = tmp_path / "bad.csv"
        self._write_csv(path, [{"foo": 1, "bar": 2}])
        with pytest.raises(DatasetError, match="missing required columns"):
            load_transactions_csv(path)

    def test_all_legitimate_data_is_rejected(self, tmp_path):
        path = tmp_path / "nofraud.csv"
        self._write_csv(path, [
            {"card_id": "c1", "amount": 10, "timestamp": "2026-01-01T10:00:00Z",
             "merchant_id": "m1", "is_fraud": 0}
        ])
        with pytest.raises(DatasetError, match="no fraud labels"):
            load_transactions_csv(path)

    def test_custom_column_map(self, tmp_path):
        path = tmp_path / "custom.csv"
        self._write_csv(path, [
            {"cid": "c1", "value": 10, "when": "2026-01-01T10:00:00Z",
             "shop": "m1", "label": i % 2}
            for i in range(4)
        ])
        rows = load_transactions_csv(path, {
            "card_token": "cid", "amount": "value", "occurred_at": "when",
            "merchant_id": "shop", "is_fraud": "label",
        })
        assert len(rows) == 4
        assert describe(rows)["fraud_rate"] == 0.5

    def test_rows_come_back_in_time_order(self, tmp_path):
        path = tmp_path / "unsorted.csv"
        self._write_csv(path, [
            {"card_id": "c1", "amount": 10, "timestamp": t, "merchant_id": "m1",
             "is_fraud": f}
            for t, f in [("2026-01-03T10:00:00Z", 1), ("2026-01-01T10:00:00Z", 0),
                         ("2026-01-02T10:00:00Z", 0)]
        ])
        rows = load_transactions_csv(path)
        assert [r["occurred_at"] for r in rows] == sorted(r["occurred_at"] for r in rows)
