"""Feature-layer tests.

The history-poisoning test is the important one here: it encodes an attack
the naive implementation lost to.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fraudguard.features.engineering import (
    FEATURE_ORDER,
    CardHistory,
    build_features,
    frame_from_transactions,
    haversine_km,
)

BANGALORE = (12.97, 77.59)
NEW_YORK = (40.71, -74.01)


def transaction(when, **overrides):
    payload = {
        "transaction_id": "txn_1",
        "amount": 50.0,
        "occurred_at": when,
        "channel": "card_present",
        "merchant_id": "mrc_1",
        "merchant_category_code": "5411",
        "merchant_country": "IN",
        "cardholder_country": "IN",
        "device_id": "dev_home",
        "latitude": BANGALORE[0],
        "longitude": BANGALORE[1],
        "is_3ds_authenticated": True,
    }
    payload.update(overrides)
    return payload


class TestGeo:
    def test_haversine_known_distance(self):
        # Bangalore to New York is roughly 13,800 km.
        assert 13_000 < haversine_km(BANGALORE, NEW_YORK) < 14_500

    def test_same_point_is_zero(self):
        assert haversine_km(BANGALORE, BANGALORE) == 0.0


class TestFeatureContract:
    def test_all_features_present_and_ordered(self):
        features = build_features(transaction(datetime.now(UTC)), CardHistory.empty())
        assert list(features.keys()) == FEATURE_ORDER

    def test_empty_history_produces_finite_values(self):
        features = build_features(transaction(datetime.now(UTC)), CardHistory.empty())
        assert all(isinstance(v, float) and v == v for v in features.values())


class TestHistoryPoisoning:
    """An attacker must not be able to normalise their own behaviour by
    spending on the card first."""

    def _card_with_home_history(self, start):
        history = CardHistory.empty()
        for i in range(8):
            history.record(
                start + timedelta(hours=i * 6), 45.0, BANGALORE, "mrc_1", "dev_home",
                trusted=True,
            )
        return history

    def test_declined_transaction_does_not_reset_location(self):
        start = datetime.now(UTC) - timedelta(days=3)
        history = self._card_with_home_history(start)
        attack_time = start + timedelta(hours=49)

        first = build_features(
            transaction(attack_time, latitude=NEW_YORK[0], longitude=NEW_YORK[1],
                        merchant_country="US", device_id="dev_attacker"),
            history,
        )
        # Attacker's attempt is declined -> recorded but NOT trusted.
        history.record(attack_time, 1.5, NEW_YORK, "mrc_9", "dev_attacker", trusted=False)

        second = build_features(
            transaction(attack_time + timedelta(minutes=2),
                        latitude=NEW_YORK[0], longitude=NEW_YORK[1],
                        merchant_country="US", device_id="dev_attacker"),
            history,
        )

        assert second["distance_from_last_km"] > 0, "distance must still measure from home"
        assert second["distance_from_last_km"] == first["distance_from_last_km"]
        assert second["new_device"] == 1.0, "attacker device must stay unfamiliar"
        assert second["implied_speed_kmh"] > 1000

    def test_declined_transaction_still_counts_toward_velocity(self):
        start = datetime.now(UTC) - timedelta(days=3)
        history = self._card_with_home_history(start)
        attack_time = start + timedelta(hours=49)

        for i in range(4):
            history.record(attack_time + timedelta(minutes=i), 1.5, NEW_YORK,
                           f"mrc_{i}", "dev_attacker", trusted=False)

        features = build_features(
            transaction(attack_time + timedelta(minutes=5)), history
        )
        assert features["txn_count_1h"] >= 4, "declined attempts are still burst signal"

    def test_approved_transaction_does_update_baseline(self):
        start = datetime.now(UTC) - timedelta(days=3)
        history = self._card_with_home_history(start)
        history.record(start + timedelta(hours=49), 45.0, BANGALORE, "mrc_1",
                       "dev_new_phone", trusted=True)

        features = build_features(
            transaction(start + timedelta(hours=50), device_id="dev_new_phone"), history
        )
        assert features["new_device"] == 0.0


class TestNoLeakage:
    def test_features_only_see_the_past(self):
        """A card's first transaction must have empty velocity counts even
        though later transactions exist in the same batch."""
        base = datetime.now(UTC) - timedelta(days=2)
        rows = [
            {**transaction(base + timedelta(hours=i)), "card_token": "card_a",
             "transaction_id": f"t{i}", "is_fraud": 0}
            for i in range(6)
        ]
        frame = frame_from_transactions(rows)
        assert frame.iloc[0]["txn_count_24h"] == 0.0
        assert frame.iloc[-1]["txn_count_24h"] > 0.0
