"""Feature construction.

Split deliberately into two groups:

  * Stateless features derive from the transaction alone. Identical at train
    and serve time by construction.
  * Stateful features need the card's recent history (velocity, deviation
    from typical spend). These are the highest-signal features in fraud
    detection and the easiest to get wrong — the same code path builds them
    for both training and serving, which is how you avoid train/serve skew.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

FEATURE_ORDER = [
    "amount_log",
    "hour_sin",
    "hour_cos",
    "is_night",
    "is_weekend",
    "channel_ecommerce",
    "channel_card_present",
    "channel_atm",
    "cross_border",
    "is_3ds_authenticated",
    "high_risk_mcc",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_24h",
    "amount_zscore_vs_card",
    "seconds_since_last_txn",
    "distance_from_last_km",
    "implied_speed_kmh",
    "distinct_merchants_24h",
    "new_device",
]

HIGH_RISK_MCCS = {"7995", "6051", "5967", "4816", "5122", "7273"}


@dataclass(slots=True)
class CardHistory:
    """Rolling state for one card. Backed by Redis in production, keyed by the
    card token so no PAN is involved.

    Entries are split into two views, and the distinction is a security
    control rather than a detail:

      * **Attempts** (every transaction, approved or not) drive the velocity
        counts. A burst of declines is itself strong signal, so declined
        attempts must still be counted.

      * **Trusted** (approved transactions only) defines the card's baseline —
        last known-good location, familiar devices, normal spend.

    Folding declined attempts into the baseline lets an attacker establish a
    new "normal" with their first transaction: the second attempt then looks
    like it came from a familiar device at the same location as the last one,
    and the strongest features collapse. Keeping the baseline clean means an
    attack has to beat the real cardholder's history every single time.
    """

    timestamps: list[datetime]
    amounts: list[float]
    coordinates: list[tuple[float, float] | None]
    merchants: list[str]
    trusted: list[bool]
    devices: set[str]

    @classmethod
    def empty(cls) -> CardHistory:
        return cls([], [], [], [], [], set())

    def window(self, now: datetime, hours: float) -> list[int]:
        cutoff = now - timedelta(hours=hours)
        return [i for i, ts in enumerate(self.timestamps) if ts >= cutoff]

    def trusted_amounts(self) -> list[float]:
        return [a for a, ok in zip(self.amounts, self.trusted, strict=True) if ok]

    def last_trusted_coordinates(self) -> tuple[float, float] | None:
        for coords, ok in zip(reversed(self.coordinates), reversed(self.trusted), strict=True):
            if ok and coords is not None:
                return coords
        return None

    def last_trusted_time(self) -> datetime | None:
        for ts, ok in zip(reversed(self.timestamps), reversed(self.trusted), strict=True):
            if ok:
                return ts
        return None

    def record(
        self,
        when: datetime,
        amount: float,
        coordinates: tuple[float, float] | None,
        merchant: str,
        device: str | None,
        trusted: bool = True,
    ) -> None:
        self.timestamps.append(when)
        self.amounts.append(amount)
        self.coordinates.append(coordinates)
        self.merchants.append(merchant)
        self.trusted.append(trusted)
        if device and trusted:
            self.devices.add(device)


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def build_features(txn: dict, history: CardHistory) -> dict[str, float]:
    """Produce the exact feature vector the model expects, in order."""
    when: datetime = txn["occurred_at"]
    amount = float(txn["amount"])
    coordinates = (
        (txn["latitude"], txn["longitude"])
        if txn.get("latitude") is not None and txn.get("longitude") is not None
        else None
    )

    hour = when.hour + when.minute / 60
    idx_1h = history.window(when, 1)
    idx_24h = history.window(when, 24)
    recent_amounts = [history.amounts[i] for i in idx_24h]

    # Baseline features use trusted (approved) history only — see CardHistory.
    card_amounts = np.asarray(history.trusted_amounts(), dtype=float)
    if card_amounts.size >= 3:
        mean, std = card_amounts.mean(), card_amounts.std()
        amount_z = (amount - mean) / std if std > 1e-6 else 0.0
    else:
        amount_z = 0.0

    # Gap since the last attempt (any) — a tight gap is burst signal.
    if history.timestamps:
        seconds_since_last = max((when - history.timestamps[-1]).total_seconds(), 0.0)
    else:
        seconds_since_last = 86_400.0

    # Travel is measured from the last known-good location, so a declined
    # attempt cannot relocate the cardholder's baseline.
    distance = 0.0
    last_coords = history.last_trusted_coordinates()
    if coordinates and last_coords:
        distance = haversine_km(last_coords, coordinates)

    last_trusted = history.last_trusted_time()
    trusted_gap = (when - last_trusted).total_seconds() if last_trusted else 86_400.0
    hours_elapsed = max(trusted_gap / 3600, 1 / 60)
    implied_speed = distance / hours_elapsed

    features = {
        "amount_log": math.log1p(amount),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "is_night": float(hour < 6 or hour >= 23),
        "is_weekend": float(when.weekday() >= 5),
        "channel_ecommerce": float(txn["channel"] == "ecommerce"),
        "channel_card_present": float(txn["channel"] == "card_present"),
        "channel_atm": float(txn["channel"] == "atm"),
        "cross_border": float(txn["merchant_country"] != txn["cardholder_country"]),
        "is_3ds_authenticated": float(txn.get("is_3ds_authenticated", False)),
        "high_risk_mcc": float(txn["merchant_category_code"] in HIGH_RISK_MCCS),
        "txn_count_1h": float(len(idx_1h)),
        "txn_count_24h": float(len(idx_24h)),
        "amount_sum_24h": float(sum(recent_amounts)),
        "amount_zscore_vs_card": float(np.clip(amount_z, -10, 10)),
        "seconds_since_last_txn": math.log1p(seconds_since_last),
        "distance_from_last_km": math.log1p(distance),
        "implied_speed_kmh": float(np.clip(implied_speed, 0, 5000)),
        "distinct_merchants_24h": float(len({history.merchants[i] for i in idx_24h})),
        "new_device": float(
            bool(txn.get("device_id")) and txn["device_id"] not in history.devices
        ),
    }
    return {name: features[name] for name in FEATURE_ORDER}


def frame_from_transactions(rows: list[dict]) -> pd.DataFrame:
    """Replay transactions chronologically per card so that every stateful
    feature only ever sees the past. This is the guard against leakage."""
    rows = sorted(rows, key=lambda r: r["occurred_at"])
    histories: dict[str, CardHistory] = {}
    built: list[dict[str, float]] = []

    for row in rows:
        token = row["card_token"]
        history = histories.setdefault(token, CardHistory.empty())
        built.append(build_features(row, history))
        # At training time the analogue of "approved" is "known-good": a
        # confirmed-fraud transaction never joins the card's baseline, exactly
        # as a declined one never does at serving time.
        history.record(
            row["occurred_at"],
            float(row["amount"]),
            (row.get("latitude"), row.get("longitude"))
            if row.get("latitude") is not None
            else None,
            row["merchant_id"],
            row.get("device_id"),
            trusted=not bool(row.get("is_fraud", 0)),
        )

    frame = pd.DataFrame(built, columns=FEATURE_ORDER)
    if "is_fraud" in rows[0]:
        frame["is_fraud"] = [r["is_fraud"] for r in rows]
    return frame
