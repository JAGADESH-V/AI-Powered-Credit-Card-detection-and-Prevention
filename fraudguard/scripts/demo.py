#!/usr/bin/env python3
"""End-to-end demonstration against the real app.

Sends a run of normal spending, then a card-testing burst from a foreign
country on a new device, and shows how the decision changes. Finishes by
tampering with the audit log to show the chain detects it.

    python scripts/demo.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("FG_ENVIRONMENT", "dev")
os.environ.setdefault("FG_HMAC_SIGNING_SECRET", "demo-hmac-secret-not-for-production")
os.environ.setdefault("FG_JWT_SECRET", "demo-jwt-secret-not-for-production-use")
os.environ.setdefault("FG_REQUIRE_HTTPS", "false")
os.environ.setdefault("FG_LOG_LEVEL", "WARNING")
os.environ.setdefault("FG_TRUSTED_HOSTS", '["testserver", "localhost", "127.0.0.1"]')
os.environ.setdefault("FG_DATABASE_URL", f"sqlite:///{tempfile.mkdtemp()}/demo.db")

from fastapi.testclient import TestClient  # noqa: E402

from fraudguard.api.app import create_app  # noqa: E402
from fraudguard.security.auth import issue_api_key  # noqa: E402
from fraudguard.security.crypto import new_nonce  # noqa: E402
from fraudguard.security.middleware import sign_request  # noqa: E402

SECRET = os.environ["FG_HMAC_SIGNING_SECRET"]
PAN = "4111111111111111"


def send(client, api_key, payload):
    body = json.dumps(payload, default=str).encode()
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        **sign_request(SECRET, "POST", "/v1/score", body, new_nonce()),
    }
    return client.post("/v1/score", content=body, headers=headers)


def txn(index, when, **overrides):
    payload = {
        "transaction_id": f"txn_demo_{index:05d}",
        "pan": PAN,
        "amount": "42.00",
        "currency": "USD",
        "occurred_at": when.isoformat(),
        "channel": "card_present",
        "merchant_id": "mrc_0100",
        "merchant_category_code": "5411",
        "merchant_country": "IN",
        "cardholder_country": "IN",
        "device_id": "dev_home",
        "latitude": 12.97,
        "longitude": 77.59,
        "is_3ds_authenticated": True,
    }
    payload.update(overrides)
    return payload


def show(label, response):
    if response.status_code != 200:
        print(f"  {label:<34} HTTP {response.status_code} {response.text[:70]}")
        return
    data = response.json()
    print(
        f"  {label:<34} {data['decision'].upper():<8} "
        f"p={data['fraud_probability']:.4f}  ****{data['card_last_four']}"
    )
    for reason in data["reason_codes"][:2]:
        print(f"      → {reason['description']} ({reason['contribution']:+.3f})")


def main() -> int:
    app = create_app()
    with TestClient(app) as client:
        raw_key, key_id, key_hash = issue_api_key()
        app.state.key_store.register("gateway-demo", key_hash, "gateway", key_id)
        analyst_raw, analyst_id, analyst_hash = issue_api_key()
        app.state.key_store.register("analyst-demo", analyst_hash, "analyst", analyst_id)

        if app.state.scorer is None:
            print("No model found. Run: python scripts/train_model.py")
            return 1

        print("=" * 72)
        print(f"FraudGuard demo — model {app.state.scorer.version}")
        print(f"decline at p >= {app.state.scorer.decline_threshold:.3f}, "
              f"review at p >= {app.state.scorer.review_threshold:.3f}")
        print("=" * 72)

        now = datetime.now(UTC) - timedelta(days=3)

        print("\n1. Establishing normal spending history")
        for i in range(12):
            when = now + timedelta(hours=i * 5)
            response = send(client, raw_key, txn(i, when, amount=f"{35 + i * 3}.00"))
            if i >= 10:
                show(f"normal purchase #{i + 1}", response)

        print("\n2. Card-testing burst — foreign country, new device, tiny amounts")
        attack_start = now + timedelta(hours=61)
        for i in range(6):
            when = attack_start + timedelta(minutes=i * 2)
            response = send(
                client, raw_key,
                txn(100 + i, when, amount="1.50", channel="ecommerce",
                    merchant_country="US", device_id="dev_attacker",
                    latitude=40.71, longitude=-74.01, is_3ds_authenticated=False),
            )
            show(f"burst attempt #{i + 1}", response)

        print("\n3. Replay protection")
        payload = txn(200, now + timedelta(hours=70))
        body = json.dumps(payload, default=str).encode()
        nonce = new_nonce()
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": raw_key,
            **sign_request(SECRET, "POST", "/v1/score", body, nonce),
        }
        first = client.post("/v1/score", content=body, headers=headers)
        replay = client.post("/v1/score", content=body, headers=headers)
        print(f"  original request                   HTTP {first.status_code}")
        print(f"  identical replay                   HTTP {replay.status_code} "
              f"({'blocked' if replay.status_code == 409 else 'NOT BLOCKED'})")

        print("\n4. Audit trail")
        audit = app.state.audit
        intact, breach = audit.verify_chain()
        count = sum(1 for _ in audit.read_all())
        print(f"  records written                    {count}")
        print(f"  chain intact                       {intact}")

        lines = audit.path.read_text().splitlines()
        if len(lines) > 3:
            lines[2] = lines[2].replace('"approve"', '"decline"')
            audit.path.write_text("\n".join(lines) + "\n")
            intact, breach = audit.verify_chain()
            print(f"  after editing one record           intact={intact} ({breach})")

        print("\n5. Analyst case queue")
        cases = client.get("/v1/cases", headers={"X-API-Key": analyst_raw}).json()
        print(f"  flagged, awaiting review            {cases['count']}")
        for case in cases["open_cases"][:3]:
            print(f"    {case['transaction_id']}  {case['fraud_probability']:.3f}  "
                  f"{case['amount']:>8.2f} {case['currency']}  {case['merchant_country']}")

        if cases["open_cases"]:
            target = cases["open_cases"][0]["transaction_id"]
            verdict = client.post(
                "/v1/cases/decision",
                json={"transaction_id": target, "confirmed_fraud": True,
                      "note": "confirmed with cardholder"},
                headers={"X-API-Key": analyst_raw},
            )
            print(f"  analyst confirms {target}   HTTP {verdict.status_code}")
            remaining = client.get("/v1/cases", headers={"X-API-Key": analyst_raw}).json()
            print(f"  queue after the verdict            {remaining['count']}")

        print("\n6. Scope enforcement")
        denied = client.post(
            "/v1/cases/decision",
            json={"transaction_id": "txn_demo_00001", "confirmed_fraud": True},
            headers={"X-API-Key": raw_key},   # gateway key: score:write only
        )
        print(f"  gateway key deciding a case        HTTP {denied.status_code} "
              f"({'blocked' if denied.status_code == 403 else 'NOT BLOCKED'})")

        print("\n7. Monitoring")
        drift = client.get("/v1/monitoring/drift", headers={"X-API-Key": analyst_raw}).json()
        print(f"  scored in window                   {drift['samples_observed']}")
        print(f"  alert rate                         {drift['scores'].get('alert_rate', 0):.2%}")
        print(f"  enough data for drift              {drift['sufficient_data']}")
        perf = client.get("/v1/monitoring/performance",
                          headers={"X-API-Key": analyst_raw}).json()
        print(f"  analyst-labelled transactions      {perf.get('labelled', 0)}")

        print("\n" + "=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
