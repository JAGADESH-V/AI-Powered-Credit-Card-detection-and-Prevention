"""End-to-end API tests.

Each test asserts the API rejects something it should reject. A fraud system
that fails open is worse than no fraud system.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime

import pytest

os.environ["FG_ENVIRONMENT"] = "dev"
os.environ["FG_HMAC_SIGNING_SECRET"] = "test-hmac-secret-not-for-production"
os.environ["FG_JWT_SECRET"] = "test-jwt-secret-not-for-production-use"
os.environ["FG_REQUIRE_HTTPS"] = "false"
# TrustedHostMiddleware would otherwise reject TestClient's "testserver" host
# with a 400 before any route runs.
os.environ["FG_TRUSTED_HOSTS"] = '["testserver", "localhost", "127.0.0.1"]'
# Each test run gets a throwaway database so decisions from one test cannot
# leak into another's assertions.
_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["FG_DATABASE_URL"] = f"sqlite:///{_DB.name}"
os.environ["FG_ARTIFACT_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient  # noqa: E402

from fraudguard.api.app import create_app  # noqa: E402
from fraudguard.security.auth import issue_api_key  # noqa: E402
from fraudguard.security.crypto import new_nonce  # noqa: E402
from fraudguard.security.middleware import sign_request  # noqa: E402

SECRET = "test-hmac-secret-not-for-production"
VALID_PAN = "4111111111111111"


def valid_transaction(**overrides) -> dict:
    payload = {
        "transaction_id": "txn_00000001",
        "pan": VALID_PAN,
        "amount": "129.50",
        "currency": "USD",
        "occurred_at": datetime.now(UTC).isoformat(),
        "channel": "ecommerce",
        "merchant_id": "mrc_0042",
        "merchant_category_code": "5411",
        "merchant_country": "IN",
        "cardholder_country": "IN",
        "device_id": "dev_00001",
        "latitude": 12.97,
        "longitude": 77.59,
        "is_3ds_authenticated": True,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        raw, key_id, key_hash = issue_api_key()
        app.state.key_store.register("gateway-test", key_hash, "gateway", key_id)
        test_client.api_key = raw

        analyst_raw, analyst_id, analyst_hash = issue_api_key()
        app.state.key_store.register("analyst-test", analyst_hash, "analyst", analyst_id)
        test_client.analyst_key = analyst_raw

        yield test_client


def post_score(client, payload: dict, *, api_key: str | None = None, nonce: str | None = None,
               sign: bool = True):
    import json

    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key is not False:
        headers["X-API-Key"] = api_key or client.api_key
    if sign:
        headers.update(sign_request(SECRET, "POST", "/v1/score", body, nonce or new_nonce()))
    return client.post("/v1/score", content=body, headers=headers)


class TestHealth:
    def test_health_is_open(self, client):
        assert client.get("/health").status_code == 200

    def test_security_headers_present(self, client):
        response = client.get("/health")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "Content-Security-Policy" in response.headers

    def test_request_id_echoed(self, client):
        assert client.get("/health").headers["X-Request-ID"]


class TestScoringSecurity:
    def test_unsigned_request_rejected(self, client):
        assert post_score(client, valid_transaction(), sign=False).status_code == 401

    def test_unauthenticated_request_rejected(self, client):
        response = post_score(client, valid_transaction(), api_key=False)
        assert response.status_code == 401

    def test_bad_api_key_rejected(self, client):
        response = post_score(client, valid_transaction(), api_key="fgk_not_a_real_key")
        assert response.status_code == 401

    def test_replayed_nonce_rejected(self, client):
        nonce = new_nonce()
        payload = valid_transaction()
        first = post_score(client, payload, nonce=nonce)
        second = post_score(client, payload, nonce=nonce)
        assert first.status_code in (200, 503)
        assert second.status_code == 409, "replay must be blocked"

    def test_tampered_body_rejected(self, client):
        """Signature covers the body, so changing the amount invalidates it."""
        import json

        payload = valid_transaction()
        body = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": client.api_key,
            **sign_request(SECRET, "POST", "/v1/score", body, new_nonce()),
        }
        payload["amount"] = "99999.00"
        tampered = json.dumps(payload).encode()
        assert client.post("/v1/score", content=tampered, headers=headers).status_code == 401


class TestInputValidation:
    def test_invalid_pan_rejected(self, client):
        response = post_score(client, valid_transaction(pan="4111111111111112"))
        assert response.status_code in (400, 422)

    def test_negative_amount_rejected(self, client):
        assert post_score(client, valid_transaction(amount="-50.00")).status_code == 422

    def test_unknown_field_rejected(self, client):
        """extra='forbid' blocks parameter smuggling."""
        response = post_score(client, valid_transaction(is_admin=True))
        assert response.status_code == 422

    def test_bad_country_code_rejected(self, client):
        assert post_score(client, valid_transaction(merchant_country="INDIA")).status_code == 422

    def test_sql_injection_in_id_rejected(self, client):
        response = post_score(client, valid_transaction(transaction_id="1'; DROP TABLE t;--"))
        assert response.status_code == 422

    def test_future_timestamp_rejected(self, client):
        from datetime import timedelta

        future = (datetime.now(UTC) + timedelta(days=2)).isoformat()
        assert post_score(client, valid_transaction(occurred_at=future)).status_code == 422


class TestScoringBehaviour:
    def test_response_never_contains_pan(self, client):
        response = post_score(client, valid_transaction())
        if response.status_code == 200:
            assert VALID_PAN not in response.text
            assert response.json()["card_last_four"] == "1111"

    def test_scope_enforced_on_case_decision(self, client):
        """A gateway key has score:write but not case:decide."""
        response = client.post(
            "/v1/cases/decision",
            json={"transaction_id": "txn_00000001", "confirmed_fraud": True},
            headers={"X-API-Key": client.api_key},
        )
        assert response.status_code == 403
