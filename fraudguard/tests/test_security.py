"""Security control tests.

These are the tests that matter most: they assert the system fails closed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("FG_ENVIRONMENT", "dev")
os.environ.setdefault("FG_HMAC_SIGNING_SECRET", "test-hmac-secret-not-for-production")
os.environ.setdefault("FG_JWT_SECRET", "test-jwt-secret-not-for-production-use")

from fraudguard.security.audit import AuditLog
from fraudguard.security.auth import (
    Scope,
    create_access_token,
    decode_access_token,
    issue_api_key,
    parse_api_key,
    verify_api_key,
)
from fraudguard.security.crypto import (
    FieldCipher,
    luhn_valid,
    mask_pan,
    tokenize_pan,
)


# --------------------------------------------------------------------------
# Cardholder data
# --------------------------------------------------------------------------
class TestCardholderData:
    def test_luhn_rejects_invalid_pan(self):
        assert luhn_valid("4111111111111111")
        assert not luhn_valid("4111111111111112")
        assert not luhn_valid("123")

    def test_tokenization_is_deterministic(self):
        a = tokenize_pan("4111111111111111")
        b = tokenize_pan("4111 1111 1111 1111")
        assert a.token == b.token, "same card must yield same token for velocity features"

    def test_different_cards_give_different_tokens(self):
        assert tokenize_pan("4111111111111111").token != tokenize_pan("5555555555554444").token

    def test_token_does_not_contain_pan(self):
        card = tokenize_pan("4111111111111111")
        assert "4111111111111111" not in card.token
        assert card.last_four == "1111"

    def test_repr_never_leaks_pan(self):
        card = tokenize_pan("4111111111111111")
        assert "4111" not in repr(card)[:-6] or "****1111" in repr(card)
        assert "****1111" in repr(card)

    def test_invalid_pan_rejected_at_tokenization(self):
        with pytest.raises(ValueError):
            tokenize_pan("4111111111111112")

    def test_log_masking(self):
        masked = mask_pan("charge on card 4111111111111111 declined")
        assert "4111111111111111" not in masked
        assert "1111" in masked


class TestFieldEncryption:
    def test_roundtrip(self):
        cipher = FieldCipher(FieldCipher.generate_key())
        assert cipher.decrypt(cipher.encrypt("sensitive value")) == "sensitive value"

    def test_tampering_is_detected(self):
        cipher = FieldCipher(FieldCipher.generate_key())
        blob = cipher.encrypt("sensitive value")
        tampered = blob[:-4] + ("AAAA" if not blob.endswith("AAAA") else "BBBB")
        with pytest.raises(ValueError, match="tampering"):
            cipher.decrypt(tampered)

    def test_wrong_key_cannot_decrypt(self):
        blob = FieldCipher(FieldCipher.generate_key()).encrypt("secret")
        with pytest.raises(ValueError):
            FieldCipher(FieldCipher.generate_key()).decrypt(blob)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
class TestAuthentication:
    def test_api_key_hash_is_not_reversible(self):
        raw, key_id, stored = issue_api_key()
        assert raw not in stored
        assert stored.startswith("$argon2")

    def test_api_key_verification(self):
        raw, key_id, stored = issue_api_key()
        parsed = parse_api_key(raw)
        assert parsed is not None
        parsed_id, secret = parsed
        assert parsed_id == key_id
        assert verify_api_key(secret, stored)
        assert not verify_api_key(secret + "x", stored)

    def test_key_id_is_public_and_secret_is_not(self):
        """The id may appear in logs; only the secret half is hashed."""
        raw, key_id, stored = issue_api_key()
        _, secret = parse_api_key(raw)
        assert key_id in raw
        assert secret not in stored
        assert key_id not in stored

    def test_malformed_key_is_rejected_cleanly(self):
        assert parse_api_key("not-a-key") is None
        assert parse_api_key("fgk_missing_separator") is None
        assert parse_api_key("fgk_.empty") is None

    def test_keys_are_unique(self):
        ids = {issue_api_key()[1] for _ in range(50)}
        assert len(ids) == 50

    def test_jwt_carries_role_scopes(self):
        principal = decode_access_token(create_access_token("analyst@bank", "analyst"))
        assert Scope.CASE_DECIDE in principal.scopes
        assert Scope.MODEL_ADMIN not in principal.scopes

    def test_gateway_cannot_decide_cases(self):
        principal = decode_access_token(create_access_token("gateway-1", "gateway"))
        assert Scope.SCORE_WRITE in principal.scopes
        assert Scope.CASE_DECIDE not in principal.scopes

    def test_tampered_token_rejected(self):
        from fastapi import HTTPException

        token = create_access_token("analyst@bank", "analyst")
        with pytest.raises(HTTPException):
            decode_access_token(token[:-6] + "aaaaaa")

    def test_alg_none_attack_rejected(self):
        """The classic JWT bypass: re-sign with alg=none."""
        import base64
        import json

        from fastapi import HTTPException

        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "none", "typ": "JWT"}).encode()
        ).decode().rstrip("=")
        body = base64.urlsafe_b64encode(
            json.dumps({"sub": "attacker", "role": "admin",
                        "scopes": ["model:admin"], "exp": int(time.time()) + 3600}).encode()
        ).decode().rstrip("=")
        with pytest.raises(HTTPException):
            decode_access_token(f"{header}.{body}.")


# --------------------------------------------------------------------------
# Audit trail
# --------------------------------------------------------------------------
class TestAuditChain:
    def test_chain_verifies_when_intact(self, tmp_path: Path):
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(5):
            log.append(actor="svc", action="transaction.scored", resource=f"txn_{i}")
        intact, breach = log.verify_chain()
        assert intact and breach is None

    def test_edited_record_is_detected(self, tmp_path: Path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        for i in range(5):
            log.append(actor="svc", action="transaction.scored", resource=f"txn_{i}",
                       metadata={"decision": "approve"})

        lines = path.read_text().splitlines()
        lines[2] = lines[2].replace('"approve"', '"decline"')
        path.write_text("\n".join(lines) + "\n")

        intact, breach = log.verify_chain()
        assert not intact
        assert "2" in breach

    def test_deleted_record_is_detected(self, tmp_path: Path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        for i in range(5):
            log.append(actor="svc", action="transaction.scored", resource=f"txn_{i}")

        lines = path.read_text().splitlines()
        del lines[2]
        path.write_text("\n".join(lines) + "\n")

        intact, _ = log.verify_chain()
        assert not intact
