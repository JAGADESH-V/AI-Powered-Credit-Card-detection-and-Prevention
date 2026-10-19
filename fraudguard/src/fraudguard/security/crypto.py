"""Cryptographic primitives for cardholder data.

Rules this module enforces:
  * A full PAN (card number) is never persisted, logged, or returned.
  * Only a keyed HMAC token (for linking transactions to the same card) and
    the last four digits (for human identification) leave this module.
  * Any other PII that must be stored is encrypted with authenticated
    encryption, so tampering in the database is detectable.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken

from fraudguard.config import get_settings

_PAN_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def luhn_valid(pan: str) -> bool:
    """Checksum validation. Rejects obviously malformed input before it
    travels any further into the system."""
    digits = [int(c) for c in pan if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


@dataclass(frozen=True, slots=True)
class CardToken:
    """The only representation of a card that is allowed to persist."""

    token: str      # deterministic, keyed — same card always yields same token
    last_four: str
    bin_prefix: str  # first 6, used as a feature (issuer / country signal)

    def __str__(self) -> str:  # defensive: keeps tokens out of casual logs
        return f"CardToken(****{self.last_four})"

    __repr__ = __str__


def tokenize_pan(pan: str) -> CardToken:
    """Turn a PAN into a non-reversible, deterministic token.

    Deterministic so that velocity features ("how many transactions on this
    card in the last hour") still work, keyed so that a stolen database of
    tokens cannot be brute-forced back to card numbers without the key.
    """
    normalized = re.sub(r"[^0-9]", "", pan)
    if not luhn_valid(normalized):
        raise ValueError("PAN failed checksum validation")

    key = get_settings().hmac_signing_secret.encode()
    if not key:
        raise RuntimeError("hmac_signing_secret is not configured")

    digest = hmac.new(key, normalized.encode(), hashlib.sha256).digest()
    return CardToken(
        token=base64.urlsafe_b64encode(digest).decode().rstrip("="),
        last_four=normalized[-4:],
        bin_prefix=normalized[:6],
    )


class FieldCipher:
    """Authenticated encryption for PII columns at rest."""

    def __init__(self, key: str | None = None) -> None:
        key = key or get_settings().field_encryption_key
        if not key:
            raise RuntimeError("field_encryption_key is not configured")
        self._fernet = Fernet(key.encode())

    @staticmethod
    def generate_key() -> str:
        return Fernet.generate_key().decode()

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise ValueError("ciphertext failed authentication — possible tampering") from exc


def mask_pan(text: str) -> str:
    """Scrub anything that looks like a card number out of free text.

    Applied to every log record as a last line of defence, so an accidental
    `logger.info(payload)` somewhere cannot leak cardholder data.
    """

    def _replace(match: re.Match[str]) -> str:
        digits = re.sub(r"[^0-9]", "", match.group())
        return f"{'*' * (len(digits) - 4)}{digits[-4:]}"

    return _PAN_RE.sub(_replace, text)


def constant_time_compare(a: str, b: str) -> bool:
    """Timing-safe comparison for secrets, tokens and signatures."""
    return hmac.compare_digest(a.encode(), b.encode())


def new_nonce() -> str:
    return secrets.token_urlsafe(24)
