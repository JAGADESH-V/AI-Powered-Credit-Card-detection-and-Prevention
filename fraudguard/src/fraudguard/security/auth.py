"""Authentication and authorization.

Two credential types, for two audiences:
  * API keys  — machine-to-machine (a payment gateway calling /score).
                Stored only as an Argon2id hash, prefixed so they can be
                recognised and revoked if they leak into a repo.
  * JWT       — short-lived bearer tokens for human analysts using the
                dashboard, carrying explicit scopes.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from fraudguard.config import get_settings

_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)
_bearer = HTTPBearer(auto_error=False)

API_KEY_PREFIX = "fgk_"


class Scope(StrEnum):
    SCORE_WRITE = "score:write"       # submit a transaction for scoring
    CASE_READ = "case:read"           # view flagged transactions
    CASE_DECIDE = "case:decide"       # confirm/dismiss fraud
    MODEL_ADMIN = "model:admin"       # reload or promote a model
    AUDIT_READ = "audit:read"


ROLE_SCOPES: dict[str, set[Scope]] = {
    "gateway": {Scope.SCORE_WRITE},
    "analyst": {Scope.CASE_READ, Scope.CASE_DECIDE},
    "auditor": {Scope.CASE_READ, Scope.AUDIT_READ},
    "admin": set(Scope),
}


class Principal(BaseModel):
    subject: str
    role: str
    scopes: set[Scope] = Field(default_factory=set)
    credential_type: str = "jwt"

    def require(self, *needed: Scope) -> None:
        if not set(needed).issubset(self.scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="insufficient scope for this operation",
            )


# --------------------------------------------------------------------------
# API keys
# --------------------------------------------------------------------------
def issue_api_key() -> tuple[str, str, str]:
    """Returns (plaintext_key, key_id, stored_hash). The plaintext is shown
    exactly once, at issue time, and is not recoverable afterwards.

    The key carries its own public identifier:

        fgk_<key_id>.<secret>

    The id is indexed in the database, so authentication is one indexed lookup
    plus one Argon2 verification. Without it, every request would have to
    verify against every stored key in turn — Argon2 is deliberately expensive,
    so that turns each call into O(n) hashing work and hands an attacker a
    trivial denial-of-service.

    Only the secret half is hashed; the id is public by design.
    """
    key_id = secrets.token_urlsafe(12)
    secret = secrets.token_urlsafe(32)
    raw = f"{API_KEY_PREFIX}{key_id}.{secret}"
    return raw, key_id, _hasher.hash(secret)


def parse_api_key(raw: str) -> tuple[str, str] | None:
    """Split a presented key into (key_id, secret). None if malformed."""
    if not raw.startswith(API_KEY_PREFIX):
        return None
    body = raw[len(API_KEY_PREFIX):]
    key_id, separator, secret = body.partition(".")
    if not separator or not key_id or not secret:
        return None
    return key_id, secret


def verify_api_key(secret: str, stored_hash: str) -> bool:
    try:
        _hasher.verify(stored_hash, secret)
        return True
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def needs_rehash(stored_hash: str) -> bool:
    """Lets you raise Argon2 parameters over time without forcing re-issue."""
    return _hasher.check_needs_rehash(stored_hash)


# --------------------------------------------------------------------------
# JWT
# --------------------------------------------------------------------------
def create_access_token(subject: str, role: str) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    scopes = ROLE_SCOPES.get(role, set())
    payload = {
        "sub": subject,
        "role": role,
        "scopes": sorted(s.value for s in scopes),
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(seconds=settings.access_token_ttl_seconds),
        "iss": settings.service_name,
        "aud": f"{settings.service_name}-api",
        "jti": secrets.token_urlsafe(16),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Principal:
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],   # explicit — blocks alg=none
            audience=f"{settings.service_name}-api",
            issuer=settings.service_name,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token expired") from None
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from None

    return Principal(
        subject=claims["sub"],
        role=claims.get("role", ""),
        scopes={Scope(s) for s in claims.get("scopes", []) if s in set(Scope)},
    )


async def current_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """Resolves either credential type into a single Principal object."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        principal = await request.app.state.key_store.authenticate(api_key)
        if principal is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
        return principal

    if credentials is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(credentials.credentials)


def requires(*scopes: Scope):
    """Route dependency: `Depends(requires(Scope.CASE_DECIDE))`."""

    async def _guard(
        principal: Annotated[Principal, Depends(current_principal)],
    ) -> Principal:
        principal.require(*scopes)
        return principal

    return _guard
