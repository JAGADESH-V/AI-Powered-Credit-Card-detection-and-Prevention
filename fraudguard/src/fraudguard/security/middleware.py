"""Transport-level protections applied to every request."""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from collections import defaultdict, deque  # noqa: F401
from typing import ClassVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from fraudguard.config import get_settings
from fraudguard.observability import get_logger, request_id_var

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Hardens the browser-facing surface (the analyst dashboard)."""

    HEADERS: ClassVar[dict[str, str]] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Content-Security-Policy": (
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; "
            "form-action 'none'"
        ),
        "Cache-Control": "no-store",
    }

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for header, value in self.HEADERS.items():
            response.headers.setdefault(header, value)
        if get_settings().require_https:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=63072000; includeSubDomains; preload",
            )
        return response


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a correlation id so a decision can be traced end to end."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request.completed",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects oversized bodies before they are parsed — cheap DoS defence."""

    def __init__(self, app: ASGIApp, max_bytes: int = 64 * 1024) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        declared = request.headers.get("content-length")
        if declared and int(declared) > self.max_bytes:
            return JSONResponse({"detail": "payload too large"}, status_code=413)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-credential rate limiting, delegated to the configured backend.

    The identity is the credential, not the IP: proxies collapse many clients
    behind one address, and an IP is trivially rotated. The credential is
    hashed so it never appears in a log line or a metric label.
    """

    EXEMPT_PATHS = ("/health", "/ready", "/metrics")

    def __init__(self, app: ASGIApp, per_minute: int) -> None:
        super().__init__(app)
        self.per_minute = per_minute

    @staticmethod
    def _identity(request: Request) -> str:
        credential = request.headers.get("X-API-Key") or request.headers.get("Authorization", "")
        if credential:
            return hashlib.sha256(credential.encode()).hexdigest()[:32]
        return request.client.host if request.client else "anonymous"

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        limiter = getattr(request.app.state, "rate_limiter", None)
        if limiter is None:
            return await call_next(request)

        identity = self._identity(request)
        allowed, remaining = limiter.allow(identity, self.per_minute, 60)

        if not allowed:
            logger.warning("rate_limit.exceeded", extra={"identity": identity})
            return JSONResponse(
                {"detail": "rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(remaining, 0))
        return response


class HMACSignatureMiddleware(BaseHTTPMiddleware):
    """Verifies that a scoring request was produced by the gateway and has not
    been replayed or modified in flight.

    The client signs: `timestamp.nonce.method.path.sha256(body)`.
    """

    PROTECTED_PREFIXES = ("/v1/score",)

    async def dispatch(self, request: Request, call_next) -> Response:
        if not request.url.path.startswith(self.PROTECTED_PREFIXES):
            return await call_next(request)

        settings = get_settings()
        signature = request.headers.get("X-Signature")
        timestamp = request.headers.get("X-Timestamp")
        nonce = request.headers.get("X-Nonce")

        if not (signature and timestamp and nonce):
            return JSONResponse({"detail": "request signature required"}, status_code=401)

        try:
            skew = abs(time.time() - float(timestamp))
        except ValueError:
            return JSONResponse({"detail": "invalid timestamp"}, status_code=401)

        if skew > settings.hmac_max_clock_skew_seconds:
            return JSONResponse({"detail": "request timestamp outside window"}, status_code=401)

        body = await request.body()
        payload = ".".join(
            [
                timestamp,
                nonce,
                request.method.upper(),
                request.url.path,
                hashlib.sha256(body).hexdigest(),
            ]
        )
        expected = hmac.new(
            settings.hmac_signing_secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected, signature):
            logger.warning("signature.invalid", extra={"path": request.url.path})
            return JSONResponse({"detail": "invalid signature"}, status_code=401)

        # Nonce is consumed only after the signature verifies, so an attacker
        # cannot burn a legitimate nonce by sending it with a bad signature.
        # The check-and-store is atomic in the Redis backend, so two concurrent
        # replays cannot both be admitted.
        store = getattr(request.app.state, "nonces", None)
        if store is not None:
            ttl = settings.hmac_max_clock_skew_seconds * 2
            if not store.check_and_store(nonce, ttl):
                logger.warning("replay.detected", extra={"nonce": nonce})
                return JSONResponse({"detail": "replayed request"}, status_code=409)

        return await call_next(request)


def sign_request(secret: str, method: str, path: str, body: bytes, nonce: str) -> dict[str, str]:
    """Client-side helper mirroring the middleware above."""
    timestamp = f"{time.time():.0f}"
    payload = ".".join(
        [timestamp, nonce, method.upper(), path, hashlib.sha256(body).hexdigest()]
    )
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return {"X-Signature": signature, "X-Timestamp": timestamp, "X-Nonce": nonce}
