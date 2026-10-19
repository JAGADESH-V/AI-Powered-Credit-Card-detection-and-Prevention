"""FastAPI application.

Middleware order is deliberate (outermost first):

    request context -> security headers -> body size -> rate limit ->
    HMAC signature -> routing

Cheap rejections happen before expensive ones. Signature verification reads
the body, so it sits behind the size limit.

Backing stores are chosen at startup: Redis and a real database when they are
reachable, in-process fallbacks otherwise. The choice is logged loudly, and
`/ready` reports it, because a silent downgrade means rate limiting and replay
protection stop working the moment you run more than one instance.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from fraudguard.config import get_settings
from fraudguard.db import redis_stores
from fraudguard.db.sql_stores import Database, ScoreRepository, SqlApiKeyStore
from fraudguard.models.drift import DriftMonitor
from fraudguard.models.scorer import FraudScorer
from fraudguard.observability import configure_logging, get_logger, request_id_var
from fraudguard.schemas import CaseDecisionIn, Decision, ScoreOut, TransactionIn
from fraudguard.security.audit import AuditLog
from fraudguard.security.auth import Principal, Scope, requires
from fraudguard.security.crypto import tokenize_pan
from fraudguard.security.middleware import (
    BodySizeLimitMiddleware,
    HMACSignatureMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    # ---- backing stores -------------------------------------------------
    client = redis_stores.connect(settings.redis_url)
    app.state.redis_available = client is not None

    if client is not None:
        app.state.rate_limiter = redis_stores.RedisRateLimiter(client)
        app.state.nonces = redis_stores.RedisNonceStore(client)
        app.state.history = redis_stores.RedisCardHistoryStore(client)
        app.state.idempotency = redis_stores.RedisIdempotencyStore(client)
    else:
        if settings.environment == "prod":
            raise RuntimeError(
                "Redis is required in production: in-process fallbacks are "
                "per-instance, so rate limiting and replay protection would "
                "silently stop working behind a load balancer."
            )
        logger.warning("stores.fallback", extra={"detail": "using in-process stores"})
        app.state.rate_limiter = redis_stores.InMemoryRateLimiter()
        app.state.nonces = redis_stores.InMemoryNonceStore()
        app.state.history = redis_stores.InMemoryCardHistoryStore()
        app.state.idempotency = redis_stores.InMemoryIdempotencyStore()

    database = Database(settings.database_url)
    database.create_all()
    app.state.database = database
    app.state.key_store = SqlApiKeyStore(database)
    app.state.scores = ScoreRepository(database)

    app.state.audit = AuditLog(Path(settings.artifact_dir) / "audit.jsonl")

    # ---- model ----------------------------------------------------------
    model_path = Path(settings.model_path)
    if model_path.exists():
        scorer = FraudScorer(model_path, settings.model_sha256, settings.review_threshold)
        app.state.scorer = scorer
        app.state.drift = DriftMonitor(scorer.reference_stats, scorer.pipeline_features)
    else:
        app.state.scorer = None
        app.state.drift = None
        logger.warning("model.missing", extra={"path": str(model_path)})

    intact, breach = app.state.audit.verify_chain()
    if not intact:
        logger.error("audit.chain_broken", extra={"detail": breach})

    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="FraudGuard",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None if settings.environment == "prod" else "/docs",
        redoc_url=None,
        openapi_url=None if settings.environment == "prod" else "/openapi.json",
    )

    # Starlette runs the last-added middleware outermost.
    app.add_middleware(HMACSignatureMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        per_minute=settings.rate_limit_per_minute,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=32 * 1024)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID", "Idempotency-Key"],
        max_age=600,
    )

    register_routes(app)
    register_error_handlers(app)
    return app


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        # Message is deliberately generic: validation detail can confirm
        # whether a card number was well-formed, which is information an
        # attacker testing stolen cards would like to have.
        return JSONResponse({"detail": "invalid request"}, status_code=400)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"path": request.url.path})
        return JSONResponse(
            {"detail": "internal error", "request_id": request_id_var.get()},
            status_code=500,
        )


def register_routes(app: FastAPI) -> None:
    # ---- operational ----------------------------------------------------
    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", include_in_schema=False)
    async def ready(request: Request) -> dict[str, object]:
        scorer = request.app.state.scorer
        if scorer is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        return {
            "status": "ready",
            "model_version": scorer.version,
            "algorithm": scorer.algorithm,
            "distributed_stores": request.app.state.redis_available,
        }

    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    async def metrics(request: Request) -> str:
        """Prometheus exposition. Scrape-only; keep it off the public listener."""
        scorer = request.app.state.scorer
        drift = request.app.state.drift
        lines = [
            "# HELP fraudguard_model_loaded Whether a model is loaded.",
            "# TYPE fraudguard_model_loaded gauge",
            f"fraudguard_model_loaded {1 if scorer else 0}",
        ]
        if drift is not None:
            summary = drift.score_summary()
            report = drift.report()
            lines += [
                "# HELP fraudguard_alert_rate Share of transactions not approved.",
                "# TYPE fraudguard_alert_rate gauge",
                f"fraudguard_alert_rate {summary.get('alert_rate', 0)}",
                "# HELP fraudguard_mean_score Mean fraud probability in the window.",
                "# TYPE fraudguard_mean_score gauge",
                f"fraudguard_mean_score {summary.get('mean_score', 0)}",
                "# HELP fraudguard_worst_feature_psi Largest feature PSI vs training.",
                "# TYPE fraudguard_worst_feature_psi gauge",
                f"fraudguard_worst_feature_psi {report.get('worst_psi', 0)}",
            ]
        intact, _ = request.app.state.audit.verify_chain()
        lines += [
            "# HELP fraudguard_audit_chain_intact Audit log integrity.",
            "# TYPE fraudguard_audit_chain_intact gauge",
            f"fraudguard_audit_chain_intact {1 if intact else 0}",
        ]
        return "\n".join(lines) + "\n"

    # ---- scoring --------------------------------------------------------
    @app.post("/v1/score", response_model=ScoreOut)
    async def score(
        request: Request,
        transaction: TransactionIn,
        principal: Annotated[Principal, Depends(requires(Scope.SCORE_WRITE))],
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> ScoreOut:
        scorer = request.app.state.scorer
        if scorer is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")

        if idempotency_key:
            cached = request.app.state.idempotency.get(idempotency_key)
            if cached is not None:
                return ScoreOut(**cached)

        # The PAN enters here and does not survive past this line.
        card = tokenize_pan(transaction.pan)
        payload = transaction.model_dump(exclude={"pan"})
        payload["card_token"] = card.token

        history = request.app.state.history.get(card.token)
        result = scorer.score(payload, history)

        # A declined transaction is recorded for velocity but never joins the
        # card's trusted baseline — otherwise an attacker's first attempt makes
        # every subsequent attempt look familiar.
        request.app.state.history.record(
            card.token, payload, trusted=result.decision is Decision.APPROVE
        )

        if request.app.state.drift is not None:
            request.app.state.drift.observe(
                result.feature_vector, result.probability, result.decision.value
            )

        response = ScoreOut(
            transaction_id=transaction.transaction_id,
            card_last_four=card.last_four,
            fraud_probability=round(result.probability, 6),
            decision=result.decision,
            reason_codes=result.reason_codes,
            model_version=result.model_version,
            scored_at=datetime.now(UTC),
            request_id=request_id_var.get(),
        )

        request.app.state.scores.save(
            transaction_id=transaction.transaction_id,
            card_token=card.token,
            card_last_four=card.last_four,
            amount=float(transaction.amount),
            currency=transaction.currency,
            merchant_id=transaction.merchant_id,
            merchant_country=transaction.merchant_country,
            channel=transaction.channel.value,
            occurred_at=transaction.occurred_at,
            fraud_probability=result.probability,
            decision=result.decision.value,
            model_version=result.model_version,
            reason_codes=json.dumps([r.model_dump() for r in result.reason_codes]),
            request_id=request_id_var.get(),
        )

        request.app.state.audit.append(
            actor=principal.subject,
            action="transaction.scored",
            resource=transaction.transaction_id,
            metadata={
                "decision": result.decision.value,
                "probability": round(result.probability, 4),
                "model_version": result.model_version,
                "card_last_four": card.last_four,
            },
        )

        if idempotency_key:
            request.app.state.idempotency.put(
                idempotency_key, response.model_dump(mode="json")
            )

        if result.decision is not Decision.APPROVE:
            logger.info(
                "transaction.flagged",
                extra={
                    "transaction_id": transaction.transaction_id,
                    "decision": result.decision.value,
                    "probability": round(result.probability, 4),
                },
            )
        return response

    # ---- case management ------------------------------------------------
    @app.get("/v1/cases")
    async def list_cases(
        request: Request,
        principal: Annotated[Principal, Depends(requires(Scope.CASE_READ))],
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> dict[str, object]:
        cases = request.app.state.scores.open_cases(limit=limit)
        return {
            "open_cases": [
                {
                    "transaction_id": case.transaction_id,
                    "card_last_four": case.card_last_four,
                    "amount": case.amount,
                    "currency": case.currency,
                    "merchant_id": case.merchant_id,
                    "merchant_country": case.merchant_country,
                    "occurred_at": case.occurred_at.isoformat(),
                    "fraud_probability": round(case.fraud_probability, 4),
                    "decision": case.decision,
                    "reason_codes": json.loads(case.reason_codes),
                }
                for case in cases
            ],
            "count": len(cases),
        }

    @app.post("/v1/cases/decision", status_code=202)
    async def record_case_decision(
        request: Request,
        decision: CaseDecisionIn,
        principal: Annotated[Principal, Depends(requires(Scope.CASE_DECIDE))],
    ) -> dict[str, str]:
        """An analyst confirming or dismissing a flag. These labels train the
        next model, so every one is attributed and audited."""
        record = request.app.state.audit.append(
            actor=principal.subject,
            action="case.decided",
            resource=decision.transaction_id,
            metadata={"confirmed_fraud": decision.confirmed_fraud, "note": decision.note},
        )
        request.app.state.scores.record_decision(
            transaction_id=decision.transaction_id,
            confirmed_fraud=decision.confirmed_fraud,
            analyst=principal.subject,
            note=decision.note,
            audit_sequence=record.sequence,
        )
        return {"status": "recorded", "audit_sequence": str(record.sequence)}

    # ---- monitoring -----------------------------------------------------
    @app.get("/v1/monitoring/drift")
    async def drift_report(
        request: Request,
        principal: Annotated[Principal, Depends(requires(Scope.CASE_READ))],
    ) -> dict:
        monitor = request.app.state.drift
        if monitor is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        return monitor.report()

    @app.get("/v1/monitoring/performance")
    async def performance(
        request: Request,
        principal: Annotated[Principal, Depends(requires(Scope.CASE_READ))],
        days: Annotated[int, Query(ge=1, le=365)] = 30,
    ) -> dict:
        since = datetime.now(UTC) - timedelta(days=days)
        return request.app.state.scores.performance_since(since)

    @app.get("/v1/audit/verify")
    async def verify_audit(
        request: Request,
        principal: Annotated[Principal, Depends(requires(Scope.AUDIT_READ))],
    ) -> dict[str, object]:
        intact, breach = request.app.state.audit.verify_chain()
        return {"intact": intact, "first_break": breach}


app = create_app()
