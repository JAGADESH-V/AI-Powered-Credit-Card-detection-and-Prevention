"""Durable storage: API keys, scored transactions, and analyst case decisions.

Two rules drive the schema:

  * **No cardholder data.** Rows carry a card token and the last four digits.
    A PAN never reaches this layer, so a database compromise does not become
    a card-data breach.

  * **Nothing is deleted.** Keys are revoked by status, never removed, so the
    audit trail keeps referring to something real. Case decisions are
    append-only for the same reason.

All queries go through SQLAlchemy's parameter binding — no string-built SQL
anywhere in this file.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from fraudguard.observability import get_logger
from fraudguard.security.auth import (
    ROLE_SCOPES,
    Principal,
    needs_rehash,
    parse_api_key,
    verify_api_key,
)

logger = get_logger(__name__)

# Verified against when a key id is unknown, so failed lookups cost the same
# time as real ones.
_DUMMY_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$"
    "c29tZXNhbHR2YWx1ZQ$8dGVzdGRpZ2VzdHZhbHVlZm9ydGltaW5n"
)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class KeyStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ApiKeyRecord(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    subject: Mapped[str] = mapped_column(String(128), index=True)
    key_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default=KeyStatus.ACTIVE, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)

    def is_usable(self, now: datetime | None = None) -> bool:
        now = now or utcnow()
        if self.status != KeyStatus.ACTIVE:
            return False
        if self.expires_at is not None:
            expires = self.expires_at
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=UTC)
            if expires <= now:
                return False
        return True


class ScoredTransaction(Base):
    """One row per scoring decision. No PAN — token and last four only."""

    __tablename__ = "scored_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    card_token: Mapped[str] = mapped_column(String(64), index=True)
    card_last_four: Mapped[str] = mapped_column(String(4))
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3))
    merchant_id: Mapped[str] = mapped_column(String(64), index=True)
    merchant_country: Mapped[str] = mapped_column(String(2))
    channel: Mapped[str] = mapped_column(String(24))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    fraud_probability: Mapped[float] = mapped_column(Float, index=True)
    decision: Mapped[str] = mapped_column(String(16), index=True)
    model_version: Mapped[str] = mapped_column(String(32), index=True)
    reason_codes: Mapped[str] = mapped_column(Text, default="[]")
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    request_id: Mapped[str] = mapped_column(String(64))

    __table_args__ = (Index("ix_scored_open_cases", "decision", "scored_at"),)


class CaseDecision(Base):
    """An analyst's verdict. These become the labels for the next model, so
    they are append-only and always attributed to a named actor."""

    __tablename__ = "case_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("scored_transactions.transaction_id"), index=True
    )
    confirmed_fraud: Mapped[bool] = mapped_column(Boolean)
    analyst: Mapped[str] = mapped_column(String(128), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    audit_sequence: Mapped[int] = mapped_column(Integer)


# ---------------------------------------------------------------------------
# repository
# ---------------------------------------------------------------------------
class Database:
    def __init__(self, url: str, echo: bool = False) -> None:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, echo=echo, future=True, connect_args=connect_args)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, future=True)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self.session_factory()


class SqlApiKeyStore:
    """Authenticates API keys and enforces expiry, revocation and rotation."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def register(
        self,
        subject: str,
        key_hash: str,
        role: str,
        key_id: str,
        valid_for_days: int | None = 90,
    ) -> ApiKeyRecord:
        if role not in ROLE_SCOPES:
            raise ValueError(f"unknown role: {role}")
        with self._db.session() as session:
            record = ApiKeyRecord(
                key_id=key_id,
                subject=subject,
                key_hash=key_hash,
                role=role,
                status=KeyStatus.ACTIVE,
                expires_at=utcnow() + timedelta(days=valid_for_days) if valid_for_days else None,
            )
            session.add(record)
            session.commit()
            return record

    def revoke(self, key_id: str, reason: str) -> bool:
        with self._db.session() as session:
            record = session.scalar(select(ApiKeyRecord).where(ApiKeyRecord.key_id == key_id))
            if record is None:
                return False
            record.status = KeyStatus.REVOKED
            record.revoked_at = utcnow()
            record.revoked_reason = reason
            session.commit()
            logger.info("apikey.revoked", extra={"key_id": key_id, "reason": reason})
            return True

    async def authenticate(self, raw_key: str) -> Principal | None:
        """One indexed lookup by key id, then one Argon2 verification.

        The verification always runs, even when the id is unknown, against a
        dummy hash — otherwise an unknown id would return fast and a known one
        slowly, letting an attacker enumerate valid key ids by timing alone.
        """
        parsed = parse_api_key(raw_key)
        if parsed is None:
            verify_api_key("decoy", _DUMMY_HASH)
            return None

        key_id, secret = parsed
        with self._db.session() as session:
            record = session.scalar(select(ApiKeyRecord).where(ApiKeyRecord.key_id == key_id))

            if record is None:
                verify_api_key(secret, _DUMMY_HASH)
                return None

            if not verify_api_key(secret, record.key_hash):
                return None

            if not record.is_usable():
                # Expired keys are marked so, rather than being silently
                # ignored, so the state is visible when someone investigates.
                if record.status == KeyStatus.ACTIVE:
                    record.status = KeyStatus.EXPIRED
                    session.commit()
                logger.warning(
                    "apikey.unusable",
                    extra={"key_id": key_id, "status": record.status},
                )
                return None

            record.last_used_at = utcnow()
            if needs_rehash(record.key_hash):
                logger.info("apikey.needs_rehash", extra={"key_id": key_id})
            session.commit()

            return Principal(
                subject=record.subject,
                role=record.role,
                scopes=ROLE_SCOPES[record.role],
                credential_type="api_key",
            )


class ScoreRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    def save(self, **fields) -> None:
        with self._db.session() as session:
            existing = session.scalar(
                select(ScoredTransaction).where(
                    ScoredTransaction.transaction_id == fields["transaction_id"]
                )
            )
            if existing is not None:
                return
            session.add(ScoredTransaction(**fields))
            session.commit()

    def open_cases(self, limit: int = 100) -> list[ScoredTransaction]:
        """Flagged transactions with no analyst verdict yet."""
        with self._db.session() as session:
            decided = select(CaseDecision.transaction_id)
            return list(
                session.scalars(
                    select(ScoredTransaction)
                    .where(ScoredTransaction.decision.in_(["review", "decline"]))
                    .where(ScoredTransaction.transaction_id.not_in(decided))
                    .order_by(ScoredTransaction.fraud_probability.desc())
                    .limit(limit)
                ).all()
            )

    def record_decision(
        self, transaction_id: str, confirmed_fraud: bool, analyst: str,
        note: str, audit_sequence: int,
    ) -> CaseDecision:
        with self._db.session() as session:
            decision = CaseDecision(
                transaction_id=transaction_id,
                confirmed_fraud=confirmed_fraud,
                analyst=analyst,
                note=note,
                audit_sequence=audit_sequence,
            )
            session.add(decision)
            session.commit()
            return decision

    def performance_since(self, since: datetime) -> dict:
        """Measured performance on *labelled* transactions only.

        Anything without an analyst verdict is excluded rather than assumed
        legitimate — treating unlabelled traffic as clean is how fraud systems
        report recall they do not have.
        """
        with self._db.session() as session:
            rows = session.execute(
                select(ScoredTransaction.decision, CaseDecision.confirmed_fraud)
                .join(
                    CaseDecision,
                    CaseDecision.transaction_id == ScoredTransaction.transaction_id,
                )
                .where(ScoredTransaction.scored_at >= since)
            ).all()

        if not rows:
            return {"labelled": 0, "note": "no analyst-labelled transactions in window"}

        flagged_and_fraud = sum(1 for d, f in rows if d != "approve" and f)
        flagged = sum(1 for d, _ in rows if d != "approve")
        fraud = sum(1 for _, f in rows if f)

        return {
            "labelled": len(rows),
            "precision": round(flagged_and_fraud / flagged, 4) if flagged else None,
            "recall": round(flagged_and_fraud / fraud, 4) if fraud else None,
            "confirmed_fraud": fraud,
        }
