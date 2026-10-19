"""Request/response contracts.

Validation here is a security control, not just hygiene: strict types and
bounded ranges stop malformed or adversarial input from reaching the feature
pipeline or the model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
SafeId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class Channel(StrEnum):
    CARD_PRESENT = "card_present"
    ECOMMERCE = "ecommerce"
    MOTO = "moto"
    RECURRING = "recurring"
    ATM = "atm"


class Decision(StrEnum):
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"


class TransactionIn(BaseModel):
    """What the payment gateway sends us.

    `pan` is accepted, immediately tokenized, and never stored or echoed.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transaction_id: SafeId
    pan: Annotated[str, Field(min_length=13, max_length=23, repr=False)]
    amount: Annotated[Decimal, Field(gt=0, le=Decimal("1000000"), decimal_places=2)]
    currency: CurrencyCode
    occurred_at: datetime
    channel: Channel
    merchant_id: SafeId
    merchant_category_code: Annotated[str, StringConstraints(pattern=r"^\d{4}$")]
    merchant_country: CountryCode
    cardholder_country: CountryCode
    device_id: SafeId | None = None
    ip_address: str | None = None
    latitude: Annotated[float, Field(ge=-90, le=90)] | None = None
    longitude: Annotated[float, Field(ge=-180, le=180)] | None = None
    is_3ds_authenticated: bool = False

    @field_validator("occurred_at")
    @classmethod
    def _not_far_future(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            v = v.replace(tzinfo=UTC)
        if (v - datetime.now(UTC)).total_seconds() > 300:
            raise ValueError("occurred_at is implausibly far in the future")
        return v


class ReasonCode(BaseModel):
    feature: str
    contribution: float
    description: str


class ScoreOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    card_last_four: str
    fraud_probability: Annotated[float, Field(ge=0, le=1)]
    decision: Decision
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    model_version: str
    scored_at: datetime
    request_id: str


class CaseDecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: SafeId
    confirmed_fraud: bool
    note: Annotated[str, Field(max_length=1000)] = ""


class TokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: Annotated[str, StringConstraints(min_length=3, max_length=64)]
    password: Annotated[str, Field(min_length=12, max_length=256, repr=False)]
