"""Request/response contracts for /predict.

The field set mirrors the example dict hard-coded in the vendored predictor's
``__main__``: that dict is the input contract.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# Bounds worst-case memory per request: the cross-join multiplies every
# string field by the brand count. Not a business constraint.
_MAX_FIELD_LEN = 256


class UserSessionRequest(BaseModel):
    session_dt: datetime = Field(..., description="Funnel start timestamp")
    conversion_dt: datetime | None = Field(
        default=None, description="Unknown at ranking time in production; accepted for schema parity with the original script"
    )
    register_date: datetime = Field(..., description="Survey submission timestamp; required — see import_preprocess")

    campaign_id: int
    page: str = Field(..., max_length=_MAX_FIELD_LEN)
    auto_city: str = Field(..., max_length=_MAX_FIELD_LEN)
    auto_country: str = Field(..., max_length=_MAX_FIELD_LEN)
    auto_state: str = Field(..., max_length=_MAX_FIELD_LEN)
    device_type: str = Field(..., max_length=_MAX_FIELD_LEN)
    sub1: str = Field(..., max_length=_MAX_FIELD_LEN)
    sub2: str = Field(..., max_length=_MAX_FIELD_LEN)
    sub3: str = Field(..., max_length=_MAX_FIELD_LEN)

    business_type: str = Field(..., max_length=_MAX_FIELD_LEN)
    credit_score: str = Field(..., max_length=_MAX_FIELD_LEN)
    industry: str = Field(..., max_length=_MAX_FIELD_LEN)
    loan_amount: str = Field(..., max_length=_MAX_FIELD_LEN)
    loan_reason: str = Field(..., max_length=_MAX_FIELD_LEN)
    monthly_revenue: str = Field(..., max_length=_MAX_FIELD_LEN)
    time_in_business: str = Field(..., max_length=_MAX_FIELD_LEN)

    fname: str = Field(..., max_length=_MAX_FIELD_LEN)
    lname: str = Field(..., max_length=_MAX_FIELD_LEN)
    cellphone: int

    @field_validator("sub1", "sub2", "sub3", mode="before")
    @classmethod
    def _coerce_sub_to_str(cls, v: object) -> str:
        return str(v)

    @field_validator("session_dt", "register_date", "conversion_dt")
    @classmethod
    def _reject_tz_aware(cls, v: datetime | None) -> datetime | None:
        # The training data and the vendored time features use naive local
        # wall-clock timestamps. strftime() on a tz-aware value would drop the
        # offset without converting, skewing them silently.
        if v is not None and v.tzinfo is not None:
            raise ValueError(
                "timestamps must be naive (no UTC offset) to match the training data's wall-clock "
                "convention — convert to that local time before sending, don't send a UTC offset"
            )
        return v

    def to_predictor_dict(self) -> dict:
        d = self.model_dump()
        d["session_dt"] = self.session_dt.strftime("%Y-%m-%d %H:%M:%S")
        d["register_date"] = self.register_date.strftime("%Y-%m-%d %H:%M:%S")
        d["conversion_dt"] = self.conversion_dt.strftime("%Y-%m-%d %H:%M:%S") if self.conversion_dt else None
        return d


class BrandRanking(BaseModel):
    rank: float
    expected_payout: float


class RankBrandsResponse(BaseModel):
    rankings: dict[str, BrandRanking]
    model_version: str
    fallback_used: bool
    latency_ms: float
    # Which rung answered: the primary transport, 'cache', or a fallback.
    # Per response, so a caller can treat a degraded ranking differently.
    payout_tier: str = "unknown"


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model_version: str | None = None
    artifacts_loaded_at: str | None = None
    tabpfn_transport_mode: str | None = None
    fallback_count: int | None = None
    # Distinguishes "artifacts loaded" from "the payout leg answers".
    canary_ok: bool | None = None
    auth_required: bool | None = None
    payout_stats: dict | None = None
    surrogate_top1_agreement: float | None = None
    detail: str | None = None
