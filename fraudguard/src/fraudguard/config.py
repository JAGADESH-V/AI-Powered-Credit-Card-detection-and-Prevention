"""Central configuration. Every secret comes from the environment or a mounted
secrets file — nothing sensitive is ever a default in this file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FG_",
        extra="ignore",
    )

    # ---- runtime -------------------------------------------------------
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    service_name: str = "fraudguard"

    # ---- paths ---------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    artifact_dir: Path = PROJECT_ROOT / "artifacts"

    # ---- model ---------------------------------------------------------
    model_path: Path = PROJECT_ROOT / "artifacts" / "model.joblib"
    model_sha256: str | None = None  # pinned hash; startup fails if mismatched
    decision_threshold: float = 0.55
    review_threshold: float = 0.30

    # ---- crypto / auth -------------------------------------------------
    # 32-byte urlsafe-base64 key for field encryption (Fernet).
    field_encryption_key: str = Field(default="", repr=False)
    # HS256 signing key for access tokens. In prod prefer RS256 + KMS.
    jwt_secret: str = Field(default="", repr=False)
    jwt_algorithm: str = "HS256"
    access_token_ttl_seconds: int = 900          # 15 min
    hmac_signing_secret: str = Field(default="", repr=False)
    hmac_max_clock_skew_seconds: int = 300       # replay window

    # ---- transport -----------------------------------------------------
    require_https: bool = True
    allowed_origins: list[str] = ["https://localhost:8501"]
    trusted_hosts: list[str] = ["localhost", "127.0.0.1"]

    # ---- rate limiting -------------------------------------------------
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 20

    # ---- storage -------------------------------------------------------
    database_url: str = Field(default="sqlite:///./fraudguard.db", repr=False)
    redis_url: str = "redis://localhost:6379/0"

    @field_validator("decision_threshold", "review_threshold")
    @classmethod
    def _in_unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
        return v

    def assert_production_ready(self) -> None:
        """Fail fast at boot rather than silently running insecurely."""
        if self.environment != "prod":
            return
        missing = [
            name
            for name in ("field_encryption_key", "jwt_secret", "hmac_signing_secret")
            if not getattr(self, name)
        ]
        if missing:
            raise RuntimeError(f"missing required prod secrets: {', '.join(missing)}")
        if not self.require_https:
            raise RuntimeError("require_https cannot be disabled in prod")
        if self.model_sha256 is None:
            raise RuntimeError("model_sha256 must be pinned in prod")
        if "*" in self.allowed_origins:
            raise RuntimeError("wildcard CORS origin is not permitted in prod")
        if self.database_url.startswith("sqlite"):
            raise RuntimeError(
                "SQLite is not suitable for production: use PostgreSQL so that "
                "concurrent writes and multi-instance deployment work correctly"
            )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.assert_production_ready()
    return settings
