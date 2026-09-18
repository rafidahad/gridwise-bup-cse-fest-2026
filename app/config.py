"""Environment-backed settings (plan §3 "Configuration", §13.4).

Nothing here reads a hard-coded credential, and nothing here ever logs one.
`describe()` returns a redacted view used by the startup check and the internal
trace, so a misconfiguration is diagnosable without printing a key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Absolute tolerance published by PS §11.5 / Guide §08. Used for every public
# numeric comparison. Internal checks use the tighter epsilon below.
TOLERANCE = 0.01
INTERNAL_EPS = 1e-6


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int_or_none(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class ProviderSettings:
    """One OpenAI-compatible inference endpoint.

    Groq and NVIDIA NIM both speak the OpenAI chat-completions format, so a
    single client shape covers the primary and the backup while they remain
    independent providers (plan §6.1).
    """

    role: str
    api_key: str
    base_url: str
    model: str
    timeout_s: float

    @property
    def configured(self) -> bool:
        return bool(self.api_key) and bool(self.base_url) and bool(self.model)

    def describe(self) -> dict[str, object]:
        return {
            "role": self.role,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_s": self.timeout_s,
            "api_key_present": bool(self.api_key),
        }


@dataclass(frozen=True)
class Settings:
    primary: ProviderSettings
    backup: ProviderSettings
    temperature: float = 0.0
    seed: int | None = 7
    max_output_tokens: int = 1200
    request_deadline_s: float = 25.0
    solver_time_limit_s: float = 10.0
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    # Bumped whenever the prompt text or the interpretation schema changes, so
    # recorded evidence and any future cache stay attributable (plan §6.3).
    prompt_version: str = "2026-09-18.1"
    schema_version: str = "1.0.0"
    providers: tuple[ProviderSettings, ...] = field(default_factory=tuple)

    def describe(self) -> dict[str, object]:
        """Redacted settings snapshot. Safe to log and safe to surface."""
        return {
            "primary": self.primary.describe(),
            "backup": self.backup.describe(),
            "temperature": self.temperature,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
            "request_deadline_s": self.request_deadline_s,
            "solver_time_limit_s": self.solver_time_limit_s,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
        }


def load_settings() -> Settings:
    primary = ProviderSettings(
        role="primary",
        api_key=_env_str("GRIDWISE_PRIMARY_API_KEY", ""),
        base_url=_env_str("GRIDWISE_PRIMARY_BASE_URL", "https://api.groq.com/openai/v1"),
        model=_env_str("GRIDWISE_PRIMARY_MODEL", "llama-3.3-70b-versatile"),
        timeout_s=_env_float("GRIDWISE_PRIMARY_TIMEOUT_S", 9.0),
    )
    backup = ProviderSettings(
        role="backup",
        api_key=_env_str("GRIDWISE_BACKUP_API_KEY", ""),
        base_url=_env_str("GRIDWISE_BACKUP_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        model=_env_str("GRIDWISE_BACKUP_MODEL", "meta/llama-3.3-70b-instruct"),
        timeout_s=_env_float("GRIDWISE_BACKUP_TIMEOUT_S", 9.0),
    )
    return Settings(
        primary=primary,
        backup=backup,
        temperature=_env_float("GRIDWISE_TEMPERATURE", 0.0),
        seed=_env_int_or_none("GRIDWISE_SEED"),
        max_output_tokens=int(_env_float("GRIDWISE_MAX_OUTPUT_TOKENS", 1200)),
        request_deadline_s=_env_float("GRIDWISE_REQUEST_DEADLINE_S", 25.0),
        solver_time_limit_s=_env_float("GRIDWISE_SOLVER_TIME_LIMIT_S", 10.0),
        host=_env_str("GRIDWISE_HOST", "0.0.0.0"),
        port=int(_env_float("GRIDWISE_PORT", 8000)),
        log_level=_env_str("GRIDWISE_LOG_LEVEL", "INFO"),
        providers=(primary, backup),
    )


# Repair reuses the primary endpoint, so its timeout is read separately.
def repair_timeout_s() -> float:
    return _env_float("GRIDWISE_REPAIR_TIMEOUT_S", 7.0)
