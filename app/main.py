"""FastAPI routes and request orchestration (PS S6; plan SS3, 4).

The status-code contract in PS S6.1 does not match FastAPI's defaults, so the
handlers below are configured explicitly rather than inherited (plan S4):

* malformed JSON or a structurally invalid body -> **400** (FastAPI would
  otherwise return 422 for both)
* a well-formed body that cannot admit a feasible schedule -> **422**
* any uncaught failure -> a controlled **500** carrying no stack trace, no
  prompt text, and no credential (Guide S04, S08)

An error must never be dressed up as a successful plan, so every failure path
returns the `ErrorResponse` shape and never a partial `hourly_plan`.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.config import load_settings
from app.schemas import (
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    SemanticValidationError,
)

logger = logging.getLogger("gridwise")

SETTINGS = load_settings()

logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

def _startup_check() -> None:
    """Fail loudly at startup, not silently on the first judged request.

    Plan S12 requires the packaged solver actually work inside the image, so
    CBC availability is probed once here and the redacted configuration is
    logged for reproducibility.
    """
    logger.info("GridWise %s starting; settings=%s", __version__, SETTINGS.describe())

    try:
        import pulp

        available = bool(pulp.PULP_CBC_CMD(msg=False).available())
    except Exception:  # pragma: no cover - reported, never raised
        logger.exception("CBC availability probe failed")
        available = False

    if available:
        logger.info("CBC solver available")
    else:
        logger.error(
            "CBC solver NOT available - optimization requests will fail. "
            "Check the coinor-cbc package and the PuLP installation."
        )

    for provider in SETTINGS.providers:
        if not provider.configured:
            logger.warning(
                "%s model is not fully configured (base_url=%s model=%s key_present=%s)",
                provider.role,
                provider.base_url,
                provider.model,
                bool(provider.api_key),
            )


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    _startup_check()
    yield


app = FastAPI(
    title="GridWise",
    version=__version__,
    lifespan=_lifespan,
    description=(
        "LLM-assisted campus energy scheduling. Operator notes are interpreted "
        "by a language model, validated deterministically, compiled into hourly "
        "bounds, solved as a linear program, and independently replayed."
    ),
)


def _error(status: int, code: str, message: str) -> JSONResponse:
    """Build the one documented error shape (plan S17)."""
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status, content=body.model_dump())


def _summarise_validation(exc: RequestValidationError) -> str:
    """Condense Pydantic errors into a short, safe, caller-useful message.

    Only the field location and the failure type are echoed. The submitted
    values are deliberately left out so a request body can never be reflected
    back into logs or responses.
    """
    parts: list[str] = []
    for error in exc.errors()[:6]:
        location = ".".join(str(item) for item in error.get("loc", ()) if item != "body")
        detail = error.get("msg", "invalid value")
        parts.append(f"{location or 'body'}: {detail}")
    if len(exc.errors()) > 6:
        parts.append(f"(+{len(exc.errors()) - 6} more)")
    return "; ".join(parts) or "request body failed structural validation"


@app.exception_handler(RequestValidationError)
async def _handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """PS S6.1: malformed JSON and structural errors are 400, not FastAPI's 422."""
    is_malformed_json = any(
        error.get("type") in {"json_invalid", "value_error.jsondecode"}
        for error in exc.errors()
    )
    code = "malformed_json" if is_malformed_json else "invalid_request"
    message = (
        "Request body is not valid JSON."
        if is_malformed_json
        else _summarise_validation(exc)
    )
    logger.info("400 %s %s", code, message)
    return _error(400, code, message)


@app.exception_handler(SemanticValidationError)
async def _handle_semantic_error(
    request: Request, exc: SemanticValidationError
) -> JSONResponse:
    """A well-formed request that no schedule can satisfy (PS S6.1, plan S4.1)."""
    logger.info("422 semantically_invalid_request %s", exc.message)
    return _error(422, "semantically_invalid_request", exc.message)


@app.exception_handler(StarletteHTTPException)
async def _handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "request could not be served"
    return _error(exc.status_code, f"http_{exc.status_code}", detail)


@app.exception_handler(Exception)
async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """Controlled 500 (PS S6.1, Guide S08 "Secret handling").

    The exception text is logged with a correlation id but never returned, so a
    provider error string or a stack trace cannot leak through the API.
    """
    incident = uuid.uuid4().hex[:12]
    logger.exception("500 internal_error incident=%s", incident)
    return _error(
        500,
        "internal_error",
        f"The service could not complete this request (incident {incident}).",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> dict[str, Any]:
    """PS S6.2 / Guide S08: HTTP 200 and exactly {"status": "ok"} when ready.

    Kept free of any network or solver work so it stays responsive while
    optimization requests are in flight (plan S11).
    """
    return {"status": "ok"}
