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

import asyncio
import logging
import time
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
from app.directives import SOLAR_PASS_MIN, CompiledLimits, Directive
from app.llm import (
    InterpretationUnavailable,
    RequestBudget,
    interpret_notes,
)
from app.optimizer import (
    InfeasibleScheduleError,
    Solution,
    SolverError,
    solve_with_solar_policy,
)
from app.schemas import (
    DirectiveType,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    OptimizeResponse,
    ScenarioRequest,
    SemanticValidationError,
    check_battery_feasibility,
)
from app.validator import ReplayFailure, validate_response

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


# ---------------------------------------------------------------------------
# Response construction
# ---------------------------------------------------------------------------


def _describe_hours(hours: tuple[int, ...]) -> str:
    """Render an hour list as compact ranges, so (2,3,4,9) reads as 2-4, 9."""
    if not hours:
        return "no hours"
    spans: list[str] = []
    start = previous = hours[0]
    for hour in hours[1:]:
        if hour == previous + 1:
            previous = hour
            continue
        spans.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = hour
    spans.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(spans)


def build_plan_summary(
    directives: list[Directive], solution: Solution, limits: CompiledLimits
) -> str:
    """Describe the strategy using only facts already verified.

    PS S10.1 asks for a short human-readable explanation. It is assembled from
    the compiled limits and the solved plan rather than from a second inference
    call, so it cannot disagree with the schedule it describes (plan S4.2).
    """
    applied: list[str] = []
    for directive in directives:
        if directive.directive_type is DirectiveType.NO_OP:
            continue
        hours = _describe_hours(directive.hours)
        adjustment = directive.adjustment or {}
        if directive.directive_type is DirectiveType.SOLAR_REDUCTION:
            applied.append(
                f"solar limited to {adjustment['factor']:g}x forecast in hour(s) {hours}"
            )
        elif directive.directive_type is DirectiveType.MINIMUM_BATTERY_RESERVE:
            applied.append(
                f"battery held at or above {adjustment['minimum_energy_kwh']:g} kWh "
                f"in hour(s) {hours}"
            )
        elif directive.directive_type is DirectiveType.NO_CHARGE_WINDOW:
            applied.append(f"charging blocked in hour(s) {hours}")
        elif directive.directive_type is DirectiveType.NO_DISCHARGE_WINDOW:
            applied.append(f"discharging blocked in hour(s) {hours}")
        elif directive.directive_type is DirectiveType.MAX_GRID_WINDOW:
            applied.append(
                f"grid import capped at {adjustment['max_grid_kwh']:g} kWh "
                f"in hour(s) {hours}"
            )

    ignored = sum(1 for d in directives if d.directive_type is DirectiveType.NO_OP)

    parts: list[str] = []
    if applied:
        parts.append("Applied " + "; ".join(applied) + ".")
    else:
        parts.append("No operator note changed the schedule.")
    if ignored:
        parts.append(
            f"{ignored} note(s) did not affect the schedule and were treated as no_op."
        )
    if limits.solar_pass == SOLAR_PASS_MIN:
        parts.append(
            "Overlapping solar reductions were relaxed to the least restrictive "
            "factor because the stricter combination admitted no feasible schedule."
        )

    peak_hour = max(solution.hourly_plan, key=lambda row: row.grid_kwh).hour
    parts.append(
        f"Shifting battery use toward cheaper hours, the plan buys "
        f"{solution.total_grid_kwh:g} kWh for {solution.total_cost_bdt:.2f} BDT, "
        f"peaks at {solution.peak_grid_kwh:g} kWh in hour {peak_hour}, and returns "
        f"the battery to its starting level."
    )
    return " ".join(parts)


def build_response(
    request: ScenarioRequest,
    directives: list[Directive],
    solution: Solution,
    limits: CompiledLimits,
) -> OptimizeResponse:
    return OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=[
            {
                "note_index": directive.note_index,
                "applies": directive.applies,
                "directive_type": directive.directive_type,
                "structured_adjustment": directive.adjustment,
                "explanation": directive.explanation,
            }
            for directive in directives
        ],
        hourly_plan=solution.hourly_plan,
        total_grid_kwh=solution.total_grid_kwh,
        total_cost_bdt=solution.total_cost_bdt,
        peak_grid_kwh=solution.peak_grid_kwh,
        plan_summary=build_plan_summary(directives, solution, limits),
    )


# ---------------------------------------------------------------------------
# The judged endpoint
# ---------------------------------------------------------------------------


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(payload: ScenarioRequest) -> OptimizeResponse:
    """Interpret, compile, solve, replay, respond (PS SS1-3).

    Structural validation has already run by the time this is entered: FastAPI
    rejects a malformed body with 400 through the handler above.
    """
    started = time.monotonic()

    # Well-formed but impossible battery parameters, caught before any spend on
    # inference (plan S4.1).
    check_battery_feasibility(payload.battery)

    budget = RequestBudget.start(SETTINGS.request_deadline_s)
    outcome = await interpret_notes(payload, SETTINGS, budget)
    interpreted_at = time.monotonic()

    # CBC blocks, so it runs off the event loop and /health stays responsive
    # while an optimization is in flight (plan S11).
    solution, limits = await asyncio.to_thread(
        solve_with_solar_policy,
        outcome.directives,
        payload.demand(),
        payload.solar(),
        payload.tariff(),
        payload.battery,
        SETTINGS.solver_time_limit_s,
    )
    solved_at = time.monotonic()

    response = build_response(payload, outcome.directives, solution, limits)

    problems = validate_response(
        payload, outcome.directives, limits.solar_pass, response
    )
    if problems:
        # The service will not return a schedule it cannot verify itself.
        raise ReplayFailure(problems)

    finished = time.monotonic()
    logger.info(
        "200 scenario=%s notes=%d interpret=%.2fs solve=%.2fs replay=%.3fs "
        "total=%.2fs solar_pass=%s solver=%s cost=%.2f | %s",
        payload.scenario_id,
        len(payload.operator_notes),
        interpreted_at - started,
        solved_at - interpreted_at,
        finished - solved_at,
        finished - started,
        limits.solar_pass,
        solution.solver_status,
        solution.total_cost_bdt,
        " | ".join(outcome.trace),
    )
    return response


# ---------------------------------------------------------------------------
# Pipeline failure handling (PS S6.1)
# ---------------------------------------------------------------------------


@app.exception_handler(InterpretationUnavailable)
async def _handle_interpretation_unavailable(
    request: Request, exc: InterpretationUnavailable
) -> JSONResponse:
    """Every model path was exhausted.

    Plan S6.2 forbids the alternatives: no partial plan, no dropped note, no
    fabricated directive. The provider error text is never echoed, so a key or
    endpoint detail cannot leak through this path.
    """
    logger.warning("500 interpretation_unavailable: %s", exc)
    return _error(
        500,
        "interpretation_unavailable",
        "The operator notes could not be interpreted reliably; no plan was produced.",
    )


@app.exception_handler(InfeasibleScheduleError)
async def _handle_infeasible(
    request: Request, exc: InfeasibleScheduleError
) -> JSONResponse:
    """No schedule satisfies this scenario together with its directives.

    Reported as 422 rather than 500: the request is well formed, but as
    interpreted it admits no valid plan, which is what PS S6.1 reserves 422 for.
    Organizer scoring scenarios are stated to be feasible (PS S08).
    """
    logger.warning("422 infeasible_scenario: %s", exc)
    return _error(
        422,
        "infeasible_scenario",
        "No schedule can satisfy this scenario together with its operator directives.",
    )


@app.exception_handler(SolverError)
async def _handle_solver_error(request: Request, exc: SolverError) -> JSONResponse:
    logger.error("500 solver_error: %s", exc)
    return _error(
        500, "solver_error", "The optimizer did not produce a proven optimal schedule."
    )


@app.exception_handler(ReplayFailure)
async def _handle_replay_failure(request: Request, exc: ReplayFailure) -> JSONResponse:
    logger.error("500 replay_failed: %s", exc.problems)
    return _error(
        500,
        "replay_failed",
        "The computed schedule failed internal verification and was not returned.",
    )
