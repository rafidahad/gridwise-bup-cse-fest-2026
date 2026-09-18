"""Request, directive, and response contracts (PS SS7, 10; plan S4).

Two validation levels are deliberately kept apart, because the required status
codes differ (PS S6.1, plan S4):

* **Structural** - missing fields, wrong types, non-finite numbers, wrong note
  count, wrong hour set. Enforced by Pydantic here, surfaced as HTTP **400**.
* **Semantic** - a well-formed request whose battery parameters cannot admit a
  feasible schedule. Enforced by `check_battery_feasibility`, HTTP **422**.

Input policies chosen where the sources are silent (plan S17 requires these be
documented rather than invented silently):

* Unknown extra fields are ignored, not rejected - an organizer input must
  never fail because it carries a field this service did not anticipate.
* Hour numerics must be finite but carry **no sign restriction**. Rejecting a
  negative tariff would forfeit an entire case if hidden data ever contains
  one, whereas accepting it stays solvable. Only the battery fields, where
  PS S9 and plan S4.1 give an explicit derivation, require non-negativity.
* Request `hours` need not arrive in ascending order; rows are indexed by
  `hour`. PS S7.2 requires uniqueness and coverage, not ordering.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import TOLERANCE

# A finite float: allow_inf_nan=False turns NaN/Infinity into a structural
# rejection rather than letting it poison the optimizer.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class DirectiveType(str, Enum):
    """The only accepted values (PS S4.1). No sixth type may ever be emitted."""

    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


OPERATIONAL_DIRECTIVES = frozenset(
    {
        DirectiveType.SOLAR_REDUCTION,
        DirectiveType.MINIMUM_BATTERY_RESERVE,
        DirectiveType.NO_CHARGE_WINDOW,
        DirectiveType.NO_DISCHARGE_WINDOW,
        DirectiveType.MAX_GRID_WINDOW,
    }
)


class BatteryAction(str, Enum):
    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"


# ---------------------------------------------------------------------------
# Request (PS S7)
# ---------------------------------------------------------------------------


class HourInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hour: int = Field(ge=0, le=23)
    demand_kwh: FiniteFloat
    solar_kwh: FiniteFloat
    tariff_bdt_per_kwh: FiniteFloat


class BatteryInput(BaseModel):
    """PS S7.3. Non-negativity here is required by plan S4.1, not invented."""

    model_config = ConfigDict(extra="ignore")

    capacity_kwh: NonNegativeFiniteFloat
    initial_energy_kwh: NonNegativeFiniteFloat
    minimum_energy_kwh: NonNegativeFiniteFloat
    max_charge_kwh_per_hour: NonNegativeFiniteFloat
    max_discharge_kwh_per_hour: NonNegativeFiniteFloat


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourInput] = Field(min_length=24, max_length=24)
    battery: BatteryInput

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, notes: list[str]) -> list[str]:
        """PS S7: notes are non-empty natural-language strings.

        The original text is preserved; only the emptiness test is whitespace
        insensitive, so the model always sees exactly what the operator wrote.
        """
        for index, note in enumerate(notes):
            if not note.strip():
                raise ValueError(f"operator_notes[{index}] must be a non-empty string")
        return notes

    @model_validator(mode="after")
    def _hours_cover_zero_to_23(self) -> ScenarioRequest:
        seen = [entry.hour for entry in self.hours]
        if len(set(seen)) != 24:
            duplicates = sorted({h for h in seen if seen.count(h) > 1})
            raise ValueError(
                f"hours must contain 24 unique entries; duplicated: {duplicates}"
            )
        missing = sorted(set(range(24)) - set(seen))
        if missing:
            raise ValueError(f"hours must cover 0 through 23; missing: {missing}")
        return self

    def by_hour(self) -> list[HourInput]:
        """Rows indexed by `hour`, independent of the order they arrived in."""
        return sorted(self.hours, key=lambda entry: entry.hour)

    def demand(self) -> list[float]:
        return [entry.demand_kwh for entry in self.by_hour()]

    def solar(self) -> list[float]:
        return [entry.solar_kwh for entry in self.by_hour()]

    def tariff(self) -> list[float]:
        return [entry.tariff_bdt_per_kwh for entry in self.by_hour()]


class SemanticValidationError(Exception):
    """A well-formed request that cannot admit a feasible schedule (HTTP 422)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def check_battery_feasibility(battery: BatteryInput) -> None:
    """Necessary battery conditions derived in plan S4.1 from PS SS9.2, 9.6.

    Neutrality forces ``E_after[23] == initial_energy_kwh`` and PS S9.2 bounds
    every ``E_after`` by ``[minimum_energy_kwh, capacity_kwh]``. Therefore::

        minimum_energy_kwh <= initial_energy_kwh <= capacity_kwh

    is necessary for any feasible schedule. This is a derived consequence, not
    an invented initial-state rule. Values are compared with the published
    tolerance so a boundary case is not rejected over float noise, and the
    supplied values are never silently clamped.

    Passing these checks does not establish feasibility - hourly demand, rates,
    and directives still decide that, and the optimizer reports it.

    A *temporary* directive reserve higher than the initial energy is NOT
    checked here: the schedule may legitimately charge up to meet it in time
    (plan S4.1). Only the permanent base reserve participates.
    """
    if battery.minimum_energy_kwh > battery.capacity_kwh + TOLERANCE:
        raise SemanticValidationError(
            "battery.minimum_energy_kwh exceeds battery.capacity_kwh, so no "
            "battery state can satisfy the reserve"
        )
    if battery.initial_energy_kwh > battery.capacity_kwh + TOLERANCE:
        raise SemanticValidationError(
            "battery.initial_energy_kwh exceeds battery.capacity_kwh"
        )
    if battery.initial_energy_kwh < battery.minimum_energy_kwh - TOLERANCE:
        raise SemanticValidationError(
            "battery.initial_energy_kwh is below battery.minimum_energy_kwh; "
            "end-of-day neutrality would force the final state below the reserve"
        )


# ---------------------------------------------------------------------------
# Response (PS S10)
# ---------------------------------------------------------------------------


class DirectiveInterpretation(BaseModel):
    """One entry per operator note, emitted in note_index order (PS S10.2)."""

    model_config = ConfigDict(extra="forbid")

    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: dict[str, Any] | None
    explanation: str


class HourlyPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class HealthResponse(BaseModel):
    """PS S6.2 requires exactly status "ok"."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """Minimal consistent error shape.

    PS S6.1 fixes the status codes but not the body, and plan S17 requires this
    choice be made and documented rather than left undefined. Nothing here ever
    carries a stack trace, a prompt, or a credential (Guide S04).
    """

    model_config = ConfigDict(extra="forbid")
    error: ErrorBody
