"""Deterministic guardrails and the Directive Compiler (PS S08; plan SS5, 7).

Two separate jobs live here, in order:

1. :func:`validate_interpretation` treats raw model output as **untrusted data**
   and decides, per note, whether it is usable. It never repairs a value, never
   substitutes a type, never invents hours, and never drops a note. Anything it
   cannot accept is reported so the caller can run the bounded repair/failover
   policy in plan SS6.1-6.2.

2. :func:`compile_limits` turns the accepted directives into a small set of
   explicit hourly bounds. The language model has no influence past this point:
   the optimizer only ever sees numbers produced here.

The compiler keeps an internal note-to-limit trace so a failed test is
explainable, without adding anything to the public response (plan S7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.config import TOLERANCE
from app.schemas import (
    OPERATIONAL_DIRECTIVES,
    BatteryInput,
    DirectiveType,
)

# Shape required for each directive type (PS S4.1). Unknown keys in model output
# are dropped rather than echoed, so the emitted adjustment is always exactly
# the required object.
REQUIRED_ADJUSTMENT_KEYS: dict[DirectiveType, tuple[str, ...]] = {
    DirectiveType.SOLAR_REDUCTION: ("hours", "factor"),
    DirectiveType.MINIMUM_BATTERY_RESERVE: ("hours", "minimum_energy_kwh"),
    DirectiveType.NO_CHARGE_WINDOW: ("hours",),
    DirectiveType.NO_DISCHARGE_WINDOW: ("hours",),
    DirectiveType.MAX_GRID_WINDOW: ("hours", "max_grid_kwh"),
}


@dataclass(frozen=True)
class Directive:
    """One validated interpretation, ready to compile or to return."""

    note_index: int
    applies: bool
    directive_type: DirectiveType
    adjustment: dict[str, Any] | None
    explanation: str

    @property
    def hours(self) -> tuple[int, ...]:
        if not self.adjustment:
            return ()
        return tuple(self.adjustment.get("hours", ()))


@dataclass
class GuardrailResult:
    """Outcome of validating a complete model response.

    ``mapping_reliable`` is the distinction plan S6.2 turns on: when entries can
    be attributed to notes with confidence, only the failing notes need
    re-interpreting; when they cannot, the whole set must be regenerated rather
    than guessed at.
    """

    directives: dict[int, Directive] = field(default_factory=dict)
    entry_problems: dict[int, str] = field(default_factory=dict)
    mapping_reliable: bool = True
    mapping_problem: str | None = None

    def complete(self, note_count: int) -> bool:
        return (
            self.mapping_reliable
            and not self.entry_problems
            and set(self.directives) == set(range(note_count))
        )

    def ordered(self, note_count: int) -> list[Directive]:
        """Accepted directives in note_index order (PS S5.1)."""
        return [self.directives[i] for i in range(note_count) if i in self.directives]


def _is_finite_number(value: Any) -> bool:
    """Reject bools explicitly: in Python ``True`` is an ``int``."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _normalise_hours(raw: Any) -> tuple[list[int] | None, str | None]:
    """Validate an hours array, normalising order and duplicates.

    Plan S7.2 permits harmless ordering normalisation only where it preserves
    the interpreted meaning. Every directive type here has set-membership
    semantics ("these hours are affected"), so sorting and de-duplicating cannot
    change what was interpreted, and it makes the emitted array satisfy the
    ascending-unique requirement in PS S5.1.

    An empty array is *not* normalised away. Per plan S7.2 it is an extraction
    failure that must be re-interpreted, never a rule that silently does nothing
    and never a rewrite to no_op.
    """
    if not isinstance(raw, list):
        return None, "structured_adjustment.hours must be an array"
    values: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            return None, "structured_adjustment.hours must contain integers"
        if isinstance(item, float):
            if not math.isfinite(item) or item != int(item):
                return None, "structured_adjustment.hours must contain whole hours"
            item = int(item)
        if not isinstance(item, int):
            return None, "structured_adjustment.hours must contain integers"
        if not 0 <= item <= 23:
            return None, f"structured_adjustment.hours value {item} is outside 0-23"
        values.append(item)
    if not values:
        return None, "structured_adjustment.hours is empty for an applicable directive"
    return sorted(set(values)), None


def _validate_adjustment(
    directive_type: DirectiveType, raw: Any, capacity_kwh: float
) -> tuple[dict[str, Any] | None, str | None]:
    """Check one structured_adjustment against the exact shape for its type."""
    if not isinstance(raw, dict):
        return None, f"structured_adjustment must be an object for {directive_type.value}"

    for key in REQUIRED_ADJUSTMENT_KEYS[directive_type]:
        if key not in raw:
            return None, f"structured_adjustment is missing '{key}'"

    hours, problem = _normalise_hours(raw.get("hours"))
    if problem is not None:
        return None, problem
    assert hours is not None

    adjustment: dict[str, Any] = {"hours": hours}

    if directive_type is DirectiveType.SOLAR_REDUCTION:
        factor = raw.get("factor")
        if not _is_finite_number(factor):
            return None, "factor must be a finite number"
        factor = float(factor)
        # PS S08: factor is the fraction of solar that REMAINS, in [0, 1].
        if not 0.0 <= factor <= 1.0:
            return None, f"factor {factor} is outside 0-1 (it is the fraction remaining)"
        adjustment["factor"] = factor

    elif directive_type is DirectiveType.MINIMUM_BATTERY_RESERVE:
        reserve = raw.get("minimum_energy_kwh")
        if not _is_finite_number(reserve):
            return None, "minimum_energy_kwh must be a finite number"
        reserve = float(reserve)
        if reserve < 0:
            return None, "minimum_energy_kwh must be non-negative"
        if reserve > capacity_kwh:
            return None, (
                f"minimum_energy_kwh {reserve} exceeds battery capacity {capacity_kwh}"
            )
        adjustment["minimum_energy_kwh"] = reserve

    elif directive_type is DirectiveType.MAX_GRID_WINDOW:
        cap = raw.get("max_grid_kwh")
        if not _is_finite_number(cap):
            return None, "max_grid_kwh must be a finite number"
        cap = float(cap)
        if cap < 0:
            return None, "max_grid_kwh must be non-negative"
        adjustment["max_grid_kwh"] = cap

    return adjustment, None


def validate_interpretation(
    entries: Any, note_count: int, battery: BatteryInput
) -> GuardrailResult:
    """Validate a complete model interpretation against PS S08.

    Returns what was accepted and what was not. Nothing is silently discarded:
    an unusable entry becomes an ``entry_problem`` for targeted repair, and an
    unattributable response sets ``mapping_reliable = False`` so the caller
    regenerates the whole set instead of guessing (plan S6.2).
    """
    result = GuardrailResult()

    if not isinstance(entries, list):
        result.mapping_reliable = False
        result.mapping_problem = "interpretation payload is not an array"
        return result

    seen_indices: list[int] = []
    unattributable = 0

    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            unattributable += 1
            continue

        raw_index = entry.get("note_index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            unattributable += 1
            continue
        if not 0 <= raw_index < note_count:
            unattributable += 1
            continue
        seen_indices.append(raw_index)

        raw_type = entry.get("directive_type")
        explanation = entry.get("explanation")
        explanation = explanation.strip() if isinstance(explanation, str) else ""

        try:
            directive_type = DirectiveType(raw_type)
        except ValueError:
            result.entry_problems[raw_index] = (
                f"directive_type {raw_type!r} is not one of the supported types"
            )
            continue

        applies = entry.get("applies")
        if not isinstance(applies, bool):
            result.entry_problems[raw_index] = "applies must be a boolean"
            continue

        adjustment_raw = entry.get("structured_adjustment", None)

        if directive_type is DirectiveType.NO_OP:
            # PS S5.1: no_op is the only directive allowed with applies = false,
            # and it must carry a null adjustment.
            if applies:
                result.entry_problems[raw_index] = "no_op must use applies = false"
                continue
            if adjustment_raw is not None:
                result.entry_problems[raw_index] = (
                    "no_op must use structured_adjustment = null"
                )
                continue
            result.directives[raw_index] = Directive(
                note_index=raw_index,
                applies=False,
                directive_type=DirectiveType.NO_OP,
                adjustment=None,
                explanation=explanation or "This note does not affect the schedule.",
            )
            continue

        if not applies:
            result.entry_problems[raw_index] = (
                f"{directive_type.value} must use applies = true"
            )
            continue

        adjustment, problem = _validate_adjustment(
            directive_type, adjustment_raw, battery.capacity_kwh
        )
        if problem is not None:
            result.entry_problems[raw_index] = problem
            continue

        result.directives[raw_index] = Directive(
            note_index=raw_index,
            applies=True,
            directive_type=directive_type,
            adjustment=adjustment,
            explanation=explanation or f"Interpreted as {directive_type.value}.",
        )

    duplicates = sorted({i for i in seen_indices if seen_indices.count(i) > 1})
    missing = sorted(set(range(note_count)) - set(seen_indices))

    if duplicates:
        result.mapping_reliable = False
        result.mapping_problem = f"duplicate note_index values: {duplicates}"
    elif unattributable:
        result.mapping_reliable = False
        result.mapping_problem = (
            f"{unattributable} entr(y/ies) could not be attributed to a note"
        )
    elif missing:
        # Missing notes are still reliably identified, so they can be repaired
        # individually rather than regenerating everything.
        for index in missing:
            result.entry_problems[index] = "no interpretation was returned for this note"

    return result


# ---------------------------------------------------------------------------
# Directive Compiler (plan S7)
# ---------------------------------------------------------------------------

#: How overlapping solar_reduction factors combine in a given hour.
SOLAR_PASS_PRODUCT = "product"
SOLAR_PASS_MIN = "min_relaxed"


@dataclass(frozen=True)
class CompiledLimits:
    """The complete set of hourly bounds handed to the optimizer."""

    effective_solar: list[float]
    minimum_reserve: list[float]
    maximum_charge: list[float]
    maximum_discharge: list[float]
    maximum_grid: list[float | None]
    solar_pass: str
    trace: tuple[str, ...]


def solar_factors_by_hour(directives: list[Directive]) -> list[list[float]]:
    """Every applicable solar factor for each hour, in note order."""
    factors: list[list[float]] = [[] for _ in range(24)]
    for directive in directives:
        if directive.directive_type is not DirectiveType.SOLAR_REDUCTION:
            continue
        assert directive.adjustment is not None
        factor = float(directive.adjustment["factor"])
        for hour in directive.hours:
            factors[hour].append(factor)
    return factors


def effective_solar_factor(factors: list[float], solar_pass: str) -> float:
    """Combine the factors applying to one hour (plan S7.1).

    With no directive the factor is 1. Where several apply, the default pass
    multiplies them; the relaxation pass takes the minimum.

    The product is the default because the two candidate rules are not
    symmetric. Using *less* solar than permitted is always legal - unused solar
    is curtailed (PS S9.4) and the shortfall comes from grid or battery - but
    using *more* than the judge's effective solar invalidates the whole case and
    forfeits its optimization credit (Guide S09). The product is the only choice
    that cannot overuse solar under either reading.

    Both passes are order-independent: multiplication and ``min`` are
    commutative, so note order can never change the schedule.
    """
    if not factors:
        return 1.0
    if solar_pass == SOLAR_PASS_MIN:
        return min(factors)
    product = 1.0
    for factor in factors:
        product *= factor
    return product


def compile_limits(
    directives: list[Directive],
    solar: list[float],
    battery: BatteryInput,
    solar_pass: str = SOLAR_PASS_PRODUCT,
) -> CompiledLimits:
    """Turn validated directives into explicit hourly bounds.

    Simultaneously applicable inequalities combine deterministically: the
    greatest reserve, the smallest grid cap, and the union of the charging and
    discharging prohibitions. If both prohibitions cover an hour, the battery
    must idle in it.
    """
    trace: list[str] = []

    # PS S11.5 treats values within 0.01 as equivalent, so an initial energy
    # marginally under the base reserve is equal to it as far as the judge is
    # concerned. Without this, neutrality would pin the final state just below
    # an exact LP bound and report a boundary scenario as infeasible. The
    # relaxation is capped at the published tolerance and never applies to a
    # materially low initial energy, which HTTP 422 already rejects upstream.
    base_reserve = float(battery.minimum_energy_kwh)
    initial = float(battery.initial_energy_kwh)
    if 0 < base_reserve - initial <= TOLERANCE:
        trace.append(
            f"base reserve {base_reserve} relaxed to the initial energy {initial}: "
            f"the gap is within the {TOLERANCE} tolerance"
        )
        base_reserve = initial

    effective_solar = [0.0] * 24
    minimum_reserve = [base_reserve] * 24
    maximum_charge = [float(battery.max_charge_kwh_per_hour)] * 24
    maximum_discharge = [float(battery.max_discharge_kwh_per_hour)] * 24
    maximum_grid: list[float | None] = [None] * 24

    factors = solar_factors_by_hour(directives)
    for hour in range(24):
        factor = effective_solar_factor(factors[hour], solar_pass)
        # Usable solar can never be negative, whatever the forecast says.
        effective_solar[hour] = max(0.0, float(solar[hour]) * factor)
        if factors[hour]:
            trace.append(
                f"hour {hour}: solar {solar[hour]} x {factor} ({solar_pass}, "
                f"factors={factors[hour]}) -> {effective_solar[hour]}"
            )

    for directive in directives:
        if directive.directive_type is DirectiveType.NO_OP:
            continue
        assert directive.adjustment is not None
        note = directive.note_index

        if directive.directive_type is DirectiveType.MINIMUM_BATTERY_RESERVE:
            reserve = float(directive.adjustment["minimum_energy_kwh"])
            for hour in directive.hours:
                if reserve > minimum_reserve[hour]:
                    trace.append(
                        f"hour {hour}: reserve {minimum_reserve[hour]} -> {reserve} "
                        f"(note {note})"
                    )
                    minimum_reserve[hour] = reserve

        elif directive.directive_type is DirectiveType.NO_CHARGE_WINDOW:
            for hour in directive.hours:
                maximum_charge[hour] = 0.0
                trace.append(f"hour {hour}: charging disabled (note {note})")

        elif directive.directive_type is DirectiveType.NO_DISCHARGE_WINDOW:
            for hour in directive.hours:
                maximum_discharge[hour] = 0.0
                trace.append(f"hour {hour}: discharging disabled (note {note})")

        elif directive.directive_type is DirectiveType.MAX_GRID_WINDOW:
            cap = float(directive.adjustment["max_grid_kwh"])
            for hour in directive.hours:
                current = maximum_grid[hour]
                if current is None or cap < current:
                    trace.append(
                        f"hour {hour}: grid cap {current} -> {cap} (note {note})"
                    )
                    maximum_grid[hour] = cap

    return CompiledLimits(
        effective_solar=effective_solar,
        minimum_reserve=minimum_reserve,
        maximum_charge=maximum_charge,
        maximum_discharge=maximum_discharge,
        maximum_grid=maximum_grid,
        solar_pass=solar_pass,
        trace=tuple(trace),
    )


def has_overlapping_solar_reduction(directives: list[Directive]) -> bool:
    """True when at least one hour carries more than one solar factor."""
    return any(len(factors) > 1 for factors in solar_factors_by_hour(directives))
