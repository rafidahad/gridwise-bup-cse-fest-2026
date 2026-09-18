"""Independent Replay Validator (PS SS8, 11; plan S9).

This module re-derives every limit from the original request and the validated
directives, rather than importing the compiler's output. The duplication is the
point: if `directives.compile_limits` and this replay agreed only because they
share code, the replay would prove nothing about the response a judge receives.

What it inspects is the **actual response object about to be returned** - the
rows, the echoed identifier, the enum values, the reported totals - not the
solver's internal state.

What it cannot do: replay proves the returned schedule obeys the directives this
service interpreted. Only the semantic tests can show those directives match the
notes, and only the organizer's harness compares them to ground truth.
"""

from __future__ import annotations

from app.config import TOLERANCE
from app.directives import SOLAR_PASS_MIN, Directive
from app.schemas import (
    BatteryAction,
    DirectiveType,
    OptimizeResponse,
    ScenarioRequest,
)


def _reconstruct_effective_solar(
    request: ScenarioRequest, directives: list[Directive], solar_pass: str
) -> list[float]:
    """Rebuild effective solar straight from the request and the directives."""
    original = request.solar()
    result: list[float] = []
    for hour in range(24):
        applicable = [
            float(d.adjustment["factor"])
            for d in directives
            if d.directive_type is DirectiveType.SOLAR_REDUCTION
            and d.adjustment is not None
            and hour in d.adjustment["hours"]
        ]
        if not applicable:
            factor = 1.0
        elif solar_pass == SOLAR_PASS_MIN:
            factor = min(applicable)
        else:
            factor = 1.0
            for value in applicable:
                factor *= value
        result.append(max(0.0, original[hour] * factor))
    return result


def _reconstruct_reserve(request: ScenarioRequest, directives: list[Directive]) -> list[float]:
    base = float(request.battery.minimum_energy_kwh)
    result = []
    for hour in range(24):
        applicable = [
            float(d.adjustment["minimum_energy_kwh"])
            for d in directives
            if d.directive_type is DirectiveType.MINIMUM_BATTERY_RESERVE
            and d.adjustment is not None
            and hour in d.adjustment["hours"]
        ]
        result.append(max([base, *applicable]))
    return result


def _hours_for(directives: list[Directive], directive_type: DirectiveType) -> set[int]:
    hours: set[int] = set()
    for directive in directives:
        if directive.directive_type is directive_type and directive.adjustment:
            hours.update(directive.adjustment["hours"])
    return hours


def _reconstruct_grid_caps(
    directives: list[Directive],
) -> list[float | None]:
    caps: list[float | None] = [None] * 24
    for directive in directives:
        if directive.directive_type is not DirectiveType.MAX_GRID_WINDOW:
            continue
        if directive.adjustment is None:
            continue
        cap = float(directive.adjustment["max_grid_kwh"])
        for hour in directive.adjustment["hours"]:
            current = caps[hour]
            caps[hour] = cap if current is None else min(current, cap)
    return caps


def validate_response(
    request: ScenarioRequest,
    directives: list[Directive],
    solar_pass: str,
    response: OptimizeResponse,
) -> list[str]:
    """Return every violation found in `response`. Empty means it may be sent."""
    problems: list[str] = []

    # 1. Scenario echo, directive coverage and order (PS SS10.1, 5.1).
    if response.scenario_id != request.scenario_id:
        problems.append("scenario_id does not echo the request")

    note_count = len(request.operator_notes)
    if len(response.directive_interpretation) != note_count:
        problems.append(
            f"expected {note_count} interpretation entries, "
            f"got {len(response.directive_interpretation)}"
        )
    for position, entry in enumerate(response.directive_interpretation):
        if entry.note_index != position:
            problems.append(
                f"interpretation[{position}] has note_index {entry.note_index}; "
                "entries must be in note_index order"
            )
        is_no_op = entry.directive_type is DirectiveType.NO_OP
        if is_no_op and (entry.applies or entry.structured_adjustment is not None):
            problems.append(
                f"interpretation[{position}]: no_op requires applies=false and a "
                "null structured_adjustment"
            )
        if not is_no_op and not entry.applies:
            problems.append(
                f"interpretation[{position}]: {entry.directive_type.value} requires "
                "applies=true"
            )
        if not is_no_op and entry.structured_adjustment is None:
            problems.append(
                f"interpretation[{position}]: {entry.directive_type.value} requires a "
                "structured_adjustment"
            )
        adjustment = entry.structured_adjustment
        if adjustment is not None:
            hours = adjustment.get("hours")
            if not isinstance(hours, list) or not hours:
                problems.append(f"interpretation[{position}]: hours must be non-empty")
            elif hours != sorted(set(hours)) or any(
                not isinstance(h, int) or not 0 <= h <= 23 for h in hours
            ):
                problems.append(
                    f"interpretation[{position}]: hours must be unique integers "
                    "0-23 in ascending order"
                )

    # 2. Exactly 24 unique plan hours covering 0-23 (PS S11.3).
    plan_hours = [row.hour for row in response.hourly_plan]
    if sorted(plan_hours) != list(range(24)):
        problems.append("hourly_plan must contain exactly one entry per hour 0-23")
        # Without a well-formed plan the remaining checks cannot be trusted.
        return problems
    rows = sorted(response.hourly_plan, key=lambda row: row.hour)
    if plan_hours != list(range(24)):
        problems.append("hourly_plan must be returned in ascending hour order")

    effective_solar = _reconstruct_effective_solar(request, directives, solar_pass)
    reserve = _reconstruct_reserve(request, directives)
    no_charge = _hours_for(directives, DirectiveType.NO_CHARGE_WINDOW)
    no_discharge = _hours_for(directives, DirectiveType.NO_DISCHARGE_WINDOW)
    grid_caps = _reconstruct_grid_caps(directives)

    demand = request.demand()
    tariff = request.tariff()
    battery = request.battery
    initial = float(battery.initial_energy_kwh)

    state = initial
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for row in rows:
        hour = row.hour

        # 3. Finite, non-negative public energy values (PS S11.3).
        for name, value in (
            ("grid_kwh", row.grid_kwh),
            ("solar_used_kwh", row.solar_used_kwh),
            ("battery_kwh", row.battery_kwh),
        ):
            if value != value or value in (float("inf"), float("-inf")):
                problems.append(f"hour {hour}: {name} is not finite")
            elif value < -TOLERANCE:
                problems.append(f"hour {hour}: {name} is negative ({value})")

        # 4. Action/magnitude consistency (PS S10.3).
        if row.battery_action is BatteryAction.IDLE and abs(row.battery_kwh) > 0:
            problems.append(
                f"hour {hour}: idle must report battery_kwh = 0, got {row.battery_kwh}"
            )

        # 5. State transition from the reconstructed previous state (PS S9.1).
        if row.battery_action is BatteryAction.CHARGE:
            expected_state = state + row.battery_kwh
        elif row.battery_action is BatteryAction.DISCHARGE:
            expected_state = state - row.battery_kwh
        else:
            expected_state = state
        if abs(row.battery_energy_after_kwh - expected_state) > TOLERANCE:
            problems.append(
                f"hour {hour}: battery_energy_after_kwh {row.battery_energy_after_kwh} "
                f"does not follow from {state} and the reported action"
            )
        state = row.battery_energy_after_kwh

        # 6. Capacity, active reserve, and rate limits (PS SS9.2, 9.3).
        if state > battery.capacity_kwh + TOLERANCE:
            problems.append(f"hour {hour}: battery exceeds capacity ({state})")
        if state < reserve[hour] - TOLERANCE:
            problems.append(
                f"hour {hour}: battery {state} is below the active minimum "
                f"{reserve[hour]}"
            )
        if (
            row.battery_action is BatteryAction.CHARGE
            and row.battery_kwh > battery.max_charge_kwh_per_hour + TOLERANCE
        ):
            problems.append(f"hour {hour}: charge exceeds the hourly rate limit")
        if (
            row.battery_action is BatteryAction.DISCHARGE
            and row.battery_kwh > battery.max_discharge_kwh_per_hour + TOLERANCE
        ):
            problems.append(f"hour {hour}: discharge exceeds the hourly rate limit")

        # 7. Effective solar reconstructed from the request and directives (PS S9.4).
        if row.solar_used_kwh > effective_solar[hour] + TOLERANCE:
            problems.append(
                f"hour {hour}: solar_used_kwh {row.solar_used_kwh} exceeds effective "
                f"solar {effective_solar[hour]}"
            )

        # 8. Energy balance (PS S9.5).
        charge = row.battery_kwh if row.battery_action is BatteryAction.CHARGE else 0.0
        discharge = (
            row.battery_kwh if row.battery_action is BatteryAction.DISCHARGE else 0.0
        )
        supply = row.grid_kwh + row.solar_used_kwh + discharge
        draw = demand[hour] + charge
        if abs(supply - draw) > TOLERANCE:
            problems.append(
                f"hour {hour}: energy balance fails ({supply} supplied vs {draw} drawn)"
            )

        # 9. Directive windows and caps (PS S5.3).
        if hour in no_charge and row.battery_action is BatteryAction.CHARGE and row.battery_kwh > TOLERANCE:
            problems.append(f"hour {hour}: charging is prohibited by a directive")
        if (
            hour in no_discharge
            and row.battery_action is BatteryAction.DISCHARGE
            and row.battery_kwh > TOLERANCE
        ):
            problems.append(f"hour {hour}: discharging is prohibited by a directive")
        cap = grid_caps[hour]
        if cap is not None and row.grid_kwh > cap + TOLERANCE:
            problems.append(
                f"hour {hour}: grid import {row.grid_kwh} exceeds the cap {cap}"
            )

        total_grid += row.grid_kwh
        total_cost += row.grid_kwh * tariff[hour]
        peak_grid = max(peak_grid, row.grid_kwh)

    # 10. End-of-day neutrality (PS S9.6).
    if abs(state - initial) > TOLERANCE:
        problems.append(
            f"final battery energy {state} does not return to the initial {initial}"
        )

    # 11. Totals recalculated from the rows actually returned (PS S11.3).
    if abs(response.total_grid_kwh - total_grid) > TOLERANCE:
        problems.append(
            f"total_grid_kwh {response.total_grid_kwh} does not match the plan "
            f"({total_grid})"
        )
    if abs(response.total_cost_bdt - total_cost) > TOLERANCE:
        problems.append(
            f"total_cost_bdt {response.total_cost_bdt} does not match the plan "
            f"({total_cost})"
        )
    if abs(response.peak_grid_kwh - peak_grid) > TOLERANCE:
        problems.append(
            f"peak_grid_kwh {response.peak_grid_kwh} does not match the plan "
            f"({peak_grid})"
        )

    return problems


class ReplayFailure(Exception):
    """The response failed its own independent replay and must not be sent.

    Reaching this means the service produced a schedule it cannot itself
    verify. Returning it anyway would risk an invalid case (Guide S09), so the
    caller converts this into a controlled error instead.
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems
