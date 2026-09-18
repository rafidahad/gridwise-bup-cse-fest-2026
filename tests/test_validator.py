"""Replay Validator tests (plan SS9, 10.2).

Every test corrupts one field of an otherwise valid response and asserts the
validator names the specific rule that was broken. A validator that only ever
agrees with the optimizer would prove nothing, so the corruptions here are the
real evidence that it checks independently.
"""

from __future__ import annotations

import copy

import pytest

from app.directives import SOLAR_PASS_MIN, SOLAR_PASS_PRODUCT, compile_limits
from app.optimizer import solve
from app.schemas import BatteryAction, DirectiveType, OptimizeResponse
from app.validator import validate_response
from tests.conftest import (
    battery,
    max_grid,
    no_charge,
    no_discharge,
    no_op,
    reserve,
    scenario,
    solar_reduction,
)


def build(directives, demand=100.0, solar=60.0, tariff=10.0, batt=None, notes=None):
    """Produce a matching (request, directives, response) triple."""
    batt = batt or battery()
    request = scenario(
        demand=demand,
        solar=solar,
        tariff=tariff,
        batt=batt,
        notes=notes or ["note"] * max(1, len(directives)),
    )
    limits = compile_limits(directives, request.solar(), batt, SOLAR_PASS_PRODUCT)
    solution = solve(request.demand(), request.tariff(), batt, limits)
    response = OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=[
            {
                "note_index": d.note_index,
                "applies": d.applies,
                "directive_type": d.directive_type,
                "structured_adjustment": d.adjustment,
                "explanation": d.explanation,
            }
            for d in directives
        ]
        or [
            {
                "note_index": 0,
                "applies": False,
                "directive_type": DirectiveType.NO_OP,
                "structured_adjustment": None,
                "explanation": "irrelevant",
            }
        ],
        hourly_plan=solution.hourly_plan,
        total_grid_kwh=solution.total_grid_kwh,
        total_cost_bdt=solution.total_cost_bdt,
        peak_grid_kwh=solution.peak_grid_kwh,
        plan_summary="test",
    )
    return request, directives, response


def corrupt(response: OptimizeResponse) -> OptimizeResponse:
    return copy.deepcopy(response)


def check(request, directives, response, solar_pass=SOLAR_PASS_PRODUCT):
    return validate_response(request, directives, solar_pass, response)


# ---------------------------------------------------------------------------
# A correct plan passes
# ---------------------------------------------------------------------------


def test_a_valid_plan_has_no_violations():
    request, directives, response = build([])
    assert check(request, directives, response) == []


def test_valid_plans_with_every_directive_type_pass():
    cases = [
        [solar_reduction(0, [12, 13], 0.25)],
        [reserve(0, [18, 19], 120)],
        [no_charge(0, [2, 3])],
        [no_discharge(0, [18, 19])],
        [max_grid(0, [18, 19], 80)],
        [no_op(0)],
    ]
    for directives in cases:
        request, directives, response = build(directives)
        assert check(request, directives, response) == [], directives


# ---------------------------------------------------------------------------
# Contract-level corruption
# ---------------------------------------------------------------------------


def test_wrong_scenario_echo_is_caught():
    request, directives, response = build([])
    bad = corrupt(response)
    bad.scenario_id = "SOMETHING-ELSE"
    assert any("scenario_id" in p for p in check(request, directives, bad))


def test_missing_plan_hour_is_caught():
    request, directives, response = build([])
    bad = corrupt(response)
    bad.hourly_plan.pop(5)
    assert any("one entry per hour" in p for p in check(request, directives, bad))


def test_duplicate_plan_hour_is_caught():
    request, directives, response = build([])
    bad = corrupt(response)
    bad.hourly_plan[5].hour = 6
    assert any("one entry per hour" in p for p in check(request, directives, bad))


def test_interpretation_out_of_note_index_order_is_caught():
    request, directives, response = build(
        [no_charge(0, [2]), no_discharge(1, [18])], notes=["a", "b"]
    )
    bad = corrupt(response)
    bad.directive_interpretation.reverse()
    assert any("note_index order" in p for p in check(request, directives, bad))


def test_no_op_with_applies_true_is_caught():
    request, directives, response = build([no_op(0)])
    bad = corrupt(response)
    bad.directive_interpretation[0].applies = True
    assert any("no_op requires" in p for p in check(request, directives, bad))


def test_operational_directive_with_applies_false_is_caught():
    request, directives, response = build([no_charge(0, [2])])
    bad = corrupt(response)
    bad.directive_interpretation[0].applies = False
    assert any("requires applies=true" in p for p in check(request, directives, bad))


def test_unordered_hours_in_the_emitted_adjustment_are_caught():
    request, directives, response = build([no_charge(0, [2, 3])])
    bad = corrupt(response)
    bad.directive_interpretation[0].structured_adjustment = {"hours": [3, 2]}
    assert any("ascending order" in p for p in check(request, directives, bad))


# ---------------------------------------------------------------------------
# Energy and battery corruption
# ---------------------------------------------------------------------------


def test_energy_imbalance_is_caught():
    request, directives, response = build([])
    bad = corrupt(response)
    bad.hourly_plan[7].grid_kwh += 25
    problems = check(request, directives, bad)
    assert any("energy balance fails" in p for p in problems)


def test_unmet_demand_is_caught():
    """Buying too little leaves demand unserved, which the balance must catch."""
    request, directives, response = build([])
    bad = corrupt(response)
    hour = next(row.hour for row in bad.hourly_plan if row.grid_kwh >= 30)
    bad.hourly_plan[hour].grid_kwh -= 30
    assert any("energy balance fails" in p for p in check(request, directives, bad))


def test_negative_grid_is_caught():
    request, directives, response = build([])
    bad = corrupt(response)
    bad.hourly_plan[7].grid_kwh = -10
    assert any("negative" in p for p in check(request, directives, bad))


def test_broken_state_transition_is_caught():
    request, directives, response = build([])
    bad = corrupt(response)
    bad.hourly_plan[7].battery_energy_after_kwh += 15
    assert any("does not follow from" in p for p in check(request, directives, bad))


def test_idle_with_non_zero_magnitude_is_caught():
    request, directives, response = build([])
    bad = corrupt(response)
    for row in bad.hourly_plan:
        if row.battery_action is BatteryAction.IDLE:
            row.battery_kwh = 5
            break
    else:  # pragma: no cover - the flat-tariff plan always idles somewhere
        pytest.skip("no idle hour in this plan")
    assert any("idle must report" in p for p in check(request, directives, bad))


def test_capacity_overrun_is_caught():
    batt = battery(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=40)
    request, directives, response = build([], batt=batt)
    bad = corrupt(response)
    bad.hourly_plan[3].battery_action = BatteryAction.CHARGE
    bad.hourly_plan[3].battery_kwh = 40
    bad.hourly_plan[3].battery_energy_after_kwh = 500
    assert any("exceeds capacity" in p for p in check(request, directives, bad))


def test_reserve_breach_is_caught():
    batt = battery(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=40)
    request, directives, response = build([], batt=batt)
    bad = corrupt(response)
    bad.hourly_plan[3].battery_action = BatteryAction.DISCHARGE
    bad.hourly_plan[3].battery_kwh = 5
    bad.hourly_plan[3].battery_energy_after_kwh = 10
    assert any("below the active minimum" in p for p in check(request, directives, bad))


def test_directive_reserve_breach_is_caught():
    batt = battery(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=40)
    directives = [reserve(0, [18, 19], 150)]
    request, directives, response = build(directives, batt=batt)
    bad = corrupt(response)
    bad.hourly_plan[18].battery_energy_after_kwh = 45
    assert any("below the active minimum" in p for p in check(request, directives, bad))


def test_rate_limit_breach_is_caught():
    batt = battery(max_charge_kwh_per_hour=50)
    request, directives, response = build([], batt=batt)
    bad = corrupt(response)
    bad.hourly_plan[3].battery_action = BatteryAction.CHARGE
    bad.hourly_plan[3].battery_kwh = 500
    assert any("rate limit" in p for p in check(request, directives, bad))


def test_broken_neutrality_is_caught():
    request, directives, response = build([])
    bad = corrupt(response)
    bad.hourly_plan[23].battery_action = BatteryAction.DISCHARGE
    bad.hourly_plan[23].battery_kwh = 20
    bad.hourly_plan[23].battery_energy_after_kwh -= 20
    bad.hourly_plan[23].grid_kwh -= 20
    assert any("does not return to the initial" in p for p in check(request, directives, bad))


# ---------------------------------------------------------------------------
# Directive corruption
# ---------------------------------------------------------------------------


def test_effective_solar_overuse_is_caught():
    directives = [solar_reduction(0, [12], 0.25)]
    request, directives, response = build(directives, solar=100.0)
    bad = corrupt(response)
    bad.hourly_plan[12].solar_used_kwh = 90
    bad.hourly_plan[12].grid_kwh = max(
        0.0, bad.hourly_plan[12].grid_kwh - (90 - response.hourly_plan[12].solar_used_kwh)
    )
    assert any("exceeds effective solar" in p for p in check(request, directives, bad))


def test_the_validator_reconstructs_the_product_rule_for_overlaps():
    """100 kWh with factors 0.8 and 0.5 leaves 40 kWh, so 45 must be rejected."""
    directives = [solar_reduction(0, [12], 0.8), solar_reduction(1, [12], 0.5)]
    request, directives, response = build(directives, solar=100.0, notes=["a", "b"])
    bad = corrupt(response)
    bad.hourly_plan[12].solar_used_kwh = 45
    bad.hourly_plan[12].grid_kwh = 55
    problems = check(request, directives, bad)
    assert any("exceeds effective solar" in p for p in problems)

    # Under the relaxed pass the same 45 kWh is legal, so the validator must
    # honour whichever pass produced the plan.
    assert not any(
        "exceeds effective solar" in p
        for p in check(request, directives, bad, SOLAR_PASS_MIN)
    )


def test_charging_in_a_no_charge_window_is_caught():
    directives = [no_charge(0, [2, 3])]
    request, directives, response = build(directives)
    bad = corrupt(response)
    bad.hourly_plan[2].battery_action = BatteryAction.CHARGE
    bad.hourly_plan[2].battery_kwh = 10
    assert any("charging is prohibited" in p for p in check(request, directives, bad))


def test_discharging_in_a_no_discharge_window_is_caught():
    directives = [no_discharge(0, [18, 19])]
    request, directives, response = build(directives)
    bad = corrupt(response)
    bad.hourly_plan[18].battery_action = BatteryAction.DISCHARGE
    bad.hourly_plan[18].battery_kwh = 10
    assert any("discharging is prohibited" in p for p in check(request, directives, bad))


def test_grid_cap_breach_is_caught():
    directives = [max_grid(0, [18, 19], 60)]
    request, directives, response = build(directives)
    bad = corrupt(response)
    bad.hourly_plan[18].grid_kwh = 200
    assert any("exceeds the cap" in p for p in check(request, directives, bad))


# ---------------------------------------------------------------------------
# Reported totals (PS S11.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "needle"),
    [
        ("total_grid_kwh", "total_grid_kwh"),
        ("total_cost_bdt", "total_cost_bdt"),
        ("peak_grid_kwh", "peak_grid_kwh"),
    ],
)
def test_totals_that_disagree_with_the_plan_are_caught(field, needle):
    request, directives, response = build([])
    bad = corrupt(response)
    setattr(bad, field, getattr(bad, field) + 100)
    assert any(needle in p for p in check(request, directives, bad))


def test_totals_within_tolerance_are_accepted():
    request, directives, response = build([])
    bad = corrupt(response)
    bad.total_cost_bdt += 0.005
    assert check(request, directives, bad) == []
