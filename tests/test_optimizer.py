"""Optimizer tests: physical rules, directive enforcement, solar policy (plan S8)."""

from __future__ import annotations

import pytest

from app.directives import (
    SOLAR_PASS_MIN,
    SOLAR_PASS_PRODUCT,
    Directive,
    compile_limits,
)
from app.optimizer import (
    InfeasibleScheduleError,
    solve,
    solve_with_solar_policy,
)
from app.schemas import BatteryAction, DirectiveType, ScenarioRequest
from tests.conftest import (
    battery,
    max_grid,
    no_charge,
    no_discharge,
    reserve,
    solar_reduction,
)

TOL = 0.01


def run(directives, demand, solar, tariff, batt):
    limits = compile_limits(directives, solar, batt, SOLAR_PASS_PRODUCT)
    return solve(demand, tariff, batt, limits), limits


def flat(value: float) -> list[float]:
    return [float(value)] * 24


# ---------------------------------------------------------------------------
# Physical rules (PS S9)
# ---------------------------------------------------------------------------


def test_energy_balance_holds_every_hour():
    solution, _ = run([], flat(100), flat(0), flat(10), battery())
    for row in solution.hourly_plan:
        charge = row.battery_kwh if row.battery_action is BatteryAction.CHARGE else 0.0
        discharge = (
            row.battery_kwh if row.battery_action is BatteryAction.DISCHARGE else 0.0
        )
        assert row.grid_kwh + row.solar_used_kwh + discharge == pytest.approx(
            100 + charge, abs=TOL
        )


def test_end_of_day_energy_returns_to_initial():
    batt = battery(initial_energy_kwh=100)
    tariff = [5.0] * 12 + [20.0] * 12  # cheap then expensive: arbitrage is tempting
    solution, _ = run([], flat(100), flat(0), tariff, batt)
    assert solution.hourly_plan[23].battery_energy_after_kwh == pytest.approx(100, abs=TOL)


def test_battery_stays_within_capacity_and_base_reserve():
    batt = battery(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=40)
    tariff = [5.0] * 12 + [20.0] * 12
    solution, _ = run([], flat(100), flat(0), tariff, batt)
    for row in solution.hourly_plan:
        assert 40 - TOL <= row.battery_energy_after_kwh <= 200 + TOL


def test_rate_limits_are_respected():
    batt = battery(max_charge_kwh_per_hour=30, max_discharge_kwh_per_hour=20)
    tariff = [5.0] * 12 + [20.0] * 12
    solution, _ = run([], flat(100), flat(0), tariff, batt)
    for row in solution.hourly_plan:
        if row.battery_action is BatteryAction.CHARGE:
            assert row.battery_kwh <= 30 + TOL
        if row.battery_action is BatteryAction.DISCHARGE:
            assert row.battery_kwh <= 20 + TOL


def test_idle_hours_report_exactly_zero_magnitude():
    solution, _ = run([], flat(100), flat(0), flat(10), battery())
    for row in solution.hourly_plan:
        if row.battery_action is BatteryAction.IDLE:
            assert row.battery_kwh == 0


def test_free_solar_displaces_grid_cost():
    """Asserts the cost, not one particular schedule.

    At a flat tariff the battery may churn for free, so several equivalent
    optima exist. PS S11.4 accepts any of them; only the total is determined:
    24 hours x (100 demand - 60 solar) x 10 BDT.
    """
    solution, _ = run([], flat(100), flat(60), flat(10), battery())
    assert solution.total_cost_bdt == pytest.approx(24 * 40 * 10, abs=TOL)
    assert solution.total_grid_kwh == pytest.approx(24 * 40, abs=TOL)


def test_surplus_solar_is_curtailed_not_exported():
    """PS S9.4: grid export is out of scope, so grid never goes negative."""
    solution, _ = run([], flat(50), flat(400), flat(10), battery())
    for row in solution.hourly_plan:
        assert row.grid_kwh >= 0
        assert row.solar_used_kwh <= 400 + TOL


def test_arbitrage_shifts_load_into_the_cheap_window():
    tariff = [2.0] * 6 + [30.0] * 18
    batt = battery(capacity_kwh=200, initial_energy_kwh=50, minimum_energy_kwh=0)
    solution, _ = run([], flat(100), flat(0), tariff, batt)
    cheap_grid = sum(row.grid_kwh for row in solution.hourly_plan[:6])
    assert cheap_grid > 600  # more than the 6 cheap hours of raw demand


# ---------------------------------------------------------------------------
# Directive enforcement
# ---------------------------------------------------------------------------


def test_no_charge_window_is_enforced():
    tariff = [2.0] * 6 + [30.0] * 18
    solution, _ = run([no_charge(0, [2, 3, 4])], flat(100), flat(0), tariff, battery())
    for hour in (2, 3, 4):
        assert solution.hourly_plan[hour].battery_action is not BatteryAction.CHARGE


def test_no_discharge_window_is_enforced():
    tariff = [2.0] * 6 + [30.0] * 18
    solution, _ = run([no_discharge(0, [18, 19])], flat(100), flat(0), tariff, battery())
    for hour in (18, 19):
        assert solution.hourly_plan[hour].battery_action is not BatteryAction.DISCHARGE


def test_minimum_reserve_directive_is_enforced():
    tariff = [2.0] * 6 + [30.0] * 18
    batt = battery(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=40)
    solution, _ = run([reserve(0, [18, 19, 20], 150)], flat(100), flat(0), tariff, batt)
    for hour in (18, 19, 20):
        assert solution.hourly_plan[hour].battery_energy_after_kwh >= 150 - TOL


def test_max_grid_window_is_enforced():
    solution, _ = run([max_grid(0, [18, 19], 60)], flat(100), flat(0), flat(10), battery())
    for hour in (18, 19):
        assert solution.hourly_plan[hour].grid_kwh <= 60 + TOL


def test_solar_reduction_caps_usable_solar():
    solution, limits = run(
        [solar_reduction(0, [12, 13], 0.25)], flat(200), flat(100), flat(10), battery()
    )
    assert limits.effective_solar[12] == pytest.approx(25)
    assert solution.hourly_plan[12].solar_used_kwh <= 25 + TOL
    assert solution.hourly_plan[14].solar_used_kwh == pytest.approx(100, abs=TOL)


def test_combined_directives_all_hold_at_once():
    tariff = [2.0] * 6 + [30.0] * 18
    batt = battery(capacity_kwh=250, initial_energy_kwh=120, minimum_energy_kwh=40)
    directives = [
        solar_reduction(0, [11, 12], 0.5),
        no_charge(1, [13, 14]),
        max_grid(2, [19, 20], 130),
    ]
    solution, limits = run(directives, flat(120), flat(80), tariff, batt)
    assert solution.hourly_plan[11].solar_used_kwh <= 40 + TOL
    for hour in (13, 14):
        assert solution.hourly_plan[hour].battery_action is not BatteryAction.CHARGE
    for hour in (19, 20):
        assert solution.hourly_plan[hour].grid_kwh <= 130 + TOL
    assert solution.hourly_plan[23].battery_energy_after_kwh == pytest.approx(120, abs=TOL)


# ---------------------------------------------------------------------------
# Infeasibility and the solar-overlap relaxation (plan S7.1)
# ---------------------------------------------------------------------------


def test_impossible_grid_cap_is_reported_as_infeasible():
    with pytest.raises(InfeasibleScheduleError):
        run([max_grid(0, list(range(24)), 1)], flat(500), flat(0), flat(10), battery())


def test_overlapping_reductions_can_still_be_feasible():
    """Integration case: overlap applies, and the schedule is still valid."""
    demand = flat(100)
    solar = [0.0] * 10 + [150.0] * 4 + [0.0] * 10
    directives = [solar_reduction(0, [11, 12], 0.8), solar_reduction(1, [12], 0.5)]
    solution, limits = run(directives, demand, solar, flat(10), battery())
    assert limits.solar_pass == SOLAR_PASS_PRODUCT
    assert limits.effective_solar[12] == pytest.approx(60.0)  # 150 * 0.8 * 0.5
    assert solution.hourly_plan[12].solar_used_kwh <= 60 + TOL


def test_product_infeasible_under_a_grid_cap_falls_back_to_min():
    """The relaxation plan S7.1 exists for: over-curtailment can be fatal too.

    At hour 12 demand is 100, the grid is capped at 50, and discharging is
    banned. Under the product rule usable solar is 100*0.8*0.5 = 40, so at most
    90 kWh can be supplied and the model is infeasible. Under min(factors) it is
    100*0.5 = 50, and 50 + 50 exactly meets demand.
    """
    demand = [0.0] * 24
    demand[12] = 100.0
    solar = [0.0] * 24
    solar[12] = 100.0
    directives = [
        solar_reduction(0, [12], 0.8),
        solar_reduction(1, [12], 0.5),
        max_grid(2, [12], 50),
        no_discharge(3, [12]),
    ]
    batt = battery(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=0)

    # The strict pass on its own is genuinely infeasible.
    strict = compile_limits(directives, solar, batt, SOLAR_PASS_PRODUCT)
    assert strict.effective_solar[12] == pytest.approx(40.0)
    with pytest.raises(InfeasibleScheduleError):
        solve(demand, flat(10), batt, strict)

    # The policy recovers it, and records which pass was used.
    solution, limits = solve_with_solar_policy(
        directives, demand, solar, flat(10), batt
    )
    assert limits.solar_pass == SOLAR_PASS_MIN
    assert limits.effective_solar[12] == pytest.approx(50.0)
    assert solution.hourly_plan[12].solar_used_kwh == pytest.approx(50.0, abs=TOL)
    assert solution.hourly_plan[12].grid_kwh == pytest.approx(50.0, abs=TOL)


def test_relaxation_is_not_attempted_without_an_overlap():
    """A genuine infeasibility must surface, not be masked by a pointless retry."""
    with pytest.raises(InfeasibleScheduleError):
        solve_with_solar_policy(
            [max_grid(0, list(range(24)), 1)],
            flat(500),
            flat(0),
            flat(10),
            battery(),
        )


def test_policy_uses_the_strict_pass_when_it_is_feasible():
    solution, limits = solve_with_solar_policy(
        [solar_reduction(0, [12], 0.8), solar_reduction(1, [12], 0.5)],
        flat(100),
        flat(100),
        flat(10),
        battery(),
    )
    assert limits.solar_pass == SOLAR_PASS_PRODUCT
    assert solution.total_cost_bdt > 0


# ---------------------------------------------------------------------------
# Public sample costs (plan S10.1)
# ---------------------------------------------------------------------------


def test_public_cases_reach_the_reference_optimal_cost(public_cases):
    """Reference directives supplied directly must produce reference costs."""
    for case in public_cases:
        request = ScenarioRequest.model_validate(case["input"])
        directives = [
            Directive(
                note_index=entry["note_index"],
                applies=entry["applies"],
                directive_type=DirectiveType(entry["directive_type"]),
                adjustment=entry["structured_adjustment"],
                explanation=entry["explanation"],
            )
            for entry in case["expected_output"]["directive_interpretation"]
        ]
        solution, _ = solve_with_solar_policy(
            directives,
            request.demand(),
            request.solar(),
            request.tariff(),
            request.battery,
        )
        expected = case["expected_output"]["total_cost_bdt"]
        assert solution.total_cost_bdt == pytest.approx(expected, abs=TOL), case["id"]
        assert solution.solver_status == "Optimal"
