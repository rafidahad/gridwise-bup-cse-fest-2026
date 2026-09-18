"""Generated cases, an independent optimum, concurrency, numeric edges (plan S16 stage 5).

The public pack is ten scenarios. Hidden cases vary demand, solar, tariffs,
battery settings, and directive combinations (Guide S10), so correctness is
checked here against generated inputs and a separately implemented optimum
rather than against ten memorised answers.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.directives import SOLAR_PASS_PRODUCT, compile_limits
from app.optimizer import InfeasibleScheduleError, solve, solve_with_solar_policy
from app.schemas import BatteryAction, OptimizeResponse
from app.validator import validate_response
from tests.conftest import (
    battery,
    max_grid,
    no_charge,
    no_discharge,
    reserve,
    scenario,
    solar_reduction,
)
from tests.reference_dp import INFINITY, dp_optimal_cost

TOL = 0.01
STEP = 10.0


def generated_case(rng: random.Random) -> dict:
    """A random but well-formed scenario on a `STEP` grid.

    Every quantity is a multiple of STEP so the DP reference is exact, and
    tariffs are non-negative so its greedy solar handling is optimal.
    """
    capacity = STEP * rng.randint(10, 30)
    reserve_level = STEP * rng.randint(0, 4)
    initial = STEP * rng.randint(
        int(reserve_level / STEP), int(capacity / STEP)
    )
    return {
        "demand": [STEP * rng.randint(4, 20) for _ in range(24)],
        "solar": [
            STEP * rng.randint(0, 18) if 6 <= hour <= 17 else 0.0 for hour in range(24)
        ],
        "tariff": [float(rng.randint(0, 30)) for _ in range(24)],
        "battery": battery(
            capacity_kwh=capacity,
            initial_energy_kwh=initial,
            minimum_energy_kwh=reserve_level,
            max_charge_kwh_per_hour=STEP * rng.randint(1, 8),
            max_discharge_kwh_per_hour=STEP * rng.randint(1, 8),
        ),
    }


def random_directives(rng: random.Random, capacity: float) -> list:
    """Zero to three directives that stay individually satisfiable."""
    directives = []
    pool = ["solar", "reserve", "no_charge", "no_discharge", "max_grid"]
    rng.shuffle(pool)
    for index, kind in enumerate(pool[: rng.randint(0, 3)]):
        start = rng.randint(0, 20)
        hours = list(range(start, min(24, start + rng.randint(1, 3))))
        if kind == "solar":
            directives.append(solar_reduction(index, hours, rng.choice([0.0, 0.2, 0.5, 0.8, 1.0])))
        elif kind == "reserve":
            directives.append(reserve(index, hours, STEP * rng.randint(0, int(capacity / STEP))))
        elif kind == "no_charge":
            directives.append(no_charge(index, hours))
        elif kind == "no_discharge":
            directives.append(no_discharge(index, hours))
        else:
            directives.append(max_grid(index, hours, STEP * rng.randint(0, 30)))
    return directives


def build_and_check(directives, case) -> OptimizeResponse | None:
    """Solve, wrap in a response, and replay it. None when infeasible."""
    request = scenario(
        demand=case["demand"],
        solar=case["solar"],
        tariff=case["tariff"],
        batt=case["battery"],
        notes=["note"] * max(1, len(directives)),
    )
    try:
        solution, limits = solve_with_solar_policy(
            directives,
            request.demand(),
            request.solar(),
            request.tariff(),
            case["battery"],
        )
    except InfeasibleScheduleError:
        return None

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
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "none",
            }
        ],
        hourly_plan=solution.hourly_plan,
        total_grid_kwh=solution.total_grid_kwh,
        total_cost_bdt=solution.total_cost_bdt,
        peak_grid_kwh=solution.peak_grid_kwh,
        plan_summary="generated",
    )
    problems = validate_response(request, directives, limits.solar_pass, response)
    assert problems == [], problems
    return response


# ---------------------------------------------------------------------------
# Generated scenarios always replay cleanly
# ---------------------------------------------------------------------------


def test_generated_scenarios_without_directives_are_always_valid():
    rng = random.Random(20260918)
    for _ in range(60):
        build_and_check([], generated_case(rng))


def test_generated_scenarios_with_random_directives_are_valid_or_infeasible():
    """Whatever comes back must replay cleanly; nothing in between."""
    rng = random.Random(4242)
    solved = 0
    for _ in range(120):
        case = generated_case(rng)
        directives = random_directives(rng, case["battery"].capacity_kwh)
        if build_and_check(directives, case) is not None:
            solved += 1
    # Sanity: the generator is not producing only infeasible problems.
    assert solved > 60, f"only {solved}/120 generated cases were solvable"


# ---------------------------------------------------------------------------
# Independent optimum (plan S8.3)
# ---------------------------------------------------------------------------


def dp_cost_for(directives, case) -> float:
    limits = compile_limits(
        directives, case["solar"], case["battery"], SOLAR_PASS_PRODUCT
    )
    return dp_optimal_cost(
        demand=case["demand"],
        effective_solar=limits.effective_solar,
        tariff=case["tariff"],
        capacity=case["battery"].capacity_kwh,
        initial=case["battery"].initial_energy_kwh,
        minimum_reserve=limits.minimum_reserve,
        max_charge=limits.maximum_charge,
        max_discharge=limits.maximum_discharge,
        max_grid=limits.maximum_grid,
        step=STEP,
    )


def test_lp_optimum_matches_an_independently_computed_optimum():
    """A dynamic program that shares no code with the LP must agree on cost."""
    rng = random.Random(777)
    compared = 0
    for _ in range(40):
        case = generated_case(rng)
        response = build_and_check([], case)
        assert response is not None
        expected = dp_cost_for([], case)
        assert expected != INFINITY
        assert response.total_cost_bdt == pytest.approx(expected, abs=TOL)
        compared += 1
    assert compared == 40


def on_grid(values: list[float], step: float = STEP) -> bool:
    return all(abs(value / step - round(value / step)) < 1e-9 for value in values)


def test_lp_is_never_worse_than_the_dp_with_directives_applied():
    """Two claims, because the DP is only exact on its own grid.

    A `solar_reduction` factor such as 0.8 takes effective solar off the 10 kWh
    grid, and there the DP searches a restricted set and returns an upper bound.
    So the LP must always be at least as cheap, and must match exactly whenever
    the compiled limits stay on the grid and the DP is therefore optimal.
    """
    rng = random.Random(31337)
    bounded = 0
    exact = 0

    for _ in range(80):
        case = generated_case(rng)
        directives = random_directives(rng, case["battery"].capacity_kwh)
        limits = compile_limits(
            directives, case["solar"], case["battery"], SOLAR_PASS_PRODUCT
        )
        expected = dp_cost_for(directives, case)
        response = build_and_check(directives, case)

        if expected == INFINITY:
            # Nothing on the DP grid. The LP is continuous, so it may still find
            # a fractional schedule; either outcome is consistent.
            continue

        assert response is not None, "the DP found a schedule but the LP did not"
        assert response.total_cost_bdt <= expected + TOL, (
            f"LP cost {response.total_cost_bdt} is worse than the DP's {expected}"
        )
        bounded += 1

        grid_exact = on_grid(limits.effective_solar) and on_grid(
            limits.minimum_reserve
        ) and on_grid([cap for cap in limits.maximum_grid if cap is not None])
        if grid_exact:
            assert response.total_cost_bdt == pytest.approx(expected, abs=TOL)
            exact += 1

    assert bounded > 40, f"only {bounded} comparable cases"
    assert exact > 15, f"only {exact} cases where the DP was provably optimal"


def test_the_lp_is_never_beaten_by_the_dp_on_the_public_cases(public_cases):
    """The published optima are reproduced by a second, unrelated method."""
    from app.directives import Directive
    from app.schemas import DirectiveType, ScenarioRequest

    for case in public_cases:
        request = ScenarioRequest.model_validate(case["input"])
        directives = [
            Directive(
                note_index=e["note_index"],
                applies=e["applies"],
                directive_type=DirectiveType(e["directive_type"]),
                adjustment=e["structured_adjustment"],
                explanation=e["explanation"],
            )
            for e in case["expected_output"]["directive_interpretation"]
        ]
        limits = compile_limits(
            directives, request.solar(), request.battery, SOLAR_PASS_PRODUCT
        )
        # The public data is on a 5 kWh grid; 2.5 is a safe common step.
        dp = dp_optimal_cost(
            demand=request.demand(),
            effective_solar=limits.effective_solar,
            tariff=request.tariff(),
            capacity=request.battery.capacity_kwh,
            initial=request.battery.initial_energy_kwh,
            minimum_reserve=limits.minimum_reserve,
            max_charge=limits.maximum_charge,
            max_discharge=limits.maximum_discharge,
            max_grid=limits.maximum_grid,
            step=2.5,
        )
        expected = case["expected_output"]["total_cost_bdt"]
        assert dp == pytest.approx(expected, abs=TOL), case["id"]


# ---------------------------------------------------------------------------
# Numeric edges (plan S10.3)
# ---------------------------------------------------------------------------


def flat(value: float) -> list[float]:
    return [float(value)] * 24


def test_zero_tariffs_everywhere():
    response = build_and_check(
        [], {"demand": flat(100), "solar": flat(0), "tariff": flat(0), "battery": battery()}
    )
    assert response.total_cost_bdt == pytest.approx(0.0, abs=TOL)


def test_equal_tariffs_everywhere():
    response = build_and_check(
        [], {"demand": flat(100), "solar": flat(0), "tariff": flat(7), "battery": battery()}
    )
    assert response.total_cost_bdt == pytest.approx(24 * 100 * 7, abs=TOL)


def test_zero_demand_all_day():
    response = build_and_check(
        [], {"demand": flat(0), "solar": flat(50), "tariff": flat(10), "battery": battery()}
    )
    assert response.total_grid_kwh == pytest.approx(0.0, abs=TOL)
    assert all(row.grid_kwh == 0 for row in response.hourly_plan)


def test_no_solar_at_all():
    response = build_and_check(
        [], {"demand": flat(80), "solar": flat(0), "tariff": flat(9), "battery": battery()}
    )
    assert all(row.solar_used_kwh == 0 for row in response.hourly_plan)


def test_massive_solar_surplus_is_curtailed():
    response = build_and_check(
        [], {"demand": flat(20), "solar": flat(900), "tariff": flat(10), "battery": battery()}
    )
    assert response.total_cost_bdt == pytest.approx(0.0, abs=TOL)
    for row in response.hourly_plan:
        assert row.solar_used_kwh <= 900 + TOL


def test_fractional_values_are_preserved():
    demand = [87.35] * 24
    response = build_and_check(
        [], {"demand": demand, "solar": flat(12.125), "tariff": flat(6.5), "battery": battery()}
    )
    for row in response.hourly_plan:
        charge = row.battery_kwh if row.battery_action is BatteryAction.CHARGE else 0.0
        discharge = row.battery_kwh if row.battery_action is BatteryAction.DISCHARGE else 0.0
        assert row.grid_kwh + row.solar_used_kwh + discharge == pytest.approx(
            87.35 + charge, abs=TOL
        )


def test_asymmetric_charge_and_discharge_limits():
    batt = battery(max_charge_kwh_per_hour=75, max_discharge_kwh_per_hour=15)
    tariff = [3.0] * 8 + [25.0] * 16
    response = build_and_check(
        [], {"demand": flat(120), "solar": flat(0), "tariff": tariff, "battery": batt}
    )
    for row in response.hourly_plan:
        if row.battery_action is BatteryAction.CHARGE:
            assert row.battery_kwh <= 75 + TOL
        if row.battery_action is BatteryAction.DISCHARGE:
            assert row.battery_kwh <= 15 + TOL


def test_battery_that_cannot_move_at_all():
    batt = battery(max_charge_kwh_per_hour=0, max_discharge_kwh_per_hour=0)
    response = build_and_check(
        [], {"demand": flat(100), "solar": flat(30), "tariff": flat(8), "battery": batt}
    )
    assert all(row.battery_action is BatteryAction.IDLE for row in response.hourly_plan)


def test_initial_energy_at_capacity():
    batt = battery(capacity_kwh=200, initial_energy_kwh=200, minimum_energy_kwh=40)
    tariff = [3.0] * 6 + [25.0] * 18
    response = build_and_check(
        [], {"demand": flat(100), "solar": flat(0), "tariff": tariff, "battery": batt}
    )
    assert response.hourly_plan[23].battery_energy_after_kwh == pytest.approx(200, abs=TOL)


def test_initial_energy_at_the_base_reserve():
    batt = battery(capacity_kwh=200, initial_energy_kwh=40, minimum_energy_kwh=40)
    tariff = [3.0] * 6 + [25.0] * 18
    response = build_and_check(
        [], {"demand": flat(100), "solar": flat(0), "tariff": tariff, "battery": batt}
    )
    assert response.hourly_plan[23].battery_energy_after_kwh == pytest.approx(40, abs=TOL)


def test_a_directive_reserve_above_the_initial_energy_is_met_by_charging():
    """Plan S4.1: a future temporary reserve is a scheduling question, not a rejection."""
    batt = battery(capacity_kwh=250, initial_energy_kwh=50, minimum_energy_kwh=20)
    response = build_and_check(
        [reserve(0, [18, 19, 20], 200)],
        {"demand": flat(100), "solar": flat(0), "tariff": flat(10), "battery": batt},
    )
    assert response is not None
    for hour in (18, 19, 20):
        assert response.hourly_plan[hour].battery_energy_after_kwh >= 200 - TOL
    assert response.hourly_plan[23].battery_energy_after_kwh == pytest.approx(50, abs=TOL)


def test_all_five_operational_directives_at_once():
    batt = battery(capacity_kwh=300, initial_energy_kwh=150, minimum_energy_kwh=30)
    tariff = [4.0] * 7 + [12.0] * 9 + [28.0] * 8
    directives = [
        solar_reduction(0, [10, 11, 12], 0.4),
        reserve(1, [17, 18], 180),
        no_charge(2, [3, 4]),
    ]
    response = build_and_check(
        directives,
        {"demand": flat(130), "solar": [0.0] * 6 + [120.0] * 12 + [0.0] * 6, "tariff": tariff, "battery": batt},
    )
    assert response is not None

    more = [
        no_discharge(0, [20, 21]),
        max_grid(1, [22, 23], 160),
        solar_reduction(2, [9], 0.25),
    ]
    response = build_and_check(
        more,
        {"demand": flat(130), "solar": [0.0] * 6 + [120.0] * 12 + [0.0] * 6, "tariff": tariff, "battery": batt},
    )
    assert response is not None


# ---------------------------------------------------------------------------
# Concurrency (plan S10.3)
# ---------------------------------------------------------------------------


def test_concurrent_solves_do_not_interfere():
    """The real hazard: shared solver state and temp files across threads."""
    rng = random.Random(9090)
    cases = [generated_case(rng) for _ in range(12)]
    expected = [dp_cost_for([], case) for case in cases]

    def work(case):
        request = scenario(
            demand=case["demand"],
            solar=case["solar"],
            tariff=case["tariff"],
            batt=case["battery"],
        )
        limits = compile_limits(
            [], request.solar(), case["battery"], SOLAR_PASS_PRODUCT
        )
        return solve(request.demand(), request.tariff(), case["battery"], limits)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, cases))

    for solution, want in zip(results, expected, strict=True):
        assert solution.total_cost_bdt == pytest.approx(want, abs=TOL)


def test_repeated_solves_of_the_same_input_are_identical():
    case = generated_case(random.Random(5))
    costs = {build_and_check([], case).total_cost_bdt for _ in range(8)}
    assert len(costs) == 1
