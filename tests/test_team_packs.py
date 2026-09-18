"""Regression cover for the team-generated test packs in `data/packs/`.

These packs are unofficial - their own `_meta` says so, and where any of them
disagrees with the Problem Statement the Problem Statement wins. They are still
worth locking down for two reasons:

* the edge pack carries reference optimal costs, so it is a second independent
  check on the optimizer beyond the ten official samples
* a prompt measurement is meaningless if the labels it scores against are
  themselves invalid

Nothing here calls a model. Interpretation accuracy is measured live by
`scripts/semantic_eval.py --packs all`.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from app.config import TOLERANCE
from app.directives import Directive, validate_interpretation
from app.optimizer import solve_with_solar_policy
from app.schemas import (
    BatteryInput,
    DirectiveType,
    OptimizeResponse,
    ScenarioRequest,
    check_battery_feasibility,
)
from app.validator import validate_response

PACKS = pathlib.Path(__file__).resolve().parents[1] / "data" / "packs"


def _load(name: str) -> dict:
    return json.loads((PACKS / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def edge_pack() -> dict:
    return _load("gridwise_edge_case_pack")


@pytest.fixture(scope="module")
def interpretation_pack() -> dict:
    return _load("interpretation_cases")


@pytest.fixture(scope="module")
def scenario_pack() -> dict:
    return _load("scenario_cases")


def directives_from(entries: list[dict]) -> list[Directive]:
    return [
        Directive(
            note_index=e["note_index"],
            applies=e["applies"],
            directive_type=DirectiveType(e["directive_type"]),
            adjustment=e["structured_adjustment"],
            explanation=e.get("explanation", ""),
        )
        for e in entries
    ]


def battery_for(capacity: float) -> BatteryInput:
    return BatteryInput(
        capacity_kwh=capacity,
        initial_energy_kwh=capacity / 2,
        minimum_energy_kwh=capacity * 0.1,
        max_charge_kwh_per_hour=capacity / 4,
        max_discharge_kwh_per_hour=capacity / 4,
    )


# ---------------------------------------------------------------------------
# Labels must be valid, or every measurement taken with them is worthless
# ---------------------------------------------------------------------------


def test_interpretation_pack_labels_pass_our_guardrails(interpretation_pack):
    for case in interpretation_pack["cases"]:
        entries = [dict(e, explanation="label") for e in case["expected"]]
        result = validate_interpretation(
            entries, len(case["notes"]), battery_for(case["battery_capacity_kwh"])
        )
        assert result.complete(len(case["notes"])), f"{case['id']}: {result.entry_problems}"


def test_scenario_pack_labels_pass_our_guardrails(scenario_pack):
    for case in scenario_pack["scenarios"]:
        entries = [dict(e, explanation="label") for e in case["expected_interpretation"]]
        result = validate_interpretation(
            entries, len(case["operator_notes"]), battery_for(case["battery"]["capacity_kwh"])
        )
        assert result.complete(len(case["operator_notes"])), case["id"]


def test_edge_pack_labels_pass_our_guardrails(edge_pack):
    for case in edge_pack["cases"]:
        entries = case["expected_output"]["directive_interpretation"]
        request = ScenarioRequest.model_validate(case["input"])
        result = validate_interpretation(
            entries, len(case["input"]["operator_notes"]), request.battery
        )
        assert result.complete(len(case["input"]["operator_notes"])), case["id"]


# ---------------------------------------------------------------------------
# The edge pack as a second independent optimizer check
# ---------------------------------------------------------------------------


def test_edge_pack_costs_are_reproduced(edge_pack):
    """Our optimizer must match the pack's reference optimum, or beat it.

    Beating it is only acceptable when our plan also replays cleanly - that is
    the difference between finding a better optimum and cheating. EDGE-15 is
    built from fractional inputs specifically to accumulate float error, and
    our plan lands ~0.05 BDT (1.1e-06 relative) under the reference there while
    satisfying every rule to 1e-14. Scoring is unaffected: Guide S07 caps
    `quality_ratio` at 1.
    """
    checked = 0
    for case in edge_pack["cases"]:
        request = ScenarioRequest.model_validate(case["input"])
        check_battery_feasibility(request.battery)
        directives = directives_from(case["expected_output"]["directive_interpretation"])

        solution, limits = solve_with_solar_policy(
            directives,
            request.demand(),
            request.solar(),
            request.tariff(),
            request.battery,
        )
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
            ],
            hourly_plan=solution.hourly_plan,
            total_grid_kwh=solution.total_grid_kwh,
            total_cost_bdt=solution.total_cost_bdt,
            peak_grid_kwh=solution.peak_grid_kwh,
            plan_summary="pack check",
        )

        problems = validate_response(request, directives, limits.solar_pass, response)
        assert problems == [], f"{case['id']}: {problems}"

        reference = float(case["expected_output"]["total_cost_bdt"])
        assert solution.total_cost_bdt <= reference + TOLERANCE, (
            f"{case['id']}: {solution.total_cost_bdt} is above the reference {reference}"
        )
        # Any shortfall must be float noise, not a missing constraint.
        assert solution.total_cost_bdt >= reference * (1 - 1e-5), (
            f"{case['id']}: {solution.total_cost_bdt} is materially below {reference}, "
            "which would mean a constraint is missing"
        )
        checked += 1
    assert checked == 18


def test_edge_pack_reference_plans_are_self_consistent(edge_pack):
    """Their published rows must agree with their published totals."""
    for case in edge_pack["cases"]:
        expected = case["expected_output"]
        tariff = {h["hour"]: h["tariff_bdt_per_kwh"] for h in case["input"]["hours"]}
        rows = expected["hourly_plan"]
        assert sum(r["grid_kwh"] for r in rows) == pytest.approx(
            expected["total_grid_kwh"], abs=TOLERANCE
        ), case["id"]
        assert sum(r["grid_kwh"] * tariff[r["hour"]] for r in rows) == pytest.approx(
            expected["total_cost_bdt"], abs=TOLERANCE
        ), case["id"]
        assert max(r["grid_kwh"] for r in rows) == pytest.approx(
            expected["peak_grid_kwh"], abs=TOLERANCE
        ), case["id"]


def test_the_analytic_anchor_is_actually_fixed_cost(scenario_pack):
    """SC-06: no solar and a flat tariff pin the cost for every valid plan.

    Neutrality forces total charge to equal total discharge, so grid energy
    equals total demand whatever the battery does. If our optimizer returns
    anything but 36400 here, either a total is mis-reported or the plan is
    invalid.
    """
    case = next(s for s in scenario_pack["scenarios"] if s["id"] == "SC-06")
    profile = scenario_pack["profiles"][case["profile"]]
    request = ScenarioRequest(
        scenario_id=case["id"],
        operator_notes=case["operator_notes"],
        hours=[
            {
                "hour": h,
                "demand_kwh": profile["demand_kwh"][h],
                "solar_kwh": profile["solar_kwh"][h],
                "tariff_bdt_per_kwh": profile["tariff_bdt_per_kwh"][h],
            }
            for h in range(24)
        ],
        battery=case["battery"],
    )
    directives = directives_from(case["expected_interpretation"])
    solution, _ = solve_with_solar_policy(
        directives, request.demand(), request.solar(), request.tariff(), request.battery
    )
    assert solution.total_cost_bdt == pytest.approx(36400, abs=TOLERANCE)
    assert solution.total_grid_kwh == pytest.approx(3640, abs=TOLERANCE)


def test_every_scenario_pack_case_solves_except_the_infeasible_one(scenario_pack):
    """SC-11 is infeasible by construction; the other ten must all solve."""
    from app.optimizer import InfeasibleScheduleError

    solved, infeasible = [], []
    for case in scenario_pack["scenarios"]:
        profile = scenario_pack["profiles"][case["profile"]]
        request = ScenarioRequest(
            scenario_id=case["id"],
            operator_notes=case["operator_notes"],
            hours=[
                {
                    "hour": h,
                    "demand_kwh": profile["demand_kwh"][h],
                    "solar_kwh": profile["solar_kwh"][h],
                    "tariff_bdt_per_kwh": profile["tariff_bdt_per_kwh"][h],
                }
                for h in range(24)
            ],
            battery=case["battery"],
        )
        directives = directives_from(case["expected_interpretation"])
        try:
            solution, limits = solve_with_solar_policy(
                directives,
                request.demand(),
                request.solar(),
                request.tariff(),
                request.battery,
            )
        except InfeasibleScheduleError:
            infeasible.append(case["id"])
            continue

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
            ],
            hourly_plan=solution.hourly_plan,
            total_grid_kwh=solution.total_grid_kwh,
            total_cost_bdt=solution.total_cost_bdt,
            peak_grid_kwh=solution.peak_grid_kwh,
            plan_summary="pack check",
        )
        assert validate_response(request, directives, limits.solar_pass, response) == [], case["id"]
        solved.append(case["id"])

    assert infeasible == ["SC-11"], f"unexpected infeasible set: {infeasible}"
    assert len(solved) == 10
