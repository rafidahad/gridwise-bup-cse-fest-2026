"""Offline check of the deterministic core against the public sample pack.

Feeds each public case's *reference* directive interpretation straight into the
compiler, optimizer, and Replay Validator, bypassing the language model. This
isolates the layers as plan S10.2 requires: a failure here is a constraint or
optimization defect, never an interpretation one.

Reports, per case, whether the plan replays cleanly and how its cost compares to
the organizer's published optimal cost. The cost column is the independent
verification plan S1 flags as outstanding.

    python scripts/verify_core.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.config import TOLERANCE
from app.directives import (
    SOLAR_PASS_PRODUCT,
    Directive,
    compile_limits,
)
from app.optimizer import solve
from app.schemas import (
    DirectiveType,
    OptimizeResponse,
    ScenarioRequest,
    check_battery_feasibility,
)
from app.validator import validate_response

CASES = pathlib.Path(__file__).resolve().parents[1] / "data" / "public_cases.json"


def reference_directives(expected: dict) -> list[Directive]:
    """Build Directive objects from a case's published interpretation."""
    directives = []
    for entry in expected["directive_interpretation"]:
        directives.append(
            Directive(
                note_index=entry["note_index"],
                applies=entry["applies"],
                directive_type=DirectiveType(entry["directive_type"]),
                adjustment=entry["structured_adjustment"],
                explanation=entry["explanation"],
            )
        )
    return directives


def main() -> int:
    pack = json.loads(CASES.read_text(encoding="utf-8"))
    failures = 0

    header = f"{'case':<12} {'status':<9} {'our cost':>12} {'reference':>12} {'delta':>10}"
    print(header)
    print("-" * len(header))

    for case in pack["cases"]:
        request = ScenarioRequest.model_validate(case["input"])
        check_battery_feasibility(request.battery)

        directives = reference_directives(case["expected_output"])
        limits = compile_limits(
            directives, request.solar(), request.battery, SOLAR_PASS_PRODUCT
        )
        solution = solve(
            request.demand(), request.tariff(), request.battery, limits,
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
            plan_summary="core verification",
        )

        problems = validate_response(
            request, directives, limits.solar_pass, response
        )
        reference_cost = float(case["expected_output"]["total_cost_bdt"])
        delta = solution.total_cost_bdt - reference_cost

        status = "ok"
        if problems:
            status = "INVALID"
            failures += 1
        elif delta > TOLERANCE:
            status = "COSTLIER"
            failures += 1
        elif delta < -TOLERANCE:
            # Cheaper than the reference while replaying cleanly would mean the
            # reference is not optimal, or a constraint is missing here.
            status = "CHEAPER"
            failures += 1

        print(
            f"{case['id']:<12} {status:<9} {solution.total_cost_bdt:>12,.2f} "
            f"{reference_cost:>12,.2f} {delta:>10,.2f}"
        )
        for problem in problems:
            print(f"    ! {problem}")

    print()
    if failures:
        print(f"FAILED: {failures} of {len(pack['cases'])} cases")
    else:
        print(f"PASSED: {len(pack['cases'])} of {len(pack['cases'])} cases "
              f"replay cleanly and match the reference optimal cost")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
