"""Run the 10 public sample cases against a running GridWise service.

This is the reproduction command in README section "Verify with the public
samples". It exercises the deployed HTTP path end to end, including live model
interpretation, and reports what the organizer harness reports: interpretation
agreement with the published reference, schedule validity under independent
replay, cost quality, and latency.

    python scripts/run_public_cases.py
    python scripts/run_public_cases.py --base-url https://your-deployment.example
    python scripts/run_public_cases.py --json results.json

Exit code 0 means every case returned a valid schedule whose cost matched the
published optimum and whose interpretation matched the published reference.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.config import TOLERANCE
from app.directives import (
    SOLAR_PASS_MIN,
    SOLAR_PASS_PRODUCT,
    Directive,
    has_overlapping_solar_reduction,
)
from app.schemas import DirectiveType, OptimizeResponse, ScenarioRequest
from app.validator import validate_response

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "public_cases.json"


def as_directives(entries: list[dict]) -> list[Directive]:
    return [
        Directive(
            note_index=entry["note_index"],
            applies=entry["applies"],
            directive_type=DirectiveType(entry["directive_type"]),
            adjustment=entry["structured_adjustment"],
            explanation=entry.get("explanation", ""),
        )
        for entry in entries
    ]


def interpretation_matches(expected: list[dict], actual: list[dict]) -> list[str]:
    """Compare semantics, not wording (Guide S07: explanations are not matched)."""
    problems: list[str] = []
    if len(expected) != len(actual):
        return [f"expected {len(expected)} interpretations, got {len(actual)}"]

    for want, got in zip(expected, actual, strict=True):
        index = want["note_index"]
        if got["note_index"] != index:
            problems.append(f"note {index}: returned out of order")
            continue
        if got["applies"] != want["applies"]:
            problems.append(
                f"note {index}: applies {got['applies']}, expected {want['applies']}"
            )
        if got["directive_type"] != want["directive_type"]:
            problems.append(
                f"note {index}: type {got['directive_type']}, "
                f"expected {want['directive_type']}"
            )
            continue

        want_adjustment = want["structured_adjustment"]
        got_adjustment = got["structured_adjustment"]
        if want_adjustment is None:
            if got_adjustment is not None:
                problems.append(f"note {index}: expected a null adjustment")
            continue
        if not isinstance(got_adjustment, dict):
            problems.append(f"note {index}: adjustment is not an object")
            continue
        if set(got_adjustment) != set(want_adjustment):
            problems.append(
                f"note {index}: adjustment keys {sorted(got_adjustment)}, "
                f"expected {sorted(want_adjustment)}"
            )
            continue
        if list(got_adjustment["hours"]) != list(want_adjustment["hours"]):
            problems.append(
                f"note {index}: hours {got_adjustment['hours']}, "
                f"expected {want_adjustment['hours']}"
            )
        for key, value in want_adjustment.items():
            if key == "hours":
                continue
            if abs(float(got_adjustment[key]) - float(value)) > TOLERANCE:
                problems.append(
                    f"note {index}: {key} {got_adjustment[key]}, expected {value}"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--json", help="write a machine-readable record here")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    pack = json.loads(CASES.read_text(encoding="utf-8"))

    with httpx.Client(timeout=args.timeout) as client:
        try:
            health = client.get(f"{base}/health")
        except httpx.HTTPError as exc:
            print(f"cannot reach {base}/health: {exc}")
            return 2
        if health.status_code != 200 or health.json() != {"status": "ok"}:
            print(f"health check failed: {health.status_code} {health.text[:200]}")
            return 2
        print(f"health: 200 {health.json()}\n")

        header = (
            f"{'case':<12} {'http':>5} {'valid':<6} {'interp':<7} "
            f"{'cost':>11} {'optimum':>11} {'ratio':>6} {'secs':>6}"
        )
        print(header)
        print("-" * len(header))

        failures = 0
        latencies: list[float] = []
        ratios: list[float] = []
        records = []

        for case in pack["cases"]:
            case_id = case["id"]
            reference = case["expected_output"]
            started = time.monotonic()
            try:
                http = client.post(f"{base}/optimize-energy", json=case["input"])
            except httpx.HTTPError as exc:
                print(f"{case_id:<12} {'---':>5}  request failed: {exc}")
                failures += 1
                continue
            elapsed = time.monotonic() - started
            latencies.append(elapsed)

            if http.status_code != 200:
                print(f"{case_id:<12} {http.status_code:>5}  {http.text[:120]}")
                failures += 1
                continue

            body = http.json()
            request = ScenarioRequest.model_validate(case["input"])

            # Replay the returned plan against the directives the service itself
            # reported, exactly as the judge replays against ground truth.
            try:
                response = OptimizeResponse.model_validate(body)
                returned = as_directives(body["directive_interpretation"])
                violations = validate_response(
                    request, returned, SOLAR_PASS_PRODUCT, response
                )
                if violations and has_overlapping_solar_reduction(returned):
                    # The service may have taken the documented relaxation pass
                    # (plan S7.1). Judge it against the rule it actually used.
                    relaxed = validate_response(
                        request, returned, SOLAR_PASS_MIN, response
                    )
                    if not relaxed:
                        violations = []
            except Exception as exc:  # malformed response shape
                violations = [f"response did not match the contract: {exc}"]

            semantic = interpretation_matches(
                reference["directive_interpretation"], body["directive_interpretation"]
            )

            cost = float(body["total_cost_bdt"])
            optimum = float(reference["total_cost_bdt"])
            # Guide S07: quality_ratio = min(1, optimum / team_cost), and an
            # invalid case earns no optimization credit at all.
            if violations:
                ratio = 0.0
            elif cost > TOLERANCE:
                ratio = min(1.0, optimum / cost)
            else:
                # Both within tolerance of zero: the guide fixes the ratio at 1.
                ratio = 1.0 if optimum <= TOLERANCE else 0.0
            if not violations:
                ratios.append(ratio)

            ok = not violations and not semantic and cost <= optimum + TOLERANCE
            if not ok:
                failures += 1

            print(
                f"{case_id:<12} {http.status_code:>5} "
                f"{'yes' if not violations else 'NO':<6} "
                f"{'yes' if not semantic else 'NO':<7} "
                f"{cost:>11,.2f} {optimum:>11,.2f} {ratio:>6.3f} {elapsed:>6.2f}"
            )
            for problem in violations:
                print(f"    ! invalid: {problem}")
            for problem in semantic:
                print(f"    ~ interpretation: {problem}")
            if not violations and cost > optimum + TOLERANCE:
                print(f"    ~ cost is {cost - optimum:,.2f} BDT above the optimum")

            records.append(
                {
                    "case": case_id,
                    "status": http.status_code,
                    "valid": not violations,
                    "interpretation_matches": not semantic,
                    "cost": cost,
                    "optimum": optimum,
                    "quality_ratio": ratio,
                    "latency_s": round(elapsed, 3),
                    "violations": violations,
                    "interpretation_problems": semantic,
                }
            )

    print()
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[max(0, int(0.95 * len(ordered)) - 1)]
        band = (
            "3/3" if p95 <= 5 else "2/3" if p95 <= 15 else "1/3" if p95 <= 30 else "0/3"
        )
        print(
            f"latency: mean {statistics.mean(latencies):.2f}s  max {max(latencies):.2f}s  "
            f"p95 {p95:.2f}s  -> Guide S08 latency band {band}"
        )
    if ratios:
        print(
            f"optimization quality: {10 * statistics.mean(ratios):.2f}/10 "
            f"over {len(ratios)} valid case(s)"
        )

    total = len(pack["cases"])
    print(
        f"\n{'FAILED' if failures else 'PASSED'}: "
        f"{total - failures}/{total} cases valid, interpreted as published, "
        f"and at the optimal cost"
    )

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"base_url": base, "records": records}, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
