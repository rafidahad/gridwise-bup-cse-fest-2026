"""Measure live interpretation accuracy for the primary and backup models.

Plan SS6, 6.3 and 10.2 require evidence that the *actual* models read unseen
language correctly - something mocked unit tests cannot show. This runs the
held-out set in `data/semantic_cases.json` against a real provider and scores it
along the same dimensions the rubric uses (Guide S07): relevance/no_op,
directive type, affected hours, numeric values and shape, and paraphrase
robustness within a cluster.

Requires real credentials in the environment. Nothing here is imported by the
service; it is an evaluation tool.

    python scripts/semantic_eval.py                 # primary model
    python scripts/semantic_eval.py --role backup   # backup model
    python scripts/semantic_eval.py --repeat 3      # consistency across runs
    python scripts/semantic_eval.py --json out.json # machine-readable record
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.config import TOLERANCE, load_settings
from app.directives import validate_interpretation
from app.llm import ProviderFailure, _call_model, build_messages
from app.schemas import ScenarioRequest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "semantic_cases.json"
PACK_DIR = ROOT / "data" / "packs"


def _load_pack(name: str) -> list[dict]:
    """Normalise a test pack into {id, cluster, battery_capacity_kwh, notes, expected}.

    Four packs with three different shapes feed the same scorer, so a prompt
    change can be measured across all of them at once. Only the interpretation
    layer is covered here; `robustness_cases.json` targets HTTP behaviour and is
    exercised by the pytest suite instead.
    """
    if name == "held-out":
        return json.loads(DATASET.read_text(encoding="utf-8"))["cases"]

    path = PACK_DIR / f"{name}.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))

    cases: list[dict] = []
    if name == "interpretation_cases":
        for case in raw["cases"]:
            cases.append(
                {
                    "id": case["id"],
                    "cluster": case.get("family", "unknown"),
                    "battery_capacity_kwh": case["battery_capacity_kwh"],
                    "notes": case["notes"],
                    "expected": case["expected"],
                }
            )
    elif name == "scenario_cases":
        for case in raw["scenarios"]:
            cases.append(
                {
                    "id": case["id"],
                    "cluster": "scenario",
                    "battery_capacity_kwh": case["battery"]["capacity_kwh"],
                    "notes": case["operator_notes"],
                    "expected": case["expected_interpretation"],
                }
            )
    elif name == "gridwise_edge_case_pack":
        for case in raw["cases"]:
            cases.append(
                {
                    "id": case["id"],
                    "cluster": "edge",
                    "battery_capacity_kwh": case["input"]["battery"]["capacity_kwh"],
                    "notes": case["input"]["operator_notes"],
                    "expected": case["expected_output"]["directive_interpretation"],
                }
            )
    return cases


PACK_NAMES = (
    "held-out",
    "interpretation_cases",
    "scenario_cases",
    "gridwise_edge_case_pack",
)


def build_request(case: dict) -> ScenarioRequest:
    """Wrap the labelled notes in a minimal valid scenario.

    The hourly numbers are irrelevant to interpretation; only the battery
    capacity matters, because a reserve may be stated as a percentage of it.
    """
    capacity = float(case["battery_capacity_kwh"])
    return ScenarioRequest(
        scenario_id=case["id"],
        operator_notes=case["notes"],
        hours=[
            {
                "hour": hour,
                "demand_kwh": 100.0,
                "solar_kwh": 80.0 if 6 <= hour <= 17 else 0.0,
                "tariff_bdt_per_kwh": 10.0,
            }
            for hour in range(24)
        ],
        battery={
            "capacity_kwh": capacity,
            "initial_energy_kwh": capacity / 2,
            "minimum_energy_kwh": capacity * 0.1,
            "max_charge_kwh_per_hour": capacity / 4,
            "max_discharge_kwh_per_hour": capacity / 4,
        },
    )


def score_entry(expected: dict, actual) -> dict[str, bool]:
    """Score one note along the rubric's four machine-checkable dimensions."""
    expected_type = expected["directive_type"]
    expected_adjustment = expected["structured_adjustment"] or {}

    if actual is None:
        return {"relevance": False, "type": False, "hours": False, "values": False}

    actual_type = actual.directive_type.value
    actual_adjustment = actual.adjustment or {}

    relevance = actual.applies == expected["applies"]
    type_ok = actual_type == expected_type

    if expected_type == "no_op":
        # There are no hours or values to get right; correctness is relevance.
        return {
            "relevance": relevance,
            "type": type_ok,
            "hours": type_ok,
            "values": type_ok and actual.adjustment is None,
        }

    hours_ok = type_ok and list(actual_adjustment.get("hours", [])) == list(
        expected_adjustment.get("hours", [])
    )

    values_ok = type_ok
    for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
        if key in expected_adjustment:
            got = actual_adjustment.get(key)
            values_ok = values_ok and isinstance(got, (int, float)) and abs(
                float(got) - float(expected_adjustment[key])
            ) <= TOLERANCE
    # The required shape must be present and carry nothing extra.
    values_ok = values_ok and set(actual_adjustment) == set(expected_adjustment)

    return {
        "relevance": relevance,
        "type": type_ok,
        "hours": hours_ok,
        "values": values_ok,
    }


def signature(actual) -> str:
    """Canonical form of an interpretation, for comparing paraphrases."""
    if actual is None:
        return "none"
    adjustment = json.dumps(actual.adjustment, sort_keys=True) if actual.adjustment else "null"
    return f"{actual.directive_type.value}|{adjustment}"


async def run_case(case: dict, provider, settings) -> tuple[list, float, str | None]:
    request = build_request(case)
    started = time.monotonic()
    try:
        entries = await _call_model(
            provider, build_messages(request), settings, provider.timeout_s
        )
    except ProviderFailure as exc:
        return [], time.monotonic() - started, str(exc)

    elapsed = time.monotonic() - started
    result = validate_interpretation(entries, len(case["notes"]), request.battery)
    actual = [result.directives.get(i) for i in range(len(case["notes"]))]
    return actual, elapsed, None


async def main_async(args: argparse.Namespace) -> int:
    settings = load_settings()
    provider = settings.primary if args.role == "primary" else settings.backup

    if not provider.configured:
        print(
            f"The {args.role} provider is not configured. Set "
            f"GRIDWISE_{args.role.upper()}_API_KEY, _BASE_URL and _MODEL "
            "(see .env.example), then re-run."
        )
        return 2

    wanted = PACK_NAMES if args.packs == "all" else tuple(args.packs.split(","))
    cases = []
    by_pack: dict[str, list[str]] = {}
    for name in wanted:
        loaded = _load_pack(name.strip())
        for case in loaded:
            case["_pack"] = name.strip()
        by_pack[name.strip()] = [c["id"] for c in loaded]
        cases.extend(loaded)

    if not cases:
        print(f"no cases loaded for --packs {args.packs}")
        return 2

    print(f"model   : {provider.model}")
    print(f"endpoint: {provider.base_url}")
    print(f"settings: temperature={settings.temperature} seed={settings.seed} "
          f"prompt={settings.prompt_version} schema={settings.schema_version}")
    print(f"packs   : " + ", ".join(f"{k} ({len(v)})" for k, v in by_pack.items()))
    print(f"cases   : {len(cases)} x {args.repeat} run(s)\n")

    totals: dict[str, list[bool]] = defaultdict(list)
    latencies: list[float] = []
    failures: list[str] = []
    cluster_signatures: dict[str, set[str]] = defaultdict(set)
    per_case_signatures: dict[str, set[str]] = defaultdict(set)
    records = []

    for run_index in range(args.repeat):
        for case in cases:
            actual, elapsed, error = await run_case(case, provider, settings)
            latencies.append(elapsed)

            if error:
                failures.append(f"{case['id']}: {error}")
                for key in ("relevance", "type", "hours", "values"):
                    totals[key].append(False)
                continue

            case_ok = True
            for expected in case["expected"]:
                index = expected["note_index"]
                scores = score_entry(expected, actual[index])
                for key, value in scores.items():
                    totals[key].append(value)
                case_ok = case_ok and all(scores.values())
                if not all(scores.values()):
                    got = actual[index]
                    failures.append(
                        f"{case['id']} note {index}: expected "
                        f"{expected['directive_type']} "
                        f"{expected['structured_adjustment']}, got "
                        f"{got.directive_type.value if got else 'nothing'} "
                        f"{got.adjustment if got else ''}"
                    )

            # Robustness bookkeeping: one signature per single-note case.
            if len(case["notes"]) == 1:
                cluster_signatures[case["cluster"]].add(signature(actual[0]))
                per_case_signatures[case["id"]].add(signature(actual[0]))

            records.append(
                {
                    "run": run_index,
                    "case": case["id"],
                    "pack": case.get("_pack", "held-out"),
                    "cluster": case["cluster"],
                    "ok": case_ok,
                    "latency_s": round(elapsed, 3),
                }
            )

    def pct(key: str) -> float:
        values = totals[key]
        return 100.0 * sum(values) / len(values) if values else 0.0

    multi_signature_clusters = {
        cluster: sorted(sigs)
        for cluster, sigs in cluster_signatures.items()
        if len(sigs) > 1
    }
    unstable_cases = {
        case_id: sorted(sigs)
        for case_id, sigs in per_case_signatures.items()
        if len(sigs) > 1
    }
    robustness = 100.0 * (
        1 - len(multi_signature_clusters) / max(1, len(cluster_signatures))
    )

    if len(by_pack) > 1:
        print("exact-case rate by pack")
        for name in by_pack:
            rows = [r for r in records if r["pack"] == name]
            if rows:
                ok = sum(1 for r in rows if r["ok"])
                print(f"  {name:<26} {ok:>4}/{len(rows):<4} {100*ok/len(rows):5.1f}%")
        print()

    print("score by rubric dimension")
    print(f"  relevance / no_op        {pct('relevance'):6.1f}%")
    print(f"  directive_type           {pct('type'):6.1f}%")
    print(f"  affected hours           {pct('hours'):6.1f}%")
    print(f"  numeric values and shape {pct('values'):6.1f}%")
    print(f"  paraphrase robustness    {robustness:6.1f}%  "
          f"({len(cluster_signatures) - len(multi_signature_clusters)}"
          f"/{len(cluster_signatures)} clusters internally consistent)")
    print()
    print(f"latency: mean {statistics.mean(latencies):.2f}s  "
          f"max {max(latencies):.2f}s  "
          f"p95 {sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]:.2f}s")

    if args.repeat > 1:
        print(f"repeat consistency: {len(per_case_signatures) - len(unstable_cases)}"
              f"/{len(per_case_signatures)} single-note cases identical across runs")
        for case_id, sigs in unstable_cases.items():
            print(f"  ~ {case_id} varied: {sigs}")

    if multi_signature_clusters:
        print("\nparaphrase disagreements (same rule, different answers):")
        for cluster, sigs in multi_signature_clusters.items():
            print(f"  ! {cluster}: {sigs}")

    if failures:
        print(f"\n{len(failures)} incorrect interpretation(s):")
        for failure in failures[:40]:
            print(f"  ! {failure}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
    else:
        print("\nall labelled notes interpreted correctly")

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(
                {
                    "role": args.role,
                    "model": provider.model,
                    "base_url": provider.base_url,
                    "temperature": settings.temperature,
                    "seed": settings.seed,
                    "prompt_version": settings.prompt_version,
                    "schema_version": settings.schema_version,
                    "repeat": args.repeat,
                    "scores": {key: pct(key) for key in
                               ("relevance", "type", "hours", "values")},
                    "paraphrase_robustness": robustness,
                    "latency_mean_s": statistics.mean(latencies),
                    "failures": failures,
                    "records": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("primary", "backup"), default="primary")
    parser.add_argument("--packs", default="all",
                        help="all, or a comma-separated subset: " + ",".join(PACK_NAMES))
    parser.add_argument("--repeat", type=int, default=1,
                        help="runs per case; >1 measures run-to-run consistency")
    parser.add_argument("--json", help="write a machine-readable record here")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
