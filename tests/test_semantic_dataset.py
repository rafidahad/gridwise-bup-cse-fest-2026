"""The held-out labelled set must be internally valid (plan S10.3).

These tests say nothing about model accuracy - they check the *labels*. A label
that the service's own guardrails would reject, or that contradicts the PS S05
conventions, would quietly corrupt every semantic measurement taken with it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from app.directives import validate_interpretation
from app.schemas import BatteryInput, DirectiveType

DATASET = pathlib.Path(__file__).resolve().parents[1] / "data" / "semantic_cases.json"
PUBLIC = pathlib.Path(__file__).resolve().parents[1] / "data" / "public_cases.json"


@pytest.fixture(scope="module")
def dataset() -> dict:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def battery_for(case: dict) -> BatteryInput:
    capacity = float(case["battery_capacity_kwh"])
    return BatteryInput(
        capacity_kwh=capacity,
        initial_energy_kwh=capacity / 2,
        minimum_energy_kwh=capacity * 0.1,
        max_charge_kwh_per_hour=capacity / 4,
        max_discharge_kwh_per_hour=capacity / 4,
    )


def test_every_label_passes_the_service_guardrails(dataset):
    for case in dataset["cases"]:
        entries = [dict(e, explanation="label") for e in case["expected"]]
        result = validate_interpretation(entries, len(case["notes"]), battery_for(case))
        assert result.complete(len(case["notes"])), f"{case['id']}: {result.entry_problems}"


def test_every_note_has_exactly_one_label(dataset):
    for case in dataset["cases"]:
        indices = sorted(e["note_index"] for e in case["expected"])
        assert indices == list(range(len(case["notes"]))), case["id"]


def test_labels_obey_the_applies_semantics(dataset):
    for case in dataset["cases"]:
        for expected in case["expected"]:
            is_no_op = expected["directive_type"] == DirectiveType.NO_OP.value
            assert expected["applies"] is not is_no_op, case["id"]
            assert (expected["structured_adjustment"] is None) is is_no_op, case["id"]


def test_all_six_directive_types_are_covered(dataset):
    covered = {
        expected["directive_type"]
        for case in dataset["cases"]
        for expected in case["expected"]
    }
    assert covered == {directive.value for directive in DirectiveType}


def test_percentage_reserves_resolve_against_the_stated_capacity(dataset):
    """Spot-check the two labels that require arithmetic, not just extraction."""
    by_id = {case["id"]: case for case in dataset["cases"]}
    percent = by_id["RESERVE-PERCENT"]
    assert percent["battery_capacity_kwh"] == 300
    assert percent["expected"][0]["structured_adjustment"]["minimum_energy_kwh"] == 180

    fraction = by_id["RESERVE-FRACTION"]
    assert fraction["battery_capacity_kwh"] == 200
    assert fraction["expected"][0]["structured_adjustment"]["minimum_energy_kwh"] == 150


def test_reduction_by_and_reduction_to_are_labelled_differently(dataset):
    """The distinction PS S08 turns on, over the same hours."""
    by_id = {case["id"]: case for case in dataset["cases"]}
    to_80 = by_id["SOLAR-TO-80"]["expected"][0]["structured_adjustment"]
    by_80 = by_id["SOLAR-BY-80"]["expected"][0]["structured_adjustment"]
    assert to_80["hours"] == by_80["hours"] == [9, 10]
    assert to_80["factor"] == 0.8
    assert by_80["factor"] == 0.2


def test_no_note_is_copied_from_the_public_sample_pack(dataset):
    """Held-out means held out: passing here cannot be memorisation."""
    public_notes = {
        note.strip().lower()
        for case in json.loads(PUBLIC.read_text(encoding="utf-8"))["cases"]
        for note in case["input"]["operator_notes"]
    }
    for case in dataset["cases"]:
        for note in case["notes"]:
            assert note.strip().lower() not in public_notes, case["id"]


def test_the_dataset_has_paraphrase_clusters_to_measure_robustness(dataset):
    clusters: dict[str, int] = {}
    for case in dataset["cases"]:
        clusters[case["cluster"]] = clusters.get(case["cluster"], 0) + 1
    multi = [name for name, count in clusters.items() if count > 1]
    assert len(multi) >= 4, "too few clusters carry more than one phrasing"
