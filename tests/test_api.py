"""API contract and end-to-end pipeline tests (PS SS6, 7, 10; plan S10.2).

Inference is stubbed with the *reference* interpretation for each public case,
so these tests exercise the real request parsing, compilation, solving, replay,
and serialisation path without depending on a live provider. Live interpretation
is measured separately by `scripts/semantic_eval.py`.
"""

from __future__ import annotations

import copy
import json

import pytest
from starlette.testclient import TestClient

from app import llm, main
from app.directives import Directive
from app.llm import InterpretationOutcome, ProviderFailure
from app.schemas import DirectiveType


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def reference_directives(case: dict) -> list[Directive]:
    return [
        Directive(
            note_index=entry["note_index"],
            applies=entry["applies"],
            directive_type=DirectiveType(entry["directive_type"]),
            adjustment=entry["structured_adjustment"],
            explanation=entry["explanation"],
        )
        for entry in case["expected_output"]["directive_interpretation"]
    ]


@pytest.fixture
def stub_interpretation(monkeypatch):
    """Replace the model call with a caller-supplied interpretation."""

    def install(directives, *, raises=None):
        async def fake(request, settings, budget):
            if raises is not None:
                raise raises
            resolved = directives(request) if callable(directives) else directives
            return InterpretationOutcome(resolved, ["stubbed"])

        monkeypatch.setattr(main, "interpret_notes", fake)

    return install


# ---------------------------------------------------------------------------
# Health (PS S6.2)
# ---------------------------------------------------------------------------


def test_health_returns_exactly_status_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_stays_available_without_any_credentials(client):
    """Readiness must not depend on a provider being reachable (Guide S08)."""
    assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Request validation: 400 for structural, 422 for semantic (PS S6.1)
# ---------------------------------------------------------------------------


def test_malformed_json_is_400(client):
    response = client.post(
        "/optimize-energy",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "malformed_json"


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda body: body.pop("scenario_id"), id="missing-scenario-id"),
        pytest.param(lambda body: body.pop("battery"), id="missing-battery"),
        pytest.param(lambda body: body.pop("hours"), id="missing-hours"),
        pytest.param(lambda body: body.update(operator_notes=[]), id="no-notes"),
        pytest.param(lambda body: body.update(operator_notes=["a", "b", "c", "d"]), id="four-notes"),
        pytest.param(lambda body: body.update(operator_notes=["   "]), id="blank-note"),
        pytest.param(lambda body: body["hours"].pop(), id="23-hours"),
        pytest.param(lambda body: body["hours"].__setitem__(5, dict(body["hours"][5], hour=6)), id="duplicate-hour"),
        pytest.param(lambda body: body["hours"].__setitem__(5, dict(body["hours"][5], hour=99)), id="hour-out-of-range"),
        pytest.param(lambda body: body["hours"].__setitem__(5, dict(body["hours"][5], demand_kwh="lots")), id="wrong-type"),
        pytest.param(lambda body: body["battery"].update(capacity_kwh=-5), id="negative-capacity"),
    ],
)
def test_structurally_invalid_requests_are_400(client, public_cases, mutation):
    body = copy.deepcopy(public_cases[0]["input"])
    mutation(body)
    response = client.post("/optimize-energy", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_400(client, public_cases, literal):
    """Sent as raw bytes: these are JSON extensions no serialiser will emit."""
    body = copy.deepcopy(public_cases[0]["input"])
    raw = json.dumps(body).replace('"tariff_bdt_per_kwh": 6', f'"tariff_bdt_per_kwh": {literal}', 1)
    response = client.post(
        "/optimize-energy",
        content=raw.encode(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_energy_kwh", 5),      # below the base reserve
        ("initial_energy_kwh", 10_000),  # above capacity
        ("minimum_energy_kwh", 10_000),  # reserve above capacity
    ],
)
def test_infeasible_battery_parameters_are_422(client, public_cases, field, value):
    body = copy.deepcopy(public_cases[0]["input"])
    body["battery"][field] = value
    response = client.post("/optimize-energy", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "semantically_invalid_request"


def test_initial_energy_exactly_on_a_boundary_is_accepted(
    client, public_cases, stub_interpretation
):
    body = copy.deepcopy(public_cases[0]["input"])
    body["battery"]["initial_energy_kwh"] = body["battery"]["minimum_energy_kwh"]
    stub_interpretation(reference_directives(public_cases[0]))
    assert client.post("/optimize-energy", json=body).status_code == 200


def test_initial_energy_within_tolerance_of_a_boundary_still_produces_a_plan(
    client, public_cases, stub_interpretation
):
    """Accepting the request is not enough - it must also solve.

    With initial energy 0.005 below the base reserve, neutrality pins the final
    state just under an exact LP bound. PS S11.5 treats that gap as zero, so the
    scenario must yield a schedule rather than an infeasibility.
    """
    case = public_cases[0]
    body = copy.deepcopy(case["input"])
    body["battery"]["initial_energy_kwh"] = body["battery"]["minimum_energy_kwh"] - 0.005
    stub_interpretation(reference_directives(case))
    response = client.post("/optimize-energy", json=body)
    assert response.status_code == 200
    final = response.json()["hourly_plan"][23]["battery_energy_after_kwh"]
    assert final == pytest.approx(body["battery"]["initial_energy_kwh"], abs=0.01)


def test_unknown_extra_fields_are_ignored_not_rejected(
    client, public_cases, stub_interpretation
):
    """An organizer input must not fail for carrying a field we did not expect."""
    body = copy.deepcopy(public_cases[0]["input"])
    body["experimental_flag"] = True
    body["hours"][0]["humidity"] = 0.8
    stub_interpretation(reference_directives(public_cases[0]))
    assert client.post("/optimize-energy", json=body).status_code == 200


def test_hours_supplied_out_of_order_are_accepted(
    client, public_cases, stub_interpretation
):
    """PS S7.2 requires uniqueness and coverage, not an input ordering."""
    case = public_cases[0]
    body = copy.deepcopy(case["input"])
    body["hours"] = list(reversed(body["hours"]))
    stub_interpretation(reference_directives(case))
    response = client.post("/optimize-energy", json=body)
    assert response.status_code == 200
    plan = response.json()["hourly_plan"]
    assert [row["hour"] for row in plan] == list(range(24))
    assert response.json()["total_cost_bdt"] == pytest.approx(
        case["expected_output"]["total_cost_bdt"], abs=0.01
    )


# ---------------------------------------------------------------------------
# Successful responses (PS S10)
# ---------------------------------------------------------------------------


def test_every_public_case_returns_the_exact_contract(
    client, public_cases, public_pack, stub_interpretation
):
    required_top = set(public_pack["_meta"]["schema_notes"]["output_required_fields"])
    required_interp = set(
        public_pack["_meta"]["schema_notes"]["directive_interpretation_required_fields"]
    )
    required_row = set(public_pack["_meta"]["schema_notes"]["hourly_plan_required_fields"])

    for case in public_cases:
        stub_interpretation(reference_directives(case))
        response = client.post("/optimize-energy", json=case["input"])
        assert response.status_code == 200, case["id"]
        body = response.json()

        assert set(body) == required_top, case["id"]
        assert body["scenario_id"] == case["input"]["scenario_id"]
        assert isinstance(body["plan_summary"], str) and body["plan_summary"]

        assert len(body["directive_interpretation"]) == len(
            case["input"]["operator_notes"]
        )
        for position, entry in enumerate(body["directive_interpretation"]):
            assert set(entry) == required_interp
            assert entry["note_index"] == position
            assert entry["directive_type"] in public_pack["_meta"]["allowed_enums"][
                "directive_type"
            ]
            if entry["directive_type"] == "no_op":
                assert entry["applies"] is False
                assert entry["structured_adjustment"] is None
            else:
                assert entry["applies"] is True
                assert entry["structured_adjustment"] is not None

        assert len(body["hourly_plan"]) == 24
        for hour, row in enumerate(body["hourly_plan"]):
            assert set(row) == required_row
            assert row["hour"] == hour
            assert row["battery_action"] in public_pack["_meta"]["allowed_enums"][
                "battery_action"
            ]
            assert row["grid_kwh"] >= 0
            assert row["solar_used_kwh"] >= 0
            assert row["battery_kwh"] >= 0
            if row["battery_action"] == "idle":
                assert row["battery_kwh"] == 0


def test_every_public_case_reaches_the_reference_optimal_cost(
    client, public_cases, stub_interpretation
):
    """Cost and total import are determined; the action sequence is not.

    `peak_grid_kwh` is deliberately not compared to the reference. It is a
    reported value, not an objective (PS S5.2), so equally optimal schedules
    can peak differently and PS S11.4 accepts any of them. That the reported
    peak matches the returned rows is checked separately.
    """
    for case in public_cases:
        stub_interpretation(reference_directives(case))
        body = client.post("/optimize-energy", json=case["input"]).json()
        expected = case["expected_output"]
        assert body["total_cost_bdt"] == pytest.approx(
            expected["total_cost_bdt"], abs=0.01
        ), case["id"]
        assert body["total_grid_kwh"] == pytest.approx(
            expected["total_grid_kwh"], abs=0.01
        ), case["id"]


def test_totals_are_recomputable_from_the_returned_rows(
    client, public_cases, stub_interpretation
):
    """PS S11.3: hourly_plan is the source of truth for the reported totals."""
    for case in public_cases:
        stub_interpretation(reference_directives(case))
        body = client.post("/optimize-energy", json=case["input"]).json()
        tariff = {h["hour"]: h["tariff_bdt_per_kwh"] for h in case["input"]["hours"]}
        rows = body["hourly_plan"]
        assert body["total_grid_kwh"] == pytest.approx(
            sum(r["grid_kwh"] for r in rows), abs=0.01
        )
        assert body["total_cost_bdt"] == pytest.approx(
            sum(r["grid_kwh"] * tariff[r["hour"]] for r in rows), abs=0.01
        )
        assert body["peak_grid_kwh"] == pytest.approx(
            max(r["grid_kwh"] for r in rows), abs=0.01
        )


def test_battery_returns_to_its_initial_level(client, public_cases, stub_interpretation):
    for case in public_cases:
        stub_interpretation(reference_directives(case))
        body = client.post("/optimize-energy", json=case["input"]).json()
        assert body["hourly_plan"][23]["battery_energy_after_kwh"] == pytest.approx(
            case["input"]["battery"]["initial_energy_kwh"], abs=0.01
        ), case["id"]


def test_response_is_json_serialisable_and_finite(
    client, public_cases, stub_interpretation
):
    stub_interpretation(reference_directives(public_cases[0]))
    body = client.post("/optimize-energy", json=public_cases[0]["input"]).json()
    text = json.dumps(body)
    assert "NaN" not in text and "Infinity" not in text


# ---------------------------------------------------------------------------
# Controlled failure (PS S6.1, Guide S08)
# ---------------------------------------------------------------------------


def test_interpretation_failure_is_a_controlled_500_without_a_partial_plan(
    client, public_cases, stub_interpretation
):
    stub_interpretation(
        [], raises=llm.InterpretationUnavailable("no valid interpretation for note(s) [0]")
    )
    response = client.post("/optimize-energy", json=public_cases[0]["input"])
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "interpretation_unavailable"
    assert "hourly_plan" not in body


def test_an_unexpected_failure_leaks_no_provider_or_stack_detail(
    public_cases, stub_interpretation
):
    """The catch-all 500 path (Guide S08 "Secret handling").

    `raise_server_exceptions=False` makes the test client behave like a real
    deployment, where the handler serves the response instead of the exception
    propagating to the test.
    """
    secret = "sk-live-000-do-not-leak"
    stub_interpretation([], raises=ProviderFailure(f"auth failed for key {secret}"))
    with TestClient(main.app, raise_server_exceptions=False) as raw_client:
        response = raw_client.post("/optimize-energy", json=public_cases[0]["input"])
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    text = response.text
    assert secret not in text
    assert "Traceback" not in text
    assert "api.groq.com" not in text
    assert "ProviderFailure" not in text


def test_an_infeasible_scenario_is_422_not_5xx(
    client, public_cases, stub_interpretation
):
    """A well-formed request that admits no schedule. Not a server fault."""
    case = public_cases[0]
    impossible = Directive(
        note_index=0,
        applies=True,
        directive_type=DirectiveType.MAX_GRID_WINDOW,
        adjustment={"hours": list(range(24)), "max_grid_kwh": 0.0},
        explanation="no grid at all",
    )
    stub_interpretation([impossible] + reference_directives(case)[1:])
    response = client.post("/optimize-energy", json=case["input"])
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "infeasible_scenario"


def test_a_plan_failing_replay_is_never_returned(
    client, public_cases, stub_interpretation, monkeypatch
):
    """The last line of defence: verification failure withholds the response."""
    case = public_cases[0]
    stub_interpretation(reference_directives(case))
    monkeypatch.setattr(
        main, "validate_response", lambda *args, **kwargs: ["hour 3: fabricated fault"]
    )
    response = client.post("/optimize-energy", json=case["input"])
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "replay_failed"
    assert "fabricated fault" not in response.text


def test_unknown_route_returns_the_documented_error_shape(client):
    response = client.post("/not-a-real-endpoint", json={})
    assert response.status_code == 404
    assert set(response.json()["error"]) == {"code", "message"}


# ---------------------------------------------------------------------------
# Repeated and concurrent requests (plan S10.3)
# ---------------------------------------------------------------------------


def test_repeated_identical_requests_are_stable(client, public_cases, stub_interpretation):
    case = public_cases[4]
    stub_interpretation(reference_directives(case))
    costs = {
        client.post("/optimize-energy", json=case["input"]).json()["total_cost_bdt"]
        for _ in range(5)
    }
    assert len(costs) == 1


def test_different_scenarios_do_not_leak_state_into_each_other(
    client, public_cases, stub_interpretation
):
    """Interleave two scenarios; each must still return its own answer."""
    first, second = public_cases[0], public_cases[6]
    by_id = {case["input"]["scenario_id"]: case for case in (first, second)}
    stub_interpretation(lambda request: reference_directives(by_id[request.scenario_id]))

    for _ in range(3):
        for case in (first, second, second, first):
            body = client.post("/optimize-energy", json=case["input"]).json()
            assert body["scenario_id"] == case["input"]["scenario_id"]
            assert body["total_cost_bdt"] == pytest.approx(
                case["expected_output"]["total_cost_bdt"], abs=0.01
            )
