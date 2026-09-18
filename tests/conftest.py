"""Shared fixtures and builders for the test suite."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.directives import Directive
from app.schemas import BatteryInput, DirectiveType, ScenarioRequest

PACK_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "public_cases.json"


@pytest.fixture(scope="session")
def public_pack() -> dict:
    return json.loads(PACK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def public_cases(public_pack: dict) -> list[dict]:
    return public_pack["cases"]


def battery(
    capacity_kwh: float = 200.0,
    initial_energy_kwh: float = 100.0,
    minimum_energy_kwh: float = 40.0,
    max_charge_kwh_per_hour: float = 50.0,
    max_discharge_kwh_per_hour: float = 50.0,
) -> BatteryInput:
    return BatteryInput(
        capacity_kwh=capacity_kwh,
        initial_energy_kwh=initial_energy_kwh,
        minimum_energy_kwh=minimum_energy_kwh,
        max_charge_kwh_per_hour=max_charge_kwh_per_hour,
        max_discharge_kwh_per_hour=max_discharge_kwh_per_hour,
    )


def solar_reduction(note_index: int, hours: list[int], factor: float) -> Directive:
    return Directive(
        note_index=note_index,
        applies=True,
        directive_type=DirectiveType.SOLAR_REDUCTION,
        adjustment={"hours": hours, "factor": factor},
        explanation="test",
    )


def reserve(note_index: int, hours: list[int], minimum_energy_kwh: float) -> Directive:
    return Directive(
        note_index=note_index,
        applies=True,
        directive_type=DirectiveType.MINIMUM_BATTERY_RESERVE,
        adjustment={"hours": hours, "minimum_energy_kwh": minimum_energy_kwh},
        explanation="test",
    )


def no_charge(note_index: int, hours: list[int]) -> Directive:
    return Directive(
        note_index=note_index,
        applies=True,
        directive_type=DirectiveType.NO_CHARGE_WINDOW,
        adjustment={"hours": hours},
        explanation="test",
    )


def no_discharge(note_index: int, hours: list[int]) -> Directive:
    return Directive(
        note_index=note_index,
        applies=True,
        directive_type=DirectiveType.NO_DISCHARGE_WINDOW,
        adjustment={"hours": hours},
        explanation="test",
    )


def max_grid(note_index: int, hours: list[int], max_grid_kwh: float) -> Directive:
    return Directive(
        note_index=note_index,
        applies=True,
        directive_type=DirectiveType.MAX_GRID_WINDOW,
        adjustment={"hours": hours, "max_grid_kwh": max_grid_kwh},
        explanation="test",
    )


def no_op(note_index: int) -> Directive:
    return Directive(
        note_index=note_index,
        applies=False,
        directive_type=DirectiveType.NO_OP,
        adjustment=None,
        explanation="test",
    )


def scenario(
    demand: list[float] | float = 100.0,
    solar: list[float] | float = 0.0,
    tariff: list[float] | float = 10.0,
    scenario_id: str = "TEST-01",
    notes: list[str] | None = None,
    batt: BatteryInput | None = None,
) -> ScenarioRequest:
    """Build a syntactically valid 24-hour request from scalars or per-hour lists."""
    def spread(value: list[float] | float) -> list[float]:
        return list(value) if isinstance(value, list) else [float(value)] * 24

    demands, solars, tariffs = spread(demand), spread(solar), spread(tariff)
    batt = batt or battery()
    return ScenarioRequest(
        scenario_id=scenario_id,
        operator_notes=notes or ["a test note"],
        hours=[
            {
                "hour": h,
                "demand_kwh": demands[h],
                "solar_kwh": solars[h],
                "tariff_bdt_per_kwh": tariffs[h],
            }
            for h in range(24)
        ],
        battery=batt,
    )
