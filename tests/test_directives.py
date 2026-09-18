"""Guardrail and Directive Compiler tests (plan SS7, 7.1, 7.2, 10.2)."""

from __future__ import annotations

import pytest

from app.directives import (
    SOLAR_PASS_MIN,
    SOLAR_PASS_PRODUCT,
    compile_limits,
    has_overlapping_solar_reduction,
    validate_interpretation,
)
from app.schemas import DirectiveType
from tests.conftest import (
    battery,
    max_grid,
    no_charge,
    no_discharge,
    no_op,
    reserve,
    solar_reduction,
)

# ---------------------------------------------------------------------------
# Guardrails on untrusted model output (PS S08, plan S7.2)
# ---------------------------------------------------------------------------


def entry(**overrides):
    base = {
        "note_index": 0,
        "applies": True,
        "directive_type": "solar_reduction",
        "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
        "explanation": "solar drops",
    }
    base.update(overrides)
    return base


def test_valid_entry_is_accepted():
    result = validate_interpretation([entry()], 1, battery())
    assert result.complete(1)
    directive = result.directives[0]
    assert directive.directive_type is DirectiveType.SOLAR_REDUCTION
    assert directive.adjustment == {"hours": [13, 14], "factor": 0.2}


def test_no_op_requires_false_applies_and_null_adjustment():
    ok = validate_interpretation(
        [entry(directive_type="no_op", applies=False, structured_adjustment=None)],
        1,
        battery(),
    )
    assert ok.complete(1)
    assert ok.directives[0].directive_type is DirectiveType.NO_OP

    for bad in (
        entry(directive_type="no_op", applies=True, structured_adjustment=None),
        entry(directive_type="no_op", applies=False, structured_adjustment={"hours": [1]}),
    ):
        result = validate_interpretation([bad], 1, battery())
        assert 0 in result.entry_problems


def test_operational_directive_must_use_applies_true():
    result = validate_interpretation([entry(applies=False)], 1, battery())
    assert 0 in result.entry_problems


def test_unsupported_directive_type_is_rejected_not_substituted():
    result = validate_interpretation([entry(directive_type="load_shift")], 1, battery())
    assert 0 in result.entry_problems
    assert 0 not in result.directives


@pytest.mark.parametrize("factor", [-0.1, 1.5, float("nan"), float("inf"), "0.2", None])
def test_out_of_range_or_non_numeric_factor_is_rejected(factor):
    result = validate_interpretation(
        [entry(structured_adjustment={"hours": [13], "factor": factor})], 1, battery()
    )
    assert 0 in result.entry_problems


def test_reserve_above_capacity_is_rejected():
    result = validate_interpretation(
        [
            entry(
                directive_type="minimum_battery_reserve",
                structured_adjustment={"hours": [18], "minimum_energy_kwh": 5_000},
            )
        ],
        1,
        battery(capacity_kwh=200),
    )
    assert 0 in result.entry_problems


def test_negative_grid_cap_is_rejected():
    result = validate_interpretation(
        [
            entry(
                directive_type="max_grid_window",
                structured_adjustment={"hours": [18], "max_grid_kwh": -5},
            )
        ],
        1,
        battery(),
    )
    assert 0 in result.entry_problems


@pytest.mark.parametrize("hours", [[24], [-1], [1.5], ["13"], [True]])
def test_out_of_range_or_non_integer_hours_are_rejected(hours):
    result = validate_interpretation(
        [entry(structured_adjustment={"hours": hours, "factor": 0.2})], 1, battery()
    )
    assert 0 in result.entry_problems


def test_empty_hours_on_applicable_directive_triggers_repair_not_no_op():
    """Plan S7.2: never silently applies nothing, and never becomes no_op."""
    result = validate_interpretation(
        [entry(structured_adjustment={"hours": [], "factor": 0.2})], 1, battery()
    )
    assert 0 in result.entry_problems
    assert 0 not in result.directives
    assert not result.complete(1)


def test_unordered_and_duplicated_hours_are_normalised():
    """Ordering normalisation preserves meaning for set-semantics directives."""
    result = validate_interpretation(
        [entry(structured_adjustment={"hours": [14, 13, 13], "factor": 0.2})],
        1,
        battery(),
    )
    assert result.directives[0].adjustment["hours"] == [13, 14]


def test_unknown_adjustment_keys_are_dropped_from_the_emitted_shape():
    result = validate_interpretation(
        [
            entry(
                structured_adjustment={
                    "hours": [13],
                    "factor": 0.2,
                    "confidence": 0.9,
                }
            )
        ],
        1,
        battery(),
    )
    assert result.directives[0].adjustment == {"hours": [13], "factor": 0.2}


def test_duplicate_note_index_makes_mapping_unreliable():
    """Plan S6.2: regenerate the set rather than guess which entry is which."""
    result = validate_interpretation([entry(), entry()], 2, battery())
    assert not result.mapping_reliable


def test_missing_note_is_a_targeted_problem_not_a_dropped_note():
    result = validate_interpretation([entry(note_index=0)], 2, battery())
    assert result.mapping_reliable
    assert 1 in result.entry_problems
    assert not result.complete(2)


def test_one_bad_entry_beside_a_good_one_keeps_the_good_one():
    result = validate_interpretation(
        [entry(note_index=0), entry(note_index=1, directive_type="teleport")],
        2,
        battery(),
    )
    assert 0 in result.directives
    assert 1 in result.entry_problems
    assert result.mapping_reliable


def test_non_list_payload_is_unreliable():
    assert not validate_interpretation({"oops": 1}, 1, battery()).mapping_reliable


# ---------------------------------------------------------------------------
# Compiler: each directive changes precisely the intended bound (plan S10.2)
# ---------------------------------------------------------------------------

SOLAR = [100.0] * 24


def test_no_directive_preserves_the_original_forecast():
    limits = compile_limits([], SOLAR, battery())
    assert limits.effective_solar == SOLAR
    assert limits.maximum_grid == [None] * 24
    assert limits.minimum_reserve == [40.0] * 24


def test_reserve_directive_raises_only_the_listed_hours():
    limits = compile_limits([reserve(0, [18, 19], 120)], SOLAR, battery())
    assert limits.minimum_reserve[18] == 120
    assert limits.minimum_reserve[19] == 120
    assert limits.minimum_reserve[17] == 40
    assert limits.minimum_reserve[20] == 40


def test_reserve_never_lowers_the_base_reserve():
    limits = compile_limits([reserve(0, [18], 10)], SOLAR, battery(minimum_energy_kwh=40))
    assert limits.minimum_reserve[18] == 40


def test_overlapping_reserves_take_the_greatest():
    limits = compile_limits(
        [reserve(0, [18, 19], 90), reserve(1, [19, 20], 130)], SOLAR, battery()
    )
    assert limits.minimum_reserve[18] == 90
    assert limits.minimum_reserve[19] == 130
    assert limits.minimum_reserve[20] == 130


def test_no_charge_window_zeroes_only_charging():
    limits = compile_limits([no_charge(0, [2, 3])], SOLAR, battery())
    assert limits.maximum_charge[2] == 0
    assert limits.maximum_charge[3] == 0
    assert limits.maximum_charge[4] == 50
    assert limits.maximum_discharge[2] == 50


def test_no_discharge_window_zeroes_only_discharging():
    limits = compile_limits([no_discharge(0, [18])], SOLAR, battery())
    assert limits.maximum_discharge[18] == 0
    assert limits.maximum_charge[18] == 50


def test_both_prohibitions_force_the_battery_to_idle():
    limits = compile_limits([no_charge(0, [5]), no_discharge(1, [5])], SOLAR, battery())
    assert limits.maximum_charge[5] == 0
    assert limits.maximum_discharge[5] == 0


def test_overlapping_grid_caps_take_the_smallest():
    limits = compile_limits(
        [max_grid(0, [19, 20], 180), max_grid(1, [20, 21], 150)], SOLAR, battery()
    )
    assert limits.maximum_grid[19] == 180
    assert limits.maximum_grid[20] == 150
    assert limits.maximum_grid[21] == 150
    assert limits.maximum_grid[18] is None


def test_no_op_changes_nothing():
    limits = compile_limits([no_op(0)], SOLAR, battery())
    baseline = compile_limits([], SOLAR, battery())
    assert limits.effective_solar == baseline.effective_solar
    assert limits.minimum_reserve == baseline.minimum_reserve
    assert limits.maximum_charge == baseline.maximum_charge
    assert limits.maximum_discharge == baseline.maximum_discharge
    assert limits.maximum_grid == baseline.maximum_grid


# ---------------------------------------------------------------------------
# Overlapping solar reductions: every test plan S7.1 requires by name
# ---------------------------------------------------------------------------


def test_overlap_multiplies_factors():
    """0.8 and 0.5 over 100 kWh yields 40 kWh, the value plan S7.1 states."""
    limits = compile_limits(
        [solar_reduction(0, [12], 0.8), solar_reduction(1, [12], 0.5)], SOLAR, battery()
    )
    assert limits.effective_solar[12] == pytest.approx(40.0)
    assert limits.solar_pass == SOLAR_PASS_PRODUCT


def test_overlap_is_independent_of_note_order():
    forward = compile_limits(
        [solar_reduction(0, [12], 0.8), solar_reduction(1, [12], 0.5)], SOLAR, battery()
    )
    reversed_ = compile_limits(
        [solar_reduction(0, [12], 0.5), solar_reduction(1, [12], 0.8)], SOLAR, battery()
    )
    assert forward.effective_solar == reversed_.effective_solar


def test_duplicate_factors_apply_twice_under_the_product_rule():
    """Explicitly asserted value, not merely compiler/validator agreement."""
    limits = compile_limits(
        [solar_reduction(0, [12], 0.5), solar_reduction(1, [12], 0.5)], SOLAR, battery()
    )
    assert limits.effective_solar[12] == pytest.approx(25.0)


def test_factor_zero_yields_zero_solar():
    limits = compile_limits(
        [solar_reduction(0, [12], 0.0), solar_reduction(1, [12], 0.5)], SOLAR, battery()
    )
    assert limits.effective_solar[12] == 0.0


def test_factor_one_leaves_another_factor_unchanged():
    limits = compile_limits(
        [solar_reduction(0, [12], 1.0), solar_reduction(1, [12], 0.3)], SOLAR, battery()
    )
    assert limits.effective_solar[12] == pytest.approx(30.0)


def test_disjoint_hours_remain_independent():
    limits = compile_limits(
        [solar_reduction(0, [10], 0.5), solar_reduction(1, [14], 0.2)], SOLAR, battery()
    )
    assert limits.effective_solar[10] == pytest.approx(50.0)
    assert limits.effective_solar[14] == pytest.approx(20.0)
    assert limits.effective_solar[12] == pytest.approx(100.0)


def test_min_pass_relaxes_to_the_least_restrictive_factor():
    limits = compile_limits(
        [solar_reduction(0, [12], 0.8), solar_reduction(1, [12], 0.5)],
        SOLAR,
        battery(),
        SOLAR_PASS_MIN,
    )
    assert limits.effective_solar[12] == pytest.approx(50.0)
    assert limits.solar_pass == SOLAR_PASS_MIN


def test_overlap_detection():
    assert has_overlapping_solar_reduction(
        [solar_reduction(0, [12], 0.8), solar_reduction(1, [12], 0.5)]
    )
    assert not has_overlapping_solar_reduction(
        [solar_reduction(0, [12], 0.8), solar_reduction(1, [13], 0.5)]
    )
    assert not has_overlapping_solar_reduction([solar_reduction(0, [12], 0.8)])


def test_effective_solar_never_goes_negative():
    limits = compile_limits([], [-5.0] * 24, battery())
    assert all(value >= 0 for value in limits.effective_solar)
