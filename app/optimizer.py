"""Linear program and response construction (PS S9; plan S8).

The model follows PS S9 exactly and adds nothing to it: no degradation cost, no
round-trip efficiency, no export revenue, no peak penalty. The published
objective is the whole objective::

    minimize  sum(grid[h] * tariff[h])  for h = 0..23

One signed continuous variable per hour carries the battery action, which keeps
the program a pure LP - no binaries - while making a simultaneous charge and
discharge structurally impossible (plan S8.2)::

    b[h] > 0  charge        b[h] < 0  discharge        b[h] = 0  idle

    -max_discharge[h] <= b[h] <= max_charge[h]
    E[h] = E[h-1] + b[h],   E[-1] = initial,   E[23] = initial
    grid[h] + solar_used[h] = demand[h] + b[h]

which is PS S9.5 rewritten: ``grid + solar_used + max(0,-b) = demand + max(0,b)``
reduces to ``grid + solar_used = demand + b``.

All 24 hours are solved together so the plan can prepare for later grid caps,
reserve windows, price changes, and end-of-day neutrality. An hour-by-hour
greedy rule cannot make those guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass

import pulp

from app.directives import (
    SOLAR_PASS_MIN,
    SOLAR_PASS_PRODUCT,
    CompiledLimits,
    Directive,
    compile_limits,
    has_overlapping_solar_reduction,
)
from app.schemas import BatteryAction, BatteryInput, HourlyPlanEntry

#: Output precision. Well inside the published 0.01 tolerance, and coarse enough
#: to drop solver noise without hiding a real violation.
OUTPUT_DECIMALS = 6

#: Below this magnitude a battery action is treated as no action at all, so that
#: an "idle" hour reports exactly 0 as PS S10.3 requires.
IDLE_EPS = 1e-7


_SOLVER_CACHE: list[object] = []


def build_solver(time_limit_s: float) -> object:
    """Pick a working CBC front end once, then reuse that choice.

    The image carries CBC twice on purpose (plan S12): the binary bundled in the
    PuLP wheel and Debian's `coinor-cbc`. `PULP_CBC_CMD` drives the former and
    `COIN_CMD` the latter, so trying both means an architecture where the
    bundled binary will not execute still gets a solver instead of a 500.
    """
    limit = max(1, int(time_limit_s))
    if _SOLVER_CACHE:
        return _SOLVER_CACHE[0](msg=False, timeLimit=limit)  # type: ignore[operator]

    for factory in (pulp.PULP_CBC_CMD, pulp.COIN_CMD):
        try:
            candidate = factory(msg=False, timeLimit=limit)
            if candidate.available():
                _SOLVER_CACHE.append(factory)
                return candidate
        except Exception:  # pragma: no cover - depends on the host
            continue
    raise SolverError("no CBC solver is available")


class InfeasibleScheduleError(Exception):
    """No schedule satisfies the request together with its directives."""


class SolverError(Exception):
    """The solver did not return a proven optimum."""


@dataclass(frozen=True)
class Solution:
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    solver_status: str


def _round(value: float) -> float:
    rounded = round(value, OUTPUT_DECIMALS)
    # Normalise -0.0, which is numerically fine but reads as an error.
    return 0.0 if rounded == 0 else rounded


def solve(
    demand: list[float],
    tariff: list[float],
    battery: BatteryInput,
    limits: CompiledLimits,
    time_limit_s: float = 10.0,
) -> Solution:
    """Solve the 24-hour problem and build a self-consistent schedule.

    Raises :class:`InfeasibleScheduleError` when the compiled model admits no
    schedule, and :class:`SolverError` when the solver stops without proving
    optimality. Neither is reported as a successful plan.
    """
    problem = pulp.LpProblem("gridwise", pulp.LpMinimize)

    grid = [
        pulp.LpVariable(f"grid_{h}", lowBound=0, upBound=limits.maximum_grid[h])
        for h in range(24)
    ]
    solar_used = [
        pulp.LpVariable(f"solar_{h}", lowBound=0, upBound=limits.effective_solar[h])
        for h in range(24)
    ]
    action = [
        pulp.LpVariable(
            f"b_{h}",
            lowBound=-limits.maximum_discharge[h],
            upBound=limits.maximum_charge[h],
        )
        for h in range(24)
    ]
    energy = [
        pulp.LpVariable(
            f"e_{h}",
            lowBound=limits.minimum_reserve[h],
            upBound=battery.capacity_kwh,
        )
        for h in range(24)
    ]

    problem += pulp.lpSum(grid[h] * tariff[h] for h in range(24)), "total_cost_bdt"

    initial = float(battery.initial_energy_kwh)
    for h in range(24):
        previous = initial if h == 0 else energy[h - 1]
        problem += energy[h] == previous + action[h], f"state_{h}"
        problem += grid[h] + solar_used[h] == demand[h] + action[h], f"balance_{h}"
    # PS S9.6: the starting charge may be moved between hours but not consumed.
    problem += energy[23] == initial, "neutrality"

    status_code = problem.solve(build_solver(time_limit_s))
    status = pulp.LpStatus[status_code]

    if status == "Infeasible":
        raise InfeasibleScheduleError("no schedule satisfies the applied directives")
    if status != "Optimal":
        # A merely feasible or interrupted result is never labelled optimal
        # (plan S8.3).
        raise SolverError(f"solver returned status {status!r}")

    raw_actions = [float(pulp.value(action[h]) or 0.0) for h in range(24)]
    raw_solar = [float(pulp.value(solar_used[h]) or 0.0) for h in range(24)]

    return _build_plan(demand, tariff, battery, limits, raw_actions, raw_solar, status)


def _build_plan(
    demand: list[float],
    tariff: list[float],
    battery: BatteryInput,
    limits: CompiledLimits,
    raw_actions: list[float],
    raw_solar: list[float],
    status: str,
) -> Solution:
    """Turn solver values into a schedule whose reported numbers are consistent.

    The returned rows - not the solver's internal values - are what the judge
    replays, so they are constructed to satisfy the rules exactly rather than
    merely transcribed:

    * near-zero actions snap to 0 so an ``idle`` hour reports exactly 0 magnitude
    * the battery state chain is re-derived by forward simulation from the
      supplied initial energy, so no accumulated solver drift survives
    * the final state is pinned to the initial energy, restoring neutrality
    * ``grid`` is then derived from the energy balance, so the balance holds by
      construction rather than by luck

    Every adjustment is bounded by solver noise, and the Replay Validator checks
    the result independently afterwards - a plan that fails replay is never
    returned (plan S9).
    """
    actions: list[float] = []
    for h in range(24):
        value = raw_actions[h]
        if abs(value) < IDLE_EPS:
            value = 0.0
        value = min(value, limits.maximum_charge[h])
        value = max(value, -limits.maximum_discharge[h])
        actions.append(_round(value))

    # Re-derive the state chain, then absorb any residual drift into the last
    # hour so end-of-day neutrality is exact.
    initial = float(battery.initial_energy_kwh)
    running = initial
    for h in range(23):
        running += actions[h]
    final_action = _round(initial - running)
    final_action = min(final_action, limits.maximum_charge[23])
    final_action = max(final_action, -limits.maximum_discharge[23])
    actions[23] = final_action

    energy_after: list[float] = []
    running = initial
    for h in range(24):
        running += actions[h]
        energy_after.append(_round(running))

    rows: list[HourlyPlanEntry] = []
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for h in range(24):
        action_value = actions[h]

        # Solar is free but never mandatory; the solver's choice is kept and
        # only clamped to the effective availability.
        solar = min(max(raw_solar[h], 0.0), limits.effective_solar[h])
        solar = _round(solar)

        grid_value = demand[h] + action_value - solar
        if grid_value < 0:
            # Only reachable from rounding noise: give back the excess solar
            # rather than reporting a negative import.
            solar = _round(max(0.0, solar + grid_value))
            grid_value = demand[h] + action_value - solar
        grid_value = _round(max(0.0, grid_value))

        if action_value > 0:
            battery_action = BatteryAction.CHARGE
            magnitude = action_value
        elif action_value < 0:
            battery_action = BatteryAction.DISCHARGE
            magnitude = -action_value
        else:
            battery_action = BatteryAction.IDLE
            magnitude = 0.0

        rows.append(
            HourlyPlanEntry(
                hour=h,
                grid_kwh=grid_value,
                solar_used_kwh=solar,
                battery_action=battery_action,
                battery_kwh=_round(magnitude),
                battery_energy_after_kwh=energy_after[h],
            )
        )

        total_grid += grid_value
        total_cost += grid_value * tariff[h]
        peak_grid = max(peak_grid, grid_value)

    return Solution(
        hourly_plan=rows,
        total_grid_kwh=_round(total_grid),
        total_cost_bdt=_round(total_cost),
        peak_grid_kwh=_round(peak_grid),
        solver_status=status,
    )


def solve_with_solar_policy(
    directives: list[Directive],
    demand: list[float],
    solar: list[float],
    tariff: list[float],
    battery: BatteryInput,
    time_limit_s: float = 10.0,
) -> tuple[Solution, CompiledLimits]:
    """Compile and solve under the documented solar-overlap policy (plan S7.1).

    Overlapping solar reductions multiply by default. Over-curtailing is not
    risk-free, though: with a `max_grid_window` also active in an overlapped
    hour and too little battery to cover the gap, the stricter ceiling can make
    an otherwise feasible scenario unsolvable, which is just as fatal as solar
    overuse. So an infeasible first pass relaxes the overlapped hours once to
    `min(factors)` and re-solves.

    The relaxation is attempted only when an overlap actually exists - without
    one the two passes are identical and retrying would just hide a genuine
    infeasibility. Both passes are deterministic and order-independent; the pass
    that produced the returned plan is recorded on the returned limits.
    """
    limits = compile_limits(directives, solar, battery, SOLAR_PASS_PRODUCT)
    try:
        return solve(demand, tariff, battery, limits, time_limit_s), limits
    except InfeasibleScheduleError:
        if not has_overlapping_solar_reduction(directives):
            raise
    relaxed = compile_limits(directives, solar, battery, SOLAR_PASS_MIN)
    return solve(demand, tariff, battery, relaxed, time_limit_s), relaxed
