"""An independent optimal-cost reference, for cross-checking the LP (plan S8.3).

This is a dynamic program over a discretised battery state. It shares no code
with `app.optimizer` - no PuLP, no linear algebra, no common formulation - so
agreement between the two is real evidence rather than two views of one bug.

Exactness. The scheduling problem is a min-cost flow on a time-expanded network:
each hour is a node, the battery is a storage arc between consecutive hours, and
grid and solar are source arcs. Its constraint matrix is totally unimodular, so
when every input is an integer multiple of `step` there is an optimal solution
whose battery levels are multiples of `step` too, and this DP finds it. Feed it
non-integral data and it returns an upper bound instead, which is still a valid
one-sided check.

Solar handling: with a non-negative tariff, using every available kWh of free
solar before buying is optimal, so the DP fills solar greedily. That assumption
is why `generated_case` only produces non-negative tariffs.
"""

from __future__ import annotations

import math

INFINITY = float("inf")


def dp_optimal_cost(
    demand: list[float],
    effective_solar: list[float],
    tariff: list[float],
    capacity: float,
    initial: float,
    minimum_reserve: list[float],
    max_charge: list[float],
    max_discharge: list[float],
    max_grid: list[float | None],
    step: float,
) -> float:
    """Minimum total cost, or +inf when no schedule exists.

    Every argument is in the same units the LP uses, and `step` is the grid
    resolution for the battery state.
    """
    levels = int(round(capacity / step)) + 1
    start = int(round(initial / step))
    if not 0 <= start < levels:
        return INFINITY

    # cost[level] = cheapest way to reach this level at the current hour.
    cost = [INFINITY] * levels
    cost[start] = 0.0

    for hour in range(24):
        nxt = [INFINITY] * levels
        floor = int(math.ceil(minimum_reserve[hour] / step - 1e-9))
        charge_steps = int(round(max_charge[hour] / step))
        discharge_steps = int(round(max_discharge[hour] / step))
        cap = max_grid[hour]

        for level in range(levels):
            if cost[level] == INFINITY:
                continue
            for delta in range(-discharge_steps, charge_steps + 1):
                target = level + delta
                if not 0 <= target < levels or target < floor:
                    continue

                draw = demand[hour] + delta * step
                if draw < -1e-9:
                    # Discharging more than the hour can absorb would require
                    # exporting, which PS S9.4 excludes.
                    continue
                solar_used = min(effective_solar[hour], max(0.0, draw))
                grid = max(0.0, draw - solar_used)
                if cap is not None and grid > cap + 1e-9:
                    continue

                candidate = cost[level] + grid * tariff[hour]
                if candidate < nxt[target]:
                    nxt[target] = candidate
        cost = nxt

    # PS S9.6: the day must end where it started.
    return cost[start]
