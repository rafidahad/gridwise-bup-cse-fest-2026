# GridWise: High-Level Solution Plan

**BUP CSE Fest 2026 — Hackathon Online Preliminary**  
**Approach:** FastAPI + Pydantic + LLM structured output + deterministic Directive Compiler + PuLP/CBC + independent Replay Validator + Docker.

This plan covers the supplied challenge requirements and the proposed solution. It is organized around correctness and completion criteria, without a build-time budget. The official round window is 7:00 PM–11:00 PM, but that event fact does not limit the planning or engineering effort described here. Runtime limits imposed by the judge still apply.

## 1. Source of truth and scope

| Source | Authority |
|---|---|
| `problemstatement.pdf`, 9 pages | Canonical challenge behavior: directives, schemas, guardrails, energy accounting, optimization, and validity. |
| `guide.pdf`, 11 pages | Canonical participation, repository, deployment, submission, performance, scoring, penalties, and tie-break rules. |
| `BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json`, version 2.0 | Ten worked public examples with expected interpretations and reference schedules; not the hidden judge set. |

References below use **PS §** for Problem Statement sections and **Guide §** for Participant Guide sections. Requirements are distinguished from proposed implementation decisions. Where the documents leave an issue unresolved, this plan records it rather than inventing an official rule.

The complete PDFs and sample pack were reviewed. The ten reference schedules passed the energy-balance, battery-transition, directive-limit, neutrality, and aggregate checks performed during planning. Their claimed optimal costs have not yet been independently verified by a new optimizer.

## 2. What the service must accomplish

**Requirements — PS §§1–5:** Build one public HTTP API service that receives the next 24 hours of synthetic campus demand, solar availability, electricity tariffs, battery settings, and 1–3 natural-language operator notes.

For each request, the service must:

1. Interpret every note using an LLM or another language-capable generative model.
2. Map each note to exactly one supported directive or `no_op`.
3. Validate the structured interpretation deterministically.
4. Apply all relevant directives to the scheduling problem.
5. Produce a valid 24-hour schedule satisfying energy and battery rules.
6. Minimize total grid electricity cost within those constraints.
7. Replay the completed schedule and return the exact response contract.

All supplied scenarios are synthetic. No live campus, utility, billing, or personal data is required or permitted by the guide's data policy.

**Design objective:** A small, inspectable service in which language understanding, constraint construction, optimization, and verification have separate responsibilities. The distinctive features are the Directive Compiler and Replay Validator, supported by clear evidence that the complete pipeline works.

## 3. Architecture and stack

```text
POST /optimize-energy
        |
        v
FastAPI + Pydantic: validate the request
        |
        v
LLM: interpret all notes into structured directives
        |
        v
Directive Compiler: validate directives and create hourly limits
        |
        v
PuLP + CBC: solve the complete 24-hour optimization problem
        |
        v
Response builder: convert actions and calculate reported values
        |
        v
Replay Validator: independently check the response to be returned
        |
        v
Exact JSON response
```

| Component | Proposed choice | Responsibility |
|---|---|---|
| Language | Python 3.12 | Application and deterministic logic. |
| API | FastAPI + Uvicorn | Required endpoints, JSON handling, controlled errors. |
| Schemas | Pydantic v2 | Request, directive, and response validation. |
| Interpretation | Primary structured-output LLM plus a tested backup language model | One primary call on the normal path; bounded repair or failover when needed. |
| Directive Compiler | Plain Python | Convert validated interpretations into hourly limits. |
| Optimization | PuLP + CBC | Minimum-cost feasible schedule. |
| Replay Validator | Independent plain Python checks | Verify actual response values and recompute totals. |
| Tests | pytest + httpx | Contract, interpretation, optimization, failure, and integration testing. |
| Configuration | Environment variables and a settings module | Primary/backup credentials and model identifiers, supported inference settings, timeouts, service port, solver configuration. |
| Deployment | Docker | Reproducible public service and pullable fallback image. |

Start with one service and no required database, queue, frontend, or agent framework. Pin the actual tested dependency versions and ensure CBC is installed and executable inside the image. Choose the model through measured interpretation accuracy and latency; this plan does not claim a particular model has already passed those measurements.

## 4. Exact API contract

**Requirements — PS §6:**

| Endpoint | Behavior |
|---|---|
| `GET /health` | HTTP 200 and `{"status":"ok"}` when ready. |
| `POST /optimize-energy` | Accept one scenario JSON object; return one interpretation-and-plan JSON object. |

| HTTP status | Required meaning |
|---|---|
| 200 | Successful health or optimization response. |
| 400 | Malformed JSON or structurally invalid request. |
| 422 | Optional for a semantically invalid but well-formed request. |
| 500 | Controlled internal error, without secrets or raw stack traces. |

**Implementation plan:** Explicitly configure validation error handling to meet this contract. Do not assume framework defaults match the required status codes. Return controlled JSON errors; an error must never masquerade as a successful plan. The documents do not define an exact error-body schema, so select and document a minimal consistent format.

### 4.1 Request fields

**Requirements — PS §7:**

| Top-level field | Type | Meaning |
|---|---|---|
| `scenario_id` | string | Synthetic scenario identifier; echoed in the response. |
| `operator_notes` | array of 1–3 strings | Non-empty natural-language notes about the same 24-hour scenario. |
| `hours` | array of exactly 24 objects | Exactly one entry for every hour 0–23. |
| `battery` | object | Battery capacity, starting energy, reserve, and rate limits. |

| Field in each `hours` entry | Type | Meaning |
|---|---|---|
| `hour` | integer | Unique hour from 0 through 23. |
| `demand_kwh` | number | Demand to supply during the hour. |
| `solar_kwh` | number | Original solar availability before directives. |
| `tariff_bdt_per_kwh` | number | Grid price during the hour. |

| Field in `battery` | Meaning |
|---|---|
| `capacity_kwh` | Maximum stored energy. |
| `initial_energy_kwh` | Energy at the start of hour 0. |
| `minimum_energy_kwh` | Base minimum reserve. |
| `max_charge_kwh_per_hour` | Maximum energy added in one hour. |
| `max_discharge_kwh_per_hour` | Maximum energy removed in one hour. |

**Validation plan:** Check required fields, types, hour uniqueness and coverage, finite numeric values, and non-empty notes. Preserve legitimate zero and fractional values. Do not impose arbitrary positive minimums or magnitude limits absent from the specification. Index input rows by `hour`; do not assume an unstated input ordering requirement. Document any additional input-policy decisions.

**Explicit battery feasibility checks:** Require non-negative capacity, base reserve, initial energy, and charge/discharge limits. For a feasible scenario, the base reserve cannot exceed capacity, and initial energy must lie between the base reserve and capacity. This last condition is derived from PS §§9.2 and 9.6, not an invented initial-state rule:

```text
minimum_energy_kwh <= E_after[23] <= capacity_kwh
E_after[23] = initial_energy_kwh
therefore minimum_energy_kwh <= initial_energy_kwh <= capacity_kwh
```

An initial energy materially below the base reserve cannot be repaired by charging in hour 0: returning to that initial level at hour 23 would violate the reserve. Distinguish the permanent base reserve from a higher temporary directive reserve; initial energy need not meet a future temporary reserve if the schedule can charge in time. Evaluate that future requirement through the optimizer.

Use the documented numeric comparison tolerance at boundaries, retain the original supplied values, and do not silently clamp the initial energy. Test values exactly on, just within tolerance of, and materially outside each boundary. A well-formed request that fails these semantic checks receives HTTP 422 under the plan's selected policy; malformed JSON or structural errors receive HTTP 400. Passing these necessary checks does not establish full feasibility, which still depends on hourly constraints and neutrality.

### 4.2 Successful response fields

**Requirements — PS §10:**

| Top-level field | Type | Meaning |
|---|---|---|
| `scenario_id` | string | Exactly the request identifier. |
| `directive_interpretation` | array | Exactly one interpretation per note, in `note_index` order. |
| `hourly_plan` | array of 24 objects | Exactly one plan entry for every hour 0–23. |
| `total_grid_kwh` | number | Sum of hourly grid imports. |
| `total_cost_bdt` | number | Sum of hourly grid import multiplied by its tariff. |
| `peak_grid_kwh` | number | Maximum hourly grid import. |
| `plan_summary` | string | Short explanation of the final strategy. |

| Interpretation field | Requirement |
|---|---|
| `note_index` | Zero-based input-note index. |
| `applies` | Boolean; false only for `no_op`. |
| `directive_type` | Exactly one supported enum value. |
| `structured_adjustment` | Required object for that type; null only for `no_op`. |
| `explanation` | Short explanation of the interpretation. |

| Hourly plan field | Requirement |
|---|---|
| `hour` | Integer 0–23. |
| `grid_kwh` | Non-negative grid energy purchased. |
| `solar_used_kwh` | Non-negative solar used, at most the effective availability. |
| `battery_action` | Exactly `charge`, `discharge`, or `idle`. |
| `battery_kwh` | Non-negative action magnitude; exactly zero for idle. |
| `battery_energy_after_kwh` | Battery energy after completing that hour. |

**Response plan:** Return hours in ascending order, validate the response schema, and derive the short summary from verified facts without a second LLM call. Keep debugging traces internal rather than adding unrequested response fields.

## 5. Supported directives and interpretation rules

**Requirements — PS §§4–5, 8:**

| Directive | Exact adjustment shape | Scheduling effect |
|---|---|---|
| `solar_reduction` | `{"hours":[...],"factor":number}` | Effective solar equals original solar multiplied by the remaining usable fraction for listed hours. |
| `minimum_battery_reserve` | `{"hours":[...],"minimum_energy_kwh":number}` | Energy after each listed hour must meet the greater of the base reserve and directive reserve. |
| `no_charge_window` | `{"hours":[...]}` | Charging equals zero in listed hours. |
| `no_discharge_window` | `{"hours":[...]}` | Discharging equals zero in listed hours. |
| `max_grid_window` | `{"hours":[...],"max_grid_kwh":number}` | Grid import in each listed hour must not exceed the cap. |
| `no_op` | `null` | No change to the optimization problem. |

Universal rules:

- Every note produces exactly one entry; indices must be `0..N-1` in order, with no missing or duplicate mappings.
- Each relevant directive has `applies = true`.
- An irrelevant note has `applies = false`, `directive_type = "no_op"`, and `structured_adjustment = null`.
- Hours are unique integers 0–23 in ascending order.
- Whole-hour windows are start-inclusive and end-exclusive: 1 PM–3 PM becomes `[13,14]`.
- Solar `factor` is the fraction remaining, finite and between 0 and 1 inclusive. An 80% reduction means `0.2`, not `0.8`.
- A directive reserve is finite, non-negative, and no greater than battery capacity.
- A grid cap is finite and non-negative.
- The LLM cannot invent demand, solar forecasts, tariffs, battery parameters, or unsupported directive types.
- Interpretations must be applied before scheduling. Correct extraction without application is still incorrect.
- Hidden scoring notes each map to one supported directive or `no_op`; valid scoring scenarios are feasible and do not require contradictory hard directives.

Examples to teach and test:

| Note meaning | Expected interpretation |
|---|---|
| Solar drops to 20% from 1 PM to 3 PM | `solar_reduction`, hours `[13,14]`, factor `0.2`. |
| Solar is reduced by 80% over that window | Same interpretation. |
| Solar is reduced to 80% | Remaining factor `0.8`. |
| Do not charge from 2 PM to 4 PM | `no_charge_window`, hours `[14,15]`. |
| Keep 120 kWh from 6 PM until 9 PM | Reserve 120 for hours `[18,19,20]`. |
| Keep 50% of a 200 kWh battery | Reserve 100 kWh for the stated hours. |
| The cafeteria menu changes tomorrow | `no_op`. |

## 6. LLM interpretation strategy

**Mandatory policy — Guide §4:** A language-capable generative model must interpret the notes, and its structured interpretation must produce the optimization constraints. Phrase matching alone is noncompliant. Using AI only for a summary or documentation does not satisfy the requirement.

**Proposed design:**

1. Interpret all 1–3 notes in one structured-output request.
2. Supply the allowed types, exact shapes, time conventions, percentage examples, and relevant scenario context, including battery capacity.
3. Treat note text as data to interpret, not instructions to change the service's behavior.
4. Keep the interpretation and explanations concise.
5. Validate the result deterministically before compilation.
6. Recover invalid model output through the per-note repair policy below; use the tested backup model when the primary provider fails or repair does not succeed.
7. Compile only after every note has a valid interpretation and the complete set passes validation. If bounded recovery fails, return a controlled error. Never silently discard notes, fabricate a directive, or use a phrase-matching substitute as the interpretation path.

Evaluate candidate models using independently labelled paraphrases, distractors, numeric conversions, and measured end-to-end latency. Record the chosen provider/model identifier, prompt version, and test results. Structured output guarantees neither semantic accuracy nor feasibility; those need separate checks.

### 6.1 Provider failover and bounded recovery

**Planned reliability feature, not an official requirement:** Include a tested backup language model in the initial delivery. Guide §4 permits backup or local models; controlled errors remain compliant but cannot replace a successful response on a valid scoring request.

Prefer a backup on an independent provider, or an already-running local model, to reduce shared outage and quota risks. A second model on the same provider may share those failures. Both models must perform genuine language interpretation, satisfy the same schema and guardrails, and pass the same semantic test suite. No long training, fine-tuning, or model download may be required during judging.

```text
Primary model succeeds and passes all checks -> compile and solve
Primary output has identifiable invalid entries -> one targeted repair attempt
Primary unavailable, or repair unsuccessful -> one backup-model attempt
Complete validated interpretation -> compile and solve
Recovery exhausted or deadline reached -> controlled JSON error
```

Use at most three application-level inference calls per request: primary, optional repair, and backup. On a primary timeout, connection failure, unusable provider response, or rate/quota failure, proceed directly to backup rather than repeatedly calling the same failing provider. Include any SDK retries in this budget. If time is insufficient for an optional repair plus a useful backup attempt and final solving/replay, skip the repair.

The normal path remains one LLM call. Configure per-call deadlines within one overall request deadline below 30 seconds, reserving time for the backup, solver, replay, and response. Cancel or disregard late responses from superseded attempts. Document measured settings; a backup call that cannot finish before the judge timeout is not effective recovery.

If all language-model paths fail, return a controlled HTTP 500 JSON error without a partial successful plan or secret-bearing provider details. This is the last resort, not the primary recovery strategy. Do not claim that fallback guarantees success or that one failure automatically forfeits every stability point; assess actual failure rate and scoring outcomes.

### 6.2 Per-note repair and complete-set validation

Validate model output at two levels: individual entries and complete note coverage/mapping. For reliably mapped entries that fail a local guardrail, retain the valid entries and ask the model to re-interpret only the failed notes. Supply the original failed-note text, original indices, relevant scenario context, interpretation rules, and concise validation feedback. Repair must infer the intended directive from the note, not merely fill fields to satisfy a schema.

If malformed output, duplicate indices, or missing mappings prevent reliable attribution, request a fresh complete interpretation instead of guessing which entry belongs to which note. The repair attempt still counts toward the same global call/deadline budget. Backup may repair the remaining reliably identified notes or regenerate the full set when mapping is unreliable.

After merging repairs, rerun all checks: exactly one entry per note, original index order, valid shapes and values, and compatibility with the original scenario. Locally valid entries are not proof of correct semantics or joint feasibility. An infeasible compiled model must never be made feasible by silently dropping a note or relaxing a hard constraint.

A single invalid directive should trigger recovery, not an immediate successful response with that directive removed. If any required interpretation remains unresolved after recovery, withhold the entire successful schedule and use the controlled failure response. The challenge defines no partial-success contract.

### 6.3 Inference settings and semantic consistency

Pin the model version/snapshot where available and record provider, model identifier, prompt version, schema version, and supported inference settings for both primary and backup. For models supporting sampling control, use a low temperature (preferably `0` for extraction) and keep other sampling settings fixed. Use a fixed seed only where the selected API/model supports it; do not send unsupported parameters. Document when a model does not expose temperature or seed controls.

These settings reduce avoidable variability; they do not guarantee deterministic model output. OpenAI describes seeded sampling as best effort rather than guaranteed determinism: [official API reference](https://developers.openai.com/api/reference/java/resources/completions/methods/create). Parameter availability must be checked for the actual selected model/API.

Run repeated exact-input tests and independently labelled paraphrase-cluster tests for both models and the failover path. Compare relevance, directive type, hours, numeric values, and downstream validity; do not require byte-identical explanations or one particular optimal action sequence. Pinning and low temperature do not substitute for semantic accuracy tests. Version or invalidate any interpretation cache when model, prompt, schema, relevant context, or interpretation policy changes.

## 7. Directive Compiler

**Purpose:** Convert validated structured directives into a small, explicit set of hourly bounds, without giving the LLM control over the optimizer.

The compiler produces:

- `effective_solar[h]`
- `minimum_reserve[h]`
- `maximum_charge[h]`
- `maximum_discharge[h]`
- `maximum_grid[h]`, with no extra cap where none is specified.

It preserves the original request and records which note changed each bound. This internal trace makes a failed test explainable without changing the public API.

Combining simultaneously applicable inequalities is deterministic: use the greatest reserve, the smallest grid cap, and the union of charging/discharging restrictions. If both action prohibitions apply in an hour, the battery must idle.

### 7.1 Defined policy for overlapping solar reductions

**Proposed runtime policy, pending any official clarification:** For each hour, multiply the remaining-solar factors of all applicable `solar_reduction` directives and apply the product once to the original forecast. With no reduction directive, use factor 1. If the resulting model is infeasible, relax that hour to the minimum applicable factor and re-solve.

```text
factors[h]           = all validated solar-reduction factors applying to hour h
effective_factor[h]  = product(factors[h]) if factors[h] is non-empty else 1
effective_solar[h]   = original_solar[h] * effective_factor[h]

if the compiled model is infeasible:
    effective_factor[h] = min(factors[h]) for every overlapped hour
    re-solve once; if still infeasible, apply normal safe-failure handling
```

For original solar of 100 kWh and overlapping factors 0.8 and 0.5, the service uses 40 kWh as the available-solar limit. Both note interpretations remain present, unchanged and in input-note order, in `directive_interpretation`; aggregation occurs only in the compiler.

**Why the product rather than the minimum.** The statement does not prescribe an overlap rule, so the choice must be made on which error the judge punishes. Effective-solar overuse invalidates the whole case and forfeits its optimization credit (Guide §09); using less solar than permitted is always legal, because unused solar is curtailed (PS §9.4) and the shortfall is met from grid or battery. The two candidate policies are therefore not symmetric:

| Our policy | If judge multiplies (limit 40) | If judge takes the minimum (limit 50) |
|---|---|---|
| Product — limit 40 | Exact match; valid | Under-uses available solar; **valid**, marginally higher cost |
| Minimum — limit 50 | Plan may draw 50 against a 40 ceiling; **case invalid** | Exact match; valid |

The product is the only option with no invalidating outcome, so it is the default. Last-note-wins is rejected outright because it makes the schedule depend on note order.

**Why the infeasibility fallback exists.** Over-curtailment is not risk-free: if a `max_grid_window` also applies in an overlapped hour and the battery cannot cover the gap, the stricter ceiling can make an otherwise feasible scenario unsolvable, which is equally fatal. The single relaxation step to `min(factors)` recovers exactly that case while keeping the strict ceiling everywhere else. Both passes are deterministic and order-independent; record which pass produced the returned plan in the internal trace.

The service must apply this policy immediately when an overlap occurs; it must not wait for an organizer response, ignore either note, or reject the request solely because solar reductions overlap.

Record the policy in the README and an internal note-to-limit trace, without adding public response fields. The Replay Validator must independently check the same documented semantics — including which pass was used — and tests must assert known expected solar limits rather than merely checking agreement between compiler and validator.

Required overlap tests: factors 0.8 and 0.5 yield 0.4; reversing note order leaves effective solar unchanged; duplicate factors apply twice under the product rule and the expected value is asserted explicitly; factor 0 yields zero; factor 1 leaves any other factor unchanged; disjoint hours remain independent; no directive preserves the original forecast. Include one integration case where overlapping restrictions still admit a feasible schedule, and one where the product is infeasible under a grid cap and the fallback to `min` produces a valid plan.

**Remaining risk:** This is a defined implementation policy, not a guarantee of matching unpublished judge semantics. The product is chosen because it cannot cause solar overuse under either reading, not because it is known to be the official rule; its cost is a slightly conservative schedule when the judge's rule is more permissive. Seek clarification before evaluation, but do not leave runtime behavior undefined while awaiting it. If an official rule arrives, update the compiler, independent replay checks, regression expectations, and documentation together.

### 7.2 Guardrail edges and rejection behavior

**Empty-hour policy:** The source specifies unique, ordered hours in range but does not explicitly state a minimum array length. As a documented semantic-validation policy, require at least one affected hour for each applicable operational directive. An LLM entry with `applies = true` and `hours = []` is sent for re-interpretation through Section 6.2; it is not accepted as a rule that silently affects nothing. Do not change it to `no_op` automatically. Only language interpretation establishing irrelevance may produce `no_op`, with a null adjustment. Record this policy as an interpretation of intended behavior rather than an explicit quoted schema requirement.

Other invalid directives must also be withheld from compilation and sent through the bounded repair/failover path in Sections 6.1–6.2. Here, “rejected” means rejected as a usable model result, not an immediate HTTP 400/422 against a valid client request. If recovery succeeds, process the complete scenario normally; if it fails, return the controlled model/internal-failure response. Never silently clip invalid factors, replace unsupported types, invent hours or numbers, or omit a note. Harmless ordering normalization is allowed only when it preserves the interpreted meaning.

## 8. Optimization model and best-solution strategy

**Required objective — PS §5.2:**

```text
minimize total_cost_bdt = sum(grid[h] * tariff[h]) for h = 0..23
```

Correctness takes precedence over cost. Grid energy and peak grid usage must be reported, but neither is an additional official optimization objective.

### 8.1 Required physical rules

**PS §9:**

```text
charge:    E_after = E_before + battery_kwh
discharge: E_after = E_before - battery_kwh
idle:      E_after = E_before and battery_kwh = 0

active_minimum[h] <= E_after[h] <= capacity

charge_amount[h] <= max_charge_kwh_per_hour
discharge_amount[h] <= max_discharge_kwh_per_hour

0 <= solar_used[h] <= effective_solar[h]
grid[h] >= 0

grid[h] + solar_used[h] + discharge_amount[h]
    = demand[h] + charge_amount[h]

E_before[0] = initial_energy_kwh
E_before[h] = E_after[h-1] for h > 0
E_after[23] = initial_energy_kwh
```

Unused solar is curtailed. Grid export is excluded. All applicable reserve, charging, discharging, and grid-cap directives are hard constraints. Restoring initial energy prevents using the starting battery as free one-time energy.

### 8.2 Proposed simple linear formulation

Use four continuous variables per hour: grid import, solar used, signed battery change, and energy after the hour.

```text
b[h] > 0: charge
b[h] < 0: discharge
b[h] = 0: idle

-maximum_discharge[h] <= b[h] <= maximum_charge[h]
E_after[h] = E_before[h] + b[h]
grid[h] + solar_used[h] = demand[h] + b[h]
```

This formulation follows the supplied lossless battery equations and produces one net action per hour without binary variables. Charging prohibition sets the upper bound on `b` to zero; discharging prohibition sets its lower bound to zero. On output, the action comes from the sign and the non-negative magnitude is `abs(b)`.

Solve all 24 hours together with PuLP/CBC. This allows the optimizer to prepare for later grid caps, reserve windows, price changes, solar surplus, and end-of-day restoration. A greedy hour-by-hour rule does not provide that guarantee.

### 8.3 Conditions for claiming the best solution

- Verify that the interpreted constraints are correct through separate language tests.
- Check solver termination status; do not label an interrupted or merely feasible result optimal.
- Replay the returned plan after serialization and numeric cleanup.
- Confirm public-case costs against references within tolerance; independently cross-check selected small or generated cases with a separate optimization method in tests.
- Do not add battery degradation costs, efficiency losses, export revenue, or weighted peak penalties absent from the official objective.
- Equivalent optimal schedules are acceptable; there is no need to reproduce reference actions byte-for-byte.

An optimal solver result establishes optimality for the model actually supplied. It cannot repair a mistaken natural-language interpretation or a missing constraint.

## 9. Independent Replay Validator and numeric handling

**Required final replay — PS §8; consistency checks — PS §11.**

The validator must independently inspect the final response using the original request and validated directives. It must not merely trust the solver's reported states or reuse all the same constraint-construction logic.

Checks:

1. Exact scenario echo, directive coverage/order, required fields, and allowed enums.
2. Exactly 24 unique plan hours covering 0–23.
3. Finite and non-negative public energy values and totals.
4. Valid battery action and zero magnitude for idle.
5. Reconstructed state transitions beginning at the supplied initial energy.
6. Capacity, base reserve, active directive reserve, and charge/discharge rates.
7. Effective solar reconstructed from the request and directives.
8. Every hourly energy balance.
9. Every no-charge, no-discharge, reserve, and grid-cap rule.
10. Final battery energy equal to initial energy.
11. Totals recalculated from the actual returned hourly rows.

```text
total_grid_kwh = sum(row.grid_kwh)
total_cost_bdt = sum(row.grid_kwh * tariff[row.hour])
peak_grid_kwh = max(row.grid_kwh)
```

The official absolute tolerance is 0.01 kWh or 0.01 BDT unless the judge package specifies a stricter value. Keep adequate output precision and use tighter internal checks. Do not assume rounding every field to two decimals is safe: accumulated state errors and tariff-weighted cost errors may exceed tolerance. Only clean tiny solver artifacts if the resulting response still passes replay; never round away a real violation.

**Limit:** Replay proves adherence to the interpreted directives. Only independent semantic tests can establish that those directives correctly represent the notes. The official judge checks against organizer ground truth.

## 10. Testing and acceptance evidence

### 10.1 Public sample coverage

| Case | Main behavior | Reference cost, BDT |
|---|---|---:|
| SAMPLE-01 | Solar reduction plus distractor | 38365 |
| SAMPLE-02 | Charging maintenance window | 42885 |
| SAMPLE-03 | Reserve expressed as battery-capacity percentage | 35480 |
| SAMPLE-04 | No-discharge window | 40495 |
| SAMPLE-05 | Temporary grid-import cap | 33950 |
| SAMPLE-06 | Solar reduction, no-charge window, and distractor | 34090 |
| SAMPLE-07 | Reserve plus transformer cap | 38550 |
| SAMPLE-08 | Separate charging and discharging outages | 37665 |
| SAMPLE-09 | 80% reduction interpreted as 20% remaining | 34873 |
| SAMPLE-10 | Evening reserve, grid cap, and distractor | 41620 |

Use each `case.input` as the request. Compare structured interpretation semantics, plan validity, and cost; explanations and optimal action sequences need not match exactly. Do not hard-code IDs, note wording, numeric values, or reference schedules into runtime behavior.

### 10.2 Separate the layers during testing

| Test layer | Evidence required |
|---|---|
| Request/response contract | Required fields, types, enums, hour coverage, note count, echo, status codes. |
| LLM interpretation | Primary and backup correctly extract relevance, type, hours, values, and shapes on independently labelled language; repeat/paraphrase tests measure semantic consistency. |
| Compiler | Each directive changes precisely the intended bounds; combinations obey every rule. |
| Optimizer | Expected directives supplied directly produce valid schedules with reference-equivalent costs. |
| Replay Validator | Deliberately corrupted plans are rejected for the specific violated rule. |
| End-to-end | Real LLM output flows through compilation, solving, replay, and exact API response. |
| Reliability | Injected primary outages trigger tested backup within the deadline; targeted repair, complete-set regeneration, both-provider failure, solver failure, repeated requests, and concurrency are controlled. |
| Reproduction | Fresh local setup and pulled Docker image work using only documented steps. |

### 10.3 Hidden-style coverage

- Paraphrases for all six directive types, using language not copied from public cases.
- Equivalent 12-hour/24-hour times, noon/midnight, stated whole-hour boundaries, and fractional descriptions such as one-fifth.
- Percentage-of-capacity reserves and the difference between reduction-by and reduction-to.
- Distractors containing campus, maintenance, or weather language without a supported current-schedule constraint.
- Multiple applicable notes, overlapping compatible restrictions, and all-no-op inputs.
- Valid zero and fractional values, solar surplus, no solar, equal or zero tariffs, asymmetric charge/discharge limits, and initial energy at a boundary.
- Missing, duplicate, out-of-range, or unordered hours; wrong field types; empty notes; non-finite numbers; malformed model JSON; unsupported types; mismatched `applies` values.
- Repeated and concurrent requests with different scenarios to detect cross-request state leakage.
- Applicable directives with empty hour arrays trigger repair; an unsuccessful repair never silently removes the note or becomes `no_op`.
- One invalid entry beside valid entries triggers targeted repair with original indices preserved; untrustworthy mappings trigger complete-set regeneration.
- Primary timeout, rate/quota failure, or invalid output exercises backup; exhaustion of all attempts returns a controlled error without leaking provider details.
- Initial energy at base reserve/capacity, just inside numeric tolerance, and materially outside; a feasible case starting below a future temporary reserve must not be rejected as an invalid initial state.
- Cold and repeated exact-input/paraphrase tests under recorded inference settings for both models; cold tests prevent caches from hiding model inconsistency.

Organizer scoring cases are feasible and use only supported directives. Robustness tests may include invalid requests, but such tests must not be confused with the promised distribution of valid scoring cases.

Keep a held-out semantic test set for final model/prompt evaluation. Mocked LLM responses are useful for deterministic unit tests but do not demonstrate mandatory live interpretation.

## 11. Runtime performance and reliability

**Requirements — Guide §8:**

| Metric | Standard |
|---|---|
| Startup readiness | `/health` returns `{"status":"ok"}` within 60 seconds of service start. |
| Per-request completion | `/optimize-energy` completes within 30 seconds. |
| Full latency credit | p95 ≤ 5 seconds: 3/3 latency points. |
| Intermediate latency credit | p95 > 5 through 15 seconds: 2/3; > 15 through 30 seconds: 1/3. |
| Excess latency | p95 > 30 seconds: 0/3; timed-out requests count as failures. |
| Valid-request stability | Valid requests should not produce 5xx, invalid JSON, or no response. |
| Failure handling | Malformed input or model/provider failure must not crash the service or leak secrets. |

**Plan:** Use one compact primary LLM call on the normal path, reused provider connections, the bounded repair/backup policy in Section 6.1, explicit solver limits, and an overall deadline below the judge timeout. Budget backup time before the primary attempt begins, and ensure SDK retries do not multiply the application's call or time budget. Measure end-to-end p95 under repeated requests, not only solver time. Report normal-path and injected-failure recovery performance separately, as well as overall performance. Keep `/health` responsive while optimization requests run.

Track stage timings, request outcomes, repair attempts, failover reasons, selected model/version, solver status, and verification failures without logging credentials or sensitive prompts. The team owns primary and backup credentials, quota, cost, rate limits, and availability throughout judging; judges are not expected to repair the dependencies. Rehearse loss of the primary provider and confirm that the backup is actually usable from the deployed environment.

Caching is optional and should follow correctness and latency evidence. If introduced, cache only validated interpretations with keys covering the note text, relevant scenario context, model, and prompt/schema version. Do not rely on repeated public inputs to meet latency targets.

## 12. Deployment, security, and repository rules

**Requirements — Guide §§2–4:**

- Submit one deployed HTTP API exposing both exact endpoints at the submitted base URL.
- Any reachable hosting platform is allowed; scoring depends on behavior, accessibility, and reproducibility.
- Judge access must require no login, dashboard, manual approval, VPN, or private network.
- Accept and return JSON and stay reachable throughout evaluation, including repeated LLM-backed requests.
- Test both endpoints from outside the development environment.
- Provide all source and dependency/configuration files.
- Create a new GitHub repository after question reveal, keep it private during the event, and make it public after the submission deadline.
- Do not commit credentials, tokens, passwords, or populated `.env` files.
- Do not expose secret values, secret-bearing prompts, or sensitive stack traces in responses or logs.
- Use only synthetic challenge data.
- AI coding assistants and public tools/libraries/APIs/SDKs are permitted under the official rulebook; core architecture and logic should be the team's own work. Credit external tools and dependencies in the README.

### Docker fallback

- Deliver an actually built and tested image, not only a Dockerfile.
- Submit a pullable registry reference with an exact tag or digest.
- Keep the image available throughout evaluation.
- Bind the service to `0.0.0.0`, expose the documented port, and supply a verified `docker run` command.
- Document required environment-variable names and inject credentials at runtime.
- Include the working solver and all runtime dependencies; bake in no secrets.
- Verify image pull, clean startup, `/health`, and at least one public sample using the documented commands.

**Proposed deployment practice:** Pin dependencies and use the same tested container for the hosted service and fallback. Keep a small startup check for configuration and solver availability. Do not start long training jobs or require organizers to alter source code.

## 13. Submission and documentation package

**Required deliverables — Guide §2:**

| Deliverable | Completion evidence |
|---|---|
| Public endpoint | Submitted base URL reaches both endpoints externally. |
| GitHub repository | Full source/configuration/dependencies and rule-compliant timing/visibility. |
| README and configuration | Self-contained instructions, sample request/response, model and solver disclosure, environment-variable names. |
| Docker fallback image | Exact accessible tag/digest, port, variable names, and verified pull/run commands. |
| Architecture/solution video | Organizer-accessible MP4 or link, maximum 3 minutes. |

The README must let an organizer reproduce the service from a clean environment without team assistance. Include:

1. Problem and architecture overview, including the actual LLM-to-constraints path.
2. Source setup and a copy-paste local quickstart.
3. Dependency/runtime versions and solver installation or packaged availability.
4. Environment-variable names with explanations and safe placeholders, never secret values.
5. Primary and backup model/provider or local model identifiers, supported inference settings, versioning, failover triggers, and deadline/call budgets.
6. LLM role, deterministic guardrails, per-note repair and final failure behavior, the empty-hour policy, the solar-overlap product rule and its infeasibility relaxation, compiler, optimizer, and replay behavior.
7. Exact start command and documented host/port.
8. `/health` and `/optimize-energy` curl examples.
9. At least one complete sample request/response and a command for the public sample suite.
10. Expected validation results and how to understand a failed test.
11. Docker pull/run fallback commands and exact image reference.
12. Known limitations, unresolved specification questions, and failure behavior.
13. Secret-handling guidance and credits for external tools/dependencies.

The video should explain the problem, architecture, solution flow, key choices, and how to run/test the system. Demonstrate that LLM interpretation becomes deterministic constraints and a verified schedule. Production-quality editing is not required. The video is mandatory but contributes no base points; it is reviewed to resolve tied total scores.

## 14. Complete scoring rubric and strategy

**Guide §§6–8: 100 points total.** Automated tests are the primary evaluation mechanism; deployment and documentation also use fixed artifact/reproducibility checks.

| Category | Points | Detailed allocation |
|---|---:|---|
| LLM Directive Interpretation | 25 | Relevance/no-op 5; directive type 5; affected hours 5; numeric values and required shape 5; paraphrase robustness 5. |
| Directive Application & Constraint Correctness | 25 | Ground-truth directive application 10; energy balance/effective solar 5; battery transitions/bounds/rates 5; action consistency/neutrality/non-negative values 5. |
| Optimization Quality | 10 | Average cost-quality ratio over optimization cases; invalid cases receive zero credit. |
| API Contract & Schema | 10 | Endpoints/status behavior 2; request validation 2; interpretation schema/order/types 3; plan/top-level schema and scenario echo 3. |
| Performance & Reliability | 10 | Health readiness 2; p95 latency 3; valid-request stability 3; controlled malformed/provider failures and secret safety 2. |
| Deployment & Docker Fallback | 10 | Live reachability 3; working pullable Docker fallback reaching health 4; clean startup/reproduction 2; no manual code fixes or judge debugging 1. |
| Documentation & Local Reproducibility | 10 | Clean quickstart 3; configuration/model documentation 2; public-sample procedure and expected result 2; architecture 1; Docker instructions 1; dependencies/limitations/secret guidance 1. |

The stated optimization formula is:

```text
quality_ratio = min(1, organizer_optimal_cost / recalculated_team_cost)
Optimization Quality = 10 * average(quality_ratio across optimization hidden cases)
```

Invalid cases receive zero optimization credit. When both costs are within numeric tolerance of zero, the guide explicitly sets the ratio to 1. Its next sentence, addressing zero organizer cost and positive team cost, is incomplete; see Section 17. Do not treat that missing text as a confirmed rule.

**Strategy:** Secure language accuracy and downstream correctness together; they account for half the points and validity gates optimization credit. Use an exact linear model to target optimal cost. Give deployment and documentation complete acceptance tests because together they account for another 20 points. Improve latency using measurements without weakening interpretation or verification.

The guide's recommended priority order is exact API contract, LLM interpretation, deterministic guardrails, directive/energy correctness, optimization, reliability/deployment/Docker, documentation/reproduction, and video tie-break readiness.

## 15. Penalties, hidden evaluation, and tie-breakers

### 15.1 Critical violations

**Guide §9:**

| Violation | Consequence |
|---|---|
| LLM absent from the operator-note interpretation path, or used only for cosmetic text | Mandatory requirement fails; not eligible for the final preliminary shortlist. |
| Relevant note misinterpreted or marked `no_op` | Interpretation credit lost for the affected note/case. |
| Ground-truth directive absent from the returned schedule | Affected case invalid for directive application; no optimization credit. |
| Energy imbalance or unmet demand | Affected case invalid; no optimization credit. |
| Battery bound, transition, or rate violation | Affected case invalid; no optimization credit. |
| Effective-solar overuse or impossible/negative energy | Affected case invalid; no optimization credit. |
| No-charge, no-discharge, reserve, or grid-cap violation | Affected case invalid; no optimization credit. |
| Final energy differs from initial energy | Affected case invalid; no optimization credit. |
| Totals disagree with the plan or repeated critical invalidity | Recalculation/scoring deduction; repeated failures may block qualification eligibility. |

### 15.2 Hidden tests

**PS §11; Guide §10:** The exact hidden cases, wording, distribution, and expected answers are unpublished. Tests vary language, whole-hour time expressions, percentages, equivalent numeric descriptions, demand, solar, tariffs, battery settings, and directive combinations. Valid scoring scenarios are feasible, contain 1–3 notes, and require only the published types.

The judge checks organizer-ground-truth interpretation and applies that ground truth when replaying the returned schedule. It does not accept a cheap schedule merely because it is consistent with an incorrect team interpretation. Free-text explanations and equivalent optimal schedules are not matched byte-for-byte. Paraphrase robustness is a measured property, not an extra response field.

### 15.3 Tie-break order

For tied total scores, the guide specifies:

1. Three-minute architecture and solution video.
2. Directive application and constraint correctness.
3. LLM directive interpretation.
4. Optimization quality.
5. API/schema validity.
6. Reliability and deployment stability.
7. Documentation and local reproducibility.
8. Exceptional engineering and verification, including robust guardrails, fallbacks, caching, testing, and implementation quality where relevant.

## 16. Development plan without a build-time budget

Each stage ends with evidence, rather than a fixed number of minutes. Deployment and documentation begin in Stage 1 and are maintained throughout. The later stages harden and verify existing artifacts rather than starting them for the first time. These are workstreams within one project, not a requirement for additional services or agents.

| Stage | Work | Completion condition |
|---|---|---|
| 1. Establish contract and delivery skeleton | Schemas, endpoint/status requirements, source traceability, ambiguity register, minimal API/health, Docker with CBC, configuration template, README quickstart, registry and deployment skeleton. | Required behavior has explicit ownership; a versioned image starts and health is externally reachable; CBC availability and documented startup are checked. This is an early smoke test, not a claim of full challenge readiness. |
| 2. Build the deterministic core | Compiler, linear model, response builder, replay validator using known directives; run inside Docker and update commands and tests in the README. | Public reference directives produce valid plans with matching optimal costs; corrupted plans are rejected; the containerized core runs without undocumented steps. |
| 3. Establish language accuracy and recovery | Primary/backup selection, structured-output prompt, inference settings, paraphrase dataset, guardrails, empty-hour policy, per-note repair, deadline-based failover. | Both live models pass public and held-out semantic tests; injected primary failure reaches a validated backup result; repeated-input results and settings are recorded. |
| 4. Integrate the API and public service | Real interpretation-to-schedule path, JSON/status handling, exact responses, secrets injected at runtime, refreshed image and deployment. | Public samples pass end-to-end externally and in the container; malformed requests, repair, and provider failover behave as documented. |
| 5. Strengthen generalization | Generated numeric cases, combinations, independent solver checks, repeat/concurrency tests, cold semantic tests, both-provider outage and latency tests. | No unexplained constraint failures; interpretation accuracy, normal/failover performance, and regressions are tracked. |
| 6. Harden and rehearse deployment | Verify pinned runtime and solver, final pullable image, external reachability, primary/backup quota, configuration and reproduction instructions. | A fresh image pull and public service meet readiness, stability, and runtime requirements without manual fixes. |
| 7. Audit and submit | Final README/credits/sample review, immutable or exact registry reference, configuration audit, video, repository timing/visibility, clean-machine rehearsal. | The full submission reproduces without undocumented steps or team intervention; endpoint, repository, image, and video remain available. |

Maintain the Docker image and README with each working milestone. A health-only skeleton does not earn the complete deployment/documentation score by itself: final acceptance requires the actual LLM-backed service, public-sample execution, and reproducible configuration. Early work reduces the risk of leaving those 20 points dependent on late integration.

Proposed small module layout:

```text
app/
  main.py          # routes and orchestration
  schemas.py       # request, directive, response contracts
  llm.py           # primary/backup interpretation, settings, bounded repair/failover
  directives.py    # guardrails and compilation
  optimizer.py     # PuLP model and solver handling
  validator.py     # independent response replay
  config.py        # environment-based settings
tests/
samples/
Dockerfile
requirements.txt
.env.example
.gitignore
.dockerignore
README.md
```

Split further only when it improves clarity or testing. Optional additions should demonstrate a correctness, reliability, or reproducibility benefit.

## 17. Unresolved specification details and decision boundaries

| Issue | What the supplied sources establish | Required treatment |
|---|---|---|
| Overlapping solar reductions | Each listed hour uses original solar multiplied by a factor; no explicit combination rule is provided for different overlapping reductions. | Runtime is defined by Section 7.1: multiply the applicable factors against original solar, defaulting to 1, with a single relaxation to the minimum factor if the model is infeasible. The product is chosen because it cannot cause effective-solar overuse under either reading. Seek organizer clarification without blocking requests, and update compiler, replay, tests, and documentation together if an official answer differs. |
| Zero-optimum scoring edge case | Both costs near zero gives ratio 1. The next sentence ends after `quality_ratio`. | Preserve the incompleteness and seek clarification. The service should still minimize cost and return valid totals. |
| Error-body shape | Status codes and safe handling are specified; an exact error JSON schema is not. | Choose and document a small consistent error body without claiming it is mandated. |
| Extra request fields and unstated input limits | Required fields are specified, but every additional validation policy is not. | Avoid rejecting valid organizer inputs through invented restrictions; document intentional policies. |
| Empty hours on an applicable directive | Uniqueness, ordering, and range are explicit; non-empty length is not explicitly stated. | Section 7.2 treats an empty list as a semantic extraction failure requiring bounded repair. Do not silently apply nothing or invent a no-op. Document this policy and update if organizers clarify otherwise. |
| Runtime/provider choices and inference settings | External APIs or local models are allowed, with team responsibility for availability; no temperature or seed is mandated. | Deliver a tested primary and backup, document supported inference settings, and measure semantic consistency and failover. Low temperature/seed are not guarantees of determinism. |

The guide references an official rulebook that was not included among the reviewed files. The requirements reproduced here are those stated in the supplied documents; any additional rulebook conditions must be checked against that separate source before submission.

## 18. Final acceptance checklist

- [ ] Both exact endpoints are reachable externally without access barriers.
- [ ] Health returns HTTP 200 and `{"status":"ok"}` when ready, within 60 seconds of startup.
- [ ] Request validation covers required fields, 1–3 notes, and all 24 unique hours; structural failures use HTTP 400.
- [ ] Explicit base-reserve/initial/capacity checks follow the final-state and neutrality rules with documented tolerance; semantic invalidity uses HTTP 422, and future temporary reserves are evaluated through scheduling.
- [ ] A real language-capable generative model interprets every note in the constraint-producing path.
- [ ] Primary and backup model identifiers/versions, supported inference settings, prompts, and schemas are documented and evaluated on held-out and repeated semantic tests.
- [ ] Every note has one correctly ordered interpretation with the exact type, hours, numeric values, shape, and `applies` semantics.
- [ ] LLM output passes deterministic guardrails before affecting optimization.
- [ ] Empty-hour and other locally invalid entries trigger bounded repair; unreliable mappings trigger complete-set regeneration; unresolved notes never disappear from a successful response.
- [ ] All five operational directive types are enforced; irrelevant notes change nothing.
- [ ] Solar overlaps use the documented product-of-factors policy with the infeasibility relaxation, unless superseded by an official clarification; order, duplicate, zero/one-factor, grid-capped-fallback, and overlapping-feasibility tests pass.
- [ ] The model uses the published cost objective and physical rules without invented economics.
- [ ] Solver status is checked, and accepted plans pass independent replay.
- [ ] Every output has 24 valid hours, correct actions/states, full energy balance, and end-of-day neutrality.
- [ ] Reported grid total, cost, and peak match the actual response rows within the applicable tolerance.
- [ ] Public cases pass; held-out paraphrases, numeric variations, and combined restrictions have recorded results.
- [ ] Malformed input and LLM/provider/solver failures are controlled and do not fabricate successful plans.
- [ ] Injected primary failure reaches the tested backup inside the overall deadline; optional repair, SDK behavior, and both-provider failure obey the global call/time budget.
- [ ] Real repeated-request p95 is measured against the 5-second full-credit target; requests meet the 30-second deadline.
- [ ] Primary and backup credentials, quota, rate limits, and availability are adequate for the judging window.
- [ ] The submitted Docker image is pullable by exact tag/digest and starts with the documented command, port, and runtime environment variables.
- [ ] A clean local/container rehearsal reaches health and completes at least one public request without manual code edits.
- [ ] README covers setup, model/provider, LLM role, guardrails, solver, commands, samples, expected results, limitations, Docker, secret safety, and credits.
- [ ] Repository creation and visibility follow the stated event policy.
- [ ] No secrets are committed, baked into images, printed in logs, or exposed through API errors.
- [ ] The accessible video is no longer than three minutes and explains the architecture and run/test flow.
- [ ] Endpoint, repository, image, and video remain available throughout evaluation.
- [ ] Unresolved source ambiguities and any organizer clarifications are recorded accurately.

## 19. Requirement-to-plan traceability

| Source sections | Coverage in this plan |
|---|---|
| PS §§0, 12 — canonical authority | Sections 1, 17. |
| PS §§1–3 — scenario, goal, processing flow | Sections 2–3. |
| PS §§4–5 — directives, clauses, objective | Sections 5–8. |
| PS §§6–7 — endpoints, codes, request | Section 4. |
| PS §8 — deterministic guardrails and safe failure | Sections 5–7, 9, 11. |
| PS §9 — battery and energy rules | Sections 8–9. |
| PS §10 — response schema | Section 4.2. |
| PS §11 — exact validation, hidden tests, tolerance | Sections 9–10, 15. |
| Guide §1 — document roles and round context | Sections 1–2 and introduction. |
| Guide §2 — required deliverables | Sections 12–13. |
| Guide §3 — deployment and reproduction | Sections 11–13. |
| Guide §4 — LLM, technology, security, repository policy | Sections 6, 12, 17. |
| Guide §5 — testing and submission checklist | Sections 10, 13, 18. |
| Guide §§6–7 — evaluation and detailed scoring | Section 14. |
| Guide §8 — interpretation, performance, API thresholds | Sections 5, 9, 11, 13. |
| Guide §9 — violations and penalties | Section 15.1. |
| Guide §10 — hidden tests and tie-breakers | Sections 15.2–15.3. |
| Guide §11 — priorities and final checklist | Sections 14, 16, 18. |
| Public sample pack — examples and comparison rules | Sections 5, 10.1. |
