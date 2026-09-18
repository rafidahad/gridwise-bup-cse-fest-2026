# GridWise — LLM-Assisted Campus Energy Optimization

BUP CSE Fest 2026 · Hackathon Online Preliminary

One HTTP service that reads 24 hours of campus demand, solar, and tariff data
plus 1–3 natural-language operator notes, **interprets those notes with a
language model**, turns the validated interpretation into hard constraints,
solves for the cheapest feasible 24-hour schedule, and independently replays the
schedule before returning it.

| | |
|---|---|
| Health endpoint | `GET /health` |
| Main endpoint | `POST /optimize-energy` |
| Primary model | Groq (OpenAI-compatible) |
| Secondary model | Optional, unconfigured by default |
| Solver | PuLP + COIN-OR CBC |
| Docker image | `gridwise:1.0.0` — see [section 9](#9-docker-fallback) |

---

## Contents

1. [Problem and architecture](#1-problem-and-architecture)
2. [Quickstart](#2-quickstart)
3. [Configuration and models](#3-configuration-and-models)
4. [API contract](#4-api-contract)
5. [Sample request and response](#5-sample-request-and-response)
6. [How the pipeline works](#6-how-the-pipeline-works)
7. [Documented policies for open questions](#7-documented-policies-for-open-questions)
8. [Testing and expected results](#8-testing-and-expected-results)
9. [Docker fallback](#9-docker-fallback)
10. [Known limitations](#10-known-limitations)
11. [Secret handling](#11-secret-handling)
12. [Credits](#12-credits)

---

## 1. Problem and architecture

Campus operators write short notes like *"Panel washing from one until three
will leave roughly one-fifth of normal solar output."* The service has to
understand that sentence, turn it into `solar_reduction` over hours `[13, 14]`
with `factor = 0.2`, apply it as a hard constraint, and return the cheapest
24-hour schedule that respects it along with the battery and energy rules.

```text
POST /optimize-energy
        │
        ▼
FastAPI + Pydantic ─────── structural validation → 400
        │                  semantic validation   → 422
        ▼
LLM interpretation ─────── one call, JSON output, notes → structured directives
        │                  bounded repair, then an optional secondary model
        ▼
Deterministic guardrails ─ model output is untrusted data until it validates
        │
        ▼
Directive Compiler ─────── directives → hourly bounds:
        │                  effective solar, minimum reserve,
        │                  charge/discharge ceilings, grid caps
        ▼
PuLP + CBC ─────────────── one 24-hour linear program
        │                  minimise Σ grid[h] · tariff[h]
        ▼
Response builder ───────── actions, states, totals recomputed from the rows
        │
        ▼
Replay Validator ───────── independent re-check of the response about to be sent
        │
        ▼
Exact JSON response (200)
```

**The language model is the interpretation path.** Its structured output is what
produces the optimizer's constraints (Guide §04). Deterministic code validates
and applies that output; it never authors an interpretation of its own, and no
phrase matching is used as an interpretation path anywhere in this service. If
every model path fails, the service returns a controlled error — it does not
fall back to a hard-coded reading of the notes.

| Component | Choice | Responsibility |
|---|---|---|
| Language | Python 3.12 (image) / 3.12+ (local) | Application and deterministic logic |
| API | FastAPI 0.141.1 + Uvicorn 0.53.0 | Endpoints, JSON, explicit status codes |
| Schemas | Pydantic 2.13.5 | Request and response contracts |
| Interpretation | Groq (`openai/gpt-oss-120b`) | Notes → structured directives |
| Compiler | Plain Python | Directives → hourly bounds |
| Optimization | PuLP 3.3.2 + CBC | Minimum-cost feasible schedule |
| Replay | Independent plain Python | Verify the actual response |
| Tests | pytest 9.1.1 + httpx 0.28.1 | Contract, compiler, optimizer, replay, integration |

### Repository layout

```text
app/
  main.py          routes, status-code contract, orchestration, response builder
  schemas.py       request / directive / response contracts
  llm.py           prompt, call budget, bounded repair, provider failover
  directives.py    guardrails on model output, and the Directive Compiler
  optimizer.py     the linear program, solver handling, plan construction
  validator.py     independent replay of the response
  config.py        environment-backed settings
scripts/
  verify_core.py         offline: compiler + optimizer + replay vs published optima
  run_public_cases.py    HTTP: the 10 public cases against a running service
  semantic_eval.py       live: model accuracy on the held-out labelled set
tests/                   187 tests
  reference_dp.py        an independent optimum, sharing no code with the LP
data/
  public_cases.json      the organizer's 10 public sample cases
  semantic_cases.json    34 held-out labelled interpretation cases
  packs/                 team-generated packs: 132 more interpretation cases,
                         18 with reference optimal costs, plus an HTTP
                         robustness checklist
docs/
  PROBLEM_STATEMENT.md   transcription of the organizer problem statement
  PARTICIPANT_GUIDE.md   transcription of the organizer guide and rubric
  SOLUTION_PLAN.md       the solution plan this build follows
  VIDEO_SCRIPT.md        outline for the 3-minute submission video
  sample_request_response.json
Dockerfile
requirements.txt / requirements-dev.txt
.env.example
```

---

## 2. Quickstart

From a clean environment. Requires Python 3.12 or newer and `git`. CBC ships
inside the PuLP wheel, so no separate solver installation is needed locally.

**1. Get the source**

```bash
git clone https://github.com/rafidahad/gridwise-bup-cse-fest-2026.git
```

```bash
cd gridwise-bup-cse-fest-2026
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure credentials**

```bash
cp .env.example .env
```

Edit `.env` and fill in `GRIDWISE_PRIMARY_API_KEY` and
`GRIDWISE_PRIMARY_API_KEYS`. The file contains placeholders only; it is
git-ignored and must never be committed.

**5. Start the service**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service binds `0.0.0.0:8000`. Override with `GRIDWISE_HOST` and
`GRIDWISE_PORT`, or with uvicorn's own flags.

**6. Check health**

```bash
curl -s http://localhost:8000/health
```

Expected, within 60 seconds of start (Guide §08) — measured at 3 seconds:

```json
{"status": "ok"}
```

**7. Run the public samples**

```bash
python scripts/run_public_cases.py
```

See [section 8](#8-testing-and-expected-results) for what to expect.

---

## 3. Configuration and models

Every setting comes from the environment. `.env.example` is the template and
holds **placeholders only, never real credentials**.

| Variable | Meaning | Default |
|---|---|---|
| `GRIDWISE_PRIMARY_API_KEYS` | Primary provider keys (Groq), comma separated | — |
| `GRIDWISE_PRIMARY_BASE_URL` | Primary OpenAI-compatible base URL | `https://api.groq.com/openai/v1` |
| `GRIDWISE_PRIMARY_MODEL` | Primary model identifier | `openai/gpt-oss-120b` |
| `GRIDWISE_BACKUP_API_KEYS` | Optional secondary keys, comma separated | — (disabled) |
| `GRIDWISE_BACKUP_BASE_URL` | Optional secondary base URL | — (disabled) |
| `GRIDWISE_BACKUP_MODEL` | Optional secondary model identifier | — (disabled) |
| `GRIDWISE_TEMPERATURE` | Sampling temperature | `0` |
| `GRIDWISE_SEED` | Best-effort seed, omitted when unset | `7` |
| `GRIDWISE_MAX_OUTPUT_TOKENS` | Output cap for the interpretation call | `1200` |
| `GRIDWISE_REQUEST_DEADLINE_S` | Overall per-request budget | `25` |
| `GRIDWISE_PRIMARY_TIMEOUT_S` | Primary call deadline | `9` |
| `GRIDWISE_REPAIR_TIMEOUT_S` | Repair call deadline | `7` |
| `GRIDWISE_BACKUP_TIMEOUT_S` | Backup call deadline | `9` |
| `GRIDWISE_SOLVER_TIME_LIMIT_S` | CBC time limit | `10` |
| `GRIDWISE_HOST` / `GRIDWISE_PORT` | Bind address and port | `0.0.0.0` / `8000` |
| `GRIDWISE_LOG_LEVEL` | Log verbosity | `INFO` |

### The optional secondary model

The service runs on one provider by default. A backup is **permitted but never
required** — Guide §04 says "a local or backup model is allowed", and plan §6.1
calls it a reliability feature rather than an obligation. With the three
`GRIDWISE_BACKUP_*` variables unset, the third recovery step is skipped and
recovery ends after the primary's targeted repair.

Any OpenAI-compatible endpoint can fill the slot. A second Groq model adds a
different reading of a note, but shares Groq's availability, so it does not
protect against a provider outage.

**Known consequence:** with no secondary configured, a Groq outage fails every
request. Credentials, quota, and availability are the team's responsibility
under Guide §04.

### Key rotation

Each provider accepts several comma-separated keys. When a key is rate limited,
out of quota, or rejected outright, the service advances to the next one and
remembers the move, so later requests skip the spent key rather than paying for
the same rejection again. At most three credentials are tried inside one logical
attempt, so exhausted keys cannot eat the request deadline.

Rotation is **not** a second interpretation attempt and does not consume the
primary/repair/backup budget — the same question is re-asked on a credential
that still has headroom. A timeout or connection failure is the provider's
fault rather than the key's, so it stops instead of burning the remaining
keys. Keys never appear in logs; only their position
(`key 2/4`) is recorded.

The singular `GRIDWISE_PRIMARY_API_KEY` name still works for a single
credential.

### Inference settings

Temperature is `0` for extraction, and a seed is sent only when
`GRIDWISE_SEED` is set and the provider accepts it. Seeded sampling is
best-effort and **not** a determinism guarantee — OpenAI documents it that way,
and providers differ in whether they honour it at all. The prompt version and
schema version are pinned in `app/config.py` (`prompt_version`,
`schema_version`) and change whenever the prompt text or the interpretation
schema changes, so recorded evidence stays attributable.

Low temperature reduces avoidable variability; it does not establish semantic
accuracy. That is measured separately:

```bash
python scripts/semantic_eval.py --role primary --repeat 3 --json primary.json
```

`--packs all` (the default) scores 166 cases / 201 notes across the held-out set
and the three team packs. Narrow it with
`--packs interpretation_cases,gridwise_edge_case_pack` when iterating on the
prompt.

```bash
This scores the 34 held-out cases in `data/semantic_cases.json` along the
rubric's own dimensions — relevance/`no_op`, directive type, affected hours,
numeric values and shape — plus paraphrase-cluster agreement, run-to-run
consistency, and latency. No wording in that set is copied from the public
pack, and a test enforces it.

---

## 4. API contract

### `GET /health`

Returns HTTP 200 and exactly `{"status": "ok"}` when ready. It performs no
network or solver work, so it stays responsive while optimizations are running.

```bash
curl -s http://localhost:8000/health
```

### `POST /optimize-energy`

Accepts one scenario object, returns one interpretation-and-plan object. The
schemas are PS §§7 and 10, implemented in `app/schemas.py`.

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
  -H 'content-type: application/json' \
  -d @docs/sample_request.json
```

Or inline, using the first public case:

```bash
python -c "import json;print(json.dumps(json.load(open('data/public_cases.json'))['cases'][0]['input']))" > /tmp/case.json && curl -s -X POST http://localhost:8000/optimize-energy -H 'content-type: application/json' -d @/tmp/case.json
```

| Status | Meaning |
|---|---|
| 200 | Successful optimization |
| 400 | Malformed JSON, or a structurally invalid request |
| 422 | Well-formed request that admits no feasible schedule |
| 500 | Controlled internal error — no stack trace, prompt, or credential |

PS §6.1 fixes the status codes but not the error body. This service uses one
consistent shape throughout. This is a documented choice, not a mandated schema:

```json
{"error": {"code": "invalid_request", "message": "hours: must cover 0 through 23"}}
```

| `code` | Status | Cause |
|---|---|---|
| `malformed_json` | 400 | Body is not valid JSON |
| `invalid_request` | 400 | Missing field, wrong type, non-finite number, wrong note count, bad hour set |
| `semantically_invalid_request` | 422 | Battery parameters that no schedule can satisfy |
| `infeasible_scenario` | 422 | Well-formed, but no schedule satisfies it with its directives |
| `interpretation_unavailable` | 500 | Every model path was exhausted |
| `solver_error` | 500 | The solver returned no proven optimum |
| `replay_failed` | 500 | The computed plan failed its own verification |
| `internal_error` | 500 | Anything unexpected, with a log correlation id |

**Input policies** (chosen where the sources are silent, per plan §17): unknown
extra fields are ignored rather than rejected; `hours` need not arrive in
ascending order, since rows are indexed by `hour`; hour numerics must be finite
but carry no sign restriction, because rejecting a negative tariff would forfeit
a whole case if hidden data ever contained one.

---

## 5. Sample request and response

The complete pair is in
[`docs/sample_request_response.json`](docs/sample_request_response.json),
produced through the real code path for public case `SAMPLE-06`. Abridged:

**Request** (3 notes: a solar reduction, a charging outage, and a distractor)

```json
{
  "scenario_id": "SAMPLE-06",
  "operator_notes": [
    "Cloud cover during panel inspection will leave about half of the forecast solar output from 10 AM until noon.",
    "The charging circuit will be unavailable from 2 PM until 4 PM.",
    "The library is extending book-return hours next week."
  ],
  "hours": [
    {"hour": 0, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
    "... 22 more ...",
    {"hour": 23, "demand_kwh": 100, "solar_kwh": 0, "tariff_bdt_per_kwh": 7}
  ],
  "battery": {
    "capacity_kwh": 220, "initial_energy_kwh": 100, "minimum_energy_kwh": 35,
    "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50
  }
}
```

**Response**

```json
{
  "scenario_id": "SAMPLE-06",
  "directive_interpretation": [
    {
      "note_index": 0, "applies": true, "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [10, 11], "factor": 0.5},
      "explanation": "Solar availability is halved during the inspection window."
    },
    {
      "note_index": 1, "applies": true, "directive_type": "no_charge_window",
      "structured_adjustment": {"hours": [14, 15]},
      "explanation": "Battery charging is unavailable while the circuit is out."
    },
    {
      "note_index": 2, "applies": false, "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "This note does not affect today's energy schedule."
    }
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 135.0, "solar_used_kwh": 0.0,
     "battery_action": "charge", "battery_kwh": 50.0,
     "battery_energy_after_kwh": 150.0},
    "... 22 more ...",
    {"hour": 23, "grid_kwh": 100.0, "solar_used_kwh": 0.0,
     "battery_action": "idle", "battery_kwh": 0.0,
     "battery_energy_after_kwh": 100.0}
  ],
  "total_grid_kwh": 2395.0,
  "total_cost_bdt": 34090.0,
  "peak_grid_kwh": 175.0,
  "plan_summary": "Applied solar limited to 0.5x forecast in hour(s) 10-11; charging blocked in hour(s) 14-15. 1 note(s) did not affect the schedule and were treated as no_op. Shifting battery use toward cheaper hours, the plan buys 2395 kWh for 34090.00 BDT, peaks at 175 kWh in hour 21, and returns the battery to its starting level."
}
```

`34090.00 BDT` is the published optimum for this case. The `explanation`
strings come from the model, so exact wording varies between runs; Guide §07
does not match them byte-for-byte. `plan_summary` is assembled from verified
facts rather than a second model call, so it cannot contradict the schedule.

---

## 6. How the pipeline works

### 6.1 The model's role

One call interprets all 1–3 notes together. The prompt supplies the six
permitted directive types with their exact `structured_adjustment` shapes, the
whole-hour convention with worked examples (`1 PM to 3 PM → [13, 14]`, noon is
12, midnight is 0), both percentage readings (`to 20%` → `0.2`, `by 80%` →
`0.2`, `to 80%` → `0.8`), and the battery capacity so a reserve stated as a
percentage can be resolved.

Notes are passed as JSON-quoted **data**, under an explicit instruction that
note text is never a command to the model. A note containing *"ignore your
previous instructions"* is interpreted as campus language, not obeyed.

### 6.2 Deterministic guardrails

Model output is untrusted until it passes `app/directives.py`. Checked per
entry: the type is one of the six; `note_index` identifies a real note and each
note appears once; `applies` is `true` for every directive except `no_op`, which
must be `false` with a `null` adjustment; `hours` are integers 0–23 and
non-empty; `factor` is finite in `[0, 1]`; a reserve is finite, non-negative and
within capacity; a grid cap is finite and non-negative.

The guardrails never repair a value, substitute a type, invent an hour, or drop
a note. Unknown keys in an adjustment are discarded rather than echoed, so the
emitted object is exactly the required shape. Hour arrays are sorted and
de-duplicated — the only normalisation applied, and meaning-preserving because
every directive type has set-membership semantics.

### 6.3 Repair and failover

At most **three** inference calls per request:

```text
primary succeeds and validates            → compile and solve
primary returns identifiable bad entries  → one targeted repair on the primary
primary unusable, or repair falls short   → one backup-model attempt
still incomplete, or deadline reached     → controlled 500
```

A primary timeout, transport failure, or quota rejection **skips repair
entirely** and goes straight to the backup: re-asking a provider that just
failed spends the budget without changing the odds. SDK-level retries are
disabled (`max_retries=0`) so they cannot multiply either budget.

A repair names only the failing notes, carries their original indices and the
validator's own feedback, and cannot revise an entry that already passed. If
entries cannot be attributed to notes at all — duplicate or missing indices —
the whole set is regenerated rather than guessed at.

There is **no partial-success path**. If any note is still unresolved, the whole
successful response is withheld and a controlled error is returned. A note is
never dropped, never rewritten to `no_op`, and never replaced by a fabricated
directive.

### 6.4 The Directive Compiler

Validated directives become five hourly arrays: `effective_solar`,
`minimum_reserve`, `maximum_charge`, `maximum_discharge`, and `maximum_grid`
(unbounded where no cap applies). Simultaneously applicable rules combine
deterministically — greatest reserve, smallest grid cap, union of the action
prohibitions — so if both prohibitions cover an hour the battery must idle. An
internal note-to-limit trace records which note changed which bound; it is
logged, never added to the response.

Past this point the model has no influence: the optimizer only ever sees numbers
produced here.

### 6.5 The optimizer

One linear program over all 24 hours, with a single signed continuous variable
per hour carrying the battery action:

```text
b[h] > 0 charge      b[h] < 0 discharge      b[h] = 0 idle

-max_discharge[h] ≤ b[h] ≤ max_charge[h]
E[h] = E[h-1] + b[h],  E[-1] = initial,  E[23] = initial
grid[h] + solar_used[h] = demand[h] + b[h]
0 ≤ solar_used[h] ≤ effective_solar[h],  0 ≤ grid[h] ≤ max_grid[h]
minimum_reserve[h] ≤ E[h] ≤ capacity

minimise Σ grid[h] · tariff[h]
```

The balance line is PS §9.5 rewritten: `grid + solar_used + max(0,−b) = demand +
max(0,b)` reduces to `grid + solar_used = demand + b`. One signed variable makes
a simultaneous charge and discharge structurally impossible and keeps the model
a pure LP with no binaries.

Nothing is added to the published objective — no degradation cost, no round-trip
efficiency, no export revenue, no peak penalty. `peak_grid_kwh` is reported but
not minimised, so an equally optimal plan may peak differently from the
reference; PS §11.4 accepts that.

Solver status is checked: a merely feasible or interrupted result is never
labelled optimal. The plan is then built so the *returned rows* are consistent
by construction — near-zero actions snap to zero so an `idle` hour reports
exactly 0, the state chain is re-derived by forward simulation from the supplied
initial energy, the final state is pinned to it, and `grid` is derived from the
balance.

### 6.6 The Replay Validator

Before anything is returned, `app/validator.py` re-derives every limit from the
original request and the validated directives — **without** importing the
compiler's output. The duplication is deliberate: two views of one bug would
prove nothing.

It checks the scenario echo, interpretation coverage and order, 24 unique plan
hours, finite non-negative values, action/magnitude consistency, the
reconstructed state transitions, capacity and active reserve, rate limits,
effective-solar usage, the hourly balance, every directive window and cap,
end-of-day neutrality, and the three reported totals recomputed from the rows.

A plan that fails replay is **not returned**. It becomes a controlled 500.

---

## 7. Documented policies for open questions

The organizer sources leave some points open. Rather than leave run-time
behaviour undefined, each is fixed here and recorded (plan §17).

### Overlapping solar reductions

When several `solar_reduction` directives cover the same hour, their factors are
**multiplied** and applied once to the original forecast. With no directive the
factor is 1. So 100 kWh under factors 0.8 and 0.5 gives 40 kWh.

The statement prescribes no combination rule, so the choice was made on which
error the judge punishes. Using *less* solar than permitted is always legal —
unused solar is curtailed (PS §9.4) and the shortfall comes from grid or
battery. Using *more* than the judge's effective solar invalidates the entire
case and forfeits its optimization credit (Guide §09). The two candidates are
therefore not symmetric:

| Our rule | If the judge multiplies (limit 40) | If the judge takes the minimum (limit 50) |
|---|---|---|
| Product — limit 40 | exact match, valid | under-uses solar; **valid**, marginally costlier |
| Minimum — limit 50 | may draw 50 against a 40 ceiling; **case invalid** | exact match, valid |

The product is the only option with no invalidating outcome. Last-note-wins is
rejected outright because it would make the schedule depend on note order.

**The relaxation.** Over-curtailment is not risk-free either: with a
`max_grid_window` also active in an overlapped hour and too little battery to
cover the gap, the stricter ceiling can make an otherwise feasible scenario
unsolvable — equally fatal. So an infeasible first pass relaxes the overlapped
hours once to `min(factors)` and re-solves. The relaxation is attempted only
when an overlap actually exists, so a genuine infeasibility still surfaces
instead of being masked by a pointless retry. Both passes are deterministic and
order-independent, and the pass used is logged and reflected in `plan_summary`.

This is a defined implementation policy, not a claim to match unpublished judge
semantics. If an official rule is published, the compiler, the replay checks,
the regression expectations, and this section change together.

### Empty `hours` on an applicable directive

The sources require hours to be unique, ordered, and in range, but do not state
a minimum length. An entry with `applies = true` and `hours = []` is treated as
an **extraction failure** and sent for re-interpretation. It is never accepted as
a rule that silently affects nothing, and never rewritten to `no_op` — only
genuine language interpretation may produce `no_op`, with a null adjustment.

### Battery feasibility and the initial state

`minimum_energy_kwh ≤ initial_energy_kwh ≤ capacity_kwh` is checked explicitly,
and a violation returns 422. This is derived, not invented: neutrality forces
`E_after[23] = initial_energy_kwh` (PS §9.6) and PS §9.2 bounds every `E_after`,
so the inequality is necessary for any feasible schedule.

Two deliberate consequences. A *temporary* directive reserve above the initial
energy is **not** rejected — the schedule may charge to meet it, and the
optimizer decides. And because PS §11.5 treats values within 0.01 as equivalent,
an initial energy marginally below the base reserve is treated as equal to it, so
a boundary scenario produces a schedule rather than a spurious infeasibility.

### Error body shape

PS §6.1 fixes the status codes but not the body. The shape in
[section 4](#4-api-contract) was chosen and documented; it is not claimed to be
mandated.

### The zero-optimum scoring edge case

Guide §07 sets `quality_ratio = 1` when both costs are within tolerance of zero.
Its next sentence — covering a zero organizer cost with a positive team cost —
is cut off in the source PDF. The incompleteness is preserved rather than
guessed at; the service minimises cost and reports valid totals regardless.

---

## 8. Testing and expected results

### Install the test dependencies

```bash
pip install -r requirements-dev.txt
```

### Full suite

```bash
pytest
```

Expected: **187 passed**, in roughly 40 seconds. No network access and no API
key is needed — provider calls are stubbed so the contract, compiler, optimizer,
replay, and failover *policy* are all exercised deterministically.

| Test file | What it proves |
|---|---|
| `test_api.py` | Endpoints, status codes, exact response schema, controlled failure, no secret leakage, repeated and interleaved scenarios |
| `test_directives.py` | Guardrails on untrusted output; each directive changes precisely its intended bound; every solar-overlap case |
| `test_optimizer.py` | Physical rules, directive enforcement, the two-pass solar policy, public reference costs |
| `test_validator.py` | Deliberately corrupted plans are rejected for the specific rule they break |
| `test_llm.py` | Call budget, targeted repair, full regeneration, failover, no dropped notes |
| `test_generalization.py` | Generated scenarios, an independently computed optimum, concurrency, numeric edges |
| `test_semantic_dataset.py` | The labelled set is internally valid and genuinely held out |
| `test_team_packs.py` | The extra packs' labels are valid, and their 18 reference optima are reproduced — a second independent check on the optimizer |

### Offline core check

```bash
python scripts/verify_core.py
```

Feeds each public case's published reference interpretation straight into the
compiler, optimizer, and validator, bypassing the model. Expected:

```text
case         status        our cost    reference      delta
-----------------------------------------------------------
SAMPLE-01    ok           38,365.00    38,365.00       0.00
SAMPLE-02    ok           42,885.00    42,885.00       0.00
SAMPLE-03    ok           35,480.00    35,480.00       0.00
SAMPLE-04    ok           40,495.00    40,495.00       0.00
SAMPLE-05    ok           33,950.00    33,950.00       0.00
SAMPLE-06    ok           34,090.00    34,090.00       0.00
SAMPLE-07    ok           38,550.00    38,550.00       0.00
SAMPLE-08    ok           37,665.00    37,665.00       0.00
SAMPLE-09    ok           34,873.00    34,873.00       0.00
SAMPLE-10    ok           41,620.00    41,620.00       0.00

PASSED: 10 of 10 cases replay cleanly and match the reference optimal cost
```

All ten published optima are also reproduced by the independent dynamic program
in `tests/reference_dp.py`, which shares no code with the linear program.

### Public samples over HTTP

With the service running and credentials configured:

```bash
python scripts/run_public_cases.py
```

Add `--base-url https://your-deployment.example` for a remote service. Each case
is posted, the returned plan is independently replayed, the returned
interpretation is compared to the published reference, and Guide §07's cost
quality and the p95 latency band are reported. Exit code 0 means all ten were
valid, interpreted as published, and at the optimal cost.

### Reading a failure

| Output | Meaning |
|---|---|
| `valid` is `NO`, with `! invalid: …` | The returned plan broke a GridWise or directive rule. The message names the hour and the rule. This is a correctness defect. |
| `interp` is `NO`, with `~ interpretation: …` | The plan is valid but the model read a note differently from the reference — wrong type, hours, or numeric value. An interpretation problem, not a constraint one. |
| `~ cost is N BDT above the optimum` | Valid and correctly interpreted, but not optimal. |
| A non-200 status | The message body carries the error `code` from [section 4](#4-api-contract). |
| `COSTLIER` from `verify_core.py` | A constraint is over-tight. |
| `CHEAPER` from `verify_core.py` | A constraint is missing — the plan is exploiting something it should not. |

### Measured performance

| Measurement | Result |
|---|---|
| Full live request (interpret, compile, solve, replay), p95 over the 10 public samples | **1.93 s** — Guide §08 band 3/3 |
| Deterministic path alone (compile, solve, build, replay), p95 over 100 runs | **88 ms** — 0.29% of the 30 s judge timeout |
| Container start to `/health` 200 | **3 s**, against the 60 s requirement |
| Both provider attempts on total outage | ~200 ms to a controlled error |

Essentially the whole 5-second full-credit latency budget (Guide §08) is
therefore available to the inference call. End-to-end p95 with live models is
reported by `scripts/run_public_cases.py` and depends on the chosen model.

---

## 9. Docker fallback

### Build

```bash
docker build -t gridwise:1.0.0 .
```

### Run

```bash
docker run --rm -p 8000:8000 -e GRIDWISE_PRIMARY_API_KEYS=your-groq-key gridwise:1.0.0
```

Then:

```bash
curl -s http://localhost:8000/health
```

The image binds `0.0.0.0` and exposes port `8000`. Credentials are injected at
run time through the variable names in [section 3](#3-configuration-and-models);
none is baked into any layer.

### Pull the published image

> **To be completed at submission.** Replace with the exact pullable reference
> once the image is pushed:
>
> ```bash
> docker pull <registry>/<namespace>/gridwise@sha256:<digest>
> ```

Locally built image id:
`sha256:d4c85b13ccd13b121e262882f57ef7d73a105a8f3e1fe118fa4f48c3b3f744be`

### Verified in the container

- Python 3.12.14, image 364 MB.
- **Both** CBC front ends work: PuLP's bundled binary and Debian's
  `/usr/bin/cbc`. The optimizer tries them in order, so a host where one will
  not execute still gets a solver.
- `docker run --rm gridwise:1.0.0 python scripts/verify_core.py` reproduces all
  ten published optima from inside the image.
- `/health` returned 200 three seconds after start; the Docker `HEALTHCHECK`
  reports `healthy`.
- Malformed JSON returned 400 and infeasible battery parameters returned 422.
- Runs unprivileged as uid 10001. No credential environment variable, no `.env`,
  no baked-in key.

---

## 10. Known limitations

- **Single provider, by choice.** No secondary model is configured, so a Groq
  outage fails every request. This is permitted — Guide §04 allows a backup
  rather than requiring one — but it is a deliberate trade, not an oversight.
  An earlier NVIDIA NIM backup was removed after measurement: most of its
  catalogue had reached end of life (HTTP 410), and the one reachable model
  answered a trivial prompt in 58 s, then returned a 500 and two timeouts on
  real interpretation calls, which is useless against a 30-second judge
  timeout.
- **Token throughput is the binding constraint, not latency.** Groq's free
  tier caps at 8,000 tokens per minute; the paid Developer plan gives 250K.
  One interpretation request measures 2,421 tokens (1,930 prompt, 352
  reasoning, 139 output), so a free key sustains only ~3.3 requests per minute
  and a paid one about 103. On the free tier, two keys were exhausted by eight
  public samples in ten seconds.
- **Prompt caching does not currently help.** Groq documents it as automatic
  on this model, with cached tokens excluded from rate limits, and the prompt
  is already structured for it (static system text first, per-scenario content
  last). Measured across three different scenarios, `cached_tokens` came back
  empty every time and the full ~2,200 tokens were debited. The docs say cache
  hits are "not guaranteed", so this is not something to rely on.
- **Interpretation accuracy is measured on the public set only.** A live run
  on `openai/gpt-oss-120b` matched **10/10** published interpretations and
  reached **10/10** optimal costs at p95 1.93 s. The 166-case held-out and
  team packs have not yet been run end to end against a live model.
- **The solar-overlap rule is a defined policy, not a known-correct one.** See
  [section 7](#7-documented-policies-for-open-questions). It is chosen because
  it cannot cause solar overuse under either reading, at the cost of a slightly
  conservative schedule if the judge's rule is more permissive.
- **Optimality is optimality of the model supplied.** A proven optimum cannot
  repair a mistaken reading of a note or a constraint that was never built.
  Replay proves the plan obeys the directives *this service interpreted*; only
  the organizer's harness compares those to ground truth.
- **`peak_grid_kwh` may differ from a reference plan.** It is reported, not
  optimised (PS §5.2), so equally optimal schedules can peak differently.
- **Provider availability is the team's responsibility** (Guide §04). Keys,
  quota, and rate limits must hold for the whole judging window.
- **The zero-optimum scoring rule is incomplete in the source PDF**, and is
  recorded rather than guessed at.
- **The official rulebook referenced by the guide was not among the supplied
  files.** Any additional conditions it carries must be checked separately.

---

## 11. Secret handling

- No API key, token, or populated `.env` is committed. `.gitignore` excludes
  `.env` and `.env.*` while keeping `.env.example`, which holds placeholders.
- No secret is baked into the Docker image; credentials are injected at run
  time, and the image was inspected to confirm it carries none.
- API error bodies never contain stack traces, prompts, provider messages, or
  configuration. Unexpected failures return a correlation id; the detail stays
  in the logs.
- Logs record stage timings, outcomes, failover reasons, model identifiers, and
  solver status. `Settings.describe()` reports whether a key is *present*, never
  its value.
- Request bodies are not echoed into validation messages — only the field
  location and the failure reason.
- Only the organizers' synthetic challenge data is used. No live campus,
  utility, billing, or personal data appears anywhere in this repository.

---

## 12. Credits

| Dependency | Role |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | HTTP framework |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| [Pydantic](https://docs.pydantic.dev/) | Schema validation |
| [PuLP](https://coin-or.github.io/pulp/) | LP modelling |
| [COIN-OR CBC](https://github.com/coin-or/Cbc) | LP solver |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | Client for Groq's OpenAI-compatible endpoint |
| [Groq](https://groq.com/) | Inference provider |
| [pytest](https://docs.pytest.org/) · [httpx](https://www.python-httpx.org/) | Testing |
| [Docker](https://www.docker.com/) · [Debian](https://www.debian.org/) | Container image and base |

The problem statement, participant guide, and public sample cases are the
organizers'. Core architecture and logic are the team's own work. AI coding
assistance was used, as permitted by Guide §04.
