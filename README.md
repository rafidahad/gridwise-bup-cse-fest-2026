# GridWise — LLM-Assisted Campus Energy Optimization

BUP CSE Fest 2026 · Hackathon Online Preliminary

A single HTTP service that receives 24 hours of campus demand, solar, and tariff
data plus 1–3 natural-language operator notes, interprets those notes with a
language model, compiles the validated interpretation into hard constraints,
solves for the cheapest feasible 24-hour schedule, and independently replays the
schedule before returning it.

| | |
|---|---|
| Health endpoint | `GET /health` |
| Main endpoint | `POST /optimize-energy` |
| Model | Groq `openai/gpt-oss-120b` |
| Solver | PuLP + COIN-OR CBC |
| Tests | 187, no API key required |
| Public samples | 10/10 valid · 10/10 optimal cost · p95 1.93 s |

---

## 1. Architecture

```text
POST /optimize-energy
        │
        ▼
FastAPI + Pydantic ─────── structural validation → 400
        │                  semantic validation   → 422
        ▼
LLM interpretation ─────── one call, JSON output, notes → structured directives
        │                  bounded repair on failure
        ▼
Deterministic guardrails ─ model output is untrusted data until it validates
        │
        ▼
Directive Compiler ─────── directives → hourly bounds: effective solar,
        │                  minimum reserve, charge/discharge ceilings, grid caps
        ▼
PuLP + CBC ─────────────── one 24-hour linear program
        │                  minimise Σ grid[h] · tariff[h]
        ▼
Replay Validator ───────── independent re-check of the response about to be sent
        │
        ▼
Exact JSON response (200)
```

The language model **is** the interpretation path: its structured output is what
produces the optimizer's constraints (Guide §04). Deterministic code validates
and applies that output but never authors an interpretation of its own, and no
phrase matching is used as an interpretation path anywhere in this service. If
the model path fails, the service returns a controlled error rather than falling
back to a hard-coded reading.

| Component | Choice | Responsibility |
|---|---|---|
| Language | Python 3.12 | Application and deterministic logic |
| API | FastAPI + Uvicorn | Endpoints, JSON, explicit status codes |
| Schemas | Pydantic v2 | Request and response contracts |
| Interpretation | Groq `openai/gpt-oss-120b` | Notes → structured directives |
| Compiler | Plain Python | Directives → hourly bounds |
| Optimization | PuLP + CBC | Minimum-cost feasible schedule |
| Replay | Independent plain Python | Verify the actual response |

### Layout

```text
app/
  main.py        routes, status-code contract, orchestration, response builder
  schemas.py     request / directive / response contracts
  llm.py         prompt, call budget, bounded repair, key rotation
  directives.py  guardrails on model output, and the Directive Compiler
  optimizer.py   the linear program, solver handling, plan construction
  validator.py   independent replay of the response
  config.py      environment-backed settings
scripts/
  verify_core.py       offline: compiler + optimizer + replay vs published optima
  run_public_cases.py  HTTP: the 10 public cases against a running service
  semantic_eval.py     live: model accuracy on 166 labelled cases
tests/                 187 tests
data/                  official public cases, held-out and extended test packs
docs/                  organizer documents, solution plan, sample request/response
```

---

## 2. Quickstart

Requires Python 3.12+. CBC ships inside the PuLP wheel, so no separate solver
installation is needed.

```bash
git clone https://github.com/rafidahad/gridwise-bup-cse-fest-2026.git
```

```bash
cd gridwise-bup-cse-fest-2026
```

```bash
python -m venv .venv && source .venv/bin/activate
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Edit `.env` and set `GRIDWISE_PRIMARY_API_KEYS` to your Groq API key. The file
contains placeholders only, is git-ignored, and must never be committed.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Verify

```bash
curl -s http://localhost:8000/health
```

Returns `{"status": "ok"}` within 60 seconds of start — measured at 3 seconds.

```bash
python scripts/run_public_cases.py
```

Posts all ten official public samples, independently replays each returned plan,
compares the interpretation against the published reference, and reports cost
quality and p95 latency. Exit code 0 means every case passed.

---

## 3. Configuration

All settings come from the environment. `.env` is read at startup and never
overrides a variable that is already exported, so `docker run -e ...` and CI
remain authoritative.

| Variable | Meaning | Default |
|---|---|---|
| `GRIDWISE_PRIMARY_API_KEYS` | Groq API keys, comma separated | — |
| `GRIDWISE_PRIMARY_BASE_URL` | OpenAI-compatible base URL | `https://api.groq.com/openai/v1` |
| `GRIDWISE_PRIMARY_MODEL` | Model identifier | `openai/gpt-oss-120b` |
| `GRIDWISE_BACKUP_API_KEYS` | Optional secondary keys | — (disabled) |
| `GRIDWISE_BACKUP_BASE_URL` | Optional secondary base URL | — (disabled) |
| `GRIDWISE_BACKUP_MODEL` | Optional secondary model | — (disabled) |
| `GRIDWISE_TEMPERATURE` | Sampling temperature | `0` |
| `GRIDWISE_SEED` | Best-effort seed | `7` |
| `GRIDWISE_MAX_OUTPUT_TOKENS` | Output cap | `1200` |
| `GRIDWISE_REQUEST_DEADLINE_S` | Overall per-request budget | `25` |
| `GRIDWISE_PRIMARY_TIMEOUT_S` | Model call deadline | `9` |
| `GRIDWISE_REPAIR_TIMEOUT_S` | Repair call deadline | `7` |
| `GRIDWISE_SOLVER_TIME_LIMIT_S` | CBC time limit | `10` |
| `GRIDWISE_HOST` / `GRIDWISE_PORT` | Bind address and port | `0.0.0.0` / `8000` |
| `GRIDWISE_LOG_LEVEL` | Log verbosity | `INFO` |

### API keys

Several comma-separated keys are accepted. Every request starts at the first
key, so **order is preference order — put the highest-limit key first**. If a key
is rate limited, out of quota, or rejected, the service advances to the next one
within that same request, up to three credentials. Rotation is not a second
interpretation attempt and does not consume the call budget; a timeout or
connection failure stops rather than burning the remaining keys. Keys never
appear in logs — only their position, as `key 2/3`.

### The optional secondary model

The service runs on one provider by default. A backup is permitted but never
required: Guide §04 says "a local or backup model is allowed", and the solution
plan treats it as a reliability feature rather than an obligation. With the
`GRIDWISE_BACKUP_*` variables unset, the third recovery step is skipped and
recovery ends after the targeted repair. Any OpenAI-compatible endpoint can fill
the slot.

### Inference settings

Temperature is `0` for extraction, and a seed is sent only when the provider
accepts one. Seeded sampling is best-effort and not a determinism guarantee. The
prompt version and schema version are pinned in `app/config.py` so recorded
evidence stays attributable.

---

## 4. API

### `GET /health`

Returns HTTP 200 and exactly `{"status": "ok"}` when ready. Performs no network
or solver work, so it stays responsive while optimizations are in flight.

### `POST /optimize-energy`

Accepts one scenario object and returns one interpretation-and-plan object. The
schemas are defined in the Problem Statement §§7 and 10 and implemented in
`app/schemas.py`.

| Status | Meaning |
|---|---|
| 200 | Successful optimization |
| 400 | Malformed JSON, or a structurally invalid request |
| 422 | Well-formed request that admits no feasible schedule |
| 500 | Controlled internal error — no stack trace, prompt, or credential |

The status codes are fixed by the specification; the error body is not. This
service uses one consistent shape:

```json
{"error": {"code": "invalid_request", "message": "hours: must cover 0 through 23"}}
```

A complete request/response pair is in
[`docs/sample_request_response.json`](docs/sample_request_response.json).

**Input policies**, chosen where the sources are silent: unknown extra fields are
ignored rather than rejected; `hours` need not arrive in ascending order, since
rows are indexed by `hour`; hour numerics must be finite but carry no sign
restriction.

---

## 5. How it works

### Interpretation

One call interprets all 1–3 notes together. The prompt supplies the six
permitted directive types with their exact shapes, the whole-hour convention
with worked examples showing that the window length is `end - start`, both
percentage readings (`to 20%` → `0.2`, `by 80%` → `0.2`), the charge/discharge
vocabulary distinction, and the battery capacity so percentage reserves resolve.
Notes are passed as JSON-quoted data under an explicit instruction that note text
is never a command to the model.

### Guardrails

Model output is untrusted until it passes `app/directives.py`: the type is one of
six, each note maps exactly once, `applies` matches `no_op` semantics, `hours` are
integers 0–23 and non-empty, `factor` is finite in `[0, 1]`, reserves are within
capacity, and grid caps are non-negative. The guardrails never repair a value,
substitute a type, invent an hour, or drop a note. Unknown adjustment keys are
discarded so the emitted object is exactly the required shape.

### Recovery

At most three inference calls per request: the primary attempt, one targeted
repair naming only the failing notes, and an optional secondary model. A provider
failure skips repair. There is no partial-success path — if any note is
unresolved, the whole response is withheld and a controlled error is returned.

### Optimization

One linear program over all 24 hours with a single signed variable per hour:

```text
b[h] > 0 charge      b[h] < 0 discharge      b[h] = 0 idle

-max_discharge[h] ≤ b[h] ≤ max_charge[h]
E[h] = E[h-1] + b[h],  E[-1] = initial,  E[23] = initial
grid[h] + solar_used[h] = demand[h] + b[h]
0 ≤ solar_used[h] ≤ effective_solar[h],  0 ≤ grid[h] ≤ max_grid[h]
minimum_reserve[h] ≤ E[h] ≤ capacity

minimise Σ grid[h] · tariff[h]
```

One signed variable makes a simultaneous charge and discharge structurally
impossible and keeps the model a pure LP. Nothing is added to the published
objective — no degradation cost, efficiency loss, export revenue, or peak
penalty. Solver status is checked: a merely feasible result is never labelled
optimal.

### Replay

Before anything is returned, `app/validator.py` re-derives every limit from the
original request without importing the compiler's output, then checks the
scenario echo, interpretation order, 24 unique hours, non-negative values,
action consistency, state transitions, capacity and reserve, rate limits, solar
usage, the hourly balance, every directive window and cap, end-of-day
neutrality, and the three reported totals. A plan that fails replay is never
returned.

---

## 6. Documented policies

Where the sources leave a question open, behaviour is fixed here rather than left
undefined at run time.

- **Overlapping solar reductions** — factors are multiplied and applied once to
  the original forecast. Using less solar than permitted is always legal, while
  using more invalidates the entire case, so the product is the only choice that
  cannot cause overuse. If the result is infeasible, the overlapped hours relax
  once to `min(factors)` and the model is re-solved.
- **Empty `hours` on an applicable directive** — treated as an extraction failure
  and re-interpreted. Never accepted as a rule affecting nothing, and never
  rewritten to `no_op`.
- **Battery feasibility** — `minimum_energy_kwh ≤ initial_energy_kwh ≤
  capacity_kwh` is required, derived from end-of-day neutrality rather than
  invented. A temporary directive reserve above the initial energy is not
  rejected; the optimizer charges to meet it.
- **Error body shape** — chosen and documented, not mandated.

---

## 7. Testing

```bash
pip install -r requirements-dev.txt
```

```bash
pytest
```

Expected: **187 passed**. No network access or API key is required — provider
calls are stubbed so the contract, compiler, optimizer, replay, and recovery
policy are all exercised deterministically.

| Suite | Coverage |
|---|---|
| `test_api.py` | Endpoints, status codes, response schema, controlled failure, no secret leakage |
| `test_directives.py` | Guardrails; each directive changes precisely its intended bound |
| `test_optimizer.py` | Physical rules, directive enforcement, public reference costs |
| `test_validator.py` | Corrupted plans rejected for the specific rule they break |
| `test_llm.py` | Call budget, repair, failover, key rotation |
| `test_generalization.py` | Generated scenarios, an independent optimum, concurrency |
| `test_team_packs.py` | Extended pack labels and 18 further reference optima |

### Offline optimizer check

```bash
python scripts/verify_core.py
```

Reproduces all ten published optimal costs from the reference interpretations,
bypassing the model. The same optima are independently reproduced by a dynamic
program in `tests/reference_dp.py` that shares no code with the linear program.

### Live interpretation accuracy

```bash
python scripts/semantic_eval.py --packs all --repeat 3 --json results.json
```

Scores 166 labelled cases across the held-out and extended packs along the
rubric's dimensions, plus paraphrase-cluster agreement and run-to-run
consistency.

### Measured performance

| Measurement | Result |
|---|---|
| Live request, p95 over the 10 public samples | 1.93 s |
| Deterministic path alone, p95 over 100 runs | 88 ms |
| Container start to `/health` 200 | 3 s |
| Tokens per interpretation request | 2,421 |

---

## 8. Docker

```bash
docker build -t gridwise:1.0.0 .
```

```bash
docker run --rm -p 8000:8000 -e GRIDWISE_PRIMARY_API_KEYS=your-groq-key gridwise:1.0.0
```

```bash
curl -s http://localhost:8000/health
```

The image binds `0.0.0.0`, exposes port 8000, runs unprivileged, and contains no
baked-in credentials. Both CBC front ends are available — PuLP's bundled binary
and Debian's `coinor-cbc` — so a host where one will not execute still has a
working solver. `docker run --rm gridwise:1.0.0 python scripts/verify_core.py`
reproduces all ten published optima from inside the image.

---

## 9. Known limitations

- **Single provider by choice.** No secondary model is configured, so a provider
  outage fails every request. This is permitted rather than an oversight, but it
  is a deliberate trade.
- **Throughput depends on the account tier.** One request measures 2,421 tokens,
  so an 8,000 TPM key sustains about 3.3 requests per minute and a 250,000 TPM
  key about 103. Provision accordingly.
- **Interpretation accuracy is measured on the public set.** A live run matched
  10/10 published interpretations and reached 10/10 optimal costs. The 166-case
  extended packs have not yet been run end to end against a live model.
- **Optimality is optimality of the model supplied.** A proven optimum cannot
  repair a mistaken reading of a note. Replay proves the plan obeys the
  directives this service interpreted; only the organizer's harness compares
  those to ground truth.
- **`peak_grid_kwh` may differ from a reference plan.** It is reported, not
  optimised, so equally optimal schedules can peak differently.

---

## 10. Security

- No API key, token, or populated `.env` is committed; `.gitignore` excludes them.
- No secret is baked into the Docker image; credentials are injected at run time.
- API error bodies contain no stack traces, prompts, or provider messages.
- Logs record timings, outcomes, model identifiers, and solver status — never
  credentials.
- Only the organizers' synthetic challenge data is used.

---

## 11. Credits

| Dependency | Role |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) · [Uvicorn](https://www.uvicorn.org/) | HTTP framework and ASGI server |
| [Pydantic](https://docs.pydantic.dev/) | Schema validation |
| [PuLP](https://coin-or.github.io/pulp/) · [COIN-OR CBC](https://github.com/coin-or/Cbc) | LP modelling and solver |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | Client for Groq's OpenAI-compatible endpoint |
| [Groq](https://groq.com/) | Inference provider |
| [pytest](https://docs.pytest.org/) · [httpx](https://www.python-httpx.org/) | Testing |
| [Docker](https://www.docker.com/) · [Debian](https://www.debian.org/) | Container image and base |

The problem statement, participant guide, and public sample cases are the
organizers'. Core architecture and logic are the team's own work. AI coding
assistance was used, as permitted by Guide §04.
