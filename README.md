# GridWise — LLM-Assisted Campus Energy Optimization

BUP CSE Fest 2026 · Hackathon Online Preliminary · `POST /optimize-energy`

A single HTTP service that reads 24 hours of campus demand, solar, and tariff
data plus 1–3 natural-language operator notes, interprets those notes with a
language model, turns the validated interpretation into hard constraints, solves
for the cheapest feasible 24-hour schedule, and independently replays the
schedule before returning it.

> **Status:** Stage 1 of 7 (contract and delivery skeleton). `GET /health` is
> live. `POST /optimize-energy` is wired in Stage 4. This notice is removed when
> the pipeline is complete.

---

## 1. Architecture

```text
POST /optimize-energy
        │
        ▼
FastAPI + Pydantic ── structural validation (400) / semantic validation (422)
        │
        ▼
LLM interpretation ── one call, structured output, notes → directives
        │                 bounded repair → independent-provider backup
        ▼
Deterministic guardrails ── untrusted model output validated before use
        │
        ▼
Directive Compiler ── directives → hourly bounds (effective solar, reserve,
        │              charge/discharge ceilings, grid caps)
        ▼
PuLP + CBC ── one 24-hour linear program, minimise Σ grid[h]·tariff[h]
        │
        ▼
Response builder ── actions, states, recomputed totals
        │
        ▼
Replay Validator ── independent re-check of the response about to be returned
        │
        ▼
Exact JSON response
```

The language model **is** the interpretation path: its structured output is what
produces the optimizer's constraints (Guide §04). Deterministic code validates
that output and applies it; it never substitutes for it. Phrase matching is not
used as an interpretation path anywhere in this service.

| Component | Choice | Responsibility |
|---|---|---|
| Language | Python 3.12 (image) | Application and deterministic logic |
| API | FastAPI + Uvicorn | Endpoints, JSON, explicit status codes |
| Schemas | Pydantic v2 | Request/response contracts |
| Interpretation | Groq (primary) + NVIDIA NIM (backup) | Notes → structured directives |
| Compiler | Plain Python | Directives → hourly bounds |
| Optimization | PuLP + CBC | Minimum-cost feasible schedule |
| Replay | Independent plain Python | Verify the actual response |
| Tests | pytest + httpx | Contract, compiler, optimizer, replay, integration |

---

## 2. Quickstart (local, from a clean environment)

Requires Python 3.12+ and `git`. CBC ships inside the PuLP wheel, so no separate
solver install is needed for local development.

```bash
git clone https://github.com/plasma-gith/GridWise.git
cd GridWise
```

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env    # then edit .env and fill in your two API keys
```

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service listens on `http://localhost:8000`.

### Verify health

```bash
curl -s http://localhost:8000/health
```

Expected, within 60 seconds of start (Guide §08):

```json
{"status": "ok"}
```

---

## 3. Configuration

Every value is supplied through the environment. `.env.example` is the template;
it contains **placeholders only, never real credentials**. Copy it to `.env`,
which is git-ignored and must never be committed (Guide §04).

| Variable | Meaning |
|---|---|
| `GRIDWISE_PRIMARY_API_KEY` | Primary provider key (Groq) |
| `GRIDWISE_PRIMARY_BASE_URL` | Primary OpenAI-compatible base URL |
| `GRIDWISE_PRIMARY_MODEL` | Primary model identifier |
| `GRIDWISE_BACKUP_API_KEY` | Backup provider key (NVIDIA NIM) |
| `GRIDWISE_BACKUP_BASE_URL` | Backup OpenAI-compatible base URL |
| `GRIDWISE_BACKUP_MODEL` | Backup model identifier |
| `GRIDWISE_TEMPERATURE` | Sampling temperature; `0` for extraction |
| `GRIDWISE_SEED` | Best-effort seed; omitted when a provider rejects it |
| `GRIDWISE_MAX_OUTPUT_TOKENS` | Output cap for the interpretation call |
| `GRIDWISE_REQUEST_DEADLINE_S` | Overall per-request budget, below the judge's 30 s |
| `GRIDWISE_PRIMARY_TIMEOUT_S` | Primary call deadline |
| `GRIDWISE_REPAIR_TIMEOUT_S` | Targeted repair call deadline |
| `GRIDWISE_BACKUP_TIMEOUT_S` | Backup call deadline |
| `GRIDWISE_SOLVER_TIME_LIMIT_S` | CBC time limit |
| `GRIDWISE_HOST` / `GRIDWISE_PORT` | Bind address and port |
| `GRIDWISE_LOG_LEVEL` | Log verbosity |

### Models and providers

Groq and NVIDIA NIM both expose OpenAI-compatible chat-completions endpoints, so
one client abstraction serves both while they remain **independent providers** —
a Groq outage, rate limit, or quota failure does not disable the backup
(plan §6.1). Both models perform genuine language interpretation, are held to the
same schema and guardrails, and are measured on the same semantic test suite.

Inference settings are recorded rather than assumed: temperature `0`, a
best-effort seed where the provider supports one, and a pinned prompt version and
schema version carried in `app/config.py`. Low temperature and seeding reduce
avoidable variability; they do **not** guarantee deterministic model output, and
semantic accuracy is measured separately.

---

## 4. API

### `GET /health`

Returns HTTP 200 and `{"status": "ok"}` when the service is ready. Does no
network or solver work, so it stays responsive while optimization requests run.

### `POST /optimize-energy`

Accepts one scenario object and returns one interpretation-and-plan object. The
exact request and response schemas are defined in PS §§7 and 10 and implemented
in `app/schemas.py`.

| Status | Meaning |
|---|---|
| 200 | Successful optimization |
| 400 | Malformed JSON, or a structurally invalid request |
| 422 | Well-formed request that cannot admit a feasible schedule |
| 500 | Controlled internal error — no stack trace, no prompt, no credential |

PS §6.1 fixes the status codes but not the error body. This service uses one
consistent shape throughout (documented here as a choice, not as a mandated
schema):

```json
{"error": {"code": "invalid_request", "message": "hours: must cover 0 through 23"}}
```

---

## 5. Documented interpretation policies

Where the organizer sources leave something open, the behaviour is fixed here
rather than left undefined at run time (plan §17).

- **Overlapping solar reductions** — factors for an hour are multiplied and
  applied once to the original forecast; with no directive the factor is 1. If
  the resulting model is infeasible, the overlapped hours relax once to
  `min(factors)` and the model is re-solved. The product is chosen because it
  cannot cause effective-solar overuse under either reading of the rule, and
  overuse invalidates a whole case (Guide §09).
- **Empty `hours` on an applicable directive** — treated as an extraction
  failure and sent for re-interpretation. It is never accepted as a rule that
  silently affects nothing, and never rewritten to `no_op`. Only genuine
  language interpretation may produce `no_op`.
- **Battery feasibility** — `minimum_energy_kwh ≤ initial_energy_kwh ≤
  capacity_kwh` is checked explicitly and rejected with 422 when violated. This
  is derived from PS §§9.2 and 9.6, not an invented initial-state rule: end-of-day
  neutrality forces `E_after[23] = initial_energy_kwh`, which must itself lie
  within the bounds. A *temporary* directive reserve above the initial energy is
  **not** rejected — the schedule may charge to meet it, and the optimizer
  decides.
- **Extra request fields** — ignored, never rejected.
- **Hour ordering in the request** — not required; rows are indexed by `hour`.

---

## 6. Repository layout

```text
app/
  main.py        routes, status-code contract, orchestration
  schemas.py     request/directive/response contracts
  config.py      environment-backed settings
docs/
  PROBLEM_STATEMENT.md   transcription of the organizer problem statement
  PARTICIPANT_GUIDE.md   transcription of the organizer guide and rubric
  SOLUTION_PLAN.md       the high-level solution plan this build follows
data/
  public_cases.json      the organizer's 10 public sample cases
tests/
scripts/
Dockerfile
requirements.txt
.env.example
```

---

## 7. Credits

Built for BUP CSE Fest 2026. External tools and dependencies used:

| Dependency | Role |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | HTTP framework |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| [Pydantic](https://docs.pydantic.dev/) | Schema validation |
| [PuLP](https://coin-or.github.io/pulp/) | LP modelling |
| [COIN-OR CBC](https://github.com/coin-or/Cbc) | LP solver |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | Client for the OpenAI-compatible Groq and NVIDIA endpoints |
| [Groq](https://groq.com/) | Primary inference provider |
| [NVIDIA NIM](https://build.nvidia.com/) | Backup inference provider |
| [pytest](https://docs.pytest.org/) / [httpx](https://www.python-httpx.org/) | Testing |

Problem statement, participant guide, and public sample cases are the
organizers'. Core architecture and logic are the team's own work. AI coding
assistance was used, as permitted by Guide §04.

## 8. Secret handling

- No API key, token, or populated `.env` is committed to this repository.
- No secret is baked into the Docker image; credentials are injected at run time.
- API error bodies carry no stack traces, prompts, or provider details.
- Logs record stage timings, outcomes, model identifiers, and solver status —
  never credentials or secret-bearing prompts.
- All scenario data used here is the organizers' synthetic challenge data.
