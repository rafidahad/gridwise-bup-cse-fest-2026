# 3-Minute Architecture & Solution Video — Script

Guide §02 requires a video of at most 3:00. It carries **no base points**; it is
reviewed only to break a tie between teams on the same total score (Guide §06),
where reviewers compare problem understanding, architecture clarity, the
LLM → guardrails → optimizer flow, and the run/test explanation.

Production quality is not judged. Screen recording with a voice-over is enough.

---

## Timing

| Segment | Time | Running |
|---|---|---|
| 1. The problem | 0:00–0:25 | 0:25 |
| 2. Architecture | 0:25–1:05 | 1:05 |
| 3. Guardrails and recovery | 1:05–1:40 | 1:40 |
| 4. Compiler and optimizer | 1:40–2:10 | 2:10 |
| 5. Replay | 2:10–2:30 | 2:30 |
| 6. Run and test it | 2:30–2:55 | 2:55 |

Leaves five seconds of headroom. Do not exceed 3:00.

---

## 1. The problem — 25s

> **Show:** a public case's `operator_notes` beside the `hours` array.

Campus operators write ordinary sentences. *"Cloud cover during panel inspection
will leave about half of the forecast solar from 10 AM until noon."* Our service
has to understand that, turn it into a hard constraint — `solar_reduction`, hours
10 and 11, factor 0.5 — apply it, and return the cheapest valid 24-hour schedule.

Two things are scored separately: reading the note correctly, and actually
obeying it. A cheap schedule built on an ignored directive scores nothing.

---

## 2. Architecture — 40s

> **Show:** the architecture diagram from the README.

Six stages. FastAPI validates the request. A language model interprets every
note into structured directives. Deterministic guardrails decide whether that
output is usable. The Directive Compiler turns it into hourly bounds. PuLP and
CBC solve all 24 hours as one linear program. And an independent Replay
Validator re-checks the response before we send it.

The key point for the mandatory requirement: **the model's structured output is
what produces the optimizer's constraints.** We do no phrase matching anywhere.
If every model path fails we return a controlled error rather than falling back
to a hard-coded reading.

---

## 3. Guardrails and recovery — 35s

> **Show:** `app/directives.py` `validate_interpretation`, then the failover
> block in `app/llm.py`.

Model output is untrusted data until it validates. We check the type is one of
the six, that each note maps exactly once, that hours are integers 0–23 and not
empty, that a solar factor is between 0 and 1, that a reserve fits inside
capacity.

The guardrails never repair a value or invent an hour. If an entry fails we ask
the model again — only for that note, with its original index and the specific
feedback. If the provider itself failed, we skip repair and go straight to a
backup on an **independent** provider, so one outage cannot take both down.

At most three calls per request. And there is no partial success: if any note is
still unresolved, we withhold the whole plan. We never drop a note or quietly
turn it into a `no_op`.

---

## 4. Compiler and optimizer — 30s

> **Show:** `compile_limits`, then the LP in `app/optimizer.py`.

The compiler produces five hourly arrays — effective solar, minimum reserve,
charge and discharge ceilings, grid caps. Overlapping rules combine
deterministically: greatest reserve, smallest cap, union of the prohibitions.
Past this point the model has no influence; the optimizer only sees numbers.

One signed variable per hour carries the battery action, so charging and
discharging cannot both happen and we stay a pure LP. All 24 hours solve
together, which is what lets the plan prepare for an evening grid cap or
end-of-day neutrality.

> **Optional, if time allows:** mention that overlapping solar reductions
> multiply, because under-using solar is always legal while over-using it
> invalidates the whole case.

---

## 5. Replay — 20s

> **Show:** `app/validator.py`, then a corrupted-plan test being rejected.

Before anything goes out, we replay it. The validator re-derives every limit
from the original request — deliberately *not* reusing the compiler's output,
because two views of one bug would prove nothing. Balance, battery states,
rates, reserves, caps, neutrality, and the totals recomputed from the rows.

A plan that fails its own replay is never returned.

---

## 6. Run and test it — 25s

> **Show:** a terminal running the commands.

```bash
docker run --rm -p 8000:8000 -e GRIDWISE_PRIMARY_API_KEY=... -e GRIDWISE_BACKUP_API_KEY=... gridwise:1.0.0
```

```bash
curl -s http://localhost:8000/health
```

```bash
python scripts/run_public_cases.py
```

Health comes up in about three seconds. All ten public cases return valid plans
at the published optimal cost. The deterministic path — compile, solve, replay —
runs at 88 milliseconds p95, so essentially the whole latency budget is left for
the model.

`pytest` runs 172 tests with no API key needed, including an independent dynamic
program that reproduces every published optimum without touching the linear
program.

---

## Checklist before uploading

- [ ] Length is 3:00 or less.
- [ ] Audible narration; the code on screen is legible at the target resolution.
- [ ] No API key, `.env` contents, or account page is ever visible on screen.
- [ ] The LLM → guardrails → optimizer flow is explicitly stated.
- [ ] The run and test commands are actually demonstrated, not just described.
- [ ] The link is accessible to organizers without a login request, and stays up
      for the whole judging window.
