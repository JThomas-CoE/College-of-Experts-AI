# demo.py — Current Operational Specification

**Document of Record — College of Experts Interactive CLI**
**Status: ACTIVE — aligned to current `demo.py` implementation**

---

## Overview

`demo.py` is the current interactive CLI for the College of Experts demo.

It is no longer just a build blueprint. This document now describes the **actual current behavior** of the implementation, including the simplifications and safeguards added during stabilization work.

Key characteristics of the current implementation:

- interactive CLI shell
- session memory and artifact persistence
- template retrieval and optional KB integration
- benchmark-style `TIER2` specialist flow for supported domains
- **process-isolated inference** via `inference_worker.py`
- **supervisor-only fallback/synthetic handling** for disabled or unavailable specialists
- `TIER3` implementation still exists in code, but **interactive dispatch currently blocks TIER3 execution** and asks the user to split multi-domain requests

The implementation source of truth is:

- [demo.py](demo.py)
- [inference_worker.py](inference_worker.py)

---

## V1 Policy Table (Source of Truth)

| Policy | Current Decision | Notes |
|---|---|---|
| Supervisor runtime model | Process-isolated worker inference | Parent process tracks model paths; worker process loads model, generates once, returns JSON, exits |
| Specialist runtime model | Process-isolated worker inference | Specialist residency in Python state is logical/path-based; actual inference is one worker process per call |
| Specialist residency policy | Benchmark-style unload after `TIER2` specialist selection | No long-lived resident specialist runtime objects in the parent process |
| Generation-state lifecycle | Per-call scratch only | No persistent KV/cache reuse across calls |
| `TIER2` synthesis | Enabled, but only when specialist-best result is not already `PASS` | Benchmark-style “keep better candidate” logic |
| `TIER3` synthesis | Implemented in code but not used by main dispatch | Interactive dispatcher blocks `TIER3` and asks user to split prompts |
| Rendering mode | Default: `intermediate+final` | Configurable via `/render` |
| Grading/retry framework | Hybrid: deterministic guards + specialist self-grading for trusted domains + supervisor grading for others | `code` and `math` use specialist self-grading; deterministic guards run first |
| Disabled specialists | `medical`, `legal` | Routed through supervisor-only specialist-capacity handling |
| Empty-output handling | Never print blank final response | Friendly fallback message used if pipeline output is empty |

**Alignment rule:** [demo.py](demo.py) should remain aligned with this table.

---

## Runtime Architecture

### 1. Coordinator process

The main CLI process owns:

- routing
- session memory
- template lookup
- KB integration
- rendering
- prompt construction
- grading and retry orchestration

It does **not** keep live OGA model/tokenizer handles for generation anymore.

### 2. Worker-based inference

Actual inference is delegated to [inference_worker.py](inference_worker.py).

Current worker contract:

- parent writes JSON request to a temp file
- worker loads the requested model
- worker performs one generation pass
- worker writes JSON response
- worker exits

This applies to both:

- supervisor calls
- specialist calls

### 3. Thinking-model handling

For thinking models such as Nanbeige:

- phase 1 runs hidden thinking up to the configured think budget
- if `</think>` never appears, the system force-closes the think block
- phase 2 reinjects the prompt plus the closed think block and asks for the visible answer

The current token budgeting was updated so the configured think budget can actually be used instead of being accidentally crowded out by answer-space reservation.

---

## Environment and Core Constraints

- intended runtime environment: `zimage`
- device default: `dml`
- embedder defaults to CPU when using `dml`
- debug trace supported via `--debug-trace`
- empty final outputs are replaced with a friendly fallback message
- final rendering is copy/paste-safe text, not dependent on boxed markdown panels

---

## Model Registry

Current logical registry in code:

```python
DOMAIN_MODELS = {
    "code":       "Qwen2.5-Coder-7B-DML",
    "sql":        "sqlcoder-7b-2-DML",
    "math":       "Qwen2.5-Math-7B-DML",
    "medical":    "BioMistral-7B-DML",
    "legal":      "law-LLM-DML",
    "web":        "Qwen2.5-Coder-7B-DML",
    "supervisor": "Nanbeige4.1-3B-ONNX-INT4",
}
```

Current disabled specialist domains:

```python
DISABLED_SPECIALIST_DOMAINS = {"medical", "legal"}
```

Meaning:

- `medical` and `legal` still classify as those domains
- but they do **not** load their domain specialists
- they run through a **supervisor acting in specialist capacity** path instead

---

## Current `ModelManager` Behavior

`ModelManager` now behaves as a path-oriented inference coordinator, not a persistent live-model container.

### Loaded at startup

- supervisor **path** is resolved and stored
- embedder is loaded in-process for templates / KB

### Specialist handling

- `begin_load_specialist()` now mostly validates/registers the target path
- `await_specialist()` is effectively a compatibility no-op in the worker architecture
- `generate_specialist()` calls the worker with the currently selected specialist path
- `unload_specialist()` clears the selected specialist path and logs the event

### Important behavioral consequence

The parent process may say a specialist is “ready”, but this means:

- the path is valid and selected
- not that a long-lived OGA runtime object remains resident in the parent process

---

## Routing and Classification

`classify_query()` is still a two-stage classifier, but several important guards have been added.

### Stage 1 heuristic guards

Current direct guards include:

- trivial greetings → `TIER1 / ['supervisor']`
- short general factual/reference questions → `TIER1 / ['supervisor']`
- veterinary health questions → `TIER2 / ['supervisor']`
- creative-writing requests (story/poem/etc.) → `TIER2 / ['supervisor']`
- single-file/self-contained HTML web tasks collapse `code+web` to `TIER2 / ['web']`

### Stage 2 supervisor classification

Used when stage 1 is not high-confidence enough.

Important current reality:

- classification is still hybrid, not purely heuristic
- stage 1 can short-circuit many obvious cases locally
- otherwise stage 2 still calls the supervisor as a JSON routing classifier
- short factual questions now have an extra guard to keep them out of pathological stage-2 all-domain drift

Output format expected:

- JSON with `tier`, `domains`, `interpretation`, `confidence`, `recheck_needed`

If parsing fails:

- fallback is `TIER2 / ['supervisor']`

### Follow-up continuity guards

Session-aware routing preserves continuity for follow-up edit prompts:

- `TIER1` misroutes can be promoted back to previous task domains
- `web` follow-ups that drift to `code` without explicit code intent can be forced back to `web`

### Current dispatcher behavior for `TIER3`

Although `run_tier3()` still exists in the file, the interactive dispatcher currently does this:

- if classified as `TIER3`
- print a split-request message
- do **not** execute the multi-domain pipeline

So for interactive use, effective routing is:

- `TIER1`
- `TIER2`
- blocked `TIER3`

---

## TIER1 — Current Behavior

`run_tier1()`:

- supervisor only
- no specialist
- no session task persistence
- uses the general supervisor answer system prompt
- if the answer comes back empty, a resilient helper retries once with a simpler direct-answer/no-think instruction

Current intent:

- keep trivial/general requests fast
- allow short factual/reference questions to be answered directly by the supervisor
- avoid blank outputs from supervisor-only routes

---

## TIER2 — Current Behavior

`run_tier2()` now has **two distinct modes**.

### A. Supervisor-only specialist-capacity route

Used when:

- `domain == 'supervisor'`, or
- the domain specialist is disabled/unavailable (`medical`, `legal`, or missing model)

This path:

- does **not** enter the normal specialist grading/retry/synthesis chain
- builds a stricter supervisor system prompt for “acting as a TIER2 specialist”
- runs one substantive supervisor answer pass with `THINK_BUDGET_T2`
- if that pass is empty, retries once with the same prompt but `think_budget=0`
- writes the result to session store as `TIER2` or `TIER2_SYNTHETIC`
- returns immediately

This is currently the path used for:

- `legal`
- `medical`
- direct supervisor-routed factual/general `TIER2` requests

### B. Benchmark-style real specialist route

Used when a real specialist is available and enabled.

Current high-level flow:

1. extract qualifiers
2. patch domain persona
3. use a direct task prompt (benchmark-style; no normal pre-specialist supervisor enhancement pass)
4. generate specialist draft
5. grade draft
6. retry once if needed
7. unload specialist after selection
8. run supervisor synthesis only if best specialist result is not already `PASS`
9. optionally run score-gated supervisor fallback

Important: this is much simpler than the earlier architecture.

---

## TIER2 Prompting Rules

### Normal specialist-backed TIER2

Current implementation prefers a direct task prompt path.

That means:

- the old always-on “supervisor prompt engineer before specialist generation” flow is **not** the normal `TIER2` path anymore
- `enhance` remains a runtime setting for compatibility, but benchmark-style direct prompting is now the dominant behavior

### Supervisor acting as specialist

There is now a dedicated stricter prompt builder for this case.

It tells the supervisor to:

- act as the requested domain specialist
- be precise, grounded, and conservative
- prefer shorter accurate answers over broad speculative ones
- avoid invented details
- avoid unnecessary comparisons and process commentary

Additional legal-specific constraints:

- if jurisdiction is unspecified, give only a general overview
- do not cite statutes, section numbers, cases, or named tests unless specifically asked and confidently known
- do not provide multi-jurisdiction surveys unless asked
- avoid edge-case details unless necessary

Additional medical-specific constraints:

- distinguish general education from individualized medical advice
- avoid fabricated findings, dosages, or certainty

Additional code-specific constraints:

- obey the requested programming language and output format exactly

---

## Grading and Retry — Current Behavior

### Deterministic guards first

Before trusting any model-based verdict, the pipeline checks for obvious failures such as:

- empty output
- prompt echo / scaffold leakage
- degenerate punctuation-heavy output
- malformed web output
- malformed code output

### Code grading

Code grading is now **language-aware**.

Current supported distinctions include:

- Python
- Rust
- JavaScript
- TypeScript
- Go
- Java
- C#
- C++
- generic fallback

Examples:

- Python expects `def ...`
- Rust expects `fn ...` / optional `pub fn ...`
- Rust checks balanced braces instead of trying Python `ast.parse`

### Specialist self-grading

Current trusted self-grade domains:

```python
SPECIALIST_SELF_GRADE_DOMAINS = {"code", "math"}
```

Behavior:

- deterministic guards run first
- then the currently selected specialist grades its own answer
- the grading prompt is now language-aware for code tasks
- non-canonical self-grade output is treated as abstention if deterministic guards found no obvious problem

### Supervisor grading

Still used for non-self-graded domains and synthesis/fallback selection.

### Retry policy

- one retry on failed or weak specialist output
- retry guidance is generated by the supervisor
- specialist remains selected for the retry
- after best-candidate selection, the specialist is unloaded

---

## Synthesis and Fallback — Current Behavior

### `TIER2` synthesis

Current rule:

- if best specialist candidate is already `PASS`, synthesis is bypassed
- otherwise supervisor synthesis is attempted

### Candidate selection

Current “keep better candidate” logic compares:

- verdict quality first
- response length second

### Supervisor fallback

If final selected output still fails quality gates:

- score-gated supervisor fallback may run

### Empty final output

If final pipeline output is still empty after all logic:

- user sees a friendly fallback message instead of a blank response

---

## Templates and KB

### Template store

Still active and backed by:

- `config/framework_templates/all_templates.json`

Current behavior:

- embeddings are precomputed/cached at startup
- template matches can be used for scaffolding/context
- template system remains active even though normal `TIER2` prompt engineering has been simplified

### Knowledge base

KB integration remains optional and graceful.

Current role:

- can be used for retry-oriented grounding
- may be unavailable without breaking the pipeline

---

## Session Memory and Artifacts

`SessionStore` remains active.

Current behavior:

- `TIER2` work preserves session continuity context
- blocked `TIER3` requests do not execute the multi-domain pipeline
- accepted generation artifacts are stored in RAM-backed structures keyed to the task
- recent task snippets can be injected into follow-up prompts
- TIER1 responses are still not treated as full session tasks in the same way as `TIER2`

The follow-up routing guards depend on this continuity data.

---

## TIER3 Status

Important current reality:

- `run_tier3()` still exists in [demo.py](demo.py)
- there is still step-buffer and synthesis code
- but the **interactive dispatcher does not call it**

Instead, if a request is classified as multi-domain:

- the CLI prints a message asking the user to split the request into smaller prompts

This document treats `TIER3` as **implemented but operationally disabled in the interactive shell**.

---

## CLI Commands and Settings

### Current startup args

- `--device`
- `--no-kb`
- `--no-enhance`
- `--no-confirm`
- `--session`
- `--embed-cpu-batch-size`
- `--embed-cpu-threads`
- `--debug-trace`

### Current shell commands

- `/quit`, `/exit`
- `/new`
- `/status`
- `/memory [N]`
- `/domains`
- `/enhance [on|off]`
- `/confirm [on|off]`
- `/kb [on|off]`
- `/render [intermediate|final]`
- `/config`
- `/config set KEY VALUE`
- `/help`

### Persistent settings file

Current config file:

- `config/demo_config.json`

Important tracked settings:

- `device`
- `enhance`
- `confirm`
- `kb`
- `session`
- `render_mode`
- `embed_cpu_batch_size`
- `embed_cpu_threads`
- `max_runtime_seq_tokens`
- `model_base_dir`

---

## Debug Trace

When enabled via `--debug-trace`, breadcrumbs are written to:

- [new Demo/acceptance_runs/debug_trace.jsonl](new%20Demo/acceptance_runs/debug_trace.jsonl)

Current trace usage includes:

- startup/shutdown
- classification and dispatch
- routing overrides
- specialist load/unload
- supervisor/specialist generation begin/end
- empty-output events

This trace is the primary tool used to diagnose:

- blank supervisor answers
- crash boundaries
- routing mistakes
- unwanted retries/synthesis passes

---

## Current Known Behavioral Notes

1. **Legal and medical currently rely on supervisor-only specialist-capacity behavior.**
2. **Supervisor-only legal answers are intentionally being pushed toward high-level, conservative summaries.**
3. **Creative writing and veterinary questions are intentionally routed away from inappropriate specialists.**
4. **`TIER3` remains blocked in the interactive path even though legacy code still exists.**
5. **Worker-based inference improves stability, but does not guarantee factual correctness. Prompt discipline is still required.**

---

## File Structure

```text
demo.py
inference_worker.py
demo.md
config/
  demo_config.json
  framework_templates/
data/
src/
acceptance_runs/
  debug_trace.jsonl
```

---

## Change Log

| Date | Change |
|---|---|
| 2026-03-04 | Initial blueprint/spec created |
| 2026-03-06 | Updated to reflect benchmark-style simplification, disabled `medical`/`legal` specialists, current `TIER2` behavior, worker-based inference, blocked interactive `TIER3`, language-aware code grading, and stricter supervisor specialist-capacity prompts |

---

*College of Experts — current operational spec for `demo.py`.*
