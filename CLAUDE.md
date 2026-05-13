# CLAUDE.md — Synchronicity

Architectural context for Claude Code sessions. Read this before touching any file.

---

## What this project is

Synchronicity is an adaptive latent-state inference system for personality assessment. It uses a Bayesian posterior over the Big Five trait space and selects questions greedily by Expected Information Gain (EIG) to reduce uncertainty as fast as possible.

**The scientific core is the Bayesian/EIG loop. Everything else is scaffolding.**

LLMs play one narrow role: generating candidate question *text*. They have no influence over posterior updates, EIG scores, or question selection logic.

Long-term direction: adaptive cognitive assessment, delirium-related research, and longitudinal latent-state inference beyond personality.

---

## Architecture

### Inference engine (do not modify without strong reason)

| File | Responsibility |
|---|---|
| `models/personality_state.py` | Gaussian belief `N(μ, Σ)` over 5 traits. Laplace posterior updates via BFGS + finite-difference Hessian. Entropy and predictive Likert probabilities. |
| `models/question_selection.py` | `expected_information_gain()` and `select_next_question_eig()`. Greedy EIG maximization over candidate pool. |

**Posterior update:** `personality_state.py:165` — `update_posterior_likert_laplace()`, called from `session_manager.py:398`.

**EIG selection:** `question_selection.py:97` — `select_next_question_eig()`, called from `session_manager.py:306`.

### Session layer

| File | Responsibility |
|---|---|
| `app/session_manager.py` | Full session lifecycle: creation, EIG-driven question selection, answer recording, posterior serialization, heldout evaluation, run logging. |
| `app/db.py` | SQLite schema and persistence. Tables: `sessions`, `responses`, `session_step_logs`, `session_run_logs`. |
| `app/main.py` | FastAPI endpoints (`/start_session`, `/next_question`, `/answer`, `/session_summary`) and env var reads. |

### Question data and generation

| File | Responsibility |
|---|---|
| `data/questions_v2.json` | Ground-truth question bank. 21 inference items + 5 heldout. Each has `id`, `text`, `w` (5D loading vector), `noise_var`, `thresholds`. |
| `models/question_bank.py` | Loads V1/V2 question schemas. |
| `models/question_generation.py` | `LLMClient` protocol, `DummyLLMClient`, `AnthropicLLMClient`, `make_llm_client()` factory, `validate_candidate_question()`, prompt builder, output parser. |
| `models/question_pool_builder.py` | Deduplicates generated candidates against seeds (cosine sim), assigns loading vectors `w` via k-NN, runs validation gate, combines with seeds into final inference pool. |
| `models/question_assignment.py` | k-NN loading-vector assignment via sentence-transformer embeddings. |

### Evaluation

| File | Responsibility |
|---|---|
| `models/real_eval.py` | Held-out log-likelihood and calibration metrics after session completes. |
| `models/simulator.py` | Synthetic user generation for benchmarking. |
| `models/evaluation.py` | Policy comparison framework (random vs fixed-order vs EIG). |
| `models/session_experiment.py` | A/B arm assignment (`seed_only` vs `seed_plus_generated`, deterministic from `session_id`). |

---

## LLM integration

**Client priority in `create_session()`:**
1. Explicit `llm_client` kwarg (tests/overrides)
2. `AnthropicLLMClient` — activated when `ANTHROPIC_API_KEY` is set
3. `DummyLLMClient` — deterministic fallback, no network calls

**Generation pipeline (only runs for `seed_plus_generated` arm):**
```
make_llm_client() → generate_candidate_questions() → _dedupe_generated_candidates()
  → assign_w_to_candidates() → validate_candidate_question() [per item] → build_session_inference_pool()
```

**Validation rules** (`question_generation.py:validate_candidate_question`):
- 10 ≤ text length ≤ 220 chars
- must end with `?`
- no sensitive terms (suicide, self-harm, abuse, trauma, religion, race, ethnic, sexual, diagnosis, politics, and variants)
- `w` vector present, non-empty, not all zeros

**Failure behavior:** Any exception in `build_generated_pool` is caught in `session_manager.create_session` and falls back to seeds-only with a `warnings.warn`. Sessions always start successfully.

---

## Engineering principles

- **Do not touch `personality_state.py` or `question_selection.py`** unless the change is specifically to the Bayesian or EIG math. These are stable and correct.
- **LLMs generate text only.** Generated questions enter the candidate pool and compete on EIG like any seed question. The LLM has no influence over which question is actually asked.
- **Prefer surgical edits over rewrites.** The architecture is intentionally layered; changes should be local.
- **Preserve fallback behavior.** Every LLM-dependent path must degrade gracefully to seeds-only. Do not break the no-API-key path.
- **No speculative abstractions.** Add complexity only when a concrete requirement demands it.
- **Static seed questions are ground truth.** `data/questions_v2.json` must remain usable as standalone fallback. Do not make the system depend on generated questions to function.

---

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | `""` | Empty/absent → `DummyLLMClient`. Set to activate live generation. |
| `LLM_MODEL` | `claude-haiku-4-5-20251001` | Any Anthropic model ID. Haiku is the default (fast, cheap). |
| `PILOT_DB_PATH` | `data/pilot.db` | SQLite database path. |
| `QUESTIONS_PATH` | `data/questions_v2.json` | Question bank file. |
| `LATENT_DIM` | `5` | Big Five dimensionality. Do not change without updating `w` vectors. |

---

## Current status (as of 2026-05-13)

- Bayesian/EIG inference loop: complete and benchmarked
- Live LLM question generation: implemented (`AnthropicLLMClient` + validation)
- A/B experiment arm logic: implemented (`seed_only` vs `seed_plus_generated`)
- Held-out evaluation and per-step telemetry: implemented
- Smoke tests passing: `test_v2_lite_session.py`, `test_real_session_backend.py`
- Frontend: not yet built

---

## Near-term roadmap

1. **Minimal frontend** — session UI that drives the `/start_session` → `/answer` loop
2. **Uncertainty visualization** — display trait posterior (μ ± σ) as the session progresses
3. **Final profile page** — probabilistic Big Five summary at session end
4. **Longitudinal sessions** — warm-start posterior from a prior session
5. **Adaptive cognitive inference** — extend latent-state framework beyond personality (delirium, cognitive load, etc.)

---

## Smoke tests

```bash
python scripts/test_v2_lite_session.py           # generation + EIG loop (no API key needed)
python scripts/test_real_session_backend.py      # full session lifecycle
python scripts/compare_policies.py               # EIG vs random vs fixed-order benchmark
ANTHROPIC_API_KEY=sk-ant-... python scripts/test_v2_lite_session.py  # live generation path
```
