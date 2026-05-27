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

**Posterior update:** `personality_state.py:165` — `update_posterior_likert_laplace()`, called from `session_manager.py`.

**EIG selection:** `question_selection.py:97` — `select_next_question_eig()`, called from `session_manager.py`.

### Session layer

| File | Responsibility |
|---|---|
| `app/session_manager.py` | Full session lifecycle: creation, EIG-driven question selection, answer recording, posterior serialization, heldout evaluation, run logging, warm-start from prior posteriors. |
| `app/db.py` | SQLite schema and persistence. Tables: `sessions`, `responses`, `session_step_logs`, `session_run_logs`, `users`, `user_current_state`, `user_posteriors`, `posterior_snapshots`, `question_performance_events`. |
| `app/main.py` | FastAPI endpoints (`/start_session`, `/next_question`, `/answer`, `/session_summary`, `/register_user`, `/user/{user_id}`, `/user/{user_id}/posterior`, `/session/{session_id}/posterior_history`) and env var reads. |

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

## Database schema

### Existing tables
| Table | Purpose |
|---|---|
| `sessions` | Full session state: posterior μ/Σ, asked/heldout IDs, inference pool, arm, pending question. Now also carries `user_id` (nullable FK) and `prior_session_id` (nullable). |
| `responses` | One row per answered question (inference or heldout). |
| `session_step_logs` | Per-step telemetry: EIG at selection, entropy before/after, source. |
| `session_run_logs` | Per-session aggregate: arm, generated candidate counts, heldout metrics. |

### Phase 1 tables (anonymous users + longitudinal state)
| Table | Purpose |
|---|---|
| `users` | Anonymous user registry. `user_id = secrets.token_urlsafe(16)`. No auth beyond possessing the token. |
| `user_current_state` | One row per user (upserted after every inference answer). The warm-start source. Used even for abandoned sessions. |
| `user_posteriors` | Append-only completed-session history. Written only when status reaches `complete`. One row per (user, completed session). |
| `posterior_snapshots` | Per-step posterior snapshots within a session. `step_idx=0` = initial prior or warm-start. One row per inference answer. |

### Phase 2 tables (question learning signal)
| Table | Purpose |
|---|---|
| `question_performance_events` | One row per inference answer. Stores full before/after posterior state (μ, Σ), predicted EIG, realized information gain, response value, question source, and `user_id`. Heldout answers are excluded. `parameter_version` is populated from the active `question_parameter_versions` row (NULL for generated questions with no version). |

### Phase 3 tables (generated candidate metadata)
| Table | Purpose |
|---|---|
| `generated_question_candidates` | One row per raw LLM candidate per session (accepted and rejected). Fields: `session_id`, `candidate_index`, `text`, `question_id` (NULL if rejected), `max_seed_similarity`, `max_kept_similarity`, `dedupe_failed`, `validation_passed`, `validation_failure_reason`, `accepted_into_pool`, `w_json`, `noise_var`, `thresholds_json`, `nn_seed_ids_json`, `nn_similarities_json`, `selected_at_step` (set when the question is actually answered). |

### Phase 4 tables (versioned question parameters)
| Table | Purpose |
|---|---|
| `question_parameter_versions` | Versioned measurement parameters per question. Fields: `question_id`, `version` (auto-incremented per question), `w_json`, `noise_var`, `thresholds_json`, `source` (`'seed'` or `'estimated'`), `estimation_method`, `n_responses_used`, `performance_summary_json`, `active` (only one active per question at a time), `created_at`. UNIQUE(question_id, version). Seeded at startup via `db.seed_question_parameters()` — idempotent. |

### Phase 5 tables (prompt/probe-generation policy tracking)
| Table | Purpose |
|---|---|
| `prompt_policy_versions` | Versioned LLM prompt strategies. Fields: `name`, `version` (auto-incremented per name), `prompt_template`, `strategy_type` (`'generic'`, `'uncertainty_targeted'`, etc.), `conditioning_mode` (`'none'`, `'posterior_only'`, `'posterior_plus_history'`, etc.), `output_schema_json` (reserved for future structured output), `active` (one globally-active policy at a time), `created_at`. Seeded at startup via `db.seed_prompt_policies()` — idempotent. |
| `llm_generation_requests` | One row per LLM candidate-generation event. Links session → policy → posterior context → rendered prompt → candidates. Fields: `session_id`, `user_id`, `step_idx`, `prompt_policy_version_id`, `posterior_mu_json`, `posterior_sigma_json`, `entropy_before`, `uncertainty_summary_json`, `question_history_summary_json`, `answer_history_summary_json`, `unresolved_tensions_json`, `prompt_rendered`, `model_name`, `n_requested`, `n_returned`, `created_at`. |
| `generated_question_candidates` | (Phase 3, augmented in Phase 5) Now also carries `generation_request_id` (FK to `llm_generation_requests`) and `prompt_policy_version_id` (FK to `prompt_policy_versions`) for full policy → request → candidate lineage. |

---

## Session flow (key paths)

**Session creation** (`session_manager.create_session`):
1. If `user_id` provided → look up `user_current_state` → warm-start `PersonalityState` from prior μ/Σ; otherwise flat prior N(0, I).
2. Build inference pool (seed-only or seed+generated per arm assignment).
3. Insert session row with `user_id` and `prior_session_id`.
4. Insert `posterior_snapshots` at `step_idx=0`.

**Inference answer** (`session_manager.record_answer`, inference pool only):
1. Bayesian posterior update via `update_posterior_likert_laplace()` (unchanged).
2. Persist updated posterior to `sessions` table (unchanged).
3. Insert `posterior_snapshots` at `step_idx=N`.
4. Upsert `user_current_state` (if session has `user_id`).
5. Look up `get_active_question_parameter_version()` → populate `parameter_version`.
6. Insert `question_performance_events` row with resolved `parameter_version`.
7. Update `generated_question_candidates.selected_at_step` if question is generated.

**Session completion** (`_finalize_heldout_evaluation_if_needed`):
1. Compute heldout metrics (unchanged).
2. Insert `session_run_logs` (unchanged).
3. Insert `user_posteriors` row (if session has `user_id`).

---

## Current status (as of 2026-05-25)

- Bayesian/EIG inference loop: complete and benchmarked
- Live LLM question generation: implemented (`AnthropicLLMClient` + validation)
- A/B experiment arm logic: implemented (`seed_only` vs `seed_plus_generated`)
- Held-out evaluation and per-step telemetry: implemented
- **Phase 1 complete:** anonymous users, longitudinal state, warm-start, posterior snapshots
- **Phase 2 complete:** `question_performance_events` — full before/after posterior per inference answer
- **Phase 3 complete:** `generated_question_candidates` — all raw LLM candidates logged per session (accepted + rejected), with dedupe scores, validation reasons, nn_seed_ids, nn_similarities, and `selected_at_step` filled when EIG picks a generated question
- **Phase 4 complete:** `question_parameter_versions` — versioned w/noise_var/thresholds; seeded at startup via `db.seed_question_parameters()`; `parameter_version` populated in performance events
- **Frontend-readiness pass complete:** CORS configured (origins from `FRONTEND_ORIGINS` env var, defaults to `localhost:3000` and `localhost:5173`); `AnswerRequest.response` tightened to `le=5`; endpoint-level per-question validation via `get_question_num_categories`; invalid `user_id` in `/start_session` returns clean 404; common error responses cleaned up; `test_frontend_readiness.py` added (21 assertions)
- **Phase 5 complete:** prompt/probe-generation policy tracking — `prompt_policy_versions`, `llm_generation_requests`, and Phase 3 lineage columns (`generation_request_id`, `prompt_policy_version_id`) on `generated_question_candidates`; `models/prompt_policy.py` with `PromptPolicy`, `render_prompt_policy`, `GENERIC_TEMPLATE`; generic policy seeded at startup; `scripts/query_prompt_policy_stats.py` (read-only analytics); `test_phase5_prompt_policy.py` (46 assertions). All three learning layers now instrumented: (1) user posterior, (2) question performance, (3) generation policy.
- Scientific core (`personality_state.py`, `question_selection.py`) untouched throughout all phases
- Frontend: not yet built

---

## Near-term roadmap

1. ~~**Phase 3**~~ — complete
2. ~~**Phase 4**~~ — complete
3. ~~**Frontend-readiness pass**~~ — complete (CORS, answer validation, user_id 404, error cleanup)
4. ~~**Phase 5**~~ — complete (prompt policy tracking, generation request logging, lineage, read-only analytics)
5. **Minimal frontend** — session UI that drives the `/start_session` → `/answer` loop
5. **Uncertainty visualization** — display trait posterior (μ ± σ) as the session progresses
6. **Final profile page** — probabilistic Big Five summary at session end
7. **Adaptive cognitive inference** — extend latent-state framework beyond personality (delirium, cognitive load, etc.)

---

## Smoke tests

```bash
# Core inference (no API key needed)
python scripts/test_v2_lite_session.py           # generation + EIG loop
python scripts/test_real_session_backend.py      # full session lifecycle
python scripts/test_eig_after_update.py          # EIG scores after posterior update
python scripts/compare_policies.py               # EIG vs random vs fixed-order benchmark

# Phase 1: users and longitudinal state
python scripts/test_phase1_users.py              # registration, warm-start, snapshots (29 assertions)

# Phase 2: question performance events
python scripts/test_phase2_performance_events.py # per-answer event logging (58 assertions)

# Phase 3: generated candidate metadata
python scripts/test_phase3_generated_candidates.py  # candidate logging, rejection tracking, selected_at_step (42 assertions)

# Phase 4: versioned question parameters
python scripts/test_phase4_question_parameter_versions.py  # seeding, versioning, record_answer integration (38 assertions)

# Phase 5: prompt policy and generation request tracking
python scripts/test_phase5_prompt_policy.py                # policy seeding, lineage, failure handling, read-only stats (46 assertions)

# Frontend readiness
python scripts/test_frontend_readiness.py                  # CORS, user validation, answer validation, flow (21 assertions)

# Live generation path
ANTHROPIC_API_KEY=sk-ant-... python scripts/test_v2_lite_session.py
```
