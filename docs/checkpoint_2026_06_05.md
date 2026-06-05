# Synchronicity — Session Checkpoint (2026-06-05)

---

## Project Goal

Synchronicity is an adaptive Bayesian personality assessment system. It maintains a 5D Gaussian posterior over Big Five traits and selects questions greedily by Expected Information Gain (EIG). The scientific core is stable and must not be changed. The current focus is closing three adaptive learning loops: (1) recalibrating question loading vectors from accumulated response data, (2) scoring and routing prompt policies for LLM-generated questions, and (3) exploratory question selection that balances EIG with novelty and exploration.

---

## Architecture

### Stable core (do not touch)
| File | Role |
|---|---|
| `models/personality_state.py` | Gaussian belief N(μ, Σ). Laplace posterior update via BFGS + finite-diff Hessian. |
| `models/question_selection.py` | `expected_information_gain()`, `select_next_question_eig()`. Greedy EIG maximization. |

### Session layer
| File | Role |
|---|---|
| `app/db.py` | Full SQLite schema (~20 tables) and all persistence helpers. |
| `app/session_manager.py` | Session lifecycle: create, select next question, record answer, finalize. |
| `app/main.py` | FastAPI endpoints. `.env` auto-loading. Seeds DB at startup. |

### Question generation
| File | Role |
|---|---|
| `models/question_generation.py` | `LLMClient` protocol, `AnthropicLLMClient`, `DummyLLMClient`, JSON output parser. |
| `models/question_pool_builder.py` | Deduplication, kNN w-vector assignment, validation gate, pool assembly. |
| `models/prompt_policy.py` | Five prompt templates: `generic`, `uncertainty_targeted`, `profile_contrast`, `tradeoff_scenario`, `anti_redundancy`. |

### Adaptive selection
| File | Role |
|---|---|
| `models/question_scoring.py` | `select_next_question_exploratory()` — composite score: `EIG×1.0 + novelty×0.15 + exploration×0.20 + policy_prior×0.05 − redundancy×0.25 − risk×1.0`. Hard composition limits via `SessionComposition`. |
| `models/policy_routing.py` | `select_prompt_policy_for_session()` — epsilon-greedy routing over `routing_enabled` policies. Decision types: `only_option`, `under_tested`, `exploration`, `exploitation`, `fallback`. |

### Learning loop scripts
| Script | Role |
|---|---|
| `scripts/calibrate_question_parameters.py` | Offline BFGS ordinal-probit re-estimation of w vectors. Standalone, also called by `run_pending_calibrations.py`. |
| `scripts/run_pending_calibrations.py` | Processes `calibration_jobs` queue. Quality gates: improvement > 1e-6, ‖w‖ ≤ 3, ≥3 unique responses, all finite. |
| `scripts/recompute_policy_scores.py` | Processes `policy_score_jobs` queue. Writes reward scores to `prompt_policy_scores`. |
| `scripts/score_prompt_policies.py` | Read-only reward scoring (does not write). |
| `scripts/query_prompt_policy_stats.py` | Read-only policy analytics. |
| `scripts/query_calibration_jobs.py`, `query_policy_scores.py`, `query_routing_decisions.py` | Read-only observability. |

### Frontend
`frontend/` — React 18 + Vite. Full session loop with `localStorage` user persistence. `QuestionCard.jsx`, `SummaryPage.jsx` (Big Five bars with ±σ uncertainty bands). Default strategy: `anchored_exploratory`.

---

## Database Schema (20 tables)

**Core session state (mutable rows):**
- `sessions` — current posterior μ/Σ, status, asked_ids, inference_pool_json (contains frozen param_version per question, session_strategy, composition limits), pending_question_id
- `user_current_state` — single upserted row per user (warm-start source)

**Append-only logs:**
- `responses` — one row per answered question (inference + heldout)
- `posterior_snapshots` — per-step μ/Σ/entropy snapshots (step 0 = prior/warm-start)
- `question_performance_events` (QPE) — per inference answer: full before/after posterior, predicted EIG, realized IG, response, parameter_version (frozen), direct lineage to generated_candidate_id / generation_request_id / prompt_policy_version_id
- `session_step_logs` — lightweight step record: question_id, source, response, eig_at_selection, entropy_before/after
- `selection_score_logs` — winner's composite score and all 6 component scores per selection step
- `session_run_logs` — per-session aggregate: arm, n_generated, heldout metrics
- `user_posteriors` — completed-session posterior history per user
- `llm_generation_requests` — per generation event: rendered prompt, model name, posterior context, n_requested/returned, status, error_message
- `generated_question_candidates` — every raw LLM candidate (accepted + rejected): text, w_json, dedupe scores, validation result, selected_at_step, calibration_status, provisional_w_source/confidence, intended_contrast, llm_suggested_traits, embedding_ref
- `policy_routing_decisions` — per-session routing decision: decision_type, epsilon, scores_considered_json (no FK on session_id — written before insert_session)

**Versioned/parameterized:**
- `question_parameter_versions` — append-only versioned w/noise_var/thresholds per question. Only one active=1 per question at a time.
- `prompt_policy_versions` — versioned prompt templates. active=1 global fallback, routing_enabled=1 for epsilon-greedy eligibility.

**Calibration/scoring infrastructure:**
- `calibration_jobs` — durable job queue with status, quality gates, old/new w, promoted version ID
- `question_calibration_runs` — permanent audit log of calibration attempts
- `policy_score_jobs` — durable job queue for policy score refresh
- `prompt_policy_scores` — append-only reward score history per policy version

**Phase 8 loop closure (just implemented):**
At session completion, `_finalize_heldout_evaluation_if_needed` queues calibration jobs for questions crossing 50-response threshold and policy score jobs for the session's linked policy. Routing replaces static `get_active_prompt_policy_version` with `select_prompt_policy_for_session`.

**Maintenance order:** always run calibration before policy scoring:
```bash
python scripts/run_pending_calibrations.py
python scripts/recompute_policy_scores.py
```

---

## Recent Changes (Phases 7–8, both committed)

### Phase 7 (commit `ed487db`)
- `models/question_scoring.py`: anchored exploratory selection — composite score, `SessionComposition` hard limits
- `scripts/calibrate_question_parameters.py`: offline BFGS ordinal-probit calibration
- `scripts/score_prompt_policies.py`, `test_phase7_exploratory_learning.py`
- DB: `selection_score_logs`, `question_calibration_runs`, `prompt_policy_scores` tables
- `generated_question_candidates` extended: calibration_status, provisional_w_source/confidence, intended_contrast, embedding_ref, risk_notes, llm_suggested_traits
- LLM output format changed from numbered list → JSON array with rich metadata per candidate
- 4 new inactive prompt templates seeded
- `anchored_exploratory` is now the default session strategy; sessions capped at 10 questions
- `app/main.py`: `.env` auto-loading added
- `frontend/`: React 18 + Vite with full session UI and Big Five summary page

### Phase 8 (committed in same session)
- `models/policy_routing.py`: epsilon-greedy `select_prompt_policy_for_session()` with decision logging
- DB: `calibration_jobs`, `policy_score_jobs`, `policy_routing_decisions` tables; `routing_enabled` column on `prompt_policy_versions`; all seeded policies get `routing_enabled=True`
- `app/db.py`: full helper functions for all three new tables; `backfill_routing_enabled()`
- `app/session_manager.py`: routing wired into `create_session`; calibration + policy score job queueing wired into `_finalize_heldout_evaluation_if_needed`; env vars `PROMPT_POLICY_EPSILON=0.20`, `PROMPT_POLICY_MIN_COMPLETED_SESSIONS=10`, `CALIBRATION_MIN_RESPONSES=50`
- `scripts/run_pending_calibrations.py`: maintenance script with --dry-run/--force/--limit
- `scripts/recompute_policy_scores.py`: maintenance script
- `scripts/query_calibration_jobs.py`, `query_policy_scores.py`, `query_routing_decisions.py`: read-only observability
- `scripts/test_phase8_loop_closure.py`: 46 smoke tests (all passing)
- `docs/adaptive_loop_closure.md`: design doc with maintenance order, known limitations

All existing phase tests (Phases 1–7) still pass.

---

## Logging Audit Findings (conducted in this session)

### Critical gaps

**C1. Candidate set at each selection step is not logged.**
`selection_score_logs` only records the *winning* question's composite score and components. The full ranked list returned by `select_next_question_exploratory` and `select_next_question_eig` is discarded. You cannot reconstruct why alternatives were rejected or compute counterfactual EIGs. The table schema already supports `selected=False` rows — no schema changes needed, only code changes.

**C2. No index on `question_performance_events(question_id)`.**
The calibration script's primary query (`WHERE question_id = ?`) does a full table scan. At 10K+ sessions (~80K QPE rows), this becomes the calibration bottleneck. Fix: `CREATE INDEX IF NOT EXISTS idx_qpe_question_id ON question_performance_events(question_id)`.

**C3. No index on `generated_question_candidates(question_id)`.**
Used by `get_generated_candidate_for_session_question` on every generated-question answer. Fix: `CREATE INDEX IF NOT EXISTS idx_gqc_question_id ON generated_question_candidates(question_id)`.

**C4. Raw LLM output text is not stored.**
`llm_generation_requests` stores `prompt_rendered` (the input) but not the raw LLM response text. Parser bugs cause permanent data loss. Fix: add `raw_response_text TEXT` column.

### Medium-priority gaps

- **Question text not snapshotted for seed questions.** Text is loaded from `data/questions_v2.json` at runtime. If the file changes, historical sessions lose their original question text.
- **LLM generation hyperparameters not logged.** `llm_generation_requests` has `model_name` but no `temperature`, `max_tokens`.
- **`session_strategy`, `min_anchor_questions`, `max_generated_probes` are buried inside `inference_pool_json`** per-item rather than as top-level session columns. Unqueryable without JSON parsing.
- **`QPE.calibration_status` always stores the column DEFAULT (`'candidate'`)**, not the actual calibration status from the inference pool dict. The `insert_question_performance_event` signature does not accept this parameter. The field is silently wrong for all rows.
- **Inconsistent parameter version identifier convention.** `QPE.parameter_version` and `question_calibration_runs` use version numbers; `calibration_jobs.current_parameter_version_id` uses row IDs.
- **`selection_score_logs.selected` is always True.** Column exists for logging non-selected candidates but is never used that way.

### Storage assessment
~28–33KB per session. Safe at 100–1K sessions, manageable at 10K (~300MB), concerning at 100K+ (~3GB for SQLite). Fastest-growing tables: `question_performance_events` (4 posterior blobs/row × 8 rows/session), `posterior_snapshots` (9 rows/session), `sessions` (large `inference_pool_json`). The 5×5 Σ matrix serialized to JSON is the dominant byte cost.

---

## Pending Concerns

1. The three adaptive loops are now instrumented and partially closed, but the **reward signal feeding policy routing is still thin** — policies need ≥10 routed sessions before exploitation begins, and realized IG is session-composition-dependent (earlier-step selections have higher baseline entropy).
2. **No automatic trigger for maintenance scripts.** `run_pending_calibrations.py` and `recompute_policy_scores.py` must be run manually. A scheduled runner (cron or similar) has not been set up.
3. **Policy routing decisions are written before `insert_session`** to avoid a circular dependency on the FK. The routing decision row therefore references a `session_id` that does not yet exist in `sessions`. The FK was intentionally dropped from `policy_routing_decisions.session_id` to allow this. This is documented in db.py comments.
4. **Thompson sampling deferred.** Policy routing uses epsilon-greedy. A TODO exists in `models/policy_routing.py` for eventual Thompson sampling once reward uncertainty can be modeled.

---

## Exact Next Implementation Task

**Fix the four critical gaps from the audit, in priority order:**

### Step 1 — Add two missing indexes (trivial, ~5 lines)
In `app/db.py:init_db`, add after the existing QPE table creation:
```python
conn.execute("CREATE INDEX IF NOT EXISTS idx_qpe_question_id ON question_performance_events(question_id);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_qpe_question_created ON question_performance_events(question_id, created_at);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_gqc_question_id ON generated_question_candidates(question_id);")
```

### Step 2 — Log the full candidate set at each selection step (~30 lines)
In `app/session_manager.py:get_next_question`, after logging the winner at line ~620:
- For `anchored_exploratory`: iterate `ranked[1:]` (or top K-1) and call `db.insert_selection_score_log(..., selected=False)` for each.
- For `classic_eig`: the `_` return from `select_next_question_eig` is a list of `(Question, float)` tuples — log the non-winner EIGs with `selected=False`. Limit to top 5 to control storage growth.

### Step 3 — Store raw LLM response text (~10 lines)
- Add `raw_response_text TEXT` column to `llm_generation_requests` via `_migrate_llm_generation_requests_phase6` (already exists, add the column there) or a new Phase 8 migration.
- Thread the raw text from `AnthropicLLMClient.complete()` return value back up through `build_generated_pool` → `create_session` → `insert_llm_generation_request`.

### Step 4 — Fix `QPE.calibration_status` (~5 lines)
- Add `calibration_status: Optional[str] = None` parameter to `db.insert_question_performance_event`.
- In `session_manager.record_answer`, look up the calibration_status for the answered question from `sess.inference_pool` (same loop that reads `param_version`) and pass it through.

After these four changes, add them to the existing Phase 8 or open a new Phase 9 commit. Run `scripts/test_phase8_loop_closure.py` and all previous phase tests to confirm nothing regressed.

---

## Hard Constraints

- **Do not modify `models/personality_state.py`** — Bayesian posterior update math is stable and correct.
- **Do not modify `models/question_selection.py`** — EIG computation is stable and correct.
- **Do not change the calibration math** in `calibrate_question_parameters.py` — the regularized ordinal-probit BFGS formulation is intentional.
- **Do not change the policy reward formula** in `recompute_policy_scores.py` (0.30 × acceptance + 0.20 × selection + 0.35 × realized_ig − penalties) — it was reviewed and accepted.
- **Preserve all fallback behavior.** Every LLM-dependent path must degrade gracefully to seeds-only. The no-API-key path (`DummyLLMClient`) must continue to work.
- **All DB migrations must be idempotent.** Use `CREATE INDEX IF NOT EXISTS`, `ALTER TABLE ... ADD COLUMN` guarded by `PRAGMA table_info` checks, never DROP or destructive ops.
- **Static seed questions in `data/questions_v2.json` are ground truth.** The system must function without generated questions.
