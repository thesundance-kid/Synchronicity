# Adaptive Loop Closure

Phase 8 closes the three learning loops that were instrumented in Phases 2–7 but
previously had no automatic feedback path.

---

## The three loops

| Loop | Signal accumulated | What changes |
|---|---|---|
| **Question calibration** | `question_performance_events` (before/after posterior, response) | Loading vectors `w` re-estimated via regularized ordinal-probit |
| **Policy scoring** | Full policy→request→candidate→QPE lineage | Reward score persisted per policy version |
| **Policy routing** | `policy_routing_decisions` (which policy was chosen per session) | Future sessions routed toward higher-scoring policies |

---

## Why calibration is queued, not run inline

Running BFGS calibration during an HTTP response would add unpredictable latency
to session completion. Instead, when a session completes, `_finalize_heldout_evaluation_if_needed`
checks whether any answered question has crossed the response threshold (default: 50)
and inserts a row into `calibration_jobs` if so. A separate maintenance script runs
the actual BFGS fit outside the request path.

Idempotency: if a pending, running, or already-succeeded job exists for the same
`(question_id, current_parameter_version_id)`, no duplicate is created.

---

## Maintenance order

Always run in this order:

```bash
# 1. Re-estimate w vectors from accumulated response data.
python scripts/run_pending_calibrations.py

# 2. Refresh policy reward scores now that realized-IG reflects better w vectors.
python scripts/recompute_policy_scores.py
```

Running them in the wrong order is safe (won't corrupt data), but the policy
reward scores will reflect slightly noisier realized-IG values if calibration
hasn't run first.

---

## Epsilon-greedy policy routing

At session creation, `select_prompt_policy_for_session()` in `models/policy_routing.py`
replaces the static `active=1` policy lookup.

**Decision logic (in order):**
1. Gather all `routing_enabled=1` policies (latest version per name).
2. If none → fall back to the globally `active` policy (`decision_type='fallback'`).
3. If exactly one → always pick it (`'only_option'`).
4. If any policy has fewer than `PROMPT_POLICY_MIN_COMPLETED_SESSIONS` (default: 10)
   routed sessions → choose uniformly among under-tested policies (`'under_tested'`).
5. Otherwise, with probability `PROMPT_POLICY_EPSILON` (default: 0.20) choose randomly
   (`'exploration'`); otherwise choose the policy with the highest stored reward score
   (`'exploitation'`).
6. If no reward scores exist for any policy → always explore.

Every decision is logged to `policy_routing_decisions` for observability.

**Why not Thompson sampling yet:**
Thompson sampling requires modeling the uncertainty of the reward estimate (a
posterior over reward). The current reward is a point estimate from a small amount
of data. Until enough sessions accumulate to make the variance meaningful, epsilon-greedy
with a conservative `min_completed_sessions` guard is sufficient and easier to debug.
A TODO is left in `policy_routing.py` for this extension.

---

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `PROMPT_POLICY_ROUTING_STRATEGY` | `epsilon_greedy` | Currently the only strategy |
| `PROMPT_POLICY_EPSILON` | `0.20` | Exploration probability |
| `PROMPT_POLICY_MIN_COMPLETED_SESSIONS` | `10` | Minimum sessions before exploitation begins per policy |
| `CALIBRATION_MIN_RESPONSES` | `50` | Responses required to queue a calibration job |

---

## Known limitations

- **Early low-volume data is noisy.** With fewer than 50 responses per question,
  calibration is gated. Policy scores are similarly unreliable with few sessions.
  The defaults are conservative for this reason.
- **Realized IG depends on posterior state and step.** It is not a clean property
  of the question alone — it depends on what was asked before. Policy scores computed
  from realized IG are therefore session-composition-dependent.
- **Policy scores should not be over-interpreted** until enough sessions accumulate
  (target: ≥100 sessions per policy before trusting exploitation decisions).
- **Thompson sampling deferred.** The reward uncertainty is not yet modeled.
- **Calibration affects future sessions only.** In-progress sessions use the parameter
  version frozen at creation time (`inference_pool_json[n].param_version`), so
  calibration does not introduce mid-session drift.

---

## Observability

```bash
# Current routing decisions (most recent 50)
python scripts/query_routing_decisions.py

# Policy scores and routing stats
python scripts/query_policy_scores.py

# Calibration job queue
python scripts/query_calibration_jobs.py
python scripts/query_calibration_jobs.py --status pending
python scripts/query_calibration_jobs.py --question-id q_01
```
