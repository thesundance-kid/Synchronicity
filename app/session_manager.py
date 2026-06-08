"""
Session manager for real-user pilot sessions.

This module orchestrates:
- choosing next question (adaptive via EIG or fixed order)
- recording answers (inference updates posterior; held-out does not)
- persisting state to SQLite for later analysis
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np

from app import db
from models.character_sketch import generate_character_sketch
from models.personality_state import PersonalityState
from models.policy_routing import select_prompt_policy_for_session
from models.question_bank import DEFAULT_THRESHOLDS, Question, load_question_pools_v2
from models.prompt_policy import PromptPolicy, render_prompt_policy
from models.question_generation import DEFAULT_LLM_MODEL, make_llm_client
from models.question_pool_builder import build_generated_pool, build_session_inference_pool
from models.question_scoring import SessionComposition, select_next_question_exploratory
from models.question_selection import expected_information_gain, select_next_question_eig
from models.real_eval import evaluate_heldout_performance
from models.session_experiment import assign_experiment_arm, should_use_generated_questions


Mode = Literal["adaptive", "fixed_order"]
Status = Literal["inference", "heldout", "complete"]
SessionStrategy = Literal["classic_eig", "anchored_exploratory"]

SELECTION_LOG_TOP_K = 5


class _AtomicConn:
    """
    Proxy a sqlite3.Connection so that all commit() calls are deferred until __exit__.
    On success, issues a single real commit. On exception, issues rollback.
    Use as a context manager inside record_answer to make all DB writes for one
    answer atomic: no partial state is committed if posterior update or event
    logging fails mid-operation.
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        return False  # do not suppress exceptions

    def commit(self) -> None:
        pass  # deferred — real commit happens in __exit__

    def rollback(self) -> None:
        self._conn.rollback()

    def __getattr__(self, name: str):
        return getattr(self._conn, name)


@dataclass(frozen=True)
class QuestionPayload:
    id: str
    text: str
    pool: Literal["inference", "heldout"]
    num_categories: int


def _session_id() -> str:
    return secrets.token_urlsafe(16)


def _state_to_jsonable(state: PersonalityState) -> Tuple[List[float], List[List[float]]]:
    return state.mu.astype(float).tolist(), state.sigma.astype(float).tolist()


def _state_from_jsonable(mu: Sequence[float], sigma: Sequence[Sequence[float]]) -> PersonalityState:
    mu_arr = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    return PersonalityState(mu_init=mu_arr, sigma_init=sigma_arr)


def _question_to_stored_dict(q: Question) -> Dict[str, Any]:
    """Serialize a Question to JSON-friendly dict (lists for arrays)."""
    return {
        "id": q.id,
        "text": q.text,
        "w": q.w.astype(float).tolist(),
        "noise_var": float(q.noise_var),
        "thresholds": q.thresholds.astype(float).tolist(),
        "source": "seed",
        "calibration_status": "calibrated",
    }


def _jsonable_question_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure w/thresholds are lists for DB JSON."""
    out = dict(d)
    w = out.get("w")
    if isinstance(w, np.ndarray):
        out["w"] = w.astype(float).tolist()
    thr = out.get("thresholds")
    if isinstance(thr, np.ndarray):
        out["thresholds"] = thr.astype(float).tolist()
    return out


def _dict_to_question(d: Dict[str, Any]) -> Question:
    """Reconstruct Question from stored dict."""
    w = np.asarray(d["w"], dtype=np.float64).reshape(-1)
    thr_raw = d.get("thresholds")
    if thr_raw is not None:
        thresholds = np.asarray(thr_raw, dtype=np.float64).reshape(-1)
    else:
        thresholds = DEFAULT_THRESHOLDS.copy()
    return Question(
        id=str(d["id"]),
        text=str(d["text"]),
        w=w,
        noise_var=float(d.get("noise_var", 1.0)),
        thresholds=thresholds,
    )


def _session_inference_questions(sess: db.SessionRow, questions_v2_path: str, dim: int) -> List[Question]:
    """Inference pool for this session: persisted V2-lite list or JSON file seeds."""
    if sess.inference_pool:
        return [_dict_to_question(d) for d in sess.inference_pool]
    inference_pool, _, _ = load_bank(questions_v2_path, expected_dim=dim)
    return inference_pool


def _question_source(question_id: str, sess: db.SessionRow) -> Literal["seed", "generated"]:
    if question_id in set(sess.generated_question_ids):
        return "generated"
    return "seed"


def _session_question_metadata(sess: db.SessionRow) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    if sess.inference_pool:
        for d in sess.inference_pool:
            qid = str(d.get("id"))
            metadata[qid] = {
                "source": d.get("source"),
                "calibration_status": d.get("calibration_status"),
                "policy_quality_prior": d.get("policy_quality_prior", 0.0),
                "risk_penalty": d.get("risk_penalty", 0.0),
            }
    return metadata


def _qmap_with_session_inference(
    sess: db.SessionRow,
    questions_v2_path: str,
    dim: int,
    extra_questions: Optional[List[Question]] = None,
) -> Dict[str, Question]:
    """Question id map including held-out from file and session inference items (incl. generated)."""
    _, _, qmap = load_bank(questions_v2_path, expected_dim=dim)
    if sess.inference_pool:
        for d in sess.inference_pool:
            q = _dict_to_question(d)
            qmap[q.id] = q
    if extra_questions:
        for q in extra_questions:
            qmap[q.id] = q
    return qmap


def _build_response_history_for_sketch(
    responses: List[db.ResponseRow],
    qmap: Dict[str, Question],
) -> List[Dict[str, Any]]:
    """Build inference response history for character sketch generation."""
    history = []
    for r in responses:
        if r.pool != "inference":
            continue
        q = qmap.get(r.question_id)
        history.append({
            "question_id": r.question_id,
            "text": q.text if q is not None else r.question_id,
            "response": int(r.response),
            "step": int(r.step),
        })
    return sorted(history, key=lambda x: x["step"])


def _load_and_merge_step_candidates(
    conn,
    sess: db.SessionRow,
    dim: int,
) -> Tuple[List[Question], Set[str], List[Dict[str, Any]]]:
    """
    Load pre-generated step candidates from the cache for the current selection step.
    Returns (questions, question_id_set, candidate_dicts_for_pool).
    """
    rows = db.get_step_candidates(conn, sess.session_id, sess.step)
    if not rows:
        return [], set(), []

    questions: List[Question] = []
    ids: Set[str] = set()
    dicts: List[Dict[str, Any]] = []
    for row in rows:
        try:
            w = np.asarray(row["w"], dtype=np.float64).reshape(-1)
            thr = np.asarray(row["thresholds"], dtype=np.float64).reshape(-1)
            q = Question(
                id=row["question_id"],
                text=row["text"],
                w=w,
                noise_var=float(row["noise_var"]),
                thresholds=thr,
            )
            questions.append(q)
            ids.add(q.id)
            pool_dict: Dict[str, Any] = {
                "id": row["question_id"],
                "text": row["text"],
                "w": w.tolist(),
                "noise_var": float(row["noise_var"]),
                "thresholds": thr.tolist(),
                "source": "generated",
                "calibration_status": "accepted_uncalibrated",
                "param_version": None,
            }
            if sess.inference_pool:
                first_d = sess.inference_pool[0]
                if first_d.get("session_strategy"):
                    pool_dict["session_strategy"] = first_d["session_strategy"]
                    pool_dict["min_anchor_questions"] = first_d.get("min_anchor_questions", 2)
                    pool_dict["max_generated_probes"] = first_d.get("max_generated_probes", 6)
            dicts.append(pool_dict)
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"Failed to load step candidate {row.get('question_id', '?')}: {exc}",
                stacklevel=2,
            )
    return questions, ids, dicts


def _queue_calibration_jobs_for_session(conn, sess: db.SessionRow) -> None:
    """Queue calibration jobs for any inference question that has crossed the threshold."""
    import os
    min_responses = int(os.environ.get("CALIBRATION_MIN_RESPONSES", "50"))
    for qid in sess.asked_ids:
        try:
            db.queue_calibration_job_if_eligible(conn, question_id=qid, min_responses=min_responses)
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(f"Failed to queue calibration job for {qid}: {exc}", stacklevel=2)


def _queue_policy_score_job_for_session(conn, session_id: str) -> None:
    """Queue a policy score refresh for the policy linked to this session."""
    try:
        requests = db.list_llm_generation_requests_for_session(conn, session_id)
        seen: set = set()
        for req in requests:
            pid = req.get("prompt_policy_version_id")
            if pid is not None and pid not in seen:
                seen.add(pid)
                db.queue_policy_score_job_if_needed(conn, prompt_policy_version_id=int(pid))
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(f"Failed to queue policy score job for session {session_id}: {exc}", stacklevel=2)


def _finalize_heldout_evaluation_if_needed(
    conn,
    *,
    session_id: str,
    questions_v2_path: str,
    dim: int,
) -> None:
    """
    After the session reaches ``complete``, compute held-out metrics using the stored
    posterior (inference-only) and persist them. No-op if not complete or already stored.
    """
    sess = db.get_session(conn, session_id)
    if sess.status != "complete":
        return
    if sess.heldout_evaluation is not None:
        return

    qmap = _qmap_with_session_inference(sess, questions_v2_path, dim)
    rows = db.list_responses(conn, session_id)
    responses_map = {r.question_id: int(r.response) for r in rows if r.pool == "heldout"}
    heldout_items = [qmap[qid] for qid in sess.heldout_ids if qid in qmap]
    posterior = _state_from_jsonable(sess.posterior_mu, sess.posterior_sigma)
    payload = evaluate_heldout_performance(posterior, heldout_items, responses_map)
    db.update_heldout_evaluation(conn, session_id=session_id, heldout_evaluation=payload)

    if db.get_session_run_log(conn, session_id) is not None:
        return
    asked = set(sess.asked_ids)
    gids = list(sess.generated_question_ids)
    n_selected = sum(1 for gid in gids if gid in asked)
    usage = {gid: (gid in asked) for gid in gids}
    db.insert_session_run_log(
        conn,
        session_id=session_id,
        arm=sess.arm,
        n_generated_candidates=int(sess.n_generated_candidates),
        n_generated_selected=n_selected,
        heldout_log_likelihood=float(payload["heldout_log_likelihood"]),
        mean_true_prob=float(payload["mean_true_prob"]),
        generated_usage=usage,
    )

    # Phase 1: persist completed-session posterior to user longitudinal history.
    if sess.user_id is not None:
        session_number = db.count_user_posteriors(conn, sess.user_id) + 1
        final_state = _state_from_jsonable(sess.posterior_mu, sess.posterior_sigma)
        db.insert_user_posterior(
            conn,
            user_id=sess.user_id,
            session_id=session_id,
            session_number=session_number,
            mu=sess.posterior_mu,
            sigma=sess.posterior_sigma,
            entropy=float(final_state.entropy()),
        )

    # Phase 8: queue calibration jobs for questions that crossed the response threshold.
    _queue_calibration_jobs_for_session(conn, sess)

    # Phase 8: queue a policy score refresh for the policy that served this session.
    _queue_policy_score_job_for_session(conn, session_id)


def load_bank(
    questions_v2_path: str,
    *,
    expected_dim: Optional[int] = None,
) -> Tuple[List[Question], List[Question], Dict[str, Question]]:
    """
    Load inference + heldout pools and return an id->Question map.
    """
    inference_pool, heldout_pool = load_question_pools_v2(questions_v2_path, expected_dim=expected_dim)
    qmap: Dict[str, Question] = {q.id: q for q in inference_pool}
    qmap.update({q.id: q for q in heldout_pool})
    return inference_pool, heldout_pool, qmap


def create_session(
    conn,
    *,
    questions_v2_path: str,
    mode: Mode,
    max_inference_questions: int,
    num_heldout: int,
    dim: int = 5,
    fixed_order_ids: Optional[Sequence[str]] = None,
    llm_client: Any = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    n_generation_candidates: int = 10,
    nn_k: int = 3,
    user_id: Optional[str] = None,
    session_strategy: SessionStrategy = "classic_eig",
    max_anchor_questions: int = 2,
    max_generated_probes: int = 6,
) -> Tuple[str, QuestionPayload]:
    """
    Create a new session and return (session_id, first_question_payload).

    V2-lite: experiment arm is assigned from ``session_id``; ``seed_plus_generated`` arm may
    append LLM-generated items (with ``w`` from nearest-neighbor assignment). The combined
    inference pool is stored and used for EIG / fixed-order selection.

    LLM client priority (highest to lowest):
    1. Explicit ``llm_client`` argument (e.g. for tests).
    2. ``AnthropicLLMClient`` built from ``llm_api_key`` / ``llm_model``.
    3. ``DummyLLMClient`` (no network calls) when neither is supplied.
    """
    inference_seed, heldout_pool, _ = load_bank(questions_v2_path, expected_dim=dim)

    if max_inference_questions <= 0:
        raise ValueError("max_inference_questions must be positive.")
    if num_heldout <= 0:
        raise ValueError("num_heldout must be positive.")
    if num_heldout > len(heldout_pool):
        raise ValueError("num_heldout exceeds heldout_pool size.")

    if mode not in ("adaptive", "fixed_order"):
        raise ValueError(f"Unknown mode: {mode}")

    session_id = _session_id()
    arm = assign_experiment_arm(session_id)

    seed_stored = [_question_to_stored_dict(q) for q in inference_seed]
    seeds_for_generation = [{"id": q.id, "text": q.text, "w": q.w} for q in inference_seed]

    # Phase 1: warm-start from user_current_state if available; otherwise flat prior.
    # Moved before LLM call so the sketch can condition question generation.
    prior_session_id: Optional[str] = None
    if user_id is not None:
        warm = db.get_user_current_state(conn, user_id)
        if warm is not None:
            state = _state_from_jsonable(warm["mu"], warm["sigma"])
            prior_session_id = warm["latest_session_id"]
        else:
            state = PersonalityState(dim=dim)
    else:
        state = PersonalityState(dim=dim)

    # Generate a prior character sketch to condition question generation.
    # Wrapped in try-except: make_llm_client() can raise ImportError when the
    # anthropic package is absent even if an API key env var is set.
    try:
        if llm_client is not None:
            _sketch_client = llm_client
        else:
            _sketch_client = make_llm_client(api_key=llm_api_key, model=llm_model)
        prior_sketch = generate_character_sketch(state, [], _sketch_client)
    except Exception as _exc:  # noqa: BLE001
        import warnings
        warnings.warn(f"Prior sketch generation failed (omitted): {_exc}", stacklevel=2)
        prior_sketch = ""

    use_generated = should_use_generated_questions(arm) or session_strategy == "anchored_exploratory"
    generated_stored: Optional[List[Dict[str, Any]]] = None
    generated_candidates_metadata: Optional[List[Dict[str, Any]]] = None

    # Phase 5: look up active prompt policy and render prompt before LLM call.
    # rendered_prompt=None → build_generated_pool falls back to build_generation_prompt.
    rendered_prompt: Optional[str] = None
    policy_version_id: Optional[int] = None
    n_returned: int = 0

    # Phase 6: explicit generation status/error for llm_generation_requests logging.
    generation_status: str = "success"
    generation_error: Optional[str] = None

    if use_generated:
        # Phase 8: epsilon-greedy routing selects among routing-enabled policies.
        # session_id is not yet in the DB, so we pass it for the routing log only;
        # the FK constraint will be satisfied when insert_session runs below.
        _policy_row = select_prompt_policy_for_session(conn, session_id=session_id)
        if _policy_row is not None:
            policy_version_id = int(_policy_row["id"])
            rendered_prompt = render_prompt_policy(
                PromptPolicy.from_row(_policy_row),
                seeds_for_generation,
                n_generation_candidates,
                uncertainty_summary=None,
                character_sketch=prior_sketch,
            )

        raw_llm_response: Optional[str] = None
        try:
            if llm_client is not None:
                client = llm_client
            else:
                client = make_llm_client(api_key=llm_api_key, model=llm_model)
            gen_result = build_generated_pool(
                client,
                seeds_for_generation,
                n_candidates=n_generation_candidates,
                k=nn_k,
                prompt=rendered_prompt,
            )
            generated_stored = [_jsonable_question_dict(x) for x in gen_result.accepted]
            for d in generated_stored:
                d["source"] = "generated"
                d["param_version"] = None
                d["calibration_status"] = str(d.get("calibration_status") or "accepted_uncalibrated")
            generated_candidates_metadata = gen_result.all_candidates_metadata
            n_returned = len(generated_candidates_metadata)
            raw_llm_response = gen_result.raw_response_text or None
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"LLM question generation failed; falling back to seeds only. Error: {exc}",
                stacklevel=2,
            )
            generated_stored = None
            generated_candidates_metadata = None
            n_returned = 0
            generation_status = "failed"
            # Truncate to avoid storing stack traces or sensitive strings.
            generation_error = str(exc)[:500]

    # Phase 6: freeze the active parameter version for each seed question at session creation
    # time. This prevents scientific drift if a new parameter version is activated mid-session;
    # record_answer will use the version captured here rather than the current active one.
    for d in seed_stored:
        _pv = db.get_active_question_parameter_version(conn, d["id"])
        d["param_version"] = _pv["version"] if _pv is not None else None

    n_gen = len(generated_stored) if generated_stored else 0
    gen_qids = [str(d["id"]) for d in generated_stored] if generated_stored else []

    final_inference_dicts = build_session_inference_pool(seed_stored, generated_stored if use_generated else None)
    for d in final_inference_dicts:
        d.setdefault("source", "generated" if str(d.get("id")) in set(gen_qids) else "seed")
        d.setdefault("calibration_status", "accepted_uncalibrated" if d.get("source") == "generated" else "calibrated")
    if session_strategy == "anchored_exploratory":
        for d in final_inference_dicts:
            d["session_strategy"] = session_strategy
            d["min_anchor_questions"] = int(max_anchor_questions)
            d["max_generated_probes"] = int(max_generated_probes)
    inference_ids = {d["id"] for d in final_inference_dicts}

    if mode == "fixed_order":
        if fixed_order_ids is None:
            fixed_order_ids = [d["id"] for d in final_inference_dicts]
        fixed_order_ids = list(fixed_order_ids)
        unknown = [qid for qid in fixed_order_ids if qid not in inference_ids]
        if unknown:
            raise ValueError(f"fixed_order_ids includes ids not in inference_pool: {unknown[:5]}")

    # Deterministic selection not required; pick held-out ids randomly.
    heldout_ids = [q.id for q in heldout_pool]
    rng = np.random.default_rng()
    rng.shuffle(heldout_ids)
    heldout_ids = heldout_ids[:num_heldout]

    mu, sigma = _state_to_jsonable(state)
    initial_entropy = float(state.entropy())

    db.insert_session(
        conn,
        session_id=session_id,
        mode=mode,
        status="inference",
        step=0,
        max_inference_questions=max_inference_questions,
        asked_ids=[],
        heldout_ids=heldout_ids,
        fixed_order_ids=list(fixed_order_ids) if fixed_order_ids is not None else None,
        posterior_mu=mu,
        posterior_sigma=sigma,
        arm=arm,
        generated_items=generated_stored,
        inference_pool=final_inference_dicts,
        n_generated_candidates=n_gen,
        generated_question_ids=gen_qids,
        user_id=user_id,
        prior_session_id=prior_session_id,
    )

    # Phase 1: record the initial prior (or warm-start) as step 0 snapshot.
    db.insert_posterior_snapshot(
        conn,
        session_id=session_id,
        step_idx=0,
        mu=mu,
        sigma=sigma,
        entropy=initial_entropy,
    )

    # Cache the prior sketch in the session row for later retrieval.
    try:
        db.update_session_sketch(conn, session_id, prior_sketch, step_idx=0)
    except Exception as _exc:  # noqa: BLE001
        import warnings
        warnings.warn(f"Failed to cache prior sketch for {session_id}: {_exc}", stacklevel=2)

    # Phase 5: log the LLM generation request (always when use_generated, even on failure).
    # Session must exist in DB before inserting this row (FK constraint).
    gen_request_id: Optional[int] = None
    if use_generated:
        import numpy as _np
        _sigma_arr = _np.asarray(sigma)
        _variances = _np.diag(_sigma_arr).tolist()
        try:
            gen_request_id = db.insert_llm_generation_request(
                conn,
                session_id=session_id,
                user_id=user_id,
                step_idx=0,
                prompt_policy_version_id=policy_version_id,
                posterior_mu=mu,
                posterior_sigma=sigma,
                entropy_before=initial_entropy,
                uncertainty_summary={
                    "entropy": initial_entropy,
                    "trait_variances": _variances,
                },
                question_history_summary=None,
                answer_history_summary=None,
                unresolved_tensions=None,
                prompt_rendered=rendered_prompt,
                model_name=llm_model or DEFAULT_LLM_MODEL,
                n_requested=n_generation_candidates,
                n_returned=n_returned,
                status=generation_status,
                error_message=generation_error,
                raw_response_text=raw_llm_response,
            )
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(f"Failed to persist LLM generation request: {exc}", stacklevel=2)

    # Phase 3: persist generated candidate metadata (all raw LLM candidates, including rejected).
    # Phase 5: attach generation_request_id and prompt_policy_version_id for full lineage.
    if generated_candidates_metadata is not None:
        for meta in generated_candidates_metadata:
            try:
                db.insert_generated_question_candidate(
                    conn,
                    session_id=session_id,
                    candidate_index=int(meta["candidate_index"]),
                    text=str(meta["text"]),
                    question_id=meta.get("question_id"),
                    max_seed_similarity=meta.get("max_seed_similarity"),
                    max_kept_similarity=meta.get("max_kept_similarity"),
                    dedupe_failed=bool(meta.get("dedupe_failed", False)),
                    validation_passed=bool(meta.get("validation_passed", False)),
                    validation_failure_reason=meta.get("validation_failure_reason"),
                    accepted_into_pool=bool(meta.get("accepted_into_pool", False)),
                    w=meta.get("w"),
                    noise_var=meta.get("noise_var"),
                    thresholds=meta.get("thresholds"),
                    nn_seed_ids=meta.get("nn_seed_ids"),
                    nn_similarities=meta.get("nn_similarities"),
                    generation_request_id=gen_request_id,
                    prompt_policy_version_id=policy_version_id,
                    embedding_model=meta.get("embedding_model"),
                    embedding_ref=meta.get("embedding_ref"),
                    intended_contrast=meta.get("intended_contrast"),
                    llm_suggested_traits=meta.get("llm_suggested_traits"),
                    expected_response_pattern=meta.get("expected_response_pattern"),
                    risk_notes=meta.get("risk_notes"),
                    provisional_w_source=meta.get("provisional_w_source"),
                    provisional_w_confidence=meta.get("provisional_w_confidence"),
                    calibration_status=str(meta.get("calibration_status") or "candidate"),
                )
            except Exception as exc:  # noqa: BLE001
                import warnings
                warnings.warn(f"Failed to persist generated question metadata: {exc}", stacklevel=2)

    q = get_next_question(conn, questions_v2_path=questions_v2_path, session_id=session_id, dim=dim)
    if q is None:
        raise RuntimeError("Failed to get first question for new session.")
    return session_id, q


def _question_payload(question: Question, *, pool: Literal["inference", "heldout"]) -> QuestionPayload:
    return QuestionPayload(id=question.id, text=question.text, pool=pool, num_categories=int(question.thresholds.size + 1))


def get_next_question(
    conn,
    *,
    questions_v2_path: str,
    session_id: str,
    dim: int = 5,
) -> Optional[QuestionPayload]:
    """
    Return the next question for the session, or None if complete.
    """
    sess = db.get_session(conn, session_id)
    inference_questions = _session_inference_questions(sess, questions_v2_path, dim)
    qmap = _qmap_with_session_inference(sess, questions_v2_path, dim)

    asked: Set[str] = set(sess.asked_ids)
    responses = db.list_responses(conn, session_id)
    answered_ids: Set[str] = {r.question_id for r in responses}

    if sess.status == "complete":
        return None

    # Inference phase: ask up to max_inference_questions and update posterior per response.
    if sess.status == "inference":
        if sess.step >= sess.max_inference_questions:
            db.update_session_state(conn, session_id=session_id, status="heldout")
            sess = db.get_session(conn, session_id)
        else:
            state = _state_from_jsonable(sess.posterior_mu, sess.posterior_sigma)
            if sess.mode == "adaptive":
                strategy = "classic_eig"
                min_anchor = 0
                max_generated = len(inference_questions)
                if sess.inference_pool:
                    first = sess.inference_pool[0]
                    strategy = str(first.get("session_strategy") or "classic_eig")
                    min_anchor = int(first.get("min_anchor_questions") or 0)
                    max_generated = int(first.get("max_generated_probes") or max_generated)

                # Load pre-generated step candidates from the cache.
                _step_questions, _step_ids, _step_dicts = _load_and_merge_step_candidates(
                    conn, sess, dim
                )
                if _step_questions:
                    inference_questions = inference_questions + _step_questions
                _all_gen_ids: Set[str] = set(sess.generated_question_ids) | _step_ids

                q_metadata = _session_question_metadata(sess)
                for _sd in _step_dicts:
                    _sqid = str(_sd.get("id", ""))
                    if _sqid and _sqid not in q_metadata:
                        q_metadata[_sqid] = {
                            "source": "generated",
                            "calibration_status": "accepted_uncalibrated",
                            "policy_quality_prior": 0.0,
                            "risk_penalty": 0.0,
                        }

                # Helper to determine source including step-cache candidates.
                def _effective_source(qid: str) -> str:
                    return "generated" if qid in _all_gen_ids else "seed"

                if strategy == "anchored_exploratory":
                    best, best_score, ranked = select_next_question_exploratory(
                        state,
                        inference_questions,
                        asked_ids=asked,
                        generated_question_ids=_all_gen_ids,
                        question_metadata=q_metadata,
                        composition=SessionComposition.scaled(
                            sess.max_inference_questions,
                            min_anchor_questions=min_anchor or 2,
                            max_generated_probes=max_generated,
                        ),
                    )
                    best_eig = 0.0
                    best_components: Dict[str, float] = {}
                    if ranked:
                        _, _, best_components = ranked[0]
                        best_eig = float(best_components.get("expected_information_gain", best_score))
                    # ranked: List[Tuple[Question, float, Dict[str, float]]]
                    _alt_ranked = [
                        (q, score, comps) for q, score, comps in ranked[1:SELECTION_LOG_TOP_K]
                    ]
                    _alt_eig_mode = False
                else:
                    best, best_eig, _eig_ranked = select_next_question_eig(state, inference_questions, asked_ids=asked)
                    best_components = {
                        "selection_score": float(best_eig),
                        "expected_information_gain": float(best_eig),
                        "semantic_novelty": 1.0,
                        "exploration_bonus": 0.0,
                        "policy_quality_prior": 0.0,
                        "redundancy_penalty": 0.0,
                        "risk_penalty": 0.0,
                    }
                    # _eig_ranked: List[Tuple[Question, float]]
                    _alt_ranked = [
                        (q, eig, {
                            "selection_score": float(eig),
                            "expected_information_gain": float(eig),
                        })
                        for q, eig in _eig_ranked[1:SELECTION_LOG_TOP_K]
                    ]
                    _alt_eig_mode = True

                # If winner is a step-cache candidate, add it to the session pool.
                if best is not None and best.id in _step_ids:
                    _cand_dict = next((d for d in _step_dicts if d["id"] == best.id), None)
                    if _cand_dict is not None:
                        _pool_ok = False
                        try:
                            db.add_candidate_to_session_pool(
                                conn,
                                session_id=sess.session_id,
                                candidate_dict=_cand_dict,
                                new_generated_id=best.id,
                            )
                            db.update_step_candidate_status(
                                conn, sess.session_id, sess.step, best.id, "selected"
                            )
                            _pool_ok = True
                        except Exception as _exc:  # noqa: BLE001
                            import warnings
                            warnings.warn(
                                f"Failed to add step candidate {best.id} to pool: {_exc}; "
                                "falling back to best seed candidate.",
                                stacklevel=2,
                            )
                        if not _pool_ok:
                            _non_cache = [
                                q for q in inference_questions
                                if q.id not in asked and q.id not in _step_ids
                            ]
                            if _non_cache:
                                best, best_eig, _ = select_next_question_eig(
                                    state, _non_cache, asked_ids=asked
                                )
                                best_components = {
                                    "selection_score": float(best_eig),
                                    "expected_information_gain": float(best_eig),
                                }
                            else:
                                best = None

                if best is None:
                    db.update_session_state(conn, session_id=session_id, status="heldout")
                    return get_next_question(conn, questions_v2_path=questions_v2_path, session_id=session_id, dim=dim)
                db.update_pending_selection(
                    conn, session_id=session_id, question_id=best.id, eig=float(best_eig)
                )
                # Deduplicate: skip logging if this step was already logged (repeated get_next_question call).
                if not db.has_selection_score_log(conn, session_id, sess.step):
                    db.insert_selection_score_log(
                        conn,
                        session_id=session_id,
                        step_idx=sess.step,
                        question_id=best.id,
                        question_source=_effective_source(best.id),
                        components=best_components,
                        calibration_status=q_metadata.get(best.id, {}).get("calibration_status"),
                        selected=True,
                        candidate_rank=0,
                    )
                    for _rank, (_alt_q, _alt_score, _alt_comps) in enumerate(_alt_ranked, start=1):
                        db.insert_selection_score_log(
                            conn,
                            session_id=session_id,
                            step_idx=sess.step,
                            question_id=_alt_q.id,
                            question_source=_effective_source(_alt_q.id),
                            components=_alt_comps,
                            calibration_status=q_metadata.get(_alt_q.id, {}).get("calibration_status"),
                            selected=False,
                            candidate_rank=_rank,
                        )
                return _question_payload(best, pool="inference")

            # fixed_order
            order = sess.fixed_order_ids or [q.id for q in inference_questions]
            for qid in order:
                if qid not in asked:
                    qn = qmap[qid]
                    eig = float(expected_information_gain(state, qn))
                    db.update_pending_selection(conn, session_id=session_id, question_id=qid, eig=eig)
                    return _question_payload(qn, pool="inference")
            db.update_session_state(conn, session_id=session_id, status="heldout")
            return get_next_question(conn, questions_v2_path=questions_v2_path, session_id=session_id, dim=dim)

    # Held-out phase: ask held-out questions selected at session start.
    if sess.status == "heldout":
        for qid in sess.heldout_ids:
            if qid not in answered_ids:
                db.update_pending_selection(conn, session_id=session_id, question_id=qid, eig=None)
                return _question_payload(qmap[qid], pool="heldout")
        db.update_session_state(conn, session_id=session_id, status="complete")
        return None

    raise RuntimeError(f"Unknown session status: {sess.status}")


def record_answer(
    conn,
    *,
    questions_v2_path: str,
    session_id: str,
    question_id: str,
    response: int,
    dim: int = 5,
) -> Dict:
    """
    Record an answer, update posterior if inference item, and return next action.

    Returns a dict payload suitable for API responses:
      - {"status": "inference"|"heldout"|"complete", "next_question": {...}|None}

    Phase 6 guarantees:
    - Only the currently pending question is accepted (no skipping ahead).
    - All DB writes are wrapped in a single atomic transaction; if any step fails
      (including the posterior update), the entire answer is rolled back.
    """
    sess = db.get_session(conn, session_id)

    # Phase 6: pending-question validation — reject stale or out-of-order answers.
    if sess.status == "complete":
        raise ValueError("Session is already complete. No further answers are accepted.")
    if sess.pending_question_id is None:
        raise ValueError(
            "No pending question for this session. Call /next_question first."
        )
    if question_id != sess.pending_question_id:
        raise ValueError(
            f"question_id '{question_id}' is not the currently pending question "
            f"(expected '{sess.pending_question_id}'). "
            "Submit answers only for the currently issued question; do not skip ahead."
        )

    inference_questions = _session_inference_questions(sess, questions_v2_path, dim)
    qmap = _qmap_with_session_inference(sess, questions_v2_path, dim)

    if question_id not in qmap:
        raise ValueError(f"Unknown question_id: {question_id}")

    q = qmap[question_id]
    K = int(q.thresholds.size + 1)
    if response < 1 or response > K:
        raise ValueError(f"response must be in [1, {K}] for question {question_id}")

    inference_ids = {qq.id for qq in inference_questions}
    pool: Literal["inference", "heldout"] = "inference" if question_id in inference_ids else "heldout"

    # Cross-phase guard: pending_question_id should never point at a heldout item during
    # inference, but guard defensively in case of data corruption.
    if sess.status == "inference" and pool == "heldout":
        raise ValueError("Cannot answer held-out items while still in inference phase.")

    step_idx = len(db.list_responses(conn, session_id))
    state_pre = _state_from_jsonable(sess.posterior_mu, sess.posterior_sigma)
    entropy_before = float(state_pre.entropy())

    eig_at: Optional[float]
    if sess.pending_question_id == question_id and sess.pending_eig is not None:
        eig_at = float(sess.pending_eig)
    elif pool == "inference":
        eig_at = float(expected_information_gain(state_pre, q))
    else:
        eig_at = None

    src = _question_source(question_id, sess)

    # Phase 6: atomic transaction — all DB writes for this answer succeed or all roll back.
    # _AtomicConn defers conn.commit() calls to a single commit at __exit__, rolling back
    # on any exception (including a failed posterior update).
    with _AtomicConn(conn) as txn:
        db.insert_response(
            txn,
            session_id=session_id,
            question_id=question_id,
            pool=pool,
            step=sess.step,
            response=int(response),
        )

        if pool == "inference":
            # Posterior update — may raise on numerical failure; rolls back insert_response.
            state_pre.update_posterior_likert_laplace(
                w=q.w,
                y=int(response),
                thresholds=q.thresholds,
                noise_var=float(q.noise_var),
            )
            entropy_after = float(state_pre.entropy())
            mu, sigma = _state_to_jsonable(state_pre)

            asked_ids = list(sess.asked_ids)
            if question_id not in asked_ids:
                asked_ids.append(question_id)

            next_step = int(sess.step) + 1
            next_status: Status = sess.status
            if next_step >= sess.max_inference_questions:
                next_status = "heldout"

            db.update_session_state(
                txn,
                session_id=session_id,
                status=next_status,
                step=next_step,
                asked_ids=asked_ids,
                posterior_mu=mu,
                posterior_sigma=sigma,
            )

            # Phase 1: per-step snapshot and longitudinal user state.
            db.insert_posterior_snapshot(
                txn,
                session_id=session_id,
                step_idx=next_step,
                mu=mu,
                sigma=sigma,
                entropy=entropy_after,
            )
            if sess.user_id is not None:
                db.upsert_user_current_state(
                    txn,
                    user_id=sess.user_id,
                    latest_session_id=session_id,
                    latest_step_idx=next_step,
                    mu=mu,
                    sigma=sigma,
                    entropy=entropy_after,
                )

            # Phase 4+6: use the parameter version frozen at session creation, not the
            # currently active version. This prevents scientific drift when parameters are
            # re-estimated after the session pool was built.
            # Phase 9: also read calibration_status from the frozen pool dict.
            pool_param_version: Optional[int] = None
            pool_calibration_status: Optional[str] = None
            if sess.inference_pool:
                for _d in sess.inference_pool:
                    if str(_d.get("id")) == question_id:
                        pool_param_version = _d.get("param_version")
                        pool_calibration_status = _d.get("calibration_status") or None
                        break

            # Phase 6: direct lineage — resolve generated_candidate_id and linked request/policy.
            generated_candidate_id: Optional[int] = None
            gen_cand_request_id: Optional[int] = None
            gen_cand_policy_version_id: Optional[int] = None
            if src == "generated":
                _gen_cand = db.get_generated_candidate_for_session_question(
                    txn, session_id, question_id
                )
                if _gen_cand is not None:
                    generated_candidate_id = _gen_cand["id"]
                    gen_cand_request_id = _gen_cand.get("generation_request_id")
                    gen_cand_policy_version_id = _gen_cand.get("prompt_policy_version_id")

            # Phase 2+6: performance event with direct lineage fields.
            # Phase 9: pass calibration_status from the frozen pool dict.
            db.insert_question_performance_event(
                txn,
                question_id=question_id,
                session_id=session_id,
                user_id=sess.user_id,
                step_idx=step_idx,
                question_source=src,
                parameter_version=pool_param_version,
                predicted_eig=eig_at,
                entropy_before=entropy_before,
                entropy_after=entropy_after,
                realized_information_gain=entropy_before - entropy_after,
                response_value=int(response),
                mu_before=sess.posterior_mu,
                sigma_before=sess.posterior_sigma,
                mu_after=mu,
                sigma_after=sigma,
                generated_candidate_id=generated_candidate_id,
                generation_request_id=gen_cand_request_id,
                prompt_policy_version_id=gen_cand_policy_version_id,
                calibration_status=pool_calibration_status,
            )

            # Phase 3: mark generated question as selected at this step.
            if src == "generated":
                try:
                    db.update_generated_candidate_selected_at_step(
                        txn,
                        session_id=session_id,
                        question_id=question_id,
                        selected_at_step=step_idx,
                    )
                except Exception as exc:  # noqa: BLE001
                    import warnings
                    warnings.warn(
                        f"Failed to update generated candidate selected_at_step: {exc}",
                        stacklevel=2,
                    )
        else:
            entropy_after = entropy_before

        db.insert_step_log(
            txn,
            session_id=session_id,
            step_idx=step_idx,
            question_id=question_id,
            source=src,
            response=int(response),
            eig_at_selection=eig_at,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
        )
        db.clear_pending_selection(txn, session_id=session_id)
    # End of atomic block — conn.commit() issued here on success, rollback on exception.

    # Post-commit: advance to the next question (sets new pending_question_id) and finalize
    # heldout evaluation if the session just completed.
    next_q = get_next_question(conn, questions_v2_path=questions_v2_path, session_id=session_id, dim=dim)
    sess2 = db.get_session(conn, session_id)
    if sess2.status == "complete":
        _finalize_heldout_evaluation_if_needed(
            conn, session_id=session_id, questions_v2_path=questions_v2_path, dim=dim
        )
        sess2 = db.get_session(conn, session_id)

    return {
        "session_id": session_id,
        "status": sess2.status,
        "step": sess2.step,
        "next_question": next_q.__dict__ if next_q is not None else None,
    }


def get_question_num_categories(
    conn,
    *,
    questions_v2_path: str,
    session_id: str,
    question_id: str,
    dim: int = 5,
) -> Optional[int]:
    """Return num_categories for question_id in this session, or None if session/question not found."""
    try:
        sess = db.get_session(conn, session_id)
    except KeyError:
        return None
    qmap = _qmap_with_session_inference(sess, questions_v2_path, dim)
    q = qmap.get(question_id)
    if q is None:
        return None
    return int(q.thresholds.size + 1)


def end_session(conn, *, session_id: str) -> None:
    """
    Mark a session complete.
    """
    db.update_session_state(conn, session_id=session_id, status="complete")


def get_session_summary(
    conn,
    *,
    questions_v2_path: str,
    session_id: str,
    dim: int = 5,
) -> Dict:
    """
    Return a JSON-serializable summary of the session state and responses.
    """
    sess = db.get_session(conn, session_id)
    qmap = _qmap_with_session_inference(sess, questions_v2_path, dim)
    responses = db.list_responses(conn, session_id)

    return {
        "session_id": sess.session_id,
        "mode": sess.mode,
        "status": sess.status,
        "step": sess.step,
        "max_inference_questions": sess.max_inference_questions,
        "arm": sess.arm,
        "generated_items": sess.generated_items,
        "inference_pool": sess.inference_pool,
        "asked_question_ids": sess.asked_ids,
        "heldout_question_ids": sess.heldout_ids,
        "fixed_order_ids": sess.fixed_order_ids,
        "posterior": {"mu": sess.posterior_mu, "sigma": sess.posterior_sigma},
        "heldout_evaluation": sess.heldout_evaluation,
        "n_generated_candidates": sess.n_generated_candidates,
        "generated_question_ids": sess.generated_question_ids,
        "step_logs": db.list_step_logs(conn, session_id),
        "selection_score_logs": db.list_selection_score_logs(conn, session_id),
        "session_run_log": db.get_session_run_log(conn, session_id),
        "generated_candidate_metadata": db.list_generated_question_candidates(conn, session_id),
        "responses": [
            {
                "question_id": r.question_id,
                "pool": r.pool,
                "step": r.step,
                "response": r.response,
                "text": qmap[r.question_id].text if r.question_id in qmap else None,
            }
            for r in responses
        ],
        "created_at": sess.created_at,
        "updated_at": sess.updated_at,
    }


def generate_step_candidates_bg(
    session_id: str,
    for_step_idx: int,
    db_path: str,
    questions_v2_path: str,
    llm_api_key: Optional[str],
    llm_model: Optional[str],
    dim: int,
) -> None:
    """
    Background task: pre-generate LLM question candidates for the next selection step.
    Opens its own DB connection so it can safely run after the HTTP response is sent.
    Stores accepted candidates in step_candidates_cache for pick-up by get_next_question().
    """
    import warnings
    conn = None
    try:
        conn = db.connect(db_path)
        db.init_db(conn)

        # Guards
        try:
            sess = db.get_session(conn, session_id)
        except KeyError:
            return
        if sess.status != "inference":
            return
        if for_step_idx >= sess.max_inference_questions:
            return
        if db.step_candidates_exist(conn, session_id, for_step_idx):
            return
        if not should_use_generated_questions(sess.arm) and not (
            sess.inference_pool and sess.inference_pool[0].get("session_strategy") == "anchored_exploratory"
        ):
            return

        # Build qmap and response history for sketch
        inference_seed, _, qmap = load_bank(questions_v2_path, expected_dim=dim)
        responses = db.list_responses(conn, session_id)
        history = _build_response_history_for_sketch(responses, qmap)

        # Generate updated character sketch from current posterior
        state = _state_from_jsonable(sess.posterior_mu, sess.posterior_sigma)
        client = make_llm_client(api_key=llm_api_key, model=llm_model)
        sketch = generate_character_sketch(state, history, client)

        # Update the stored sketch
        try:
            db.update_session_sketch(conn, session_id, sketch, step_idx=sess.step)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Failed to update session sketch for {session_id}: {exc}", stacklevel=2)

        # Select an active policy for rendering the prompt
        policy_row = db.get_active_prompt_policy_version(conn)
        seeds_for_gen = [{"id": q.id, "text": q.text, "w": q.w.tolist()} for q in inference_seed]
        rendered_prompt: Optional[str] = None
        policy_version_id: Optional[int] = None
        if policy_row is not None:
            policy_version_id = int(policy_row["id"])
            rendered_prompt = render_prompt_policy(
                PromptPolicy.from_row(policy_row),
                seeds_for_gen,
                2,
                uncertainty_summary=None,
                character_sketch=sketch,
            )

        gen_status = "success"
        gen_error: Optional[str] = None
        n_returned = 0
        gen_result = None
        try:
            gen_result = build_generated_pool(
                client, seeds_for_gen, n_candidates=2, k=3, prompt=rendered_prompt
            )
            n_returned = len(gen_result.all_candidates_metadata)
        except Exception as exc:  # noqa: BLE001
            gen_status = "failed"
            gen_error = str(exc)[:500]

        # Log the generation request
        sigma_arr = np.asarray(sess.posterior_sigma)
        variances = np.diag(sigma_arr).tolist()
        gen_request_id: Optional[int] = None
        try:
            gen_request_id = db.insert_llm_generation_request(
                conn,
                session_id=session_id,
                user_id=sess.user_id,
                step_idx=for_step_idx,
                prompt_policy_version_id=policy_version_id,
                posterior_mu=sess.posterior_mu,
                posterior_sigma=sess.posterior_sigma,
                entropy_before=float(state.entropy()),
                uncertainty_summary={"entropy": float(state.entropy()), "trait_variances": variances},
                question_history_summary=None,
                answer_history_summary=None,
                unresolved_tensions=None,
                prompt_rendered=rendered_prompt,
                model_name=llm_model or DEFAULT_LLM_MODEL,
                n_requested=2,
                n_returned=n_returned,
                status=gen_status,
                error_message=gen_error,
                raw_response_text=gen_result.raw_response_text if gen_result else None,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Failed to log generation request for {session_id} step {for_step_idx}: {exc}",
                stacklevel=2,
            )

        # Store accepted candidates in step_candidates_cache
        if gen_result is not None:
            accepted_dicts = [_jsonable_question_dict(x) for x in gen_result.accepted]
            for i, cand in enumerate(accepted_dicts):
                q_id = str(cand.get("id") or f"sc_{session_id[:6]}_{for_step_idx}_{i}")
                try:
                    db.insert_step_candidate(
                        conn,
                        session_id=session_id,
                        for_step_idx=for_step_idx,
                        question_id=q_id,
                        text=str(cand.get("text", "")),
                        w=list(cand.get("w") or []),
                        noise_var=float(cand.get("noise_var") or 1.0),
                        thresholds=list(cand.get("thresholds") or [-1.5, -0.5, 0.5, 1.5]),
                        nn_seed_ids=cand.get("nn_seed_ids"),
                        nn_similarities=cand.get("nn_similarities"),
                        sketch_mu=sess.posterior_mu,
                        sketch_sigma=sess.posterior_sigma,
                        character_sketch=sketch,
                        generation_request_id=gen_request_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(f"Failed to insert step candidate for {session_id}: {exc}", stacklevel=2)

    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(
            f"generate_step_candidates_bg failed for {session_id} step {for_step_idx}: {exc}",
            stacklevel=2,
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
