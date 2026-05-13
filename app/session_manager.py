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
from models.personality_state import PersonalityState
from models.question_bank import DEFAULT_THRESHOLDS, Question, load_question_pools_v2
from models.question_generation import make_llm_client
from models.question_pool_builder import build_generated_pool, build_session_inference_pool
from models.question_selection import expected_information_gain, select_next_question_eig
from models.real_eval import evaluate_heldout_performance
from models.session_experiment import assign_experiment_arm, should_use_generated_questions


Mode = Literal["adaptive", "fixed_order"]
Status = Literal["inference", "heldout", "complete"]


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


def _qmap_with_session_inference(sess: db.SessionRow, questions_v2_path: str, dim: int) -> Dict[str, Question]:
    """Question id map including held-out from file and session inference items (incl. generated)."""
    _, _, qmap = load_bank(questions_v2_path, expected_dim=dim)
    if sess.inference_pool:
        for d in sess.inference_pool:
            q = _dict_to_question(d)
            qmap[q.id] = q
    return qmap


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

    use_generated = should_use_generated_questions(arm)
    generated_stored: Optional[List[Dict[str, Any]]] = None
    if use_generated:
        if llm_client is not None:
            client = llm_client
        else:
            client = make_llm_client(api_key=llm_api_key, model=llm_model)
        try:
            gen_list = build_generated_pool(
                client,
                seeds_for_generation,
                n_candidates=n_generation_candidates,
                k=nn_k,
            )
            generated_stored = [_jsonable_question_dict(x) for x in gen_list]
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"LLM question generation failed; falling back to seeds only. Error: {exc}",
                stacklevel=2,
            )
            generated_stored = None

    final_inference_dicts = build_session_inference_pool(seed_stored, generated_stored if use_generated else None)
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

    state = PersonalityState(dim=dim)
    mu, sigma = _state_to_jsonable(state)

    n_gen = len(generated_stored) if generated_stored else 0
    gen_qids = [str(d["id"]) for d in generated_stored] if generated_stored else []

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
    )

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
                best, best_eig, _ = select_next_question_eig(state, inference_questions, asked_ids=asked)
                if best is None:
                    db.update_session_state(conn, session_id=session_id, status="heldout")
                    return get_next_question(conn, questions_v2_path=questions_v2_path, session_id=session_id, dim=dim)
                db.update_pending_selection(
                    conn, session_id=session_id, question_id=best.id, eig=float(best_eig)
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
    """
    sess = db.get_session(conn, session_id)
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

    # Prevent answering out-of-phase heldout items during inference unless requested.
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

    # Write response first (so it's always logged, even if update fails later).
    db.insert_response(
        conn,
        session_id=session_id,
        question_id=question_id,
        pool=pool,
        step=sess.step if pool == "inference" else sess.step,
        response=int(response),
    )

    if pool == "inference":
        # Update posterior snapshot.
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
            conn,
            session_id=session_id,
            status=next_status,
            step=next_step,
            asked_ids=asked_ids,
            posterior_mu=mu,
            posterior_sigma=sigma,
        )
    else:
        entropy_after = entropy_before

    db.insert_step_log(
        conn,
        session_id=session_id,
        step_idx=step_idx,
        question_id=question_id,
        source=src,
        response=int(response),
        eig_at_selection=eig_at,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
    )
    db.clear_pending_selection(conn, session_id=session_id)

    # Held-out answers do not update posterior; we just advance via get_next_question().
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
        "session_run_log": db.get_session_run_log(conn, session_id),
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

