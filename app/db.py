"""
SQLite persistence for the real-user pilot session backend.

This is intentionally minimal (no ORM). It stores:
- sessions: current session state + latest posterior snapshot (mu/sigma)
- responses: all recorded item responses (inference + heldout)
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _utc_ts() -> int:
    return int(time.time())


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _loads(s: Optional[str], default: Any) -> Any:
    if s is None:
        return default
    return json.loads(s)


@dataclass(frozen=True)
class SessionRow:
    session_id: str
    mode: str
    status: str
    step: int
    max_inference_questions: int
    asked_ids: List[str]
    heldout_ids: List[str]
    fixed_order_ids: Optional[List[str]]
    posterior_mu: List[float]
    posterior_sigma: List[List[float]]
    created_at: int
    updated_at: int
    # V2-lite experiment + inference pool (None on legacy rows before migration)
    arm: str
    generated_items: Optional[List[Any]]
    inference_pool: Optional[List[Any]]
    # Filled when session completes; uses posterior after inference, not updated by held-out answers
    heldout_evaluation: Optional[Dict[str, Any]]
    # Logging (V2-lite)
    n_generated_candidates: int
    generated_question_ids: List[str]
    pending_question_id: Optional[str]
    pending_eig: Optional[float]
    # Phase 1: anonymous users and longitudinal state
    user_id: Optional[str]
    prior_session_id: Optional[str]


@dataclass(frozen=True)
class ResponseRow:
    session_id: str
    question_id: str
    pool: str  # "inference" | "heldout"
    step: int
    response: int
    created_at: int


def connect(db_path: str) -> sqlite3.Connection:
    """
    Open a SQLite connection with sensible defaults.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _migrate_sessions_user(conn: sqlite3.Connection) -> None:
    """Add user_id and prior_session_id columns to sessions (Phase 1)."""
    info = conn.execute("PRAGMA table_info(sessions);").fetchall()
    col_names = {row[1] for row in info}
    if "user_id" not in col_names:
        conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT;")
    if "prior_session_id" not in col_names:
        conn.execute("ALTER TABLE sessions ADD COLUMN prior_session_id TEXT;")


def _migrate_sessions_v2(conn: sqlite3.Connection) -> None:
    """Add V2-lite columns to existing databases."""
    info = conn.execute("PRAGMA table_info(sessions);").fetchall()
    col_names = {row[1] for row in info}
    if "arm" not in col_names:
        conn.execute("ALTER TABLE sessions ADD COLUMN arm TEXT NOT NULL DEFAULT 'seed_only';")
    if "generated_items_json" not in col_names:
        conn.execute("ALTER TABLE sessions ADD COLUMN generated_items_json TEXT;")
    if "inference_pool_json" not in col_names:
        conn.execute("ALTER TABLE sessions ADD COLUMN inference_pool_json TEXT;")
    if "heldout_eval_json" not in col_names:
        conn.execute("ALTER TABLE sessions ADD COLUMN heldout_eval_json TEXT;")
    if "n_generated_candidates" not in col_names:
        conn.execute("ALTER TABLE sessions ADD COLUMN n_generated_candidates INTEGER NOT NULL DEFAULT 0;")
    if "generated_question_ids_json" not in col_names:
        conn.execute("ALTER TABLE sessions ADD COLUMN generated_question_ids_json TEXT;")
    if "pending_question_id" not in col_names:
        conn.execute("ALTER TABLE sessions ADD COLUMN pending_question_id TEXT;")
    if "pending_eig" not in col_names:
        conn.execute("ALTER TABLE sessions ADD COLUMN pending_eig REAL;")


def init_db(conn: sqlite3.Connection) -> None:
    """
    Create tables if they do not exist.
    """
    # users must be created before sessions so the FK constraint is valid on new DBs.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
          user_id    TEXT    PRIMARY KEY,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
          session_id TEXT PRIMARY KEY,
          mode TEXT NOT NULL,
          status TEXT NOT NULL,
          step INTEGER NOT NULL,
          max_inference_questions INTEGER NOT NULL,
          asked_ids_json TEXT NOT NULL,
          heldout_ids_json TEXT NOT NULL,
          fixed_order_ids_json TEXT,
          posterior_mu_json TEXT NOT NULL,
          posterior_sigma_json TEXT NOT NULL,
          arm TEXT NOT NULL DEFAULT 'seed_only',
          generated_items_json TEXT,
          inference_pool_json TEXT,
          heldout_eval_json TEXT,
          n_generated_candidates INTEGER NOT NULL DEFAULT 0,
          generated_question_ids_json TEXT,
          pending_question_id TEXT,
          pending_eig REAL,
          user_id TEXT REFERENCES users(user_id),
          prior_session_id TEXT,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );
        """
    )
    _migrate_sessions_v2(conn)
    _migrate_sessions_user(conn)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_step_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT NOT NULL,
          step_idx INTEGER NOT NULL,
          question_id TEXT NOT NULL,
          source TEXT NOT NULL,
          response INTEGER NOT NULL,
          eig_at_selection REAL,
          entropy_before REAL NOT NULL,
          entropy_after REAL NOT NULL,
          created_at INTEGER NOT NULL,
          FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_run_logs (
          session_id TEXT PRIMARY KEY,
          arm TEXT NOT NULL,
          n_generated_candidates INTEGER NOT NULL,
          n_generated_selected INTEGER NOT NULL,
          heldout_log_likelihood REAL,
          mean_true_prob REAL,
          generated_usage_json TEXT,
          created_at INTEGER NOT NULL,
          FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT NOT NULL,
          question_id TEXT NOT NULL,
          pool TEXT NOT NULL,
          step INTEGER NOT NULL,
          response INTEGER NOT NULL,
          created_at INTEGER NOT NULL,
          UNIQUE(session_id, question_id),
          FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """
    )

    # Phase 1: longitudinal user state tables
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_current_state (
          user_id           TEXT    PRIMARY KEY,
          latest_session_id TEXT    NOT NULL,
          latest_step_idx   INTEGER NOT NULL,
          mu_json           TEXT    NOT NULL,
          sigma_json        TEXT    NOT NULL,
          entropy           REAL    NOT NULL,
          updated_at        INTEGER NOT NULL,
          FOREIGN KEY(user_id)           REFERENCES users(user_id),
          FOREIGN KEY(latest_session_id) REFERENCES sessions(session_id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_posteriors (
          id             INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id        TEXT    NOT NULL,
          session_id     TEXT    NOT NULL UNIQUE,
          session_number INTEGER NOT NULL,
          mu_json        TEXT    NOT NULL,
          sigma_json     TEXT    NOT NULL,
          entropy        REAL    NOT NULL,
          created_at     INTEGER NOT NULL,
          FOREIGN KEY(user_id)    REFERENCES users(user_id),
          FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS posterior_snapshots (
          id         INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT    NOT NULL,
          step_idx   INTEGER NOT NULL,
          mu_json    TEXT    NOT NULL,
          sigma_json TEXT    NOT NULL,
          entropy    REAL    NOT NULL,
          created_at INTEGER NOT NULL,
          UNIQUE(session_id, step_idx),
          FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """
    )

    # Phase 2: per-answer learning signal for question-level performance analysis.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS question_performance_events (
          id                        INTEGER PRIMARY KEY AUTOINCREMENT,
          question_id               TEXT    NOT NULL,
          session_id                TEXT    NOT NULL,
          user_id                   TEXT,
          step_idx                  INTEGER NOT NULL,
          question_source           TEXT    NOT NULL,
          parameter_version         INTEGER,
          predicted_eig             REAL,
          entropy_before            REAL    NOT NULL,
          entropy_after             REAL    NOT NULL,
          realized_information_gain REAL    NOT NULL,
          response_value            INTEGER NOT NULL,
          mu_before_json            TEXT    NOT NULL,
          sigma_before_json         TEXT    NOT NULL,
          mu_after_json             TEXT    NOT NULL,
          sigma_after_json          TEXT    NOT NULL,
          created_at                INTEGER NOT NULL,
          FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


def insert_session(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    mode: str,
    status: str,
    step: int,
    max_inference_questions: int,
    asked_ids: Sequence[str],
    heldout_ids: Sequence[str],
    fixed_order_ids: Optional[Sequence[str]],
    posterior_mu: Sequence[float],
    posterior_sigma: Sequence[Sequence[float]],
    arm: str = "seed_only",
    generated_items: Optional[Any] = None,
    inference_pool: Optional[Any] = None,
    heldout_evaluation: Optional[Any] = None,
    n_generated_candidates: int = 0,
    generated_question_ids: Optional[Sequence[str]] = None,
    created_at: Optional[int] = None,
    user_id: Optional[str] = None,
    prior_session_id: Optional[str] = None,
) -> None:
    now = _utc_ts() if created_at is None else int(created_at)
    gen_ids = list(generated_question_ids) if generated_question_ids is not None else []
    conn.execute(
        """
        INSERT INTO sessions (
          session_id, mode, status, step, max_inference_questions,
          asked_ids_json, heldout_ids_json, fixed_order_ids_json,
          posterior_mu_json, posterior_sigma_json,
          arm, generated_items_json, inference_pool_json, heldout_eval_json,
          n_generated_candidates, generated_question_ids_json,
          pending_question_id, pending_eig,
          user_id, prior_session_id,
          created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            session_id,
            mode,
            status,
            int(step),
            int(max_inference_questions),
            _dumps(list(asked_ids)),
            _dumps(list(heldout_ids)),
            _dumps(list(fixed_order_ids)) if fixed_order_ids is not None else None,
            _dumps(list(posterior_mu)),
            _dumps([list(row) for row in posterior_sigma]),
            arm,
            _dumps(generated_items) if generated_items is not None else None,
            _dumps(inference_pool) if inference_pool is not None else None,
            _dumps(heldout_evaluation) if heldout_evaluation is not None else None,
            int(n_generated_candidates),
            _dumps(gen_ids),
            None,
            None,
            user_id,
            prior_session_id,
            now,
            now,
        ),
    )
    conn.commit()


def update_session_state(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    status: Optional[str] = None,
    step: Optional[int] = None,
    asked_ids: Optional[Sequence[str]] = None,
    posterior_mu: Optional[Sequence[float]] = None,
    posterior_sigma: Optional[Sequence[Sequence[float]]] = None,
) -> None:
    """
    Patch session state fields (only those provided).
    """
    updates: List[str] = []
    params: List[Any] = []

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if step is not None:
        updates.append("step = ?")
        params.append(int(step))
    if asked_ids is not None:
        updates.append("asked_ids_json = ?")
        params.append(_dumps(list(asked_ids)))
    if posterior_mu is not None:
        updates.append("posterior_mu_json = ?")
        params.append(_dumps(list(posterior_mu)))
    if posterior_sigma is not None:
        updates.append("posterior_sigma_json = ?")
        params.append(_dumps([list(row) for row in posterior_sigma]))

    updates.append("updated_at = ?")
    params.append(_utc_ts())

    if not updates:
        return

    params.append(session_id)
    sql = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?;"
    conn.execute(sql, params)
    conn.commit()


def update_heldout_evaluation(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    heldout_evaluation: Dict[str, Any],
) -> None:
    """Persist held-out metrics (posterior frozen after inference)."""
    conn.execute(
        """
        UPDATE sessions
        SET heldout_eval_json = ?, updated_at = ?
        WHERE session_id = ?;
        """,
        (_dumps(heldout_evaluation), _utc_ts(), session_id),
    )
    conn.commit()


def update_pending_selection(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    question_id: str,
    eig: Optional[float],
) -> None:
    """Store which question was issued and its EIG (if applicable)."""
    conn.execute(
        """
        UPDATE sessions
        SET pending_question_id = ?, pending_eig = ?, updated_at = ?
        WHERE session_id = ?;
        """,
        (question_id, eig, _utc_ts(), session_id),
    )
    conn.commit()


def clear_pending_selection(conn: sqlite3.Connection, *, session_id: str) -> None:
    conn.execute(
        """
        UPDATE sessions
        SET pending_question_id = NULL, pending_eig = NULL, updated_at = ?
        WHERE session_id = ?;
        """,
        (_utc_ts(), session_id),
    )
    conn.commit()


def insert_step_log(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    step_idx: int,
    question_id: str,
    source: str,
    response: int,
    eig_at_selection: Optional[float],
    entropy_before: float,
    entropy_after: float,
) -> None:
    conn.execute(
        """
        INSERT INTO session_step_logs (
          session_id, step_idx, question_id, source, response,
          eig_at_selection, entropy_before, entropy_after, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            session_id,
            int(step_idx),
            question_id,
            source,
            int(response),
            eig_at_selection,
            float(entropy_before),
            float(entropy_after),
            _utc_ts(),
        ),
    )
    conn.commit()


def insert_session_run_log(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    arm: str,
    n_generated_candidates: int,
    n_generated_selected: int,
    heldout_log_likelihood: Optional[float],
    mean_true_prob: Optional[float],
    generated_usage: Dict[str, bool],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO session_run_logs (
          session_id, arm, n_generated_candidates, n_generated_selected,
          heldout_log_likelihood, mean_true_prob, generated_usage_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            session_id,
            arm,
            int(n_generated_candidates),
            int(n_generated_selected),
            heldout_log_likelihood,
            mean_true_prob,
            _dumps(generated_usage),
            _utc_ts(),
        ),
    )
    conn.commit()


def list_step_logs(conn: sqlite3.Connection, session_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT session_id, step_idx, question_id, source, response,
               eig_at_selection, entropy_before, entropy_after, created_at
        FROM session_step_logs
        WHERE session_id = ?
        ORDER BY step_idx ASC, id ASC;
        """,
        (session_id,),
    ).fetchall()
    return [
        {
            "session_id": r["session_id"],
            "step_idx": int(r["step_idx"]),
            "question_id": r["question_id"],
            "source": r["source"],
            "response": int(r["response"]),
            "eig_at_selection": float(r["eig_at_selection"]) if r["eig_at_selection"] is not None else None,
            "entropy_before": float(r["entropy_before"]),
            "entropy_after": float(r["entropy_after"]),
            "created_at": int(r["created_at"]),
        }
        for r in rows
    ]


def get_session_run_log(conn: sqlite3.Connection, session_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM session_run_logs WHERE session_id = ?;",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "session_id": row["session_id"],
        "arm": row["arm"],
        "n_generated_candidates": int(row["n_generated_candidates"]),
        "n_generated_selected": int(row["n_generated_selected"]),
        "heldout_log_likelihood": float(row["heldout_log_likelihood"]) if row["heldout_log_likelihood"] is not None else None,
        "mean_true_prob": float(row["mean_true_prob"]) if row["mean_true_prob"] is not None else None,
        "generated_usage": _loads(row["generated_usage_json"], {}),
        "created_at": int(row["created_at"]),
    }


def get_session(conn: sqlite3.Connection, session_id: str) -> SessionRow:
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?;",
        (session_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Session not found: {session_id}")

    keys = row.keys()
    arm = str(row["arm"]) if "arm" in keys and row["arm"] is not None else "seed_only"
    gen_raw = row["generated_items_json"] if "generated_items_json" in keys else None
    inf_raw = row["inference_pool_json"] if "inference_pool_json" in keys else None
    generated_items = _loads(gen_raw, None) if gen_raw else None
    inference_pool = _loads(inf_raw, None) if inf_raw else None
    eval_raw = row["heldout_eval_json"] if "heldout_eval_json" in keys else None
    heldout_evaluation = _loads(eval_raw, None) if eval_raw else None

    n_gen_cand = (
        int(row["n_generated_candidates"])
        if "n_generated_candidates" in keys and row["n_generated_candidates"] is not None
        else 0
    )
    if "generated_question_ids_json" in keys and row["generated_question_ids_json"]:
        generated_question_ids = list(_loads(row["generated_question_ids_json"], []))
    else:
        generated_question_ids = []
    pending_qid = row["pending_question_id"] if "pending_question_id" in keys else None
    pending_eig = row["pending_eig"] if "pending_eig" in keys else None
    if pending_eig is not None:
        pending_eig = float(pending_eig)
    user_id = str(row["user_id"]) if "user_id" in keys and row["user_id"] is not None else None
    prior_session_id = (
        str(row["prior_session_id"])
        if "prior_session_id" in keys and row["prior_session_id"] is not None
        else None
    )

    return SessionRow(
        session_id=row["session_id"],
        mode=row["mode"],
        status=row["status"],
        step=int(row["step"]),
        max_inference_questions=int(row["max_inference_questions"]),
        asked_ids=list(_loads(row["asked_ids_json"], [])),
        heldout_ids=list(_loads(row["heldout_ids_json"], [])),
        fixed_order_ids=_loads(row["fixed_order_ids_json"], None),
        posterior_mu=list(_loads(row["posterior_mu_json"], [])),
        posterior_sigma=list(_loads(row["posterior_sigma_json"], [])),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        arm=arm,
        generated_items=generated_items,
        inference_pool=inference_pool,
        heldout_evaluation=heldout_evaluation,
        n_generated_candidates=n_gen_cand,
        generated_question_ids=generated_question_ids,
        pending_question_id=str(pending_qid) if pending_qid else None,
        pending_eig=pending_eig,
        user_id=user_id,
        prior_session_id=prior_session_id,
    )


def list_responses(conn: sqlite3.Connection, session_id: str) -> List[ResponseRow]:
    rows = conn.execute(
        """
        SELECT session_id, question_id, pool, step, response, created_at
        FROM responses
        WHERE session_id = ?
        ORDER BY created_at ASC, id ASC;
        """,
        (session_id,),
    ).fetchall()
    return [
        ResponseRow(
            session_id=r["session_id"],
            question_id=r["question_id"],
            pool=r["pool"],
            step=int(r["step"]),
            response=int(r["response"]),
            created_at=int(r["created_at"]),
        )
        for r in rows
    ]


def insert_response(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    question_id: str,
    pool: str,
    step: int,
    response: int,
) -> None:
    conn.execute(
        """
        INSERT INTO responses (session_id, question_id, pool, step, response, created_at)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (session_id, question_id, pool, int(step), int(response), _utc_ts()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Phase 1: users, longitudinal state, posterior snapshots
# ---------------------------------------------------------------------------

def insert_user(conn: sqlite3.Connection, *, user_id: str) -> None:
    now = _utc_ts()
    conn.execute(
        "INSERT INTO users (user_id, created_at, updated_at) VALUES (?, ?, ?);",
        (user_id, now, now),
    )
    conn.commit()


def get_user(conn: sqlite3.Connection, user_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM users WHERE user_id = ?;", (user_id,)).fetchone()
    if row is None:
        return None
    return {
        "user_id": row["user_id"],
        "created_at": int(row["created_at"]),
        "updated_at": int(row["updated_at"]),
    }


def upsert_user_current_state(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    latest_session_id: str,
    latest_step_idx: int,
    mu: Sequence[float],
    sigma: Sequence[Sequence[float]],
    entropy: float,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO user_current_state
          (user_id, latest_session_id, latest_step_idx, mu_json, sigma_json, entropy, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            user_id,
            latest_session_id,
            int(latest_step_idx),
            _dumps(list(mu)),
            _dumps([list(r) for r in sigma]),
            float(entropy),
            _utc_ts(),
        ),
    )
    conn.commit()


def get_user_current_state(conn: sqlite3.Connection, user_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM user_current_state WHERE user_id = ?;", (user_id,)
    ).fetchone()
    if row is None:
        return None
    return {
        "user_id": row["user_id"],
        "latest_session_id": row["latest_session_id"],
        "latest_step_idx": int(row["latest_step_idx"]),
        "mu": list(_loads(row["mu_json"], [])),
        "sigma": list(_loads(row["sigma_json"], [])),
        "entropy": float(row["entropy"]),
        "updated_at": int(row["updated_at"]),
    }


def insert_user_posterior(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    session_id: str,
    session_number: int,
    mu: Sequence[float],
    sigma: Sequence[Sequence[float]],
    entropy: float,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO user_posteriors
          (user_id, session_id, session_number, mu_json, sigma_json, entropy, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            user_id,
            session_id,
            int(session_number),
            _dumps(list(mu)),
            _dumps([list(r) for r in sigma]),
            float(entropy),
            _utc_ts(),
        ),
    )
    conn.commit()


def count_user_posteriors(conn: sqlite3.Connection, user_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM user_posteriors WHERE user_id = ?;", (user_id,)
    ).fetchone()
    return int(row[0]) if row else 0


def get_latest_user_posterior(conn: sqlite3.Connection, user_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT * FROM user_posteriors WHERE user_id = ?
        ORDER BY session_number DESC LIMIT 1;
        """,
        (user_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "user_id": row["user_id"],
        "session_id": row["session_id"],
        "session_number": int(row["session_number"]),
        "mu": list(_loads(row["mu_json"], [])),
        "sigma": list(_loads(row["sigma_json"], [])),
        "entropy": float(row["entropy"]),
        "created_at": int(row["created_at"]),
    }


def insert_posterior_snapshot(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    step_idx: int,
    mu: Sequence[float],
    sigma: Sequence[Sequence[float]],
    entropy: float,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO posterior_snapshots
          (session_id, step_idx, mu_json, sigma_json, entropy, created_at)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            session_id,
            int(step_idx),
            _dumps(list(mu)),
            _dumps([list(r) for r in sigma]),
            float(entropy),
            _utc_ts(),
        ),
    )
    conn.commit()


def list_posterior_snapshots(conn: sqlite3.Connection, session_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT step_idx, mu_json, sigma_json, entropy, created_at
        FROM posterior_snapshots
        WHERE session_id = ?
        ORDER BY step_idx ASC;
        """,
        (session_id,),
    ).fetchall()
    return [
        {
            "step_idx": int(r["step_idx"]),
            "mu": list(_loads(r["mu_json"], [])),
            "sigma": list(_loads(r["sigma_json"], [])),
            "entropy": float(r["entropy"]),
            "created_at": int(r["created_at"]),
        }
        for r in rows
    ]


def list_sessions_for_user(conn: sqlite3.Connection, user_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT session_id, mode, status, step, max_inference_questions, arm,
               prior_session_id, created_at, updated_at
        FROM sessions
        WHERE user_id = ?
        ORDER BY created_at ASC;
        """,
        (user_id,),
    ).fetchall()
    return [
        {
            "session_id": r["session_id"],
            "mode": r["mode"],
            "status": r["status"],
            "step": int(r["step"]),
            "max_inference_questions": int(r["max_inference_questions"]),
            "arm": r["arm"],
            "prior_session_id": r["prior_session_id"],
            "created_at": int(r["created_at"]),
            "updated_at": int(r["updated_at"]),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Phase 2: question_performance_events
# ---------------------------------------------------------------------------

def _qpe_row_to_dict(r) -> Dict[str, Any]:
    return {
        "id": int(r["id"]),
        "question_id": r["question_id"],
        "session_id": r["session_id"],
        "user_id": r["user_id"],
        "step_idx": int(r["step_idx"]),
        "question_source": r["question_source"],
        "parameter_version": (
            int(r["parameter_version"]) if r["parameter_version"] is not None else None
        ),
        "predicted_eig": (
            float(r["predicted_eig"]) if r["predicted_eig"] is not None else None
        ),
        "entropy_before": float(r["entropy_before"]),
        "entropy_after": float(r["entropy_after"]),
        "realized_information_gain": float(r["realized_information_gain"]),
        "response_value": int(r["response_value"]),
        "mu_before": list(_loads(r["mu_before_json"], [])),
        "sigma_before": list(_loads(r["sigma_before_json"], [])),
        "mu_after": list(_loads(r["mu_after_json"], [])),
        "sigma_after": list(_loads(r["sigma_after_json"], [])),
        "created_at": int(r["created_at"]),
    }


def insert_question_performance_event(
    conn: sqlite3.Connection,
    *,
    question_id: str,
    session_id: str,
    user_id: Optional[str],
    step_idx: int,
    question_source: str,
    parameter_version: Optional[int],
    predicted_eig: Optional[float],
    entropy_before: float,
    entropy_after: float,
    realized_information_gain: float,
    response_value: int,
    mu_before: Sequence[float],
    sigma_before: Sequence[Sequence[float]],
    mu_after: Sequence[float],
    sigma_after: Sequence[Sequence[float]],
) -> None:
    conn.execute(
        """
        INSERT INTO question_performance_events (
            question_id, session_id, user_id, step_idx,
            question_source, parameter_version,
            predicted_eig, entropy_before, entropy_after, realized_information_gain,
            response_value,
            mu_before_json, sigma_before_json, mu_after_json, sigma_after_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            question_id,
            session_id,
            user_id,
            int(step_idx),
            question_source,
            parameter_version,
            float(predicted_eig) if predicted_eig is not None else None,
            float(entropy_before),
            float(entropy_after),
            float(realized_information_gain),
            int(response_value),
            _dumps(list(mu_before)),
            _dumps([list(r) for r in sigma_before]),
            _dumps(list(mu_after)),
            _dumps([list(r) for r in sigma_after]),
            _utc_ts(),
        ),
    )
    conn.commit()


def list_question_performance_events(
    conn: sqlite3.Connection, session_id: str
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT * FROM question_performance_events
        WHERE session_id = ?
        ORDER BY step_idx ASC, id ASC;
        """,
        (session_id,),
    ).fetchall()
    return [_qpe_row_to_dict(r) for r in rows]


def list_question_performance_events_for_question(
    conn: sqlite3.Connection, question_id: str
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT * FROM question_performance_events
        WHERE question_id = ?
        ORDER BY created_at ASC, id ASC;
        """,
        (question_id,),
    ).fetchall()
    return [_qpe_row_to_dict(r) for r in rows]

