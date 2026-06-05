"""
FastAPI app for the real-user pilot backend.

Endpoints:
- POST /start_session
- GET  /next_question/{session_id}
- POST /answer
- GET  /session_summary/{session_id}
- POST /register_user
- GET  /user/{user_id}
- GET  /user/{user_id}/posterior
- GET  /session/{session_id}/posterior_history
"""

from __future__ import annotations

import os
import secrets
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import db
from app.session_manager import (
    Mode,
    create_session,
    get_next_question,
    get_question_num_categories,
    get_session_summary,
    record_answer,
)
from models.prompt_policy import GENERIC_TEMPLATE


def _load_local_env() -> None:
    """
    Load simple KEY=VALUE pairs from the project-root .env file if present.
    Existing shell environment variables take precedence.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v.strip() else default


_load_local_env()

DB_PATH = _env("PILOT_DB_PATH", os.path.join("data", "pilot.db"))
QUESTIONS_PATH = _env("QUESTIONS_PATH", os.path.join("data", "questions_v2.json"))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "5"))
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY", "")   # empty string → DummyLLMClient
LLM_MODEL = _env("LLM_MODEL", "")                   # empty string → DEFAULT_LLM_MODEL

# Comma-separated allowed origins for the React dev server.
# Override via FRONTEND_ORIGINS env var; defaults to common Vite and CRA ports.
_FRONTEND_ORIGINS_RAW = _env("FRONTEND_ORIGINS", "http://localhost:3000,http://localhost:5173")
FRONTEND_ORIGINS = [o.strip() for o in _FRONTEND_ORIGINS_RAW.split(",") if o.strip()]


app = FastAPI(title="Synchronicity Pilot Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    db.seed_question_parameters(conn, QUESTIONS_PATH)
    db.seed_prompt_policies(conn, GENERIC_TEMPLATE)
    db.seed_exploratory_prompt_policies(conn)
    conn.close()


class StartSessionRequest(BaseModel):
    mode: Mode = Field(..., description='Session mode: "adaptive" or "fixed_order"')
    max_inference_questions: int = Field(8, ge=1, le=50)
    num_heldout: int = Field(2, ge=1, le=50)
    fixed_order_ids: Optional[list[str]] = None
    user_id: Optional[str] = None
    session_strategy: str = Field("anchored_exploratory")
    max_anchor_questions: int = Field(2, ge=1, le=8)
    max_generated_probes: int = Field(6, ge=0, le=8)


class QuestionModel(BaseModel):
    id: str
    text: str
    pool: str
    num_categories: int


class StartSessionResponse(BaseModel):
    session_id: str
    first_question: QuestionModel


@app.post("/start_session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    try:
        if req.session_strategy == "anchored_exploratory" and (
            req.max_inference_questions + req.num_heldout > 10
        ):
            raise HTTPException(
                status_code=422,
                detail="anchored_exploratory sessions are capped at 10 total questions",
            )
        # Reject a provided user_id that does not exist — do not silently treat it as anonymous.
        if req.user_id:
            user = db.get_user(conn, req.user_id)
            if user is None:
                raise HTTPException(status_code=404, detail="User not found")
        session_id, first_q = create_session(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            mode=req.mode,
            max_inference_questions=req.max_inference_questions,
            num_heldout=req.num_heldout,
            dim=LATENT_DIM,
            fixed_order_ids=req.fixed_order_ids,
            llm_api_key=ANTHROPIC_API_KEY or None,
            llm_model=LLM_MODEL or None,
            user_id=req.user_id or None,
            session_strategy=req.session_strategy,
            max_anchor_questions=req.max_anchor_questions,
            max_generated_probes=req.max_generated_probes,
        )
        return StartSessionResponse(session_id=session_id, first_question=QuestionModel(**first_q.__dict__))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        conn.close()


@app.get("/next_question/{session_id}")
def next_question(session_id: str):
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    try:
        q = get_next_question(conn, questions_v2_path=QUESTIONS_PATH, session_id=session_id, dim=LATENT_DIM)
        if q is None:
            return {"session_id": session_id, "status": "complete", "next_question": None}
        sess = db.get_session(conn, session_id)
        return {"session_id": session_id, "status": sess.status, "next_question": q.__dict__}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        conn.close()


class AnswerRequest(BaseModel):
    session_id: str
    question_id: str
    # 5-point Likert; further validated per-question against actual num_categories.
    response: int = Field(..., ge=1, le=5)


@app.post("/answer")
def answer(req: AnswerRequest):
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    try:
        # Validate response value against the actual question's num_categories before
        # calling record_answer, so the client gets a clean 422 with a descriptive message.
        num_cat = get_question_num_categories(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            session_id=req.session_id,
            question_id=req.question_id,
            dim=LATENT_DIM,
        )
        if num_cat is not None and (req.response < 1 or req.response > num_cat):
            raise HTTPException(
                status_code=422,
                detail=f"response must be between 1 and {num_cat} (question has {num_cat} categories)",
            )
        return record_answer(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            session_id=req.session_id,
            question_id=req.question_id,
            response=req.response,
            dim=LATENT_DIM,
        )
    except HTTPException:
        raise
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Duplicate answer for this session/question")
    except KeyError as e:
        raise HTTPException(status_code=404, detail=e.args[0] if e.args else "Session or question not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        conn.close()


@app.get("/session_summary/{session_id}")
def session_summary(session_id: str):
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    try:
        return get_session_summary(conn, questions_v2_path=QUESTIONS_PATH, session_id=session_id, dim=LATENT_DIM)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 1: user registration and longitudinal endpoints
# ---------------------------------------------------------------------------

@app.post("/register_user")
def register_user():
    user_id = secrets.token_urlsafe(16)
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    try:
        db.insert_user(conn, user_id=user_id)
        return {"user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        conn.close()


@app.get("/user/{user_id}")
def get_user(user_id: str):
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    try:
        user = db.get_user(conn, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        sessions = db.list_sessions_for_user(conn, user_id)
        return {"user_id": user_id, "created_at": user["created_at"], "sessions": sessions}
    finally:
        conn.close()


@app.get("/user/{user_id}/posterior")
def get_user_posterior(user_id: str):
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    try:
        user = db.get_user(conn, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        row = db.get_latest_user_posterior(conn, user_id)
        if row is None:
            raise HTTPException(status_code=404, detail="No completed sessions for this user")
        return row
    finally:
        conn.close()


@app.get("/session/{session_id}/posterior_history")
def get_posterior_history(session_id: str):
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    try:
        db.get_session(conn, session_id)
        snapshots = db.list_posterior_snapshots(conn, session_id)
        return {"session_id": session_id, "snapshots": snapshots}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        conn.close()
