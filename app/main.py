"""
FastAPI app for the real-user pilot backend.

Endpoints:
- POST /start_session
- GET  /next_question/{session_id}
- POST /answer
- GET  /session_summary/{session_id}
"""

from __future__ import annotations

import os
import secrets
import sqlite3
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import db
from app.session_manager import Mode, create_session, get_next_question, get_session_summary, record_answer


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v.strip() else default


DB_PATH = _env("PILOT_DB_PATH", os.path.join("data", "pilot.db"))
QUESTIONS_PATH = _env("QUESTIONS_PATH", os.path.join("data", "questions_v2.json"))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "5"))
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY", "")   # empty string → DummyLLMClient
LLM_MODEL = _env("LLM_MODEL", "")                   # empty string → DEFAULT_LLM_MODEL


app = FastAPI(title="Synchronicity Pilot Backend", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    conn.close()


class StartSessionRequest(BaseModel):
    mode: Mode = Field(..., description='Session mode: "adaptive" or "fixed_order"')
    max_inference_questions: int = Field(5, ge=1, le=50)
    num_heldout: int = Field(5, ge=1, le=50)
    fixed_order_ids: Optional[list[str]] = None
    user_id: Optional[str] = None


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
        )
        return StartSessionResponse(session_id=session_id, first_question=QuestionModel(**first_q.__dict__))
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
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        conn.close()


class AnswerRequest(BaseModel):
    session_id: str
    question_id: str
    response: int = Field(..., ge=1, le=10)


@app.post("/answer")
def answer(req: AnswerRequest):
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    try:
        return record_answer(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            session_id=req.session_id,
            question_id=req.question_id,
            response=req.response,
            dim=LATENT_DIM,
        )
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail="Duplicate answer for this session/question") from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
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
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
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
        # Verify session exists before fetching snapshots.
        db.get_session(conn, session_id)
        snapshots = db.list_posterior_snapshots(conn, session_id)
        return {"session_id": session_id, "snapshots": snapshots}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        conn.close()

