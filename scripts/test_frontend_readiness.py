#!/usr/bin/env python3
"""
Frontend-readiness smoke tests.

Covers:
- CORS middleware is configured
- CORS headers are present on responses to allowed origins
- POST /register_user creates a user_id
- POST /start_session with valid user_id succeeds
- POST /start_session with no user_id (anonymous) succeeds
- POST /start_session with nonexistent user_id returns 404
- Invalid answer values (above and below scale) are rejected with 4xx
- Valid answer values are accepted
- Anonymous/unlinked session drives to completion
- Linked-user session drives to completion
"""

from __future__ import annotations

import atexit
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Set a temp DB before importing app.main — DB_PATH is a module-level constant.
_tmp_db_fd, _tmp_db_path = tempfile.mkstemp(suffix=".db")
os.close(_tmp_db_fd)
os.environ["PILOT_DB_PATH"] = _tmp_db_path


@atexit.register
def _cleanup() -> None:
    try:
        os.unlink(_tmp_db_path)
    except OSError:
        pass


from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from app.main import app  # import after env var is set

client = TestClient(app)

_PASS = 0
_FAIL = 0


def _ok(name: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  PASS  {name}")


def _fail(name: str, msg: str) -> None:
    global _FAIL
    _FAIL += 1
    print(f"  FAIL  {name}: {msg}")


def _assert(cond: bool, name: str, msg: str = "") -> None:
    if cond:
        _ok(name)
    else:
        _fail(name, msg or "assertion failed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start_session(user_id: Optional[str] = None, **kwargs) -> dict:
    body = {"mode": "adaptive", "max_inference_questions": 3, "num_heldout": 2, **kwargs}
    if user_id is not None:
        body["user_id"] = user_id
    return client.post("/start_session", json=body)


def _drive_to_completion(session_id: str, first_question: dict) -> None:
    """Answer every question (inference then heldout) with response=3."""
    q = first_question
    while q is not None:
        resp = client.post("/answer", json={
            "session_id": session_id,
            "question_id": q["id"],
            "response": 3,
        })
        assert resp.status_code == 200, f"answer failed: {resp.text}"
        nq = resp.json().get("next_question")
        q = nq


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cors_middleware_configured():
    has_cors = any(getattr(m, "cls", None) is CORSMiddleware for m in app.user_middleware)
    _assert(has_cors, "cors/middleware_in_stack")


def test_cors_headers_on_allowed_origin():
    resp = client.post("/register_user", headers={"Origin": "http://localhost:3000"})
    lower = {k.lower(): v for k, v in resp.headers.items()}
    _assert(
        "access-control-allow-origin" in lower,
        "cors/allow_origin_header_present",
        f"headers: {list(lower.keys())}",
    )
    _assert(
        lower.get("access-control-allow-origin") == "http://localhost:3000",
        "cors/origin_correctly_reflected",
        f"got: {lower.get('access-control-allow-origin')}",
    )


def test_register_user() -> str:
    resp = client.post("/register_user")
    _assert(resp.status_code == 200, "register_user/status_200", f"got {resp.status_code}")
    data = resp.json()
    _assert("user_id" in data, "register_user/user_id_in_response")
    user_id = data["user_id"]
    _assert(isinstance(user_id, str) and len(user_id) > 0, "register_user/user_id_nonempty_str")
    return user_id


def test_start_session_valid_user(user_id: str) -> tuple[str, dict]:
    resp = _start_session(user_id=user_id)
    _assert(resp.status_code == 200, "start_session/valid_user/status_200", f"got {resp.status_code}: {resp.text}")
    data = resp.json()
    _assert("session_id" in data, "start_session/valid_user/has_session_id")
    _assert("first_question" in data, "start_session/valid_user/has_first_question")
    return data["session_id"], data["first_question"]


def test_start_session_anonymous() -> tuple[str, dict]:
    resp = _start_session()  # no user_id
    _assert(resp.status_code == 200, "start_session/anonymous/status_200", f"got {resp.status_code}: {resp.text}")
    data = resp.json()
    _assert("session_id" in data, "start_session/anonymous/has_session_id")
    _assert("first_question" in data, "start_session/anonymous/has_first_question")
    return data["session_id"], data["first_question"]


def test_start_session_nonexistent_user():
    resp = _start_session(user_id="nonexistent_user_id_that_does_not_exist")
    _assert(
        resp.status_code == 404,
        "start_session/nonexistent_user/status_404",
        f"got {resp.status_code}: {resp.text}",
    )


def test_answer_invalid_high(session_id: str, first_question: dict):
    """Response above num_categories is rejected."""
    resp = client.post("/answer", json={
        "session_id": session_id,
        "question_id": first_question["id"],
        "response": 6,  # above 5-point scale
    })
    _assert(
        resp.status_code in (400, 422),
        "answer/invalid_high/rejected",
        f"got {resp.status_code}: {resp.text}",
    )


def test_answer_invalid_low(session_id: str, first_question: dict):
    """Response below 1 is rejected."""
    resp = client.post("/answer", json={
        "session_id": session_id,
        "question_id": first_question["id"],
        "response": 0,
    })
    _assert(
        resp.status_code in (400, 422),
        "answer/invalid_low/rejected",
        f"got {resp.status_code}: {resp.text}",
    )


def test_answer_valid(session_id: str, first_question: dict) -> dict:
    """Valid response=3 is accepted and returns a next_question or completion."""
    resp = client.post("/answer", json={
        "session_id": session_id,
        "question_id": first_question["id"],
        "response": 3,
    })
    _assert(resp.status_code == 200, "answer/valid/status_200", f"got {resp.status_code}: {resp.text}")
    data = resp.json()
    _assert("status" in data, "answer/valid/has_status")
    return data


def test_anonymous_session_completes(session_id: str, first_question: dict):
    """Anonymous session can be driven all the way to complete status."""
    _drive_to_completion(session_id, first_question)
    resp = client.get(f"/session_summary/{session_id}")
    _assert(resp.status_code == 200, "anon_complete/summary_200")
    _assert(resp.json()["status"] == "complete", "anon_complete/status_complete",
            f"got {resp.json().get('status')}")


def test_linked_session_completes(session_id: str, first_question: dict, after_answer: dict):
    """Linked-user session (with one answer already given) drives to complete."""
    # Advance from after_answer's next_question onward.
    q = after_answer.get("next_question")
    if q is not None:
        _drive_to_completion(session_id, q)
    resp = client.get(f"/session_summary/{session_id}")
    _assert(resp.status_code == 200, "linked_complete/summary_200")
    _assert(resp.json()["status"] == "complete", "linked_complete/status_complete",
            f"got {resp.json().get('status')}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Frontend Readiness Tests ===\n")

    # CORS
    test_cors_middleware_configured()
    test_cors_headers_on_allowed_origin()

    # User registration
    user_id = test_register_user()

    # Session creation
    linked_session_id, linked_first_q = test_start_session_valid_user(user_id)
    anon_session_id, anon_first_q = test_start_session_anonymous()
    test_start_session_nonexistent_user()

    # Answer validation — use anonymous session (does not consume linked session's first question).
    # Create a separate session for validation checks so we don't pollute the flow sessions.
    val_resp = _start_session()
    assert val_resp.status_code == 200
    val_session_id = val_resp.json()["session_id"]
    val_first_q = val_resp.json()["first_question"]

    test_answer_invalid_high(val_session_id, val_first_q)
    test_answer_invalid_low(val_session_id, val_first_q)

    # Valid answer on the linked session (saves one step for the completion test below).
    after_answer = test_answer_valid(linked_session_id, linked_first_q)

    # End-to-end flow completion
    test_anonymous_session_completes(anon_session_id, anon_first_q)
    test_linked_session_completes(linked_session_id, linked_first_q, after_answer)

    print(f"\n{'='*40}")
    total = _PASS + _FAIL
    print(f"Results: {_PASS}/{total} passed, {_FAIL} failed")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
