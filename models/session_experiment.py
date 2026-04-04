"""
Deterministic experiment arm assignment from ``session_id`` (no external RNG state).
"""

from __future__ import annotations

import hashlib


def assign_experiment_arm(session_id: str) -> str:
    """
    Map ``session_id`` to an experiment arm with an approximate 50/50 split.

    Uses SHA-256 of the UTF-8 session id so the assignment is stable across
    processes (unlike built-in ``hash(str)``, which can vary between runs).

    Returns:
        ``"seed_only"`` or ``"seed_plus_generated"``.
    """
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    # First byte → bit parity gives ~50/50
    if digest[0] & 1:
        return "seed_only"
    return "seed_plus_generated"


def should_use_generated_questions(arm: str) -> bool:
    """Return True if the arm includes LLM-generated pool items."""
    return arm == "seed_plus_generated"
