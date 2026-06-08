"""
Minimal provider-agnostic layer for generating Likert-style candidate questions (V2-lite).

Swap in any client that implements ``LLMClient`` (``complete(prompt) -> str``).
"""

from __future__ import annotations

import re
import json
from typing import List, Protocol, Tuple, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Model default — override with LLM_MODEL env var.
# Haiku is fast and cheap; sufficient for short question generation tasks.
# ---------------------------------------------------------------------------
DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Safety / quality gate
# ---------------------------------------------------------------------------
_SENSITIVE_TERMS: frozenset = frozenset(
    {
        "suicide",
        "self-harm",
        "self harm",
        "abuse",
        "trauma",
        "religion",
        "religious",
        "race",
        "racial",
        "ethnic",
        "sexual",
        "sexuality",
        "diagnosis",
        "diagnose",
        "politics",
        "political",
        "politician",
    }
)


def validate_candidate_question(q: dict) -> Tuple[bool, str]:
    """
    Return ``(True, "")`` if the candidate passes all quality and safety checks,
    or ``(False, reason)`` on the first failing check.

    Rules (all must pass):
    - text length between 10 and 220 characters (inclusive)
    - text ends with a question mark
    - text does not contain any sensitive / unsafe terms
    - ``w`` key is present, non-empty, and not all zeros
    """
    text = str(q.get("text") or "").strip()

    if len(text) < 10:
        return False, "text too short (< 10 chars)"
    if len(text) > 220:
        return False, "text too long (> 220 chars)"
    if not text.endswith("?"):
        return False, "text does not end with a question mark"

    text_lower = text.lower()
    for term in _SENSITIVE_TERMS:
        if term in text_lower:
            return False, f"contains sensitive term: '{term}'"

    w = q.get("w")
    if w is None:
        return False, "missing loading vector 'w'"
    try:
        w_arr = np.asarray(w, dtype=np.float64).reshape(-1)
    except Exception:
        return False, "loading vector 'w' could not be converted to array"
    if w_arr.size == 0:
        return False, "loading vector 'w' is empty"
    if np.all(w_arr == 0.0):
        return False, "loading vector 'w' is all zeros"

    return True, ""


@runtime_checkable
class LLMClient(Protocol):
    """Small protocol for text completion; any object with ``complete`` works."""

    def complete(self, prompt: str) -> str:
        """Return model text completion for the given prompt."""
        ...


def build_generation_prompt(seed_questions: List[dict], n_candidates: int) -> str:
    """
    Build a prompt that asks for new personality-assessment questions.

    Args:
        seed_questions: Each dict should have at least a ``"text"`` key (question wording).
        n_candidates: How many new questions to generate.
    """
    seed_lines = []
    for i, q in enumerate(seed_questions, start=1):
        text = (q.get("text") or "").strip()
        seed_lines.append(f"  {i}. {text}")

    seeds_block = "\n".join(seed_lines) if seed_lines else "  (none)"

    return f"""You are helping design a short personality questionnaire.

Generate exactly {n_candidates} new candidate questions. Each question must be:

- A single direct question suitable for a Likert scale (e.g. Strongly disagree … Strongly agree)
- End with a question mark
- One idea per question only (no double-barreled questions; do not combine two unrelated claims with "and" or "or")
- Simple, natural, everyday wording
- Clearly different in meaning from every seed question below (no paraphrases or near-duplicates of the seeds)
- Must not mention sensitive topics such as suicide, self-harm, abuse, trauma, religion, race, sexual content, medical diagnosis, or politics

Seed questions (do not repeat or closely mimic these):
{seeds_block}

Output format:
Return only a JSON array. Each item must have:
- "text": the question text
- "intended_contrast": what profiles this item is meant to distinguish
- "expected_response_pattern": how those profiles should differ
- "risk_notes": brief safety/neutrality note
- "suggested_traits": optional list of trait names involved
"""


def parse_candidate_output(raw_text: str) -> List[dict]:
    """
    Parse model output into candidate dicts.

    Structured JSON arrays are preferred. Numbered-list output remains supported
    for deterministic dummy clients and older prompt policies.
    Lines like ``1. I enjoy ...`` or ``2) I prefer ...`` are recognized.
    """
    if not raw_text or not raw_text.strip():
        return []

    raw = raw_text.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
            parsed = parsed["questions"]
        if isinstance(parsed, list):
            items = []
            for row in parsed:
                if isinstance(row, str):
                    text = row.strip()
                    if text:
                        items.append({"text": text})
                elif isinstance(row, dict):
                    text = str(row.get("text") or row.get("question") or "").strip()
                    if text:
                        items.append({
                            "text": text,
                            "intended_contrast": row.get("intended_contrast"),
                            "expected_response_pattern": row.get("expected_response_pattern"),
                            "risk_notes": row.get("risk_notes"),
                            "suggested_traits": row.get("suggested_traits") or row.get("llm_suggested_traits"),
                        })
            return items
    except json.JSONDecodeError:
        pass

    lines = raw.splitlines()
    items: List[dict] = []
    # Match start of line: optional whitespace, digits, then . or ), then space
    pattern = re.compile(r"^\s*(\d+)\s*[\.)]\s*(.+)$")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            body = m.group(2).strip()
            if body:
                items.append({"text": body})
        else:
            # Continuation of previous item (wrapped line)
            if items:
                items[-1]["text"] = (items[-1]["text"] + " " + line).strip()

    return items


def generate_candidate_questions(
    llm_client: LLMClient,
    seed_questions: List[dict],
    n_candidates: int = 10,
    prompt: str | None = None,
) -> Tuple[List[dict], str]:
    """
    Build prompt, call the client, parse numbered output into candidate dicts.

    Args:
        llm_client: Any object implementing ``complete(prompt: str) -> str``.
        seed_questions: Seed items (typically with ``"text"``).
        n_candidates: How many candidates to ask for.
        prompt: Pre-rendered prompt string. If provided, skips ``build_generation_prompt``.
                Used by ``render_prompt_policy`` callers (Phase 5).

    Returns:
        Tuple of (candidates, raw_response_text). ``candidates`` is a list of dicts
        (may be fewer than n_candidates if parsing drops lines). ``raw_response_text``
        is the raw string returned by the LLM before parsing.
    """
    if prompt is None:
        prompt = build_generation_prompt(seed_questions, n_candidates)
    raw = llm_client.complete(prompt)
    return parse_candidate_output(raw), raw


class DummyLLMClient:
    """
    Test double: returns a fixed numbered list of plausible personality questions.
    Used as a fallback when no API key is configured.
    """

    _HARDCODED = """1. Do you like to spend time alone to recharge?
2. Do you often notice small changes in your environment?
3. Do you stay calm when plans change at the last minute?
4. Do you enjoy learning skills that are completely new to you?
5. Do you find it easy to empathize with someone you disagree with?
6. Do you prefer clear instructions over figuring things out on your own?
7. Do you sometimes put off tasks even when you know you should start them?
8. Do you feel comfortable speaking up in a group discussion?
9. Do you value fairness even when it slows a decision down?
10. Do you tend to reflect on your day before going to sleep?"""

    def complete(self, prompt: str) -> str:
        if "JSON array" in prompt:
            return self._HARDCODED
        return "No profile available — running without LLM."


class AnthropicLLMClient:
    """
    Production LLM client backed by the Anthropic Messages API.

    Requires the ``anthropic`` package (``pip install anthropic``) and a valid
    ``api_key``.  The model is configurable; defaults to ``DEFAULT_LLM_MODEL``
    which can be overridden via the ``LLM_MODEL`` environment variable.
    """

    DEFAULT_TIMEOUT = 30.0  # seconds; prevents hanging requests from blocking session creation

    def __init__(self, api_key: str, model: str = DEFAULT_LLM_MODEL, timeout: float = DEFAULT_TIMEOUT) -> None:
        try:
            import anthropic as _anthropic  # local import keeps dependency optional
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicLLMClient. "
                "Install it with: pip install anthropic"
            ) from exc
        self._client = _anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._model = model

    def complete(self, prompt: str) -> str:
        try:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as exc:
            raise RuntimeError(
                f"Anthropic API call failed ({type(exc).__name__}): {exc}"
            ) from exc


def make_llm_client(
    api_key: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """
    Return an ``AnthropicLLMClient`` when *api_key* is provided, otherwise a
    ``DummyLLMClient`` (no network calls, deterministic output).

    Args:
        api_key: Anthropic API key. If ``None`` or empty, falls back to dummy.
        model:   Model ID to pass to ``AnthropicLLMClient``. Defaults to
                 ``DEFAULT_LLM_MODEL`` (``claude-haiku-4-5-20251001``).
    """
    if api_key:
        return AnthropicLLMClient(api_key=api_key, model=model or DEFAULT_LLM_MODEL)
    return DummyLLMClient()


if __name__ == "__main__":
    seeds = [
        {"id": "o_01", "text": "I enjoy exploring new ideas and concepts."},
        {"id": "c_01", "text": "I keep my workspace organized."},
    ]
    n = 10
    built = build_generation_prompt(seeds, n)
    print("=== Built prompt (excerpt) ===")
    print(built[:800] + ("..." if len(built) > 800 else ""))
    print()

    dummy = DummyLLMClient()
    candidates = generate_candidate_questions(dummy, seeds, n_candidates=n)
    print("=== Parsed candidates ===")
    for i, c in enumerate(candidates, start=1):
        print(f"{i}. {c.get('text', '')}")
