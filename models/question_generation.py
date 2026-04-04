"""
Minimal provider-agnostic layer for generating Likert-style candidate questions (V2-lite).

Swap in any client that implements ``LLMClient`` (``complete(prompt) -> str``).
"""

from __future__ import annotations

import re
from typing import List, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Small protocol for text completion; any object with ``complete`` works."""

    def complete(self, prompt: str) -> str:
        """Return model text completion for the given prompt."""
        ...


def build_generation_prompt(seed_questions: List[dict], n_candidates: int) -> str:
    """
    Build a prompt that asks for new first-person Likert-style statements.

    Args:
        seed_questions: Each dict should have at least a ``"text"`` key (question wording).
        n_candidates: How many new statements to generate.
    """
    seed_lines = []
    for i, q in enumerate(seed_questions, start=1):
        text = (q.get("text") or "").strip()
        seed_lines.append(f"  {i}. {text}")

    seeds_block = "\n".join(seed_lines) if seed_lines else "  (none)"

    return f"""You are helping design a short personality questionnaire.

Generate exactly {n_candidates} new candidate items. Each item must be:

- A single first-person statement suitable for a Likert scale (e.g. Strongly disagree … Strongly agree)
- One idea per statement only (no double-barreled questions; do not combine two unrelated claims with "and" or "or" in one item)
- Simple, natural, everyday wording
- Clearly different in meaning from every seed question below (no paraphrases or near-duplicates of the seeds)

Seed questions (do not repeat or closely mimic these):
{seeds_block}

Output format:
- Number each line starting with the number, a period, and a space (e.g. "1. ...", "2. ...").
- Output only the numbered list, no other commentary.
"""


def parse_candidate_output(raw_text: str) -> List[dict]:
    """
    Parse model output assumed to be a numbered list into ``[{{"text": "..."}}, ...]``.

    Lines like ``1. I enjoy ...`` or ``2) I prefer ...`` are recognized.
    """
    if not raw_text or not raw_text.strip():
        return []

    lines = raw_text.strip().splitlines()
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
) -> List[dict]:
    """
    Build prompt, call the client, parse numbered output into candidate dicts.

    Args:
        llm_client: Any object implementing ``complete(prompt: str) -> str``.
        seed_questions: Seed items (typically with ``"text"``).
        n_candidates: How many candidates to ask for.

    Returns:
        List of ``{{"text": "..."}}`` dicts (may be fewer if parsing drops lines).
    """
    prompt = build_generation_prompt(seed_questions, n_candidates)
    raw = llm_client.complete(prompt)
    return parse_candidate_output(raw)


class DummyLLMClient:
    """
    Test double: returns a fixed numbered list of plausible Likert-style questions.
    """

    _HARDCODED = """1. I like to spend time alone to recharge.
2. I often notice small changes in my environment.
3. I stay calm when plans change at the last minute.
4. I enjoy learning skills that are completely new to me.
5. I find it easy to empathize with someone I disagree with.
6. I prefer clear instructions over figuring things out on my own.
7. I sometimes put off tasks even when I know I should start them.
8. I feel comfortable speaking up in a group discussion.
9. I value fairness even when it slows a decision down.
10. I tend to reflect on my day before going to sleep."""

    def complete(self, prompt: str) -> str:
        _ = prompt  # API-compatible; dummy ignores prompt
        return self._HARDCODED


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
