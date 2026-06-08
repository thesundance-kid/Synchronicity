"""
Generate a user-facing narrative portrait from a completed session.

Called once on first GET /session/{session_id}/narrative and cached in the DB.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.personality_state import PersonalityState
    from models.question_generation import LLMClient


def _build_narrative_prompt(
    response_history: List[Dict[str, Any]],
    character_sketch: Optional[str],
) -> str:
    if response_history:
        qa_lines = []
        for i, r in enumerate(response_history, start=1):
            text = r.get("text") or r.get("question_id", "?")
            val = r.get("response", "?")
            qa_lines.append(f"{i}. {text} → {val}/5")
        qa_block = "\n".join(qa_lines)
    else:
        qa_block = "(No responses recorded.)"

    sketch_block = (character_sketch or "Not available.").strip()

    return f"""You have just observed someone complete a personality assessment session. Below is everything you have about them.

QUESTIONS ASKED AND RESPONSES (1=strongly disagree, 5=strongly agree):
{qa_block}

INTERNAL CHARACTER SKETCH (generated during the session):
{sketch_block}

Your task: write a 3-4 paragraph portrait of this person for them to read at the end of their session.

Rules:
- Paragraph 1: their overall behavioral style and how they tend to show up in the world.
- Paragraph 2: their relationship with structure, goals, and how they handle uncertainty or change.
- Paragraph 3: their social tendencies and emotional texture — how they typically relate to others and to stress.
- Paragraph 4 (optional but preferred): a nuanced observation about tension or complexity in their profile — something that doesn't fit neatly into paragraphs 1-3.
- Tone: warm, direct, specific. Write as though you are a thoughtful observer describing a real person, not producing a generic horoscope.
- Do NOT use Big Five trait labels (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) anywhere in the output.
- Do NOT use clinical or psychological jargon.
- Do NOT start with "You are someone who..." or any preamble. Begin directly with a substantive observation.
- Write in second person ("you", "your").
- Output only the portrait paragraphs. No title, no preamble, no closing remarks."""


def generate_narrative(
    response_history: List[Dict[str, Any]],
    character_sketch: Optional[str],
    llm_client: "LLMClient",
) -> str:
    """
    Generate a user-facing narrative portrait.

    Args:
        response_history: List of dicts with keys: question_id, text, response (int), step.
        character_sketch:  The latest internal character sketch (may be None).
        llm_client:        Any object implementing complete(prompt) -> str.

    Returns:
        Plain-text narrative (3-4 paragraphs). Falls back gracefully on exception.
    """
    try:
        prompt = _build_narrative_prompt(response_history, character_sketch)
        result = llm_client.complete(prompt)
        text = (result or "").strip()
        return text if text else _fallback_narrative()
    except Exception:
        return _fallback_narrative()


def _fallback_narrative() -> str:
    return (
        "Your responses across this session reveal a nuanced picture that resists easy "
        "categorization. The patterns suggest someone who approaches situations with their "
        "own internal logic — sometimes predictable, sometimes surprising even to themselves.\n\n"
        "How you handle structure and uncertainty came through clearly in several answers: "
        "there is a preference, but also flexibility when the situation calls for it.\n\n"
        "Your relationship with others and with your own emotional landscape appears thoughtful. "
        "You seem to move between engagement and reflection depending on context.\n\n"
        "The full portrait would benefit from more data — this session captured the outline, "
        "but the deeper texture emerges over time and across situations."
    )
