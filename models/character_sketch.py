"""
Generate a natural-language behavioral portrait (character sketch) from a
PersonalityState + response history.  Used internally to condition per-step
question generation — never shown directly to users.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from models.personality_state import PersonalityState
    from models.question_generation import LLMClient

# ---------------------------------------------------------------------------
# Trait descriptor tables
# ---------------------------------------------------------------------------
_TRAIT_NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

_TRAIT_HIGH_DESC = [
    "drawn to novel ideas and abstract thinking",
    "organized, deliberate, follows through",
    "energized by social engagement",
    "prioritizes harmony, cooperative, trusting",
    "emotionally reactive, sensitive to stress",
]

_TRAIT_LOW_DESC = [
    "prefers the concrete, familiar, and proven",
    "flexible, spontaneous, resists structure",
    "internally oriented, prefers depth over breadth",
    "direct, skeptical, prioritizes truth over comfort",
    "emotionally stable, calm under pressure",
]


def _trait_descriptor(mu_val: float, i: int) -> str:
    if mu_val > 1.0:
        return _TRAIT_HIGH_DESC[i]
    if mu_val < -1.0:
        return _TRAIT_LOW_DESC[i]
    if mu_val > 0.3:
        return "leaning toward: " + _TRAIT_HIGH_DESC[i]
    if mu_val < -0.3:
        return "leaning toward: " + _TRAIT_LOW_DESC[i]
    return "mixed or ambiguous signal so far"


def _is_effectively_prior(state: "PersonalityState") -> bool:
    """True when the state has not been updated from the flat prior N(0, 4I)."""
    mu_arr = np.asarray(state.mu, dtype=np.float64)
    sigma_arr = np.asarray(state.sigma, dtype=np.float64)
    flat_prior = 4.0 * np.eye(mu_arr.shape[0])
    return bool(
        np.allclose(mu_arr, 0.0, atol=1e-6) and np.allclose(sigma_arr, flat_prior, atol=1e-6)
    )


def _build_sketch_prompt(
    state: "PersonalityState",
    response_history: List[Dict[str, Any]],
    is_prior: bool,
) -> str:
    if is_prior:
        return (
            "You are helping design a personality assessment. We are about to start a new "
            "session with no prior information about this person. Generate a single concise "
            "paragraph (3-5 sentences) describing a completely generic respondent — someone "
            "whose personality is entirely unknown. Do not assume any trait leans high or low. "
            "Write in third person, using behavioral language only (no Big Five labels). "
            "This will be used internally to seed question generation.\n\n"
            "Output only the paragraph. No preamble, no title."
        )

    mu = np.asarray(state.mu, dtype=np.float64)
    sigma = np.asarray(state.sigma, dtype=np.float64)
    variances = np.diag(sigma)

    descriptor_lines = []
    for i, name in enumerate(_TRAIT_NAMES):
        desc = _trait_descriptor(float(mu[i]), i)
        descriptor_lines.append(f"  - {name}: {desc}")
    descriptors_block = "\n".join(descriptor_lines)

    uncertain_idx = list(np.argsort(-variances))
    uncertain_names = [
        _TRAIT_NAMES[i] for i in uncertain_idx if float(variances[i]) > 2.0
    ][:2]
    if uncertain_names:
        uncertainty_line = f"Most uncertain (highest variance): {', '.join(uncertain_names)}."
    else:
        uncertainty_line = "Posterior is reasonably converged across all traits."

    if not response_history:
        pattern_line = "No responses recorded yet."
    else:
        n = len(response_history)
        extreme = [r for r in response_history if r["response"] in (1, 5)]
        neutral = [r for r in response_history if r["response"] == 3]
        if extreme:
            pattern_line = (
                f"{len(extreme)} of {n} responses were strong (1 or 5); "
                f"{len(neutral)} were neutral (3)."
            )
        else:
            pattern_line = (
                f"Responses have been moderate; {len(neutral)} of {n} were neutral (3)."
            )

    return f"""You are helping design a personality assessment. Below is what we currently know about a respondent based on their answers so far.

CURRENT UNDERSTANDING:
{descriptors_block}

UNCERTAINTY:
{uncertainty_line}

RESPONSE PATTERN:
{pattern_line}

Your task: write a single concise paragraph (3-5 sentences) describing this person's behavioral tendencies based on the above. Write in third person. Use behavioral and situational language only — no Big Five trait names, no clinical terms, no jargon. Capture what is known confidently and note what remains unclear.

Output only the paragraph. No preamble, no title."""


def generate_character_sketch(
    state: "PersonalityState",
    response_history: List[Dict[str, Any]],
    llm_client: "LLMClient",
) -> str:
    """
    Generate an internal behavioral portrait from the current posterior.

    Args:
        state:            Current PersonalityState (mu, sigma over 5 traits).
        response_history: List of dicts with keys: question_id, text, response (int), step.
        llm_client:       Any object implementing complete(prompt) -> str.

    Returns:
        A plain-text paragraph suitable for insertion into question-generation prompts.
        Falls back to a safe default string on any exception.
    """
    try:
        prior = _is_effectively_prior(state)
        prompt = _build_sketch_prompt(state, response_history, is_prior=prior)
        result = llm_client.complete(prompt)
        text = (result or "").strip()
        return text if text else "No profile available — running without LLM."
    except Exception:
        return "No profile available — running without LLM."
