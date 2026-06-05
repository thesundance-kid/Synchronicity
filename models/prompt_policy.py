"""
Prompt policy abstraction for LLM candidate question generation.

A PromptPolicy encapsulates the strategy, template, and conditioning mode used to
generate personality-assessment candidate questions. Policies are versioned in the
prompt_policy_versions table and are the third learning layer of the system:

  Layer 1 — user posterior (what we know about this person)
  Layer 2 — question performance (which questions are informative)
  Layer 3 — prompt/probe-generation policy (which LLM strategies produce useful questions)

Supported strategy_type values:
  generic                                    current default: blind generation from seeds
  uncertainty_targeted                       focuses on highest-variance dimensions
  anti_redundancy                            avoids semantic overlap with recent questions
  tradeoff_scenario                          surfaces competing hypotheses
  contradiction_probe                        probes internal consistency
  projection_probe                           asks about others to reduce self-presentation bias
  private_axis_discovery                     explores undiscovered latent dimensions

Supported conditioning_mode values:
  none                                       only seeds + n_candidates
  posterior_only                             adds current trait estimates
  posterior_plus_history                     adds prior question/answer history
  posterior_plus_unresolved_tensions         adds flagged tensions
  posterior_plus_prior_question_performance  adds per-question EIG/RIG stats
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PromptPolicy:
    """Lightweight in-memory representation of a prompt_policy_versions row."""
    id: int
    name: str
    version: int
    strategy_type: str
    prompt_template: str
    conditioning_mode: str
    output_schema_json: Optional[str]
    active: bool
    created_at: int

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "PromptPolicy":
        return cls(
            id=int(row["id"]),
            name=str(row["name"]),
            version=int(row["version"]),
            strategy_type=str(row["strategy_type"]),
            prompt_template=str(row["prompt_template"]),
            conditioning_mode=str(row["conditioning_mode"]),
            output_schema_json=row.get("output_schema_json"),
            active=bool(row.get("active", True)),
            created_at=int(row.get("created_at", 0)),
        )


# ---------------------------------------------------------------------------
# GENERIC_TEMPLATE — functionally identical to build_generation_prompt().
# Placeholders: {n_candidates}, {seeds_block}.
# Future conditioning placeholders ({posterior_context}, etc.) are present in
# richer templates; render_prompt_policy always provides them so unknown keys
# are silently ignored via defaultdict.
# ---------------------------------------------------------------------------
GENERIC_TEMPLATE = """\
You are helping design a short personality questionnaire.

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


UNCERTAINTY_TARGETED_TEMPLATE = GENERIC_TEMPLATE + """

Prefer questions that distinguish people along high-uncertainty trait combinations.
{uncertainty_context}
"""


PROFILE_CONTRAST_TEMPLATE = GENERIC_TEMPLATE + """

Prefer concrete scenario questions that separate two plausible personality profiles,
especially when a single trait label would be too simplistic.
{posterior_context}
"""


TRADEOFF_SCENARIO_TEMPLATE = GENERIC_TEMPLATE + """

Prefer everyday tradeoff scenarios where different latent profiles would naturally
choose different responses. Avoid obvious trait-name wording.
"""


ANTI_REDUNDANCY_TEMPLATE = GENERIC_TEMPLATE + """

Prioritize semantic novelty. Avoid paraphrases of seed questions and avoid asking
about the same situation with only surface-level wording changes.
"""


# ---------------------------------------------------------------------------
# Context builders — simple, deterministic; no secondary LLM calls.
# ---------------------------------------------------------------------------

def _build_seeds_block(seed_questions: List[dict]) -> str:
    lines = []
    for i, q in enumerate(seed_questions, start=1):
        text = (q.get("text") or "").strip()
        lines.append(f"  {i}. {text}")
    return "\n".join(lines) if lines else "  (none)"


def _build_posterior_context(posterior_summary: Optional[Dict[str, Any]]) -> str:
    if not posterior_summary:
        return ""
    mu = posterior_summary.get("mu") or []
    trait_names = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    lines = ["Current trait estimates (Big Five):"]
    for trait, val in zip(trait_names, mu):
        v = float(val)
        direction = "above average" if v > 0.3 else "below average" if v < -0.3 else "near average"
        lines.append(f"  {trait}: {v:.2f} ({direction})")
    return "\n".join(lines)


def _build_uncertainty_context(uncertainty_summary: Optional[Dict[str, Any]]) -> str:
    if not uncertainty_summary:
        return ""
    entropy = uncertainty_summary.get("entropy")
    variances = uncertainty_summary.get("trait_variances") or []
    trait_names = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    header = f"Current uncertainty — entropy: {float(entropy):.3f}" if entropy is not None else "Current uncertainty:"
    lines = [header]
    for trait, var in zip(trait_names, variances):
        lines.append(f"  {trait} variance: {float(var):.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_prompt_policy(
    policy: PromptPolicy,
    seed_questions: List[dict],
    n_candidates: int,
    *,
    posterior_summary: Optional[Dict[str, Any]] = None,
    uncertainty_summary: Optional[Dict[str, Any]] = None,
    question_history_summary: Optional[Dict[str, Any]] = None,
    answer_history_summary: Optional[Dict[str, Any]] = None,
    unresolved_tensions: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Render the policy's prompt template with the given context.

    Placeholders supported in templates (all optional; missing keys render as ""):
      {n_candidates}             how many candidates to request
      {seeds_block}              numbered list of seed question texts
      {posterior_context}        trait-estimate summary (posterior_only and richer modes)
      {uncertainty_context}      entropy/variance summary (posterior_only and richer modes)
      {question_history_context} prior question history (future)
      {answer_history_context}   prior answer history (future)
      {tensions_context}         unresolved tensions (future)

    Uses defaultdict so templates with unknown or future placeholders degrade gracefully.
    """
    seeds_block = _build_seeds_block(seed_questions)

    mode = policy.conditioning_mode
    posterior_context = ""
    uncertainty_context = ""
    if mode in (
        "posterior_only",
        "posterior_plus_history",
        "posterior_plus_unresolved_tensions",
        "posterior_plus_prior_question_performance",
    ):
        posterior_context = _build_posterior_context(posterior_summary)
        uncertainty_context = _build_uncertainty_context(uncertainty_summary)

    # Future conditioning contexts: placeholder strings ready for future policies.
    question_history_context = ""   # TODO: posterior_plus_history
    answer_history_context = ""     # TODO: posterior_plus_history
    tensions_context = ""           # TODO: posterior_plus_unresolved_tensions

    context: Dict[str, Any] = defaultdict(str, {
        "n_candidates": str(n_candidates),
        "seeds_block": seeds_block,
        "posterior_context": posterior_context,
        "uncertainty_context": uncertainty_context,
        "question_history_context": question_history_context,
        "answer_history_context": answer_history_context,
        "tensions_context": tensions_context,
    })
    return policy.prompt_template.format_map(context)
