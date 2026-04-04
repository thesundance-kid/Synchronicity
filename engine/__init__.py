"""
Engine package — reserved for V2 acquisition extensions.

V1 acquisition (EIG-based greedy selection, Laplace posterior updates) lives in
`models/question_selection.py` and `models/personality_state.py`.

Planned V2 additions:
  - LLM-in-the-loop question synthesis integrated into the live inference pass
  - Non-greedy look-ahead acquisition (e.g. batch EIG, rollout policies)
  - Hierarchical priors for population-level trait correlations
  - Streaming posterior updates for long-horizon longitudinal sessions
"""
