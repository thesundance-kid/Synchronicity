const BASE_URL = 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export function registerUser() {
  return request('/register_user', { method: 'POST' })
}

export function startSession(userId) {
  return request('/start_session', {
    method: 'POST',
    body: JSON.stringify({
      mode: 'adaptive',
      max_inference_questions: 8,
      num_heldout: 2,
      session_strategy: 'anchored_exploratory',
      max_anchor_questions: 2,
      max_generated_probes: 6,
      user_id: userId,
    }),
  })
}

export function submitAnswer(sessionId, questionId, response) {
  return request('/answer', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, question_id: questionId, response }),
  })
}

export function getSessionSummary(sessionId) {
  return request(`/session_summary/${sessionId}`)
}

export function getSessionNarrative(sessionId) {
  return request(`/session/${sessionId}/narrative`)
}
