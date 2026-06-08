import { useState, useEffect, useCallback } from 'react'
import { registerUser, startSession, submitAnswer, getSessionSummary, getSessionNarrative } from './api'
import QuestionCard from './components/QuestionCard'
import SummaryPage from './components/SummaryPage'

const USER_ID_KEY = 'synchronicity_user_id'
const MAX_INFERENCE = 8
const NUM_HELDOUT = 2
const TOTAL_QUESTIONS = MAX_INFERENCE + NUM_HELDOUT

export default function App() {
  const [phase, setPhase] = useState('init') // init | session | complete | error
  const [sessionId, setSessionId] = useState(null)
  const [question, setQuestion] = useState(null)
  const [questionNumber, setQuestionNumber] = useState(1)
  const [summary, setSummary] = useState(null)
  const [narrative, setNarrative] = useState(null)
  const [narrativeLoading, setNarrativeLoading] = useState(false)
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  const initSession = useCallback(async () => {
    setPhase('init')
    setError(null)
    setSummary(null)
    setNarrative(null)
    setNarrativeLoading(false)
    setQuestionNumber(1)
    try {
      let userId = localStorage.getItem(USER_ID_KEY)
      if (!userId) {
        const data = await registerUser()
        userId = data.user_id
        localStorage.setItem(USER_ID_KEY, userId)
      }
      const data = await startSession(userId)
      setSessionId(data.session_id)
      setQuestion(data.first_question)
      setPhase('session')
    } catch (e) {
      // If stored user_id is invalid (e.g. from old DB), clear it and retry once
      if (e.message.includes('User not found') || e.message.includes('404')) {
        localStorage.removeItem(USER_ID_KEY)
        try {
          const reg = await registerUser()
          localStorage.setItem(USER_ID_KEY, reg.user_id)
          const data = await startSession(reg.user_id)
          setSessionId(data.session_id)
          setQuestion(data.first_question)
          setPhase('session')
          return
        } catch (e2) {
          setError(e2.message)
        }
      } else {
        setError(e.message)
      }
      setPhase('error')
    }
  }, [])

  useEffect(() => {
    initSession()
  }, [initSession])

  async function handleAnswer(response) {
    if (submitting) return
    setSubmitting(true)
    setError(null)
    try {
      const data = await submitAnswer(sessionId, question.id, response)
      if (data.status === 'complete' || data.next_question === null) {
        const summaryData = await getSessionSummary(sessionId)
        setSummary(summaryData)
        setPhase('complete')
        // Fetch narrative in parallel — loading state shown in SummaryPage
        setNarrativeLoading(true)
        getSessionNarrative(sessionId)
          .then((nd) => setNarrative(nd.narrative))
          .catch(() => setNarrative(null))
          .finally(() => setNarrativeLoading(false))
      } else {
        setQuestion(data.next_question)
        setQuestionNumber((n) => n + 1)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  if (phase === 'init') {
    return (
      <div className="app">
        <div className="center-screen">
          <div className="spinner" />
          <p className="loading-text">Starting assessment…</p>
        </div>
      </div>
    )
  }

  if (phase === 'error') {
    return (
      <div className="app">
        <div className="center-screen">
          <p className="error-msg">Could not connect to backend: {error}</p>
          <p className="error-hint">Make sure the backend is running on port 8000.</p>
          <button className="btn-primary" onClick={initSession}>
            Try again
          </button>
        </div>
      </div>
    )
  }

  if (phase === 'complete') {
    return (
      <div className="app">
        <SummaryPage
          summary={summary}
          narrative={narrative}
          narrativeLoading={narrativeLoading}
          onRestart={initSession}
        />
      </div>
    )
  }

  return (
    <div className="app">
      <header className="header">
        <span className="header-title">Synchronicity</span>
        <span className="header-sub">Adaptive Personality Assessment</span>
      </header>
      <main className="main">
        <QuestionCard
          question={question}
          questionNumber={questionNumber}
          totalQuestions={TOTAL_QUESTIONS}
          onAnswer={handleAnswer}
          submitting={submitting}
          error={error}
        />
      </main>
    </div>
  )
}
