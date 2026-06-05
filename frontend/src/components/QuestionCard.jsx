const LIKERT_LABELS = {
  1: 'Strongly\nDisagree',
  2: 'Disagree',
  3: 'Neutral',
  4: 'Agree',
  5: 'Strongly\nAgree',
}

export default function QuestionCard({
  question,
  questionNumber,
  totalQuestions,
  onAnswer,
  submitting,
  error,
}) {
  const progress = Math.round(((questionNumber - 1) / totalQuestions) * 100)

  return (
    <div className="card">
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>

      <p className="step-label">
        Question {questionNumber} of {totalQuestions}
      </p>

      <p className="question-text">{question.text}</p>

      <div className={`likert-row${submitting ? ' likert-disabled' : ''}`}>
        {Array.from({ length: question.num_categories }, (_, i) => i + 1).map((val) => (
          <button
            key={val}
            className="likert-btn"
            onClick={() => onAnswer(val)}
            disabled={submitting}
            aria-label={`${val} — ${LIKERT_LABELS[val] ?? val}`}
          >
            <span className="likert-num">{val}</span>
            <span className="likert-label">
              {(LIKERT_LABELS[val] ?? String(val)).split('\n').map((line, i) => (
                <span key={i} className="label-line">{line}</span>
              ))}
            </span>
          </button>
        ))}
      </div>

      {submitting && <p className="status-text">Updating…</p>}
      {error && !submitting && <p className="error-msg">{error}</p>}
    </div>
  )
}
