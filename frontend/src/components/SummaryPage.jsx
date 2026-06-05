const TRAITS = [
  { name: 'Openness', short: 'O', desc: 'Curiosity and openness to new ideas' },
  { name: 'Conscientiousness', short: 'C', desc: 'Organization and self-discipline' },
  { name: 'Extraversion', short: 'E', desc: 'Sociability and positive energy' },
  { name: 'Agreeableness', short: 'A', desc: 'Cooperation and empathy' },
  { name: 'Neuroticism', short: 'N', desc: 'Tendency toward negative affect' },
]

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x))
}

// Map latent value (prior N(0, 4I), σ=2) → 0–100 via sigmoid approximation of Φ(μ/2)
function latentToPercent(mu) {
  return clamp(((Math.tanh(mu * 0.45) + 1) / 2) * 100, 2, 98)
}

function traitSigma(sigmaMatrix, i) {
  if (!sigmaMatrix || !sigmaMatrix[i]) return 2
  return Math.sqrt(Math.max(0, sigmaMatrix[i][i]))
}

function ordinalSuffix(n) {
  const r = n % 100
  if (r >= 11 && r <= 13) return `${n}th`
  const s = ['th', 'st', 'nd', 'rd']
  return `${n}${s[(n % 10 < 4 ? n % 10 : 0)]}`
}

export default function SummaryPage({ summary, onRestart }) {
  const mu = summary?.posterior?.mu ?? []
  const sigma = summary?.posterior?.sigma ?? []
  const responses = summary?.responses ?? []
  const inferenceAnswers = responses.filter((r) => r.pool === 'inference').length

  return (
    <div className="summary">
      <div className="summary-top">
        <h1 className="summary-title">Assessment Complete</h1>
        <p className="summary-sub">
          Your Big Five personality profile, estimated from {inferenceAnswers} adaptive questions.
        </p>
      </div>

      <div className="trait-list">
        {TRAITS.map((trait, i) => {
          const pct = latentToPercent(mu[i] ?? 0)
          const sd = traitSigma(sigma, i)
          // Uncertainty band: ±σ mapped to same scale
          const sdPct = clamp((sd / 4) * 100, 0, 40)
          const bandLeft = clamp(pct - sdPct, 2, 96)
          const bandWidth = clamp(sdPct * 2, 2, 98 - bandLeft)

          return (
            <div key={trait.name} className="trait-row">
              <div className="trait-meta">
                <span className="trait-name">{trait.name}</span>
                <span className="trait-desc">{trait.desc}</span>
              </div>
              <div className="trait-bar-wrap">
                <div className="trait-track">
                  <div className="trait-uncertainty" style={{ left: `${bandLeft}%`, width: `${bandWidth}%` }} />
                  <div className="trait-fill" style={{ width: `${pct}%` }} />
                </div>
                <span className="trait-pct">{ordinalSuffix(Math.round(pct))}</span>
              </div>
            </div>
          )
        })}
      </div>

      <p className="summary-note">
        Percentiles are approximate. Higher is not better — these are descriptive, not evaluative.
      </p>

      <button className="btn-primary" onClick={onRestart}>
        Start new session
      </button>
    </div>
  )
}
