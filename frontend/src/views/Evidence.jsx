import { Badge, Collapsible, Disclaimer, EmptyState } from '../components/common.jsx';
import {
  STATE_LABELS, STATE_TONE, RECOMMENDATION_LABELS, RECOMMENDATION_TONE, ORIGIN_LABELS, recommendationOf, stateOf,
} from '../evidence.js';

export default function Evidence({ result, caseId, navigate }) {
  const ev = result && result.reliability_evidence_layer;
  if (!result || !ev) {
    return (
      <main className="tg-page" id="main">
        <EmptyState
          title={result ? 'Evidence layer unavailable' : 'No analysis yet'}
          action={<button type="button" className="tg-btn tg-btn-primary" onClick={() => navigate(result ? 'results' : 'upload')}>{result ? 'Back to results' : 'Upload MRI'}</button>}
        >
          {result ? 'The server did not return a reliability evidence report for this result.' : 'Run an analysis to generate a reliability evidence report.'}
        </EmptyState>
      </main>
    );
  }
  const rec = recommendationOf(ev);
  const signals = Array.isArray(ev.signals) ? ev.signals : [];
  const counts = ev.validity_counts || {};
  const prov = ev.provenance || {};
  const th = ev.thresholds || {};
  const iv = result.input_validation;
  return (
    <main className="tg-page" id="main">
      <div className="tg-page-head">
        <div>
          <h1>Reliability evidence layer</h1>
          <p className="tg-fine">Case {caseId}. Explains why this prediction is, or is not, flagged for human review. A workflow recommendation, not a clinical statement.</p>
        </div>
        <button type="button" className="tg-btn tg-btn-ghost" onClick={() => navigate('results')}>Back to results</button>
      </div>

      <section className={`tg-card tg-rec tg-rec-${RECOMMENDATION_TONE[rec]}`} aria-labelledby="rec-h">
        <h2 id="rec-h">Review recommendation</h2>
        <p className="tg-rec-value" data-testid="recommendation">{RECOMMENDATION_LABELS[rec]}</p>
        <code className="tg-code">{rec}</code>
        <h3>Reasons</h3>
        <ul className="tg-list">
          {(ev.reasons || []).map((r, i) => <li key={`${r.signal}-${i}`}>{r.text}{r.signal ? <span className="tg-fine"> ({r.signal})</span> : null}</li>)}
          {(!ev.reasons || ev.reasons.length === 0) && <li>No reasons were provided.</li>}
        </ul>
        <p className="tg-fine">&ldquo;No additional review flagged&rdquo; never means the segmentation is correct or safe.</p>
      </section>

      <section aria-labelledby="sig-h" className="tg-section">
        <h2 id="sig-h">Evidence signals</h2>
        <div className="tg-grid tg-grid-2">
          {signals.map((s) => {
            const st = stateOf(s);
            return (
              <article key={s.id} className="tg-card tg-signal">
                <header>
                  <h3>{s.label}</h3>
                  <Badge tone={STATE_TONE[st]}>{STATE_LABELS[st]}</Badge>
                </header>
                <p className="tg-signal-value">{s.display ?? (Number.isFinite(s.value) ? String(s.value) : 'n/a')}</p>
                {s.triggered && <p className="tg-trigger">Review rule triggered</p>}
                {!s.scientifically_validated && <p className="tg-fine">Not validated as an error predictor.</p>}
                <Collapsible title="Details">
                  <p>{s.explanation}</p>
                  <p className="tg-fine">Origin: {ORIGIN_LABELS[s.origin] || s.origin}</p>
                  {s.trigger_rule && <p className="tg-fine">Rule: {s.trigger_rule}</p>}
                </Collapsible>
              </article>
            );
          })}
        </div>
      </section>

      <div className="tg-grid tg-grid-2">
        <section className="tg-card" aria-labelledby="iv-h">
          <h2 id="iv-h">Input validation</h2>
          {iv ? (
            <ul className="tg-list">
              <li>Volume shape: {Array.isArray(iv.shape) ? iv.shape.join(' × ') : 'n/a'} ({iv.num_slices} slices)</li>
              <li>Modalities: {(iv.modalities || []).join(', ')}</li>
              <li>Spatially consistent: {iv.affine_consistent ? 'yes' : 'no'}</li>
              <li>Non-finite voxel fraction: {Number.isFinite(iv.nonfinite_fraction) ? iv.nonfinite_fraction.toExponential(1) : 'n/a'}</li>
              {(iv.warnings || []).map((w) => <li key={w} className="tg-msg-unchecked">Warning: {w}</li>)}
            </ul>
          ) : <p>No input validation summary was returned.</p>}
        </section>
        <section className="tg-card" aria-labelledby="vs-h">
          <h2 id="vs-h">Validity summary</h2>
          <ul className="tg-list">
            {Object.keys(STATE_LABELS).map((k) => (
              <li key={k}><Badge tone={STATE_TONE[k]}>{STATE_LABELS[k]}</Badge> <strong>{counts[k] ?? 0}</strong></li>
            ))}
          </ul>
        </section>
      </div>

      <Collapsible title="Research limitations" defaultOpen>
        <ul className="tg-list">{(ev.limitations || []).map((l) => <li key={l}>{l}</li>)}</ul>
      </Collapsible>
      <Collapsible title="Thresholds and score breakdown">
        <p>{th.status || 'Thresholds not reported.'}</p>
        {th.verdict_cutoffs && <p className="tg-fine">Verdict cut-offs: {JSON.stringify(th.verdict_cutoffs)}. Source: {th.source}.</p>}
      </Collapsible>
      <Collapsible title="Model and evaluation provenance">
        <dl className="tg-metrics">
          <div><dt>Checkpoint file</dt><dd>{prov.checkpoint_file || 'n/a'}</dd></div>
          <div><dt>SHA-256</dt><dd className="tg-wrap">{prov.checkpoint_sha256 || 'n/a'}</dd></div>
          <div><dt>Checkpoint epoch</dt><dd>{prov.checkpoint_epoch ?? 'n/a'}</dd></div>
          <div><dt>Architecture</dt><dd>{prov.architecture || 'n/a'}</dd></div>
          <div><dt>Familiarity reference patients</dt><dd>{Array.isArray(prov.familiarity_reference_patients) ? prov.familiarity_reference_patients.join(', ') : 'n/a'}</dd></div>
        </dl>
        <p className="tg-fine">{prov.historical_evaluation}</p>
      </Collapsible>
      <Disclaimer>{result.disclaimer}</Disclaimer>
    </main>
  );
}
