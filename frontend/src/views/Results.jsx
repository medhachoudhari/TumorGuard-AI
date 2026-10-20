import { useState } from 'react';
import { verdictClass, verdictLabel, isKnownVerdict } from '../verdict.js';
import { RECOMMENDATION_LABELS, RECOMMENDATION_TONE, recommendationOf } from '../evidence.js';
import { Badge, Collapsible, Disclaimer, EmptyState } from '../components/common.jsx';

const f = (x, d = 4) => (Number.isFinite(x) ? Number(x).toFixed(d) : 'n/a');
const ZOOMS = [1, 1.5, 2, 3];

function Panel({ title, src, alt, filename, children, zoom }) {
  return (
    <figure className="tg-panel">
      <figcaption>{title}</figcaption>
      <div className="tg-panel-view">
        {src ? (
          <img src={src} alt={alt} style={{ width: `${zoom * 100}%` }} />
        ) : (
          <p className="tg-fine" role="status">Image unavailable in this result.</p>
        )}
      </div>
      {children}
      {src && <a className="tg-link" href={src} download={filename}>Download PNG</a>}
    </figure>
  );
}

export default function Results({
  result, caseId, canReslice, resliceReason, reslicing, resliceError,
  onSlice, onAuto, onNew, onReportHtml, onReportJson, navigate,
}) {
  const [zoomIdx, setZoomIdx] = useState(0);
  const [jump, setJump] = useState('');
  if (!result) {
    return (
      <main className="tg-page" id="main">
        <EmptyState title="No analysis yet" action={<button type="button" className="tg-btn tg-btn-primary" onClick={() => navigate('upload')}>Upload MRI</button>}>
          Run an analysis to see the MRI slice, segmentation overlay and uncertainty map here.
        </EmptyState>
      </main>
    );
  }
  const zoom = ZOOMS[zoomIdx];
  const n = Number.isFinite(result.num_slices) ? result.num_slices : null;
  const idx = result.slice_index;
  const prevDisabled = !canReslice || reslicing || idx <= 0;
  const nextDisabled = !canReslice || reslicing || (n !== null && idx >= n - 1);
  const vc = verdictClass(result.verdict);
  const ev = result.reliability_evidence_layer;
  const rec = recommendationOf(ev);
  const bd = result.reliability_breakdown;
  const hm = result.image_info && result.image_info.heatmap_max_variance;
  const submitJump = (e) => {
    e.preventDefault();
    const v = Number.parseInt(jump, 10);
    if (Number.isInteger(v) && v >= 0 && (n === null || v < n)) { onSlice(v); setJump(''); }
  };
  const jumpValid = jump !== '' && Number.isInteger(Number(jump)) && Number(jump) >= 0 && (n === null || Number(jump) < n);

  return (
    <main className="tg-page" id="main" aria-busy={reslicing}>
      <div className="tg-page-head">
        <div>
          <h1>Analysis results</h1>
          <p className="tg-fine">
            Case {caseId} · slice {idx}{n !== null ? ` of ${n - 1}` : ''} · {result.slice_selection_mode || 'unknown'} selection ·
            {' '}{result.prediction_method ? `${result.prediction_method.num_passes}-pass MC Dropout mean, threshold ${result.prediction_method.operator} ${result.prediction_method.threshold}` : 'prediction method not reported'}
          </p>
        </div>
        <div className="tg-row">
          <button type="button" className="tg-btn tg-btn-secondary" onClick={onReportHtml}>Download report (HTML)</button>
          <button type="button" className="tg-btn tg-btn-secondary" onClick={onReportJson}>Download data (JSON)</button>
          <button type="button" className="tg-btn tg-btn-primary" onClick={onNew}>New analysis</button>
        </div>
      </div>

      {resliceError && <div className="tg-alert tg-alert-bad" role="alert">{resliceError}</div>}

      <div className="tg-toolbar" role="group" aria-label="Slice and zoom controls">
        <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" disabled={prevDisabled} title={!canReslice ? resliceReason : undefined} onClick={() => onSlice(idx - 1)}>Previous slice</button>
        <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" disabled={nextDisabled} title={!canReslice ? resliceReason : undefined} onClick={() => onSlice(idx + 1)}>Next slice</button>
        <form onSubmit={submitJump} className="tg-inline-form">
          <label htmlFor="slice-jump" className="tg-fine">Go to slice</label>
          <input id="slice-jump" type="number" inputMode="numeric" min={0} max={n !== null ? n - 1 : undefined} value={jump} disabled={!canReslice || reslicing}
            onChange={(e) => setJump(e.target.value)} placeholder={String(idx)} />
          <button type="submit" className="tg-btn tg-btn-ghost tg-btn-sm" disabled={!canReslice || reslicing || !jumpValid}>Go</button>
        </form>
        <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" disabled={!canReslice || reslicing || result.slice_selection_mode === 'automatic'} onClick={onAuto}>Auto slice</button>
        <span className="tg-sep" aria-hidden="true" />
        <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" disabled={zoomIdx === 0} onClick={() => setZoomIdx((z) => Math.max(0, z - 1))} aria-label="Zoom out">−</button>
        <span className="tg-fine" aria-live="polite">{Math.round(zoom * 100)}%</span>
        <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" disabled={zoomIdx === ZOOMS.length - 1} onClick={() => setZoomIdx((z) => Math.min(ZOOMS.length - 1, z + 1))} aria-label="Zoom in">+</button>
        <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" disabled={zoomIdx === 0} onClick={() => setZoomIdx(0)}>Reset zoom</button>
        {!canReslice && <span className="tg-fine">Slice controls disabled: {resliceReason}</span>}
        {reslicing && <span className="tg-fine" role="status">Re-running inference for the new slice…</span>}
      </div>

      <div className={`tg-panels${reslicing ? ' is-busy' : ''}`}>
        <Panel title="FLAIR MRI slice" src={result.images.mri} alt="FLAIR MRI slice used as model input" filename={`flair_slice_${idx}.png`} zoom={zoom} />
        <Panel title="U-Net segmentation overlay" src={result.images.segmentation_overlay} alt="Predicted tumour mask in red over the FLAIR slice" filename={`segmentation_slice_${idx}.png`} zoom={zoom}>
          <p className="tg-fine">Red: predicted mask ({Number.isFinite(result.mask_pixels) ? `${result.mask_pixels} px` : 'size n/a'}). {result.mask_pixels === 0 ? 'No tumour pixels predicted on this slice.' : ''}</p>
        </Panel>
        <Panel title="MC Dropout uncertainty" src={result.images.uncertainty_heatmap} alt="Predictive variance heatmap over the FLAIR slice" filename={`uncertainty_slice_${idx}.png`} zoom={zoom}>
          <div className="tg-legend" aria-label="Uncertainty colour scale">
            <span>0</span><span className="tg-legend-bar" aria-hidden="true" /><span>{Number.isFinite(hm) ? hm.toExponential(2) : 'max'}</span>
          </div>
          <p className="tg-fine">Variance of the sigmoid probability across passes, scaled per slice (0 to this slice&apos;s maximum).</p>
        </Panel>
      </div>

      <div className="tg-grid tg-grid-2">
        <section className={`tg-card tg-verdict ${vc}`} aria-labelledby="verdict-h">
          <h2 id="verdict-h">Heuristic reliability verdict</h2>
          <p className="tg-verdict-value">{verdictLabel(result.verdict)}</p>
          {!isKnownVerdict(result.verdict) && <p className="tg-fine">The server returned an unrecognised or missing verdict, so this result is flagged for manual review.</p>}
          <p>Reliability score <strong>{f(result.reliability_score)}</strong></p>
          <p className="tg-fine">A weighted heuristic of familiarity, entropy, variance and pass divergence. It is not a calibrated probability and not evidence of clinical safety.</p>
          <Collapsible title="Score details">
            {bd ? (
              <table className="tg-table">
                <thead><tr><th scope="col">Signal</th><th scope="col">Confidence</th><th scope="col">Weight</th><th scope="col">Contribution</th></tr></thead>
                <tbody>
                  {Object.entries(bd).map(([k, v]) => (
                    <tr key={k}><td>{k.replace('_', ' ')}</td><td>{f(v.confidence)}</td><td>{f(v.weight, 2)}</td><td>{f(v.contribution)}</td></tr>
                  ))}
                </tbody>
              </table>
            ) : <p>Breakdown unavailable (invalid or missing inputs).</p>}
            <p className="tg-fine">Cut-offs: HIGH ≥ 0.70, MODERATE ≥ 0.50, LOW ≥ 0.30, otherwise manual review. Experimental.</p>
          </Collapsible>
        </section>

        <section className="tg-card" aria-labelledby="metrics-h">
          <h2 id="metrics-h">Signals</h2>
          <dl className="tg-metrics">
            <div><dt>Familiarity</dt><dd>{f(result.familiarity_score)}</dd></div>
            <div><dt>Uncertainty (MC variance)</dt><dd>{f(result.uncertainty_score, 6)}</dd></div>
            <div><dt>Predictive entropy</dt><dd>{f(result.predictive_entropy)}</dd></div>
            <div><dt>Max pass divergence</dt><dd>{f(result.max_probability_difference)}</dd></div>
          </dl>
          <div className="tg-evidence-summary">
            <Badge tone={RECOMMENDATION_TONE[rec]}>{RECOMMENDATION_LABELS[rec]}</Badge>
            {ev ? <p className="tg-fine">{(ev.reasons && ev.reasons[0] && ev.reasons[0].text) || 'See the evidence layer for details.'}</p>
              : <p className="tg-fine">The evidence layer was not returned by the server.</p>}
            <button type="button" className="tg-btn tg-btn-secondary" onClick={() => navigate('evidence')}>View evidence layer</button>
          </div>
        </section>
      </div>
      <Disclaimer>{result.disclaimer}</Disclaimer>
    </main>
  );
}
