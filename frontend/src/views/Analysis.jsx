import { useEffect, useState } from 'react';
import { formatBytes } from '../nifti.js';
import { Disclaimer } from '../components/common.jsx';

const SERVER_STAGES = ['Spatial and content validation', 'Preprocessing', 'Ten-pass MC Dropout inference', 'Uncertainty and reliability evidence', 'Rendering results'];

export default function Analysis({ progress, error, onCancel, onRetry, onBack }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (progress.phase === 'idle' || error) return undefined;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(id);
  }, [progress.phase, error]);
  const pct = progress.total > 0 ? Math.round((progress.loaded / progress.total) * 100) : null;
  const uploading = progress.phase === 'uploading';
  const processing = progress.phase === 'processing';
  return (
    <main className="tg-page" id="main">
      <h1>Analysing MRI scan</h1>
      {error ? (
        <div className="tg-alert tg-alert-bad" role="alert">
          <p><strong>Analysis failed.</strong> {error}</p>
          <div className="tg-row">
            <button type="button" className="tg-btn tg-btn-primary" onClick={onRetry}>Retry</button>
            <button type="button" className="tg-btn tg-btn-ghost" onClick={onBack}>Back to upload</button>
          </div>
        </div>
      ) : (
        <section className="tg-card" aria-live="polite">
          <ol className="tg-progress">
            <li className="is-done"><span>Client header check</span><em>done</em></li>
            <li className={uploading ? 'is-active' : 'is-done'}>
              <span>Uploading files</span>
              <em>{uploading ? (pct === null ? 'in progress' : `${pct}% (${formatBytes(progress.loaded)} of ${formatBytes(progress.total)})`) : 'done'}</em>
              {uploading && pct !== null && <progress max="100" value={pct} aria-label="Upload progress" />}
            </li>
            <li className={processing ? 'is-active' : ''}>
              <span>Server pipeline</span>
              <em>{processing ? 'running' : 'waiting'}</em>
              <ul className="tg-fine">
                {SERVER_STAGES.map((s) => <li key={s}>{s}</li>)}
              </ul>
              <p className="tg-fine">The server does not report individual stages, so none is shown as completed until the result arrives.</p>
            </li>
          </ol>
          <p className="tg-fine">Elapsed: {elapsed}s. Duration depends on your hardware.</p>
          <button type="button" className="tg-btn tg-btn-ghost" onClick={onCancel}>Cancel</button>
        </section>
      )}
      <Disclaimer />
    </main>
  );
}
