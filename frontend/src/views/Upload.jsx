import { useRef, useState } from 'react';
import { MODALITIES, formatBytes } from '../nifti.js';
import { Disclaimer } from '../components/common.jsx';

function Slot({ modality, file, inspection, onFile, onRemove, disabled }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);
  const inputId = `file-${modality.key}`;
  const take = (list) => { const f = list && list[0]; if (f) onFile(modality.key, f); };
  const state = inspection ? inspection.state : file ? 'checking' : 'empty';
  return (
    <div
      className={`tg-slot is-${state}${drag ? ' is-drag' : ''}`}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); if (!disabled) take(e.dataTransfer.files); }}
    >
      <div>
        <h3><label htmlFor={inputId}>{modality.label}</label></h3>
        <p className="tg-fine">{modality.desc}</p>
      </div>
      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept=".nii,.nii.gz,application/gzip"
        className="tg-visually-hidden"
        disabled={disabled}
        onChange={(e) => { take(e.target.files); e.target.value = ''; }}
      />
      {file ? (
        <div className="tg-slot-file">
          <span className="tg-filename" title={file.name}>{file.name}</span>
          <span className="tg-fine">{formatBytes(file.size)}</span>
          <p className={`tg-slot-msg tg-msg-${state}`} role="status">
            {state === 'checking' && 'Checking header…'}
            {state === 'valid' && `Valid NIfTI volume ${inspection.dims.join(' × ')}`}
            {state === 'unchecked' && inspection.message}
            {state === 'error' && inspection.message}
          </p>
          <div className="tg-row">
            <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" disabled={disabled} onClick={() => inputRef.current?.click()}>Replace</button>
            <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" disabled={disabled} onClick={() => onRemove(modality.key)} aria-label={`Remove ${modality.label} file`}>Remove</button>
          </div>
        </div>
      ) : (
        <div className="tg-slot-empty">
          <p>Drag and drop a <code>.nii</code> or <code>.nii.gz</code> file here</p>
          <button type="button" className="tg-btn tg-btn-secondary" disabled={disabled} onClick={() => inputRef.current?.click()}>
            Choose {modality.label} file
          </button>
        </div>
      )}
    </div>
  );
}

export default function Upload({ files, inspections, validation, maxMb, online, busy, error, onFile, onRemove, onClear, onStart }) {
  const reason = !validation.canSubmit
    ? (validation.pending ? 'Checking files…' : 'Add all three valid volumes to continue.')
    : online === false ? 'The analysis server is offline.' : null;
  const disabled = !validation.canSubmit || online === false || busy;
  return (
    <main className="tg-page" id="main">
      <h1>Upload brain MRI</h1>
      <p className="tg-lead">Provide co-registered 3D NIfTI volumes (<code>.nii</code> / <code>.nii.gz</code>): FLAIR, T1CE and T2. Total upload limit {maxMb} MB.</p>
      {error && <div className="tg-alert tg-alert-bad" role="alert">{error}</div>}
      <div className="tg-grid tg-grid-3">
        {MODALITIES.map((m) => (
          <Slot key={m.key} modality={m} file={files[m.key]} inspection={inspections[m.key]} onFile={onFile} onRemove={onRemove} disabled={busy} />
        ))}
      </div>
      <section className="tg-card" aria-labelledby="val-h">
        <h2 id="val-h">Validation</h2>
        {validation.issues.length === 0 && validation.canSubmit ? (
          <p className="tg-msg-valid" role="status">All checks passed: three NIfTI volumes with matching dimensions, {formatBytes(validation.totalBytes)} in total.</p>
        ) : (
          <ul className="tg-list" role="status">
            {validation.issues.map((i) => <li key={i} className="tg-msg-error">{i}</li>)}
            {validation.issues.length === 0 && <li>{validation.pending ? 'Checking files…' : 'Waiting for files.'}</li>}
          </ul>
        )}
        <p className="tg-fine">Header checks run in your browser. The server repeats them and also verifies alignment, finite values and content.</p>
      </section>
      <div className="tg-row tg-row-end">
        <button type="button" className="tg-btn tg-btn-ghost" onClick={onClear} disabled={busy || !(files.flair || files.t1ce || files.t2)}>Clear all</button>
        <button type="button" className="tg-btn tg-btn-primary tg-btn-lg" disabled={disabled} aria-describedby={reason ? 'start-reason' : undefined} onClick={onStart}>
          Start analysis
        </button>
      </div>
      {reason && <p id="start-reason" className="tg-fine tg-right">{reason}</p>}
      <Disclaimer />
    </main>
  );
}
