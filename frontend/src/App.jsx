import { useState, useEffect, useRef } from 'react';
import { Brain, Scissors, Dices, Layers, ShieldCheck } from 'lucide-react';
import './App.css';

const BACKEND = 'http://127.0.0.1:5000';

const MODALITIES = [
  { key: 'flair', label: 'FLAIR', desc: 'Fluid Attenuated Inversion Recovery' },
  { key: 't1ce', label: 'T1CE',  desc: 'T1 Contrast Enhanced' },
  { key: 't2',   label: 'T2',    desc: 'T2 Weighted' },
];

function verdictClass(v) {
  if (!v) return '';
  if (v === 'SAFE') return 'v-safe';
  if (v.includes('NEEDS')) return 'v-attention';
  return 'v-manual';
}

/* ─────────────────────────────────────────────────────────────
   Upload Page
───────────────────────────────────────────────────────────── */
function UploadPage({ onResult }) {
  const [online, setOnline]   = useState(null);
  const [files,  setFiles]    = useState({ flair: null, t1ce: null, t2: null });
  const [error,  setError]    = useState(null);
  const [loading, setLoading] = useState(false);
  const inputRefs = { flair: useRef(), t1ce: useRef(), t2: useRef() };

  // Poll health
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${BACKEND}/health`);
        const d = await r.json();
        setOnline(r.ok && d.model_loaded && d.training_stats_loaded);
      } catch { setOnline(false); }
    };
    check();
    const iv = setInterval(check, 6000);
    return () => clearInterval(iv);
  }, []);

  const allFilled = files.flair && files.t1ce && files.t2;

  const handleFile = (key, e) => {
    const f = e.target.files[0] || null;
    setFiles(p => ({ ...p, [key]: f }));
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!allFilled) { setError('Please select all three MRI modalities.'); return; }
    const ext = f => f.name.endsWith('.nii') || f.name.endsWith('.nii.gz');
    for (const { key } of MODALITIES) {
      if (!ext(files[key])) {
        setError(`${key.toUpperCase()} must be a .nii or .nii.gz file.`);
        return;
      }
    }
    setError(null);
    setLoading(true);
    const fd = new FormData();
    for (const { key } of MODALITIES) fd.append(key, files[key]);
    try {
      const r = await fetch(`${BACKEND}/predict`, { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || 'Inference failed.');
      onResult(d);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-page">
      {/* ── Nav ── */}
      <nav className="page-nav">
        <div className="brand-mark">
          <span className="brand-name">TumorGuard AI</span>
          <span className="brand-sub">Uncertainty-Aware Brain Tumor Segmentation</span>
        </div>
        <div className={`status-pill ${online ? 'online' : 'offline'}`}>
          <span className="dot" />
          {online === null ? 'Connecting…' : online ? 'AI Engine Online' : 'AI Engine Offline'}
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="hero">

        <h1 className="hero-title">
          See the segmentation.<br />
          Know when the AI is uncertain.
        </h1>
        <p className="hero-sub">
          AI-assisted MRI segmentation with uncertainty quantification and familiarity analysis.
        </p>

        {/* Glass upload panel */}
        <div className="upload-panel">
          {error && <div className="upload-error">⚠ {error}</div>}

          <div className="upload-row">
            {MODALITIES.map(({ key, label, desc }) => (
              <label key={key} className={`upload-slot ${files[key] ? 'filled' : ''}`}>
                <input
                  ref={inputRefs[key]}
                  type="file"
                  accept=".nii,.nii.gz"
                  onChange={e => handleFile(key, e)}
                />
                <span className="slot-label">{label}</span>
                <span className="slot-icon">{files[key] ? '✓' : '⬆'}</span>
                <span className={`slot-filename ${files[key] ? 'ready' : ''}`}>
                  {files[key] ? files[key].name : desc}
                </span>
              </label>
            ))}
          </div>

          <button
            className="btn-analyze"
            onClick={handleAnalyze}
            disabled={!allFilled || !online || loading}
          >
            {loading ? 'Analyzing MRI…' : 'Analyze MRI →'}
          </button>

          <p className="upload-hint">Supports NIfTI (.nii / .nii.gz)</p>
        </div>
      </section>

      {/* Loading overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="spinner-ring" />
          <p className="loading-label">Analyzing MRI Study…</p>
          <span className="loading-sub">Running MC Dropout segmentation &amp; reliability analysis</span>
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────
   Results Page
───────────────────────────────────────────────────────────── */
function ResultsPage({ result, onReset }) {
  const [online, setOnline] = useState(true);

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${BACKEND}/health`);
        const d = await r.json();
        setOnline(r.ok && d.model_loaded && d.training_stats_loaded);
      } catch { setOnline(false); }
    };
    check();
    const iv = setInterval(check, 6000);
    return () => clearInterval(iv);
  }, []);

  const vc = verdictClass(result.verdict);

  const metrics = [
    { label: 'Selected Slice',     value: result.slice_index },
    { label: 'Familiarity',        value: result.familiarity_score?.toFixed(4) },
    { label: 'Predictive Entropy', value: result.predictive_entropy?.toFixed(4) },
    { label: 'Reliability Score',  value: result.reliability_score?.toFixed(4) },
  ];

  const flowSteps = [
    { icon: <Brain size={24} strokeWidth={1.5} />, label: 'MRI' },
    { icon: <Scissors size={24} strokeWidth={1.5} />, label: 'Segmentation' },
    { icon: <Dices size={24} strokeWidth={1.5} />, label: 'MC Dropout' },
    { icon: <Layers size={24} strokeWidth={1.5} />, label: 'Familiarity' },
    { icon: <ShieldCheck size={24} strokeWidth={1.5} />, label: 'Reliability' },
  ];

  return (
    <div className="results-page">
      <div className="results-inner">
        {/* ── Nav ── */}
        <nav className="results-nav">
          <div className="results-brand">
            <span className="r-name">TumorGuard AI</span>
            <span className="r-sub">AI Reliability Analysis</span>
          </div>
          <div className="nav-right">
            <div className={`status-pill ${online ? 'online' : 'offline'}`}>
              <span className="dot" />
              {online ? 'AI Engine Online' : 'AI Engine Offline'}
            </div>
            <button className="btn-new" onClick={onReset}>New Analysis</button>
          </div>
        </nav>

        {/* ── Verdict ── */}
        <div className={`verdict-section ${vc}`}>
          <div className="verdict-content">
            <p className="verdict-eyebrow">AI Segmentation Reliability</p>
            <div className="verdict-value">{result.verdict}</div>
            <p className="verdict-caption">
              Reliability combines two signals: how familiar the scan is to the training distribution and how consistent the model's predictions are across stochastic inference passes.
            </p>
            <div className="verdict-equation">
              <span className="eq-term">Model Familiarity</span>
              <span className="eq-op">+</span>
              <span className="eq-term">Prediction Uncertainty</span>
              <span className="eq-op">→</span>
              <span className="eq-result">AI Segmentation Reliability</span>
            </div>
          </div>
          <div className="verdict-score-badge">
            <div className="score-number">{result.reliability_score?.toFixed(2)}</div>
            <div className="score-label">Score</div>
          </div>
        </div>

        {/* ── Metrics ── */}
        <div className="metrics-strip-container">
          <div className="metrics-strip">
            {metrics.map(m => (
              <div key={m.label} className="metric-cell">
                <span className="mc-value">{m.value}</span>
                <span className="mc-label">{m.label}</span>
              </div>
            ))}
          </div>
          <div className="metrics-secondary">
            <span className="mc-label">MC Dropout Uncertainty</span>
            <span className="mc-value-small">{result.uncertainty_score?.toFixed(6)}</span>
          </div>
        </div>

        {/* ── Visualizations ── */}
        <div className="viz-section">
          <div className="section-header">
            <span className="section-eyebrow">MRI Analysis</span>
            <span className="section-meta">Selected slice · {result.slice_index}</span>
            <div className="section-divider" />
          </div>
          <div className="viz-grid">
            <div className="viz-card">
              <div className="viz-header">FLAIR MRI</div>
              <div className="viz-image-wrap">
                <img src={result.images.mri} alt="FLAIR MRI slice" />
              </div>
            </div>
            <div className="viz-card">
              <div className="viz-header">AI Segmentation</div>
              <div className="viz-image-wrap">
                <img src={result.images.segmentation_overlay} alt="AI segmentation overlay" />
              </div>
            </div>
            <div className="viz-card">
              <div className="viz-header">Uncertainty</div>
              <div className="viz-image-wrap">
                <img src={result.images.uncertainty_heatmap} alt="MC Dropout uncertainty map" />
              </div>
              <p className="viz-caption">AI uncertainty visualization — not a biological heat map.</p>
            </div>
          </div>
        </div>

        {/* ── How it works ── */}
        <div className="how-section">
          <div className="section-header">
            <span className="section-eyebrow">How It Works</span>
            <div className="section-divider" />
          </div>
          <div className="how-flow">
            {flowSteps.map((s, i) => (
              <>
                <div key={s.label} className="flow-step">
                  <div className="flow-bubble">{s.icon}</div>
                  <div className="flow-label">{s.label}</div>
                </div>
                {i < flowSteps.length - 1 && (
                  <span key={`arrow-${i}`} className="flow-arrow">→</span>
                )}
              </>
            ))}
          </div>
          <p className="how-desc">
            TumorGuard evaluates whether the segmentation result is sufficiently reliable based on
            model uncertainty and similarity to the training distribution.
          </p>
        </div>

        {/* ── Disclaimer ── */}
        <div className="disclaimer">
          <p className="disclaimer-title">⚠ Medical &amp; Research Disclaimer</p>
          <p className="disclaimer-text">
            {result.disclaimer ||
              'This is a research prototype for AI segmentation reliability visualization. It does NOT diagnose cancer. It does NOT replace radiologist review. The verdict describes model reliability, not patient health status.'}
          </p>
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────
   Root
───────────────────────────────────────────────────────────── */
export default function App() {
  const [result, setResult] = useState(null);
  return result
    ? <ResultsPage result={result} onReset={() => setResult(null)} />
    : <UploadPage  onResult={setResult} />;
}
