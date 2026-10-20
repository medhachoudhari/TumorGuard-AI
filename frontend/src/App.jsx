import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './App.css';
import Nav from './components/Nav.jsx';
import Landing from './views/Landing.jsx';
import Upload from './views/Upload.jsx';
import Analysis from './views/Analysis.jsx';
import Results from './views/Results.jsx';
import Evidence from './views/Evidence.jsx';
import History from './views/History.jsx';
import Research from './views/Research.jsx';
import About from './views/About.jsx';
import { useRoute } from './router.js';
import { checkHealth, predict, DEFAULT_MAX_UPLOAD_MB } from './api.js';
import { inspectFile, combineValidation } from './nifti.js';
import { makeRecord, newCaseId, loadStored, saveStored } from './history.js';
import { buildReportHTML, buildReportJSON, downloadText } from './report.js';

const EMPTY_FILES = { flair: null, t1ce: null, t2: null };
const IDLE = { phase: 'idle', loaded: 0, total: 0 };

function restoredResult(record) {
  return {
    status: 'success',
    slice_index: record.sliceIndex,
    num_slices: record.numSlices,
    verdict: record.verdict,
    reliability_score: record.reliabilityScore,
    images: { mri: null, segmentation_overlay: null, uncertainty_heatmap: null },
    reliability_evidence_layer: null,
    disclaimer: 'Restored from session history: scores only, no images.',
  };
}

export default function App() {
  const [route, navigate] = useRoute();
  const [online, setOnline] = useState(null);
  const [maxMb, setMaxMb] = useState(DEFAULT_MAX_UPLOAD_MB);
  const [files, setFiles] = useState(EMPTY_FILES);
  const [inspections, setInspections] = useState({});
  const [progress, setProgress] = useState(IDLE);
  const [analysisError, setAnalysisError] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [result, setResult] = useState(null);
  const [caseId, setCaseId] = useState(null);
  const [liveCaseId, setLiveCaseId] = useState(null);
  const [reslicing, setReslicing] = useState(false);
  const [resliceError, setResliceError] = useState(null);
  const [records, setRecords] = useState(() => loadStored());
  const busyRef = useRef(false);
  const abortRef = useRef(null);
  const inspectToken = useRef({});

  // Real server health (also supplies the upload limit).
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      const h = await checkHealth();
      if (!alive) return;
      setOnline(h.online);
      if (h.limits && Number.isFinite(h.limits.max_upload_mb)) setMaxMb(h.limits.max_upload_mb);
    };
    tick();
    const id = setInterval(tick, 6000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  useEffect(() => { saveStored(records); }, [records]);

  const validation = useMemo(() => combineValidation(files, inspections, maxMb), [files, inspections, maxMb]);

  const onFile = useCallback(async (key, file) => {
    setUploadError(null);
    setFiles((f) => ({ ...f, [key]: file }));
    setInspections((i) => { const n = { ...i }; delete n[key]; return n; });
    const token = {};
    inspectToken.current[key] = token;
    const ins = await inspectFile(file);
    if (inspectToken.current[key] === token) setInspections((i) => ({ ...i, [key]: ins }));
  }, []);

  const onRemove = useCallback((key) => {
    inspectToken.current[key] = null;
    setFiles((f) => ({ ...f, [key]: null }));
    setInspections((i) => { const n = { ...i }; delete n[key]; return n; });
  }, []);

  const onClear = useCallback(() => {
    inspectToken.current = {};
    setFiles(EMPTY_FILES);
    setInspections({});
    setUploadError(null);
  }, []);

  const run = useCallback(async (sliceIndex, fresh) => {
    if (busyRef.current) return; // prevent duplicate submissions
    busyRef.current = true;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const id = fresh ? newCaseId() : liveCaseId;
    if (fresh) {
      setAnalysisError(null);
      setUploadError(null);
      setProgress({ phase: 'uploading', loaded: 0, total: 0 });
      navigate('analysis');
    } else {
      setReslicing(true);
      setResliceError(null);
    }
    try {
      const res = await predict({
        files,
        sliceIndex,
        signal: ctrl.signal,
        onUploadProgress: (p) => { if (fresh) setProgress({ phase: 'uploading', ...p }); },
        onUploadComplete: () => { if (fresh) setProgress((p) => ({ ...p, phase: 'processing' })); },
      });
      setResult(res);
      if (fresh) {
        setCaseId(id);
        setLiveCaseId(id);
        setRecords((r) => [makeRecord(id, res), ...r]);
      } else {
        setRecords((r) => r.map((x) => (x.id === id ? makeRecord(id, res, new Date(x.createdAt)) : x)));
      }
      setProgress(IDLE);
      if (fresh) navigate('results');
    } catch (e) {
      setProgress(IDLE);
      if (e.code === 'ABORTED') {
        if (fresh) navigate('upload');
      } else if (fresh) {
        setAnalysisError(e.message);
      } else {
        setResliceError(e.message);
      }
    } finally {
      busyRef.current = false;
      abortRef.current = null;
      setReslicing(false);
    }
  }, [files, liveCaseId, navigate]);

  const onStart = useCallback(() => { run(null, true); }, [run]);
  const onCancel = useCallback(() => { abortRef.current?.abort(); }, []);
  const onSlice = useCallback((i) => { run(i, false); }, [run]);
  const onAuto = useCallback(() => { run(null, false); }, [run]);

  const onNew = useCallback(() => { onClear(); navigate('upload'); }, [onClear, navigate]);

  const onView = useCallback((id) => {
    const rec = records.find((r) => r.id === id);
    if (!rec) return;
    setResult(rec.result || restoredResult(rec));
    setCaseId(rec.id);
    setResliceError(null);
    navigate('results');
  }, [records, navigate]);

  const onDelete = useCallback((id) => { setRecords((r) => r.filter((x) => x.id !== id)); }, []);
  const onClearHistory = useCallback(() => { setRecords([]); }, []);

  const filesLoaded = Boolean(files.flair && files.t1ce && files.t2);
  const isLive = caseId !== null && caseId === liveCaseId;
  const canReslice = isLive && filesLoaded && online !== false;
  const resliceReason = !isLive ? 'the original files are not loaded for this case'
    : !filesLoaded ? 'the original files were cleared' : online === false ? 'the server is offline' : '';

  const busy = progress.phase !== 'idle';
  let view = route;
  if (route === 'analysis' && !busy && !analysisError) view = 'upload';

  return (
    <>
      <a className="tg-skip" href="#main">Skip to content</a>
      <Nav route={route} navigate={navigate} online={online} />
      {view === 'home' && <Landing navigate={navigate} />}
      {view === 'upload' && (
        <Upload files={files} inspections={inspections} validation={validation} maxMb={maxMb} online={online}
          busy={busy} error={uploadError} onFile={onFile} onRemove={onRemove} onClear={onClear} onStart={onStart} />
      )}
      {view === 'analysis' && (
        <Analysis progress={progress} error={analysisError} onCancel={onCancel}
          onRetry={() => { setAnalysisError(null); run(null, true); }}
          onBack={() => { setAnalysisError(null); navigate('upload'); }} />
      )}
      {view === 'results' && (
        <Results result={result} caseId={caseId} canReslice={canReslice} resliceReason={resliceReason}
          reslicing={reslicing} resliceError={resliceError} onSlice={onSlice} onAuto={onAuto} onNew={onNew}
          onReportHtml={() => downloadText(`tumorguard_${caseId}.html`, buildReportHTML(result, caseId), 'text/html')}
          onReportJson={() => downloadText(`tumorguard_${caseId}.json`, `${JSON.stringify(buildReportJSON(result, caseId), null, 2)}\n`, 'application/json')}
          navigate={navigate} />
      )}
      {view === 'evidence' && <Evidence result={result} caseId={caseId} navigate={navigate} />}
      {view === 'history' && <History records={records} onView={onView} onDelete={onDelete} onClear={onClearHistory} navigate={navigate} />}
      {view === 'research' && <Research />}
      {view === 'about' && <About navigate={navigate} />}
      <footer className="tg-footer">
        <p>TumorGuard AI is a research prototype. Not clinically validated; not for diagnosis or treatment decisions.</p>
      </footer>
    </>
  );
}
