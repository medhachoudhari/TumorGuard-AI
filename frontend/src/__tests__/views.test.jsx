import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Results from '../views/Results.jsx';
import Evidence from '../views/Evidence.jsx';
import History from '../views/History.jsx';
import Research from '../views/Research.jsx';
import Nav from '../components/Nav.jsx';
import Upload from '../views/Upload.jsx';
import Analysis from '../views/Analysis.jsx';
import Landing from '../views/Landing.jsx';
import About from '../views/About.jsx';
import { makeRecord } from '../history.js';

const PNG = 'data:image/png;base64,AAAA';
const evidence = (over = {}) => ({
  recommendation: 'HUMAN_REVIEW_RECOMMENDED',
  reasons: [{ signal: 'reliability_score', text: 'Verdict is MODERATE RELIABILITY' }],
  signals: [
    { id: 'a', label: 'Signal A', display: '0.1', value: 0.1, state: 'AVAILABLE_AND_VALID', origin: 'model_output', triggered: true, trigger_rule: 'rule A', explanation: 'Explain A', scientifically_validated: false },
    { id: 'b', label: 'Signal B', display: null, value: null, state: 'MISSING_OR_UNAVAILABLE', origin: 'heuristic_indicator', triggered: false, explanation: 'Explain B', scientifically_validated: false },
  ],
  validity_counts: { AVAILABLE_AND_VALID: 1, AVAILABLE_BUT_INVALID: 0, MISSING_OR_UNAVAILABLE: 1, NOT_YET_SCIENTIFICALLY_VALIDATED: 0 },
  limitations: ['Limit one'], thresholds: { status: 'experimental', verdict_cutoffs: { HIGH: 0.7 }, source: 'core' },
  provenance: { checkpoint_file: 'm.pt', checkpoint_sha256: 'ab'.repeat(32), checkpoint_epoch: 20, familiarity_reference_patients: ['001'], historical_evaluation: 'hist' },
  ...over,
});
const result = (over = {}) => ({
  status: 'success', slice_index: 5, num_slices: 10, slice_selection_mode: 'automatic', verdict: 'HIGH RELIABILITY',
  reliability_score: 0.8, familiarity_score: 0.5, uncertainty_score: 0.0001, predictive_entropy: 0.02, max_probability_difference: 0.5, mask_pixels: 10,
  prediction_method: { num_passes: 10, threshold: 0.5, operator: '>' }, image_info: { heatmap_max_variance: 0.01 },
  images: { mri: PNG, segmentation_overlay: PNG, uncertainty_heatmap: PNG },
  reliability_breakdown: { familiarity: { confidence: 0.5, weight: 0.4, contribution: 0.2 } },
  reliability_evidence_layer: evidence(), disclaimer: 'Disclaimer text', ...over,
});
const resultsProps = (over = {}) => ({
  result: result(), caseId: 'TG-1', canReslice: true, resliceReason: '', reslicing: false, resliceError: null,
  onSlice: vi.fn(), onAuto: vi.fn(), onNew: vi.fn(), onReportHtml: vi.fn(), onReportJson: vi.fn(), navigate: vi.fn(), ...over,
});

describe('Results controls', () => {
  it('slice buttons call onSlice with neighbouring indices', async () => {
    const p = resultsProps(); render(<Results {...p} />);
    await userEvent.click(screen.getByRole('button', { name: 'Next slice' })); expect(p.onSlice).toHaveBeenLastCalledWith(6);
    await userEvent.click(screen.getByRole('button', { name: 'Previous slice' })); expect(p.onSlice).toHaveBeenLastCalledWith(4);
  });
  it('enforces slice bounds', () => {
    const { rerender } = render(<Results {...resultsProps({ result: result({ slice_index: 0 }) })} />);
    expect(screen.getByRole('button', { name: 'Previous slice' })).toBeDisabled();
    rerender(<Results {...resultsProps({ result: result({ slice_index: 9 }) })} />);
    expect(screen.getByRole('button', { name: 'Next slice' })).toBeDisabled();
  });
  it('jump-to-slice only submits valid indices', async () => {
    const p = resultsProps(); render(<Results {...p} />);
    const go = screen.getByRole('button', { name: 'Go' });
    await userEvent.type(screen.getByLabelText('Go to slice'), '99'); expect(go).toBeDisabled();
    await userEvent.clear(screen.getByLabelText('Go to slice')); await userEvent.type(screen.getByLabelText('Go to slice'), '3');
    await userEvent.click(go); expect(p.onSlice).toHaveBeenCalledWith(3);
  });
  it('disables slice controls with a reason when files are not loaded', () => {
    render(<Results {...resultsProps({ canReslice: false, resliceReason: 'the original files are not loaded for this case' })} />);
    expect(screen.getByRole('button', { name: 'Next slice' })).toBeDisabled();
    expect(screen.getByText(/Slice controls disabled: the original files/)).toBeInTheDocument();
  });
  it('Auto slice is disabled while already automatic and works when explicit', async () => {
    const { rerender } = render(<Results {...resultsProps()} />);
    expect(screen.getByRole('button', { name: 'Auto slice' })).toBeDisabled();
    const p = resultsProps({ result: result({ slice_selection_mode: 'explicit' }) }); rerender(<Results {...p} />);
    await userEvent.click(screen.getByRole('button', { name: 'Auto slice' })); expect(p.onAuto).toHaveBeenCalled();
  });
  it('zoom controls respect limits', async () => {
    render(<Results {...resultsProps()} />);
    expect(screen.getByRole('button', { name: 'Zoom out' })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(screen.getByText('150%')).toBeInTheDocument();
    for (let i = 0; i < 5; i += 1) if (!screen.getByRole('button', { name: 'Zoom in' }).disabled) await userEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(screen.getByRole('button', { name: 'Zoom in' })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: 'Reset zoom' })); expect(screen.getByText('100%')).toBeInTheDocument();
  });
  it('action buttons fire their handlers', async () => {
    const p = resultsProps(); render(<Results {...p} />);
    await userEvent.click(screen.getByRole('button', { name: 'Download report (HTML)' })); expect(p.onReportHtml).toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Download data (JSON)' })); expect(p.onReportJson).toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'New analysis' })); expect(p.onNew).toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'View evidence layer' })); expect(p.navigate).toHaveBeenCalledWith('evidence');
  });
  it('shows an empty state with no result', async () => {
    const p = resultsProps({ result: null }); render(<Results {...p} />);
    await userEvent.click(screen.getByRole('button', { name: 'Upload MRI' })); expect(p.navigate).toHaveBeenCalledWith('upload');
  });
  it('unknown verdict is shown as manual review, never favourable', () => {
    render(<Results {...resultsProps({ result: result({ verdict: 'SAFE' }) })} />);
    expect(screen.getByText('MANUAL REVIEW REQUIRED')).toBeInTheDocument();
    expect(screen.getByText(/unrecognised or missing verdict/)).toBeInTheDocument();
    expect(document.querySelector('.tg-verdict').className).toContain('v-manual');
  });
  it('known verdicts use the matching style', () => {
    const cls = (v) => { const { unmount } = render(<Results {...resultsProps({ result: result({ verdict: v }) })} />); const c = document.querySelector('.tg-verdict').className; unmount(); return c; };
    expect(cls('HIGH RELIABILITY')).toContain('v-safe'); expect(cls('MODERATE RELIABILITY')).toContain('v-attention');
    expect(cls('LOW RELIABILITY')).toContain('v-manual');
  });
  it('missing images and missing evidence layer degrade gracefully', () => {
    render(<Results {...resultsProps({ result: result({ images: { mri: null, segmentation_overlay: null, uncertainty_heatmap: null }, reliability_evidence_layer: null, reliability_breakdown: null, reliability_score: NaN }) })} />);
    expect(screen.getAllByText('Image unavailable in this result.')).toHaveLength(3);
    expect(screen.getByText(/evidence layer was not returned/)).toBeInTheDocument();
    expect(screen.getByText('Assessment unavailable')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Download PNG' })).toBeNull();
  });
  it('shows reslice errors and re-run state', () => {
    render(<Results {...resultsProps({ resliceError: 'Server said no', reslicing: true })} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Server said no');
    expect(screen.getByText(/Re-running inference/)).toBeInTheDocument();
  });
});

describe('Evidence view', () => {
  it('renders recommendation, reasons, signals and expands details', async () => {
    render(<Evidence result={result()} caseId="TG-1" navigate={vi.fn()} />);
    expect(screen.getByTestId('recommendation')).toHaveTextContent('Human review recommended');
    expect(screen.getByText(/Verdict is MODERATE RELIABILITY/)).toBeInTheDocument();
    expect(screen.getByText('Review rule triggered')).toBeInTheDocument();
    const details = document.querySelectorAll('.tg-signal details');
    expect(details).toHaveLength(2);
    await userEvent.click(within(details[0]).getByText('Details')); expect(details[0].open).toBe(true);
    expect(screen.getByText('Explain A')).toBeVisible();
    expect(screen.getByText('Available and valid', { selector: '.tg-signal span' })).toBeInTheDocument();
  });
  it('shows an empty state without a report', async () => {
    const nav = vi.fn(); render(<Evidence result={result({ reliability_evidence_layer: null })} caseId="x" navigate={nav} />);
    await userEvent.click(screen.getByRole('button', { name: 'Back to results' })); expect(nav).toHaveBeenCalledWith('results');
  });
  it('an unknown recommendation is displayed as unavailable, not safe', () => {
    render(<Evidence result={result({ reliability_evidence_layer: evidence({ recommendation: 'SAFE' }) })} caseId="x" navigate={vi.fn()} />);
    expect(screen.getByTestId('recommendation')).toHaveTextContent('Assessment unavailable');
  });
});

describe('History view', () => {
  const recs = Array.from({ length: 7 }, (_, i) => makeRecord(`TG-${i}`, result({ slice_index: i }), new Date(2026, 0, i + 1)));
  it('paginates, searches, views, and confirms deletion', async () => {
    const onView = vi.fn(); const onDelete = vi.fn(); const onClear = vi.fn();
    render(<History records={recs} onView={onView} onDelete={onDelete} onClear={onClear} navigate={vi.fn()} />);
    expect(document.querySelectorAll('tbody tr')).toHaveLength(5);
    await userEvent.click(screen.getByRole('button', { name: 'Next page' })); expect(document.querySelectorAll('tbody tr')).toHaveLength(2);
    await userEvent.click(screen.getByRole('button', { name: 'Previous page' }));
    await userEvent.type(screen.getByLabelText('Search'), 'TG-3'); expect(document.querySelectorAll('tbody tr')).toHaveLength(1);
    await userEvent.click(screen.getByRole('button', { name: 'View case TG-3' })); expect(onView).toHaveBeenCalledWith('TG-3');
    await userEvent.click(screen.getByRole('button', { name: 'Delete case TG-3' }));
    expect(onDelete).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' })); expect(onDelete).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Delete case TG-3' }));
    await userEvent.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: 'Delete' })); expect(onDelete).toHaveBeenCalledWith('TG-3');
    await userEvent.click(screen.getByRole('button', { name: 'Clear history' }));
    await userEvent.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: 'Clear history' })); expect(onClear).toHaveBeenCalled();
  });
  it('honest empty state and session-only notice', () => {
    render(<History records={[]} onView={vi.fn()} onDelete={vi.fn()} onClear={vi.fn()} navigate={vi.fn()} />);
    expect(screen.getByText('No analyses in this session')).toBeInTheDocument();
    expect(screen.getByText(/Nothing is saved on a server/)).toBeInTheDocument();
  });
});

describe('Nav, Landing, About, Research', () => {
  it('nav links navigate and the mobile toggle opens/closes', async () => {
    const nav = vi.fn(); render(<Nav route="home" navigate={nav} online />);
    for (const [label, r] of [['Home', 'home'], ['Upload', 'upload'], ['Results', 'results'], ['History', 'history'], ['Research', 'research'], ['About', 'about']]) {
      await userEvent.click(screen.getByRole('link', { name: label })); expect(nav).toHaveBeenLastCalledWith(r);
    }
    expect(screen.getByRole('link', { name: 'Home' })).toHaveAttribute('aria-current', 'page');
    await userEvent.click(screen.getByRole('button', { name: 'New analysis' })); expect(nav).toHaveBeenLastCalledWith('upload');
    const t = screen.getByRole('button', { name: 'Open menu' });
    await userEvent.click(t); expect(screen.getByRole('button', { name: 'Close menu' })).toHaveAttribute('aria-expanded', 'true');
    await userEvent.click(screen.getByRole('link', { name: 'About' })); expect(screen.getByRole('button', { name: 'Open menu' })).toHaveAttribute('aria-expanded', 'false');
  });
  it('status pill reflects server state', () => {
    const { rerender } = render(<Nav route="home" navigate={vi.fn()} online={null} />);
    expect(screen.getByText('Checking server…')).toBeInTheDocument();
    rerender(<Nav route="home" navigate={vi.fn()} online={false} />); expect(screen.getByText('Server offline')).toBeInTheDocument();
  });
  it('landing buttons work and statistics are labelled historical', async () => {
    const nav = vi.fn(); const scroll = vi.fn(); Element.prototype.scrollIntoView = scroll;
    render(<Landing navigate={nav} />);
    await userEvent.click(screen.getByRole('button', { name: 'Get started' })); expect(nav).toHaveBeenCalledWith('upload');
    await userEvent.click(screen.getByRole('button', { name: 'Learn more' })); expect(scroll).toHaveBeenCalled();
    expect(screen.getByText(/not independently reproduced/)).toBeInTheDocument();
  });
  it('about buttons navigate and the GitHub link is real', async () => {
    const nav = vi.fn(); render(<About navigate={nav} />);
    await userEvent.click(screen.getByRole('button', { name: 'Research and validation' })); expect(nav).toHaveBeenCalledWith('research');
    await userEvent.click(screen.getByRole('button', { name: 'Start an analysis' })); expect(nav).toHaveBeenCalledWith('upload');
    expect(screen.getByRole('link', { name: 'GitHub' })).toHaveAttribute('href', 'https://github.com/medhachoudhari/TumorGuard-AI');
  });
  it('research tabs switch with click and arrow keys', async () => {
    render(<Research />);
    for (const [name, h] of [['Methodology', /Methodology/], ['Limitations', /Known limitations/], ['Citations', /References/]]) {
      await userEvent.click(screen.getByRole('tab', { name })); expect(screen.getByRole('heading', { name: h })).toBeInTheDocument();
    }
    await userEvent.click(screen.getByRole('tab', { name: 'Model performance' })); screen.getByRole('tab', { name: 'Model performance' }).focus();
    await userEvent.keyboard('{ArrowRight}'); expect(screen.getByRole('tab', { name: 'Methodology' })).toHaveAttribute('aria-selected', 'true');
    for (const a of (await userEvent.click(screen.getByRole('tab', { name: 'Citations' })), screen.getAllByRole('link'))) {
      expect(a.getAttribute('href')).toMatch(/^https:\/\//); expect(a).toHaveAttribute('rel', expect.stringContaining('noopener'));
    }
  });
});

describe('Upload and Analysis views', () => {
  const base = () => ({ files: { flair: null, t1ce: null, t2: null }, inspections: {}, validation: { issues: [], canSubmit: false, pending: false, totalBytes: 0 }, maxMb: 512, online: true, busy: false, error: null,
    onFile: vi.fn(), onRemove: vi.fn(), onClear: vi.fn(), onStart: vi.fn() });
  it('Start is disabled with a reason until validation passes', async () => {
    const p = base(); const { rerender } = render(<Upload {...p} />);
    const start = screen.getByRole('button', { name: 'Start analysis' }); expect(start).toBeDisabled();
    expect(screen.getByText('Add all three valid volumes to continue.')).toBeInTheDocument();
    rerender(<Upload {...p} validation={{ issues: [], canSubmit: true, pending: false, totalBytes: 10 }} />);
    await userEvent.click(screen.getByRole('button', { name: 'Start analysis' })); expect(p.onStart).toHaveBeenCalled();
    rerender(<Upload {...p} online={false} validation={{ issues: [], canSubmit: true, pending: false, totalBytes: 10 }} />);
    expect(screen.getByRole('button', { name: 'Start analysis' })).toBeDisabled(); expect(screen.getByText('The analysis server is offline.')).toBeInTheDocument();
  });
  it('file selection, drop, replace and remove call the right handlers', async () => {
    const p = base(); p.files.flair = new File(['x'], 'f.nii.gz'); p.inspections.flair = { state: 'valid', dims: [1, 2, 3] };
    render(<Upload {...p} />);
    const f = new File(['y'], 't1.nii.gz');
    await userEvent.upload(document.getElementById('file-t1ce'), f); expect(p.onFile).toHaveBeenCalledWith('t1ce', f);
    const drop = new File(['z'], 't2.nii.gz'); const slot = document.getElementById('file-t2').closest('.tg-slot');
    const ev = new Event('drop', { bubbles: true, cancelable: true }); ev.dataTransfer = { files: [drop] }; slot.dispatchEvent(ev);
    expect(p.onFile).toHaveBeenCalledWith('t2', drop);
    await userEvent.click(screen.getByRole('button', { name: 'Remove FLAIR file' })); expect(p.onRemove).toHaveBeenCalledWith('flair');
    expect(screen.getByRole('button', { name: 'Replace' })).toBeEnabled();
    await userEvent.click(screen.getByRole('button', { name: 'Clear all' })); expect(p.onClear).toHaveBeenCalled();
    expect(screen.getByText(/Valid NIfTI volume 1 × 2 × 3/)).toBeInTheDocument();
  });
  it('disables controls while busy', () => {
    render(<Upload {...base()} busy />); expect(screen.getByRole('button', { name: 'Choose FLAIR file' })).toBeDisabled();
  });
  it('analysis view shows real upload progress, cancel, error and retry', async () => {
    const cancel = vi.fn(); const retry = vi.fn(); const back = vi.fn();
    const { rerender } = render(<Analysis progress={{ phase: 'uploading', loaded: 50, total: 100 }} error={null} onCancel={cancel} onRetry={retry} onBack={back} />);
    expect(screen.getByLabelText('Upload progress')).toHaveAttribute('value', '50');
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' })); expect(cancel).toHaveBeenCalled();
    rerender(<Analysis progress={{ phase: 'processing', loaded: 100, total: 100 }} error={null} onCancel={cancel} onRetry={retry} onBack={back} />);
    expect(screen.getByText(/does not report individual stages/)).toBeInTheDocument();
    rerender(<Analysis progress={{ phase: 'idle', loaded: 0, total: 0 }} error="Boom" onCancel={cancel} onRetry={retry} onBack={back} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Boom');
    await userEvent.click(screen.getByRole('button', { name: 'Retry' })); expect(retry).toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Back to upload' })); expect(back).toHaveBeenCalled();
  });
});
