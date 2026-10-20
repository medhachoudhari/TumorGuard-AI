// App-level flow with a MOCKED api module (no real server). Real-backend behaviour is covered by e2e/functional_audit.mjs.
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import zlib from 'node:zlib';

const PNG = 'data:image/png;base64,AAAA';
const body = (slice = 3) => ({
  status: 'success', slice_index: slice, num_slices: 10, slice_selection_mode: 'automatic', verdict: 'LOW RELIABILITY', reliability_score: 0.4,
  familiarity_score: 0.1, uncertainty_score: 0.0001, predictive_entropy: 0.02, max_probability_difference: 0.5, mask_pixels: 0,
  images: { mri: PNG, segmentation_overlay: PNG, uncertainty_heatmap: PNG }, reliability_evidence_layer: null, disclaimer: 'd',
});
let calls;
vi.mock('../api.js', async (orig) => {
  const real = await orig();
  return { ...real, checkHealth: vi.fn(async () => ({ online: true, limits: { max_upload_mb: 512 } })), predict: vi.fn() };
});
import { predict } from '../api.js';
import App from '../App.jsx';

function nifti() {
  const b = Buffer.alloc(352); b.writeInt32LE(348, 0); b.writeInt16LE(3, 40); [8, 8, 8].forEach((d, i) => b.writeInt16LE(d, 42 + 2 * i)); b.write('n+1', 344, 'latin1');
  return b;
}
const f = (n) => new File([zlib.gzipSync(nifti())], n);

async function fillAndStart() {
  await userEvent.click(screen.getAllByRole('link', { name: 'Upload' })[0]);
  for (const k of ['flair', 't1ce', 't2']) await userEvent.upload(document.getElementById(`file-${k}`), f(`${k}.nii.gz`));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Start analysis' })).toBeEnabled());
  await userEvent.click(screen.getByRole('button', { name: 'Start analysis' }));
}

beforeEach(() => { calls = 0; window.location.hash = '#/home'; predict.mockReset(); });

describe('App flow (mocked API)', () => {
  it('upload -> results -> history -> view', async () => {
    predict.mockImplementation(async () => { calls += 1; return body(); });
    render(<App />);
    await fillAndStart();
    await screen.findByRole('heading', { name: 'Analysis results' });
    expect(calls).toBe(1);
    expect(screen.getByText('LOW RELIABILITY')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Next slice' }));
    await waitFor(() => expect(predict).toHaveBeenLastCalledWith(expect.objectContaining({ sliceIndex: 4 })));
    await userEvent.click(screen.getAllByRole('link', { name: 'History' })[0]);
    expect(await screen.findAllByRole('row')).toHaveLength(2); // header + one case (re-slice updates the same record)
    await userEvent.click(screen.getByRole('button', { name: /View case/ }));
    await screen.findByRole('heading', { name: 'Analysis results' });
  });
  it('failure shows the error and Retry succeeds', async () => {
    const { ApiError } = await import('../api.js');
    predict.mockRejectedValueOnce(new ApiError('Spatial mismatch from server', { code: 'SPATIAL_MISMATCH', status: 400 })).mockResolvedValueOnce(body());
    render(<App />);
    await fillAndStart();
    expect(await screen.findByRole('alert')).toHaveTextContent('Spatial mismatch from server');
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await screen.findByRole('heading', { name: 'Analysis results' });
  });
  it('cancel returns to upload without an error', async () => {
    predict.mockImplementation(({ signal }) => new Promise((_res, rej) => signal.addEventListener('abort', () => rej(Object.assign(new Error('c'), { code: 'ABORTED' })))));
    render(<App />);
    await fillAndStart();
    await userEvent.click(await screen.findByRole('button', { name: 'Cancel' }));
    await screen.findByRole('heading', { name: 'Upload brain MRI' });
    expect(screen.queryByRole('alert')).toBeNull();
  });
  it('a second submit while busy is ignored (no duplicate request)', async () => {
    let resolve; predict.mockImplementation(() => new Promise((r) => { resolve = r; }));
    render(<App />);
    await fillAndStart();
    await screen.findByRole('button', { name: 'Cancel' });
    expect(predict).toHaveBeenCalledTimes(1);
    resolve(body()); await screen.findByRole('heading', { name: 'Analysis results' });
  });
  it('results without a run show an empty state; direct #/analysis falls back to upload', async () => {
    window.location.hash = '#/results';
    render(<App />);
    expect(await screen.findByText('No analysis yet')).toBeInTheDocument();
    window.location.hash = '#/analysis';
    await screen.findByRole('heading', { name: 'Upload brain MRI' });
    expect(within(document.body).queryByText('Analysing MRI scan')).toBeNull();
  });
  it('offline server disables Start with a reason', async () => {
    const api = await import('../api.js'); api.checkHealth.mockResolvedValueOnce({ online: false, limits: null });
    render(<App />);
    await userEvent.click(screen.getAllByRole('link', { name: 'Upload' })[0]);
    expect(await screen.findByText('Server offline')).toBeInTheDocument();
  });
});
