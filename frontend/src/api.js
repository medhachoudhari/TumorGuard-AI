// Client for the existing Flask API (/health, /predict). No simulated results.
export const BACKEND = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_BACKEND_URL) || 'http://127.0.0.1:5000';
export const DEFAULT_MAX_UPLOAD_MB = 512;

export class ApiError extends Error {
  constructor(message, { code = 'ERROR', status = 0 } = {}) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }
}

const PNG_PREFIX = 'data:image/png;base64,';

export async function checkHealth({ baseUrl = BACKEND, signal } = {}) {
  try {
    const res = await fetch(`${baseUrl}/health`, { signal });
    if (!res.ok) return { online: false, limits: null };
    const body = await res.json();
    return {
      online: body.status === 'ok' && body.model_loaded === true,
      limits: body.limits || null,
    };
  } catch {
    return { online: false, limits: null };
  }
}

/** Validate the response and make missing pieces explicit instead of crashing the UI. */
export function normalizeResult(body) {
  if (!body || typeof body !== 'object' || body.status !== 'success') {
    throw new ApiError('The server returned an unexpected response.', { code: 'BAD_RESPONSE' });
  }
  const images = body.images && typeof body.images === 'object' ? body.images : {};
  const pick = (k) => (typeof images[k] === 'string' && images[k].startsWith(PNG_PREFIX) ? images[k] : null);
  return {
    ...body,
    images: {
      mri: pick('mri'),
      segmentation_overlay: pick('segmentation_overlay'),
      uncertainty_heatmap: pick('uncertainty_heatmap'),
    },
    reliability_evidence_layer: body.reliability_evidence_layer && typeof body.reliability_evidence_layer === 'object'
      ? body.reliability_evidence_layer : null,
  };
}

/**
 * POST the three modalities (and an optional explicit slice) to /predict.
 * Uses XMLHttpRequest so that upload progress is real.
 */
export function predict({ files, sliceIndex = null, baseUrl = BACKEND, onUploadProgress, onUploadComplete, signal } = {}) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const k of ['flair', 't1ce', 't2']) {
      if (!files || !files[k]) {
        reject(new ApiError(`Missing ${k.toUpperCase()} file.`, { code: 'MISSING_MODALITY' }));
        return;
      }
      form.append(k, files[k], files[k].name);
    }
    if (Number.isInteger(sliceIndex)) form.append('slice_index', String(sliceIndex));

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${baseUrl}/predict`);
    xhr.upload.onprogress = (e) => { if (onUploadProgress) onUploadProgress({ loaded: e.loaded, total: e.lengthComputable ? e.total : 0 }); };
    xhr.upload.onload = () => { if (onUploadComplete) onUploadComplete(); };
    xhr.onerror = () => reject(new ApiError('Could not reach the analysis server. Is the backend running?', { code: 'NETWORK' }));
    xhr.onabort = () => reject(new ApiError('Analysis cancelled.', { code: 'ABORTED' }));
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { body = null; }
      if (xhr.status !== 200) {
        const msg = body && typeof body.error === 'string' ? body.error : `The server returned an error (HTTP ${xhr.status}).`;
        reject(new ApiError(msg, { code: (body && body.error_code) || 'HTTP_ERROR', status: xhr.status }));
        return;
      }
      try { resolve(normalizeResult(body)); } catch (e) { reject(e); }
    };
    if (signal) {
      if (signal.aborted) { reject(new ApiError('Analysis cancelled.', { code: 'ABORTED' })); return; }
      signal.addEventListener('abort', () => xhr.abort(), { once: true });
    }
    xhr.send(form);
  });
}
