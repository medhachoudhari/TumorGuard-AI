// Client-side NIfTI-1 header check. Reads only the first 348 bytes (decompressing
// .nii.gz as a stream), so no whole-volume load happens in the browser.
export const MODALITIES = [
  { key: 'flair', label: 'FLAIR', desc: 'Fluid Attenuated Inversion Recovery' },
  { key: 't1ce', label: 'T1CE', desc: 'T1-weighted, contrast enhanced' },
  { key: 't2', label: 'T2', desc: 'T2-weighted' },
];

export function hasNiftiExtension(name) {
  const n = String(name || '').toLowerCase();
  return n.endsWith('.nii') || n.endsWith('.nii.gz');
}

export function formatBytes(n) {
  if (!Number.isFinite(n)) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function parseNiftiHeader(bytes) {
  if (!bytes || bytes.length < 348) return { ok: false, error: 'File is too short to be a NIfTI volume.' };
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let little = true;
  if (dv.getInt32(0, true) !== 348) {
    if (dv.getInt32(0, false) === 348) little = false;
    else return { ok: false, error: 'Not a NIfTI-1 file (invalid header).' };
  }
  const ndim = dv.getInt16(40, little);
  if (ndim < 1 || ndim > 7) return { ok: false, error: 'Invalid NIfTI dimensionality.' };
  const dims = [];
  for (let i = 1; i <= ndim; i += 1) dims.push(dv.getInt16(40 + 2 * i, little));
  const magic = String.fromCharCode(bytes[344], bytes[345], bytes[346]);
  if (magic !== 'n+1' && magic !== 'ni1') return { ok: false, error: 'Not a NIfTI-1 file (bad magic).' };
  return { ok: true, ndim, dims };
}

async function readFirstBytes(file, n) {
  const head = new Uint8Array(await file.slice(0, 2).arrayBuffer());
  const gz = head.length === 2 && head[0] === 0x1f && head[1] === 0x8b;
  if (!gz) return new Uint8Array(await file.slice(0, n).arrayBuffer());
  if (typeof DecompressionStream === 'undefined' || typeof file.stream !== 'function') return null;
  const reader = file.stream().pipeThrough(new DecompressionStream('gzip')).getReader();
  const out = new Uint8Array(n);
  let got = 0;
  while (got < n) {
    const { value, done } = await reader.read();
    if (done) break;
    const take = Math.min(value.length, n - got);
    out.set(value.subarray(0, take), got);
    got += take;
  }
  reader.cancel().catch(() => {});
  return got >= n ? out : out.subarray(0, got);
}

/** Inspect one file. state: 'valid' | 'error' | 'unchecked' (header could not be read here). */
export async function inspectFile(file) {
  if (!file) return { state: 'empty' };
  if (!hasNiftiExtension(file.name)) return { state: 'error', message: 'Unsupported file type. Use .nii or .nii.gz.' };
  if (file.size === 0) return { state: 'error', message: 'The file is empty.' };
  try {
    const bytes = await readFirstBytes(file, 348);
    if (!bytes) return { state: 'unchecked', message: 'Header check is unavailable in this browser; the server will validate the file.' };
    const h = parseNiftiHeader(bytes);
    if (!h.ok) return { state: 'error', message: h.error };
    if (h.ndim !== 3) return { state: 'error', message: `Expected a 3D volume, found ${h.ndim} dimensions.` };
    return { state: 'valid', dims: h.dims };
  } catch {
    return { state: 'error', message: 'The file could not be read as NIfTI.' };
  }
}

/** Combine per-file inspections into overall validation. */
export function combineValidation(files, inspections, maxMb) {
  const issues = [];
  const present = MODALITIES.filter((m) => files[m.key]);
  const missing = MODALITIES.filter((m) => !files[m.key]).map((m) => m.label);
  if (missing.length) issues.push(`Missing: ${missing.join(', ')}.`);
  for (const m of present) {
    const ins = inspections[m.key];
    if (ins && ins.state === 'error') issues.push(`${m.label}: ${ins.message}`);
  }
  const total = present.reduce((s, m) => s + files[m.key].size, 0);
  if (maxMb && total > maxMb * 1024 * 1024) issues.push(`Total size ${formatBytes(total)} exceeds the ${maxMb} MB limit.`);
  const dims = present.map((m) => inspections[m.key]).filter((i) => i && i.state === 'valid').map((i) => i.dims.join('x'));
  if (new Set(dims).size > 1) issues.push('Spatial mismatch: the modalities have different volume dimensions.');
  const pending = present.some((m) => !inspections[m.key]);
  const allValid = present.length === 3 && issues.length === 0 && !pending;
  return { issues, canSubmit: allValid, totalBytes: total, pending };
}
