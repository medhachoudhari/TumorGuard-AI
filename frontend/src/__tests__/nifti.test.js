// @vitest-environment node
import { describe, it, expect } from 'vitest';
import zlib from 'node:zlib';
import { parseNiftiHeader, inspectFile, combineValidation, hasNiftiExtension, formatBytes } from '../nifti.js';

function header({ dims = [96, 96, 40], ndim = 3, little = true, magic = 'n+1' } = {}) {
  const b = Buffer.alloc(352);
  const w32 = little ? b.writeInt32LE.bind(b) : b.writeInt32BE.bind(b);
  const w16 = little ? b.writeInt16LE.bind(b) : b.writeInt16BE.bind(b);
  w32(348, 0);
  w16(ndim, 40);
  dims.forEach((d, i) => w16(d, 42 + 2 * i));
  b.write(magic, 344, 'latin1');
  return b;
}
const file = (buf, name) => new File([buf], name);

describe('NIfTI header parsing', () => {
  it('reads little and big endian headers', () => {
    for (const little of [true, false]) {
      const h = parseNiftiHeader(new Uint8Array(header({ little })));
      expect(h).toEqual({ ok: true, ndim: 3, dims: [96, 96, 40] });
    }
  });
  it('rejects short, garbage and bad-magic input', () => {
    expect(parseNiftiHeader(new Uint8Array(10)).ok).toBe(false);
    expect(parseNiftiHeader(new Uint8Array(400)).ok).toBe(false);
    expect(parseNiftiHeader(new Uint8Array(header({ magic: 'xyz' }))).ok).toBe(false);
  });
});

describe('inspectFile', () => {
  it('accepts plain and gzipped 3D volumes', async () => {
    const plain = await inspectFile(file(header(), 'a.nii'));
    expect(plain).toEqual({ state: 'valid', dims: [96, 96, 40] });
    const gz = await inspectFile(file(zlib.gzipSync(header()), 'a.nii.gz'));
    expect(gz).toEqual({ state: 'valid', dims: [96, 96, 40] });
  });
  it('rejects wrong type, empty, 4D and corrupt gzip', async () => {
    expect((await inspectFile(file('x', 'a.txt'))).state).toBe('error');
    expect((await inspectFile(file('', 'a.nii'))).state).toBe('error');
    expect((await inspectFile(file(header({ ndim: 4, dims: [8, 8, 8, 2] }), 'a.nii'))).state).toBe('error');
    expect((await inspectFile(file(Buffer.from([0x1f, 0x8b, 1, 2, 3]), 'a.nii.gz'))).state).toBe('error');
    expect((await inspectFile(null)).state).toBe('empty');
  });
});

describe('combineValidation', () => {
  const f = (size = 10) => ({ size });
  const ok = (d) => ({ state: 'valid', dims: d });
  it('requires all three files and detects spatial mismatch and size limit', () => {
    expect(combineValidation({ flair: f(), t1ce: null, t2: null }, {}, 512).canSubmit).toBe(false);
    const files = { flair: f(), t1ce: f(), t2: f() };
    expect(combineValidation(files, { flair: ok([1, 1, 1]), t1ce: ok([1, 1, 1]), t2: ok([1, 1, 1]) }, 512).canSubmit).toBe(true);
    const mm = combineValidation(files, { flair: ok([1, 1, 1]), t1ce: ok([1, 1, 1]), t2: ok([1, 1, 2]) }, 512);
    expect(mm.canSubmit).toBe(false);
    expect(mm.issues.join()).toMatch(/Spatial mismatch/);
    const big = { flair: f(400 * 1024 ** 2), t1ce: f(100 * 1024 ** 2), t2: f(100 * 1024 ** 2) };
    expect(combineValidation(big, { flair: ok([1, 1, 1]), t1ce: ok([1, 1, 1]), t2: ok([1, 1, 1]) }, 512).issues.join()).toMatch(/exceeds/);
  });
  it('is pending until every present file is inspected', () => {
    const v = combineValidation({ flair: f(), t1ce: f(), t2: f() }, { flair: ok([1, 1, 1]) }, 512);
    expect(v.canSubmit).toBe(false);
    expect(v.pending).toBe(true);
  });
  it('helpers', () => {
    expect(hasNiftiExtension('X.NII.GZ')).toBe(true);
    expect(hasNiftiExtension('x.nii.txt')).toBe(false);
    expect(formatBytes(1536)).toBe('1.5 KB');
  });
});
