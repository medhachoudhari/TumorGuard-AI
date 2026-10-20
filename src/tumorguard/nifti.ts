import pako from 'pako';

export interface NiftiVolume {
  width: number;
  height: number;
  slices: number;
  data: Float32Array; // 3D volume linearized (slice * width * height + y * width + x)
  dataType: number;
  sliceSpacing?: number;
}

/**
 * Checks if a buffer is gzip compressed (starts with 0x1f, 0x8b)
 */
function isGzip(buffer: Uint8Array): boolean {
  return buffer.length > 2 && buffer[0] === 0x1f && buffer[1] === 0x8b;
}

/**
 * Parses NIfTI-1 file buffer (.nii or .nii.gz)
 */
export function parseNifti(buffer: ArrayBuffer): NiftiVolume {
  let uint8 = new Uint8Array(buffer);

  if (isGzip(uint8)) {
    try {
      uint8 = pako.ungzip(uint8);
    } catch (e) {
      throw new Error(`Failed to decompress gzip NIfTI archive: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  const view = new DataView(uint8.buffer, uint8.byteOffset, uint8.byteLength);

  // Check endianness: sizeof_hdr at offset 0 should be 348
  let littleEndian = true;
  let sizeofHdr = view.getInt32(0, true);
  if (sizeofHdr !== 348) {
    sizeofHdr = view.getInt32(0, false);
    if (sizeofHdr === 348) {
      littleEndian = false;
    } else {
      throw new Error(`Invalid NIfTI header size (${sizeofHdr}). Expected 348.`);
    }
  }

  // Dimensions
  const dim0 = view.getInt16(40, littleEndian); // dimensionality
  const nx = view.getInt16(42, littleEndian) || 128;
  const ny = view.getInt16(44, littleEndian) || 128;
  const nz = view.getInt16(46, littleEndian) || 1;

  const datatype = view.getInt16(70, littleEndian);
  const bitpix = view.getInt16(72, littleEndian);
  const voxOffset = Math.max(352, Math.round(view.getFloat32(108, littleEndian)));

  const totalVoxels = nx * ny * nz;
  const floatData = new Float32Array(totalVoxels);

  const rawBytes = uint8.subarray(voxOffset);

  // Parse according to datatype
  if (datatype === 2) {
    // uint8
    for (let i = 0; i < totalVoxels && i < rawBytes.length; i++) {
      floatData[i] = rawBytes[i];
    }
  } else if (datatype === 4) {
    // int16
    const int16View = new Int16Array(rawBytes.buffer, rawBytes.byteOffset, Math.min(totalVoxels, Math.floor(rawBytes.byteLength / 2)));
    for (let i = 0; i < int16View.length; i++) {
      floatData[i] = int16View[i];
    }
  } else if (datatype === 512) {
    // uint16
    const uint16View = new Uint16Array(rawBytes.buffer, rawBytes.byteOffset, Math.min(totalVoxels, Math.floor(rawBytes.byteLength / 2)));
    for (let i = 0; i < uint16View.length; i++) {
      floatData[i] = uint16View[i];
    }
  } else if (datatype === 16) {
    // float32
    const float32View = new Float32Array(rawBytes.buffer, rawBytes.byteOffset, Math.min(totalVoxels, Math.floor(rawBytes.byteLength / 4)));
    for (let i = 0; i < float32View.length; i++) {
      floatData[i] = float32View[i];
    }
  } else {
    // Fallback: read as 16-bit integers
    const fallbackCount = Math.min(totalVoxels, Math.floor(rawBytes.byteLength / 2));
    const fallback = new Int16Array(rawBytes.buffer, rawBytes.byteOffset, fallbackCount);
    for (let i = 0; i < fallback.length; i++) {
      floatData[i] = fallback[i];
    }
  }

  return {
    width: nx,
    height: ny,
    slices: nz,
    data: floatData,
    dataType: datatype,
  };
}

/**
 * Extracts a specific 2D axial slice from a 3D NIfTI volume,
 * resampled to target size (128x128).
 */
export function extractSlice(
  volume: NiftiVolume,
  sliceIdx: number,
  targetSize: number = 128
): Float32Array {
  const { width, height, slices, data } = volume;
  const clampedSlice = Math.max(0, Math.min(slices - 1, sliceIdx));
  const sliceStride = width * height;
  const sliceOffset = clampedSlice * sliceStride;

  const out = new Float32Array(targetSize * targetSize);

  for (let ty = 0; ty < targetSize; ty++) {
    const srcY = Math.floor((ty / targetSize) * height);
    for (let tx = 0; tx < targetSize; tx++) {
      const srcX = Math.floor((tx / targetSize) * width);
      const val = data[sliceOffset + srcY * width + srcX] || 0;
      out[ty * targetSize + tx] = val;
    }
  }

  return out;
}

/**
 * Automatically finds the slice index with maximum brain area / contrast
 */
export function findBestSlice(volume: NiftiVolume): number {
  const { width, height, slices, data } = volume;
  if (slices <= 1) return 0;

  let bestSlice = Math.floor(slices / 2);
  let maxVariance = -1;

  for (let s = Math.floor(slices * 0.2); s < Math.floor(slices * 0.85); s++) {
    const offset = s * width * height;
    let sum = 0;
    let sumSq = 0;
    let nonZeroCount = 0;
    const stride = Math.max(1, Math.floor((width * height) / 500)); // Sample 500 pixels per slice

    for (let i = 0; i < width * height; i += stride) {
      const v = data[offset + i];
      if (v > 0) {
        sum += v;
        sumSq += v * v;
        nonZeroCount++;
      }
    }

    if (nonZeroCount > 30) {
      const mean = sum / nonZeroCount;
      const variance = sumSq / nonZeroCount - mean * mean;
      if (variance > maxVariance) {
        maxVariance = variance;
        bestSlice = s;
      }
    }
  }

  return bestSlice;
}
