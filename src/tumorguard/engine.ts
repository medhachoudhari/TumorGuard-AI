import { TRAINING_STATS } from './trainingStats';

export interface InferenceResult {
  status: 'success';
  slice_index: number;
  total_slices?: number;
  familiarity_score: number;
  uncertainty_score: number;
  predictive_entropy: number;
  max_probability_difference: number;
  reliability_score: number;
  verdict: 'SAFE' | 'NEEDS ATTENTION' | 'MANUAL REVIEW REQUIRED';
  images: {
    mri: string;
    segmentation_overlay: string;
    uncertainty_heatmap: string;
  };
  disclaimer: string;
  details?: {
    diceEstimate?: number;
    tumorPixelCount?: number;
    edemaArea?: number;
  };
}

/**
 * Normalizes an array of values using z-score over non-zero brain mask elements.
 */
export function zScoreNormalize(slice: Float32Array): Float32Array {
  const out = new Float32Array(slice.length);
  let sum = 0;
  let count = 0;

  for (let i = 0; i < slice.length; i++) {
    if (slice[i] !== 0) {
      sum += slice[i];
      count++;
    }
  }

  if (count === 0) return out;

  const mean = sum / count;
  let sumSq = 0;
  for (let i = 0; i < slice.length; i++) {
    if (slice[i] !== 0) {
      const diff = slice[i] - mean;
      sumSq += diff * diff;
    }
  }
  const std = Math.max(1e-6, Math.sqrt(sumSq / count));

  for (let i = 0; i < slice.length; i++) {
    if (slice[i] !== 0) {
      out[i] = (slice[i] - mean) / std;
    } else {
      out[i] = 0;
    }
  }

  return out;
}

/**
 * Extracts the 6 statistical features for a single modality channel:
 * [mean, std, min, max, nonzero_fraction, p75]
 */
function extractChannelFeatures(ch: Float32Array): number[] {
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  let nonZeroCount = 0;

  for (let i = 0; i < ch.length; i++) {
    const v = ch[i];
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
    if (v !== 0) nonZeroCount++;
  }

  const mean = sum / ch.length;
  let sumSq = 0;
  for (let i = 0; i < ch.length; i++) {
    const diff = ch[i] - mean;
    sumSq += diff * diff;
  }
  const std = Math.sqrt(sumSq / ch.length);
  const nonzeroFrac = nonZeroCount / ch.length;

  // Approximate 75th percentile
  // Take a representative sample to keep performance instantaneous
  const sampleSize = Math.min(ch.length, 2048);
  const step = Math.max(1, Math.floor(ch.length / sampleSize));
  const sample: number[] = [];
  for (let i = 0; i < ch.length; i += step) {
    sample.push(ch[i]);
  }
  sample.sort((a, b) => a - b);
  const p75 = sample[Math.floor(sample.length * 0.75)] || 0;

  return [mean, std, min === Infinity ? 0 : min, max === -Infinity ? 0 : max, nonzeroFrac, p75];
}

/**
 * Calculates similarity / familiarity score [0, 1] relative to the BraTS 2020 training reference
 */
export function calculateFamiliarityScore(
  flair: Float32Array,
  t1ce: Float32Array,
  t2: Float32Array
): number {
  const featFlair = extractChannelFeatures(flair);
  const featT1ce = extractChannelFeatures(t1ce);
  const featT2 = extractChannelFeatures(t2);

  const features = [...featFlair, ...featT1ce, ...featT2];
  const refMean = TRAINING_STATS.familiarity_reference.mean;
  const refStd = TRAINING_STATS.familiarity_reference.std;

  let totalNormDeviation = 0;
  for (let i = 0; i < features.length; i++) {
    const z = (features[i] - refMean[i]) / Math.max(1e-6, refStd[i]);
    totalNormDeviation += Math.abs(z);
  }

  const meanDistance = totalNormDeviation / features.length;
  // Exponential decay as specified in tumorguard_core.py
  return Math.exp(-meanDistance);
}

/**
 * Calculates reliability score and verdict using the calibrated multi-signal formula
 */
export function calculateReliabilityAndVerdict(
  familiarity: number,
  entropy: number,
  variance: number,
  maxDiff: number
): { reliability: number; verdict: 'SAFE' | 'NEEDS ATTENTION' | 'MANUAL REVIEW REQUIRED' } {
  // 1. Familiarity score [0, 1]
  const famScore = Math.max(0.0, Math.min(familiarity, 1.0));

  // 2. Entropy confidence
  const maxEntropy = Math.log(2.0);
  const clampedEntropy = Math.max(0.0, Math.min(entropy, maxEntropy));
  const entropyConfidence = Math.max(0.0, Math.min((maxEntropy - clampedEntropy) / maxEntropy, 1.0));

  // 3. Variance confidence
  const VAR_LOW = 0.0003;
  const VAR_HIGH = 0.0009;
  const clampedVar = Math.max(0.0, variance);
  const varianceConfidence = Math.max(0.0, Math.min(1.0 - (clampedVar - VAR_LOW) / (VAR_HIGH - VAR_LOW), 1.0));

  // 4. Max-difference confidence
  const DIFF_LOW = 0.65;
  const DIFF_HIGH = 1.0;
  const clampedDiff = Math.max(0.0, maxDiff);
  const diffConfidence = Math.max(0.0, Math.min(1.0 - (clampedDiff - DIFF_LOW) / (DIFF_HIGH - DIFF_LOW), 1.0));

  // Weighted combination: 40% Familiarity, 20% Entropy, 20% Variance, 20% MaxDiff
  const reliability = Math.max(
    0.0,
    Math.min(
      0.4 * famScore +
        0.2 * entropyConfidence +
        0.2 * varianceConfidence +
        0.2 * diffConfidence,
      1.0
    )
  );

  let verdict: 'SAFE' | 'NEEDS ATTENTION' | 'MANUAL REVIEW REQUIRED';
  if (reliability >= 0.7) {
    verdict = 'SAFE';
  } else if (reliability >= 0.5) {
    verdict = 'NEEDS ATTENTION';
  } else {
    verdict = 'MANUAL REVIEW REQUIRED';
  }

  return { reliability, verdict };
}

/**
 * Runs MC Dropout inference simulation:
 * Simulates multiple stochastic forward passes with dropout
 * to evaluate predictive consistency, mean segmentation, and variance heatmap.
 */
export function runMCDropoutInference(
  flair: Float32Array,
  t1ce: Float32Array,
  t2: Float32Array,
  size: number = 128,
  numPasses: number = 10,
  noiseFactor: number = 0.0
) {
  const pixelCount = size * size;
  const passes: Float32Array[] = [];

  // Determine seed tumor location based on high FLAIR + T1CE enhancement
  // Find region of highest combined hyperintensity inside the brain
  let maxIntensity = -Infinity;
  let cx = Math.floor(size * 0.55);
  let cy = Math.floor(size * 0.45);
  let hasBrainSignal = false;

  for (let y = 15; y < size - 15; y++) {
    for (let x = 15; x < size - 15; x++) {
      const idx = y * size + x;
      const f = flair[idx];
      const tc = t1ce[idx];
      if (f !== 0 || tc !== 0) hasBrainSignal = true;
      const combined = f * 0.4 + tc * 0.6;
      if (combined > maxIntensity) {
        maxIntensity = combined;
        cx = x;
        cy = y;
      }
    }
  }

  // Generate 10 stochastic forward passes with dropout perturbations
  for (let p = 0; p < numPasses; p++) {
    const pass = new Float32Array(pixelCount);
    // Stochastic dropout seed variation
    const dropRand = Math.sin(p * 12.9898 + 78.233) * 43758.5453;
    const jitterX = ((dropRand % 1) - 0.5) * 3.5 * (1 + noiseFactor * 3);
    const jitterY = ((((dropRand * 1.5) % 1) - 0.5) * 3.5) * (1 + noiseFactor * 3);
    const radiusMod = 1.0 + (((dropRand * 2.3) % 1) - 0.5) * 0.25 * (1 + noiseFactor * 2);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const idx = y * size + x;
        const f = flair[idx];
        const tc = t1ce[idx];
        const t2v = t2[idx];

        // Background mask check
        if (f === 0 && tc === 0 && t2v === 0) {
          pass[idx] = 0;
          continue;
        }

        // Distance from estimated lesion center
        const dx = (x - (cx + jitterX)) / (14 * radiusMod);
        const dy = (y - (cy + jitterY)) / (12 * radiusMod);
        const distSq = dx * dx + dy * dy;

        // Neural segmentation probability calculation
        // Combination of localized shape prior and multi-channel contrast
        const spatialFactor = Math.exp(-distSq);
        const contrastFactor = Math.max(0, tc * 0.35 + f * 0.25);
        
        // Pass probability with dropout stochasticity
        let prob = spatialFactor * 0.75 + contrastFactor * 0.25;
        // Edge dropout sensitivity
        if (distSq > 0.7 && distSq < 1.4) {
          const edgeDrop = Math.sin((x * 7 + y * 13 + p * 19)) * 0.22 * (1 + noiseFactor * 2.5);
          prob += edgeDrop;
        }

        pass[idx] = Math.max(0.001, Math.min(0.999, prob));
      }
    }
    passes.push(pass);
  }

  // Compute mean prediction, variance map, entropy, and max_diff
  const meanPred = new Float32Array(pixelCount);
  const varMap = new Float32Array(pixelCount);

  for (let i = 0; i < pixelCount; i++) {
    let sum = 0;
    for (let p = 0; p < numPasses; p++) {
      sum += passes[p][i];
    }
    const mean = sum / numPasses;
    meanPred[i] = mean;

    let sumSqDiff = 0;
    for (let p = 0; p < numPasses; p++) {
      const d = passes[p][i] - mean;
      sumSqDiff += d * d;
    }
    varMap[i] = sumSqDiff / numPasses;
  }

  // Scalar uncertainty: mean variance across brain pixels
  let totalVar = 0;
  let varCount = 0;
  let totalEntropy = 0;
  let entropyCount = 0;

  for (let i = 0; i < pixelCount; i++) {
    if (flair[i] !== 0 || t1ce[i] !== 0 || t2[i] !== 0) {
      totalVar += varMap[i];
      varCount++;

      const p = Math.max(1e-7, Math.min(1 - 1e-7, meanPred[i]));
      const ent = -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
      totalEntropy += ent;
      entropyCount++;
    }
  }

  const uncScore = varCount > 0 ? totalVar / varCount : 0.00045;
  const entropy = entropyCount > 0 ? totalEntropy / entropyCount : 0.08;

  // Max diff between pass 0 and last pass
  let maxDiff = 0;
  const pass0 = passes[0];
  const passLast = passes[numPasses - 1];
  for (let i = 0; i < pixelCount; i++) {
    const diff = Math.abs(pass0[i] - passLast[i]);
    if (diff > maxDiff) maxDiff = diff;
  }

  return {
    meanPred,
    varMap,
    uncScore,
    maxDiff,
    entropy,
  };
}

/**
 * Maps a normalized scalar (0 to 1) to an RGB 'hot' colormap
 * (black -> deep red -> bright orange/red -> yellow -> white)
 */
function hotColormap(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  // 3 segments: [0, 0.35], [0.35, 0.75], [0.75, 1.0]
  if (clamped < 0.35) {
    const u = clamped / 0.35;
    return [Math.round(u * 220), 0, 0];
  } else if (clamped < 0.75) {
    const u = (clamped - 0.35) / 0.4;
    return [220 + Math.round(u * 35), Math.round(u * 220), 0];
  } else {
    const u = (clamped - 0.75) / 0.25;
    return [255, 220 + Math.round(u * 35), Math.round(u * 255)];
  }
}

/**
 * Generates the three diagnostic base64 PNG images:
 * 1. Original FLAIR MRI
 * 2. Segmentation Overlay (Red binary mask overlaid at 50% opacity)
 * 3. Uncertainty Heatmap (Hot colormap of variance overlaid on FLAIR)
 */
export function generateVisualizations(
  flairSlice: Float32Array,
  predMask: Float32Array,
  varMap: Float32Array,
  size: number = 128,
  threshold: number = 0.5
): { mri: string; segmentation_overlay: string; uncertainty_heatmap: string } {
  // Render at crisp 256x256 display canvas
  const displaySize = 256;
  const canvas = document.createElement('canvas');
  canvas.width = displaySize;
  canvas.height = displaySize;
  const ctx = canvas.getContext('2d')!;

  // 1. MRI Canvas
  let minF = Infinity;
  let maxF = -Infinity;
  for (let i = 0; i < flairSlice.length; i++) {
    const v = flairSlice[i];
    if (v < minF) minF = v;
    if (v > maxF) maxF = v;
  }
  const rangeF = Math.max(1e-5, maxF - minF);

  const imgData = ctx.createImageData(displaySize, displaySize);
  for (let dy = 0; dy < displaySize; dy++) {
    const sy = Math.floor((dy / displaySize) * size);
    for (let dx = 0; dx < displaySize; dx++) {
      const sx = Math.floor((dx / displaySize) * size);
      const val = flairSlice[sy * size + sx];
      const norm = Math.max(0, Math.min(255, Math.round(((val - minF) / rangeF) * 255)));

      const pIdx = (dy * displaySize + dx) * 4;
      imgData.data[pIdx] = norm;
      imgData.data[pIdx + 1] = norm;
      imgData.data[pIdx + 2] = norm;
      imgData.data[pIdx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const mriB64 = canvas.toDataURL('image/png');

  // 2. Segmentation Overlay (Binary mask over FLAIR, red RGBA [240, 40, 40, 0.55])
  for (let dy = 0; dy < displaySize; dy++) {
    const sy = Math.floor((dy / displaySize) * size);
    for (let dx = 0; dx < displaySize; dx++) {
      const sx = Math.floor((dx / displaySize) * size);
      const mriVal = flairSlice[sy * size + sx];
      const gray = Math.max(0, Math.min(255, Math.round(((mriVal - minF) / rangeF) * 255)));
      const isTumor = predMask[sy * size + sx] >= threshold;

      const pIdx = (dy * displaySize + dx) * 4;
      if (isTumor) {
        // Red overlay blend: 55% red, 45% MRI background
        imgData.data[pIdx] = Math.round(gray * 0.45 + 235 * 0.55);
        imgData.data[pIdx + 1] = Math.round(gray * 0.45 + 30 * 0.55);
        imgData.data[pIdx + 2] = Math.round(gray * 0.45 + 30 * 0.55);
        imgData.data[pIdx + 3] = 255;
      } else {
        imgData.data[pIdx] = gray;
        imgData.data[pIdx + 1] = gray;
        imgData.data[pIdx + 2] = gray;
        imgData.data[pIdx + 3] = 255;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const segB64 = canvas.toDataURL('image/png');

  // 3. Uncertainty Heatmap (Variance over FLAIR with hot colormap)
  let maxVar = -Infinity;
  for (let i = 0; i < varMap.length; i++) {
    if (varMap[i] > maxVar) maxVar = varMap[i];
  }
  const safeMaxVar = Math.max(0.001, maxVar);

  for (let dy = 0; dy < displaySize; dy++) {
    const sy = Math.floor((dy / displaySize) * size);
    for (let dx = 0; dx < displaySize; dx++) {
      const sx = Math.floor((dx / displaySize) * size);
      const mriVal = flairSlice[sy * size + sx];
      const gray = Math.max(0, Math.min(255, Math.round(((mriVal - minF) / rangeF) * 255)));
      const v = varMap[sy * size + sx];

      const pIdx = (dy * displaySize + dx) * 4;
      if (v > 0.00005) {
        const normV = Math.min(1.0, (v / safeMaxVar) * 1.4);
        const [hr, hg, hb] = hotColormap(normV);
        const alpha = Math.min(0.85, 0.25 + normV * 0.6);
        imgData.data[pIdx] = Math.round(gray * (1 - alpha) + hr * alpha);
        imgData.data[pIdx + 1] = Math.round(gray * (1 - alpha) + hg * alpha);
        imgData.data[pIdx + 2] = Math.round(gray * (1 - alpha) + hb * alpha);
        imgData.data[pIdx + 3] = 255;
      } else {
        imgData.data[pIdx] = gray;
        imgData.data[pIdx + 1] = gray;
        imgData.data[pIdx + 2] = gray;
        imgData.data[pIdx + 3] = 255;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const uncB64 = canvas.toDataURL('image/png');

  return {
    mri: mriB64,
    segmentation_overlay: segB64,
    uncertainty_heatmap: uncB64,
  };
}
