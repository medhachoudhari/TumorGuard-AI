export interface SampleCase {
  id: string;
  name: string;
  tag: string;
  verdictExpected: 'SAFE' | 'NEEDS ATTENTION' | 'MANUAL REVIEW REQUIRED';
  description: string;
  flair: Float32Array;
  t1ce: Float32Array;
  t2: Float32Array;
  sliceIndex: number;
}

/**
 * Procedurally generates an anatomically realistic axial brain slice
 * with skull, CSF ventricles, white/gray matter, and a pathological lesion.
 */
function generateSyntheticBrain(
  size: number = 128,
  options: {
    tumorCenter?: [number, number];
    tumorRadius?: number;
    edemaRadius?: number;
    enhancement?: number;
    noiseLevel?: number;
    motionArtifact?: boolean;
    modality: 'flair' | 't1ce' | 't2';
  }
): Float32Array {
  const {
    tumorCenter = [70, 56],
    tumorRadius = 14,
    edemaRadius = 24,
    enhancement = 1.0,
    noiseLevel = 0.04,
    motionArtifact = false,
    modality,
  } = options;

  const data = new Float32Array(size * size);
  const cx = size / 2;
  const cy = size / 2;
  const a = size * 0.42; // semi-major axis (brain skull width)
  const b = size * 0.46; // semi-minor axis (brain skull height)

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = y * size + x;

      // Normalized coordinates relative to center
      const nx = (x - cx) / a;
      const ny = (y - cy) / b;
      const rSkull = nx * nx + ny * ny;

      if (rSkull > 1.0) {
        // Outside skull
        data[idx] = 0;
        continue;
      }

      // Inside brain parenchyma
      let signal = 0.48; // baseline white/gray matter

      // Gyri / sulci convolutions
      const sulci = Math.sin(nx * 14 + ny * 6) * Math.cos(ny * 16 - nx * 8) * 0.08;
      signal += sulci;

      // Bilateral lateral ventricles (CSF)
      const vLy = (y - (cy - 4)) / 16;
      const vLx = (x - (cx - 7)) / 5;
      const vRx = (x - (cx + 7)) / 5;
      const inVentricleL = vLx * vLx + vLy * vLy < 1.0;
      const inVentricleR = vRx * vRx + vLy * vLy < 1.0;

      if (inVentricleL || inVentricleR) {
        if (modality === 'flair') {
          signal = 0.05; // CSF is suppressed in FLAIR!
        } else if (modality === 't2') {
          signal = 0.95; // CSF is hyperintense in T2
        } else {
          signal = 0.12; // CSF is hypointense in T1CE
        }
      }

      // Lesion: Distance to tumor center
      const tdx = x - tumorCenter[0];
      const tdy = y - tumorCenter[1];
      const distToTumor = Math.sqrt(tdx * tdx + tdy * tdy);

      // Edema rim (FLAIR hyperintense, T2 bright)
      if (distToTumor <= edemaRadius) {
        const edemaT = 1.0 - distToTumor / edemaRadius;
        if (modality === 'flair') {
          signal += edemaT * 0.55;
        } else if (modality === 't2') {
          signal += edemaT * 0.42;
        } else {
          signal -= edemaT * 0.1; // mild edema hypointensity on T1
        }
      }

      // Tumor active core / enhancing margin
      if (distToTumor <= tumorRadius) {
        const coreT = 1.0 - distToTumor / tumorRadius;
        if (modality === 't1ce') {
          // Ring enhancement on T1CE
          const ringDist = Math.abs(distToTumor - tumorRadius * 0.65);
          const ring = Math.exp(-(ringDist * ringDist) / 8);
          signal += (ring * 0.75 + coreT * 0.35) * enhancement;
        } else if (modality === 'flair') {
          signal += coreT * 0.4;
        } else if (modality === 't2') {
          signal += coreT * 0.3;
        }
      }

      // Skull bone / scalp border
      if (rSkull > 0.92 && rSkull <= 1.0) {
        signal = modality === 't1ce' ? 0.65 : 0.25;
      }

      // Motion artifact stripes (simulating patient movement along phase-encode axis)
      if (motionArtifact) {
        const ghosting = Math.sin(y * 0.7) * 0.28;
        signal += ghosting;
      }

      // Gaussian-like noise
      const rand = (Math.sin(x * 93.1 + y * 45.7) * 43758.5453) % 1;
      signal += rand * noiseLevel;

      data[idx] = Math.max(0, signal);
    }
  }

  return data;
}

/**
 * 3 Authentic sample cases representing distinct reliability domains
 */
export const SAMPLE_CASES: SampleCase[] = [
  {
    id: 'sample_brats_001',
    name: 'Patient BraTS20_001',
    tag: 'Canonical Glioblastoma (High Familiarity)',
    verdictExpected: 'SAFE',
    description:
      'Standard high-grade glioblastoma with distinct FLAIR peritumoral edema, ring-enhancing T1CE rim, and high signal-to-noise ratio. Very close to the training distribution.',
    flair: generateSyntheticBrain(128, {
      modality: 'flair',
      tumorCenter: [72, 54],
      tumorRadius: 15,
      edemaRadius: 26,
      enhancement: 1.1,
      noiseLevel: 0.03,
      motionArtifact: false,
    }),
    t1ce: generateSyntheticBrain(128, {
      modality: 't1ce',
      tumorCenter: [72, 54],
      tumorRadius: 15,
      edemaRadius: 26,
      enhancement: 1.25,
      noiseLevel: 0.03,
      motionArtifact: false,
    }),
    t2: generateSyntheticBrain(128, {
      modality: 't2',
      tumorCenter: [72, 54],
      tumorRadius: 15,
      edemaRadius: 26,
      enhancement: 1.0,
      noiseLevel: 0.03,
      motionArtifact: false,
    }),
    sliceIndex: 78,
  },
  {
    id: 'sample_brats_014',
    name: 'Patient BraTS20_014',
    tag: 'Diffuse Infiltrative Boundary (Borderline)',
    verdictExpected: 'NEEDS ATTENTION',
    description:
      'Infiltrative lower-contrast lesion with irregular margins, subtle contrast uptake, and higher stochastic MC Dropout variance across the diffuse peripheral infiltration zone.',
    flair: generateSyntheticBrain(128, {
      modality: 'flair',
      tumorCenter: [56, 68],
      tumorRadius: 12,
      edemaRadius: 20,
      enhancement: 0.65,
      noiseLevel: 0.06,
      motionArtifact: false,
    }),
    t1ce: generateSyntheticBrain(128, {
      modality: 't1ce',
      tumorCenter: [56, 68],
      tumorRadius: 12,
      edemaRadius: 20,
      enhancement: 0.55,
      noiseLevel: 0.06,
      motionArtifact: false,
    }),
    t2: generateSyntheticBrain(128, {
      modality: 't2',
      tumorCenter: [56, 68],
      tumorRadius: 12,
      edemaRadius: 20,
      enhancement: 0.6,
      noiseLevel: 0.06,
      motionArtifact: false,
    }),
    sliceIndex: 65,
  },
  {
    id: 'sample_brats_outlier',
    name: 'Patient BraTS20_Outlier',
    tag: 'OOD / Motion Artifact (Distribution Shift)',
    verdictExpected: 'MANUAL REVIEW REQUIRED',
    description:
      'Severe phase-encode motion ghosting and heavy noise deviation from standard BraTS imaging protocol. Triggers low familiarity score and high predictive variance.',
    flair: generateSyntheticBrain(128, {
      modality: 'flair',
      tumorCenter: [48, 50],
      tumorRadius: 16,
      edemaRadius: 22,
      enhancement: 0.7,
      noiseLevel: 0.16,
      motionArtifact: true,
    }),
    t1ce: generateSyntheticBrain(128, {
      modality: 't1ce',
      tumorCenter: [48, 50],
      tumorRadius: 16,
      edemaRadius: 22,
      enhancement: 0.65,
      noiseLevel: 0.16,
      motionArtifact: true,
    }),
    t2: generateSyntheticBrain(128, {
      modality: 't2',
      tumorCenter: [48, 50],
      tumorRadius: 16,
      edemaRadius: 22,
      enhancement: 0.65,
      noiseLevel: 0.16,
      motionArtifact: true,
    }),
    sliceIndex: 82,
  },
];
