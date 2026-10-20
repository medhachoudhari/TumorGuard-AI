// Maps backend reliability verdicts (tumorguard_core.VERDICTS) to presentation.
// Anything missing, malformed or unrecognized is shown conservatively as
// "MANUAL REVIEW REQUIRED" and never as a favourable state.

export const MANUAL_REVIEW = 'MANUAL REVIEW REQUIRED';

export const VERDICT_CLASS = Object.freeze({
  'HIGH RELIABILITY': 'v-safe',
  'MODERATE RELIABILITY': 'v-attention',
  'LOW RELIABILITY': 'v-manual',
  [MANUAL_REVIEW]: 'v-manual',
});

export function verdictClass(verdict) {
  if (typeof verdict !== 'string') return 'v-manual';
  return Object.hasOwn(VERDICT_CLASS, verdict) ? VERDICT_CLASS[verdict] : 'v-manual';
}

export function isKnownVerdict(verdict) {
  return typeof verdict === 'string' && Object.hasOwn(VERDICT_CLASS, verdict);
}

export function verdictLabel(verdict) {
  return isKnownVerdict(verdict) ? verdict : MANUAL_REVIEW;
}
