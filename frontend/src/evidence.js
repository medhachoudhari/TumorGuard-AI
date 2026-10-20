// Display vocabulary for the Reliability Evidence Layer (values come from the backend).
export const STATE_LABELS = {
  AVAILABLE_AND_VALID: 'Available and valid',
  AVAILABLE_BUT_INVALID: 'Available but invalid',
  MISSING_OR_UNAVAILABLE: 'Missing or unavailable',
  NOT_YET_SCIENTIFICALLY_VALIDATED: 'Not yet scientifically validated',
};
export const STATE_TONE = {
  AVAILABLE_AND_VALID: 'ok',
  AVAILABLE_BUT_INVALID: 'bad',
  MISSING_OR_UNAVAILABLE: 'warn',
  NOT_YET_SCIENTIFICALLY_VALIDATED: 'info',
};
export const RECOMMENDATION_LABELS = {
  HUMAN_REVIEW_RECOMMENDED: 'Human review recommended',
  NO_ADDITIONAL_REVIEW_FLAGGED: 'No additional review flagged',
  ASSESSMENT_UNAVAILABLE: 'Assessment unavailable',
};
export const RECOMMENDATION_TONE = {
  HUMAN_REVIEW_RECOMMENDED: 'bad',
  NO_ADDITIONAL_REVIEW_FLAGGED: 'info',
  ASSESSMENT_UNAVAILABLE: 'warn',
};
export const ORIGIN_LABELS = {
  model_output: 'Model output',
  deterministic_transform_of_model_output: 'Mathematical transformation of model output',
  heuristic_indicator: 'Heuristic indicator',
  input_check: 'Input check',
};

export function recommendationOf(report) {
  const r = report && report.recommendation;
  return Object.hasOwn(RECOMMENDATION_LABELS, r) ? r : 'ASSESSMENT_UNAVAILABLE';
}
export function stateOf(signal) {
  const s = signal && signal.state;
  return Object.hasOwn(STATE_LABELS, s) ? s : 'MISSING_OR_UNAVAILABLE';
}
