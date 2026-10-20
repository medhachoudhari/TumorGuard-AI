import assert from 'node:assert/strict';
import { verdictClass, verdictLabel, isKnownVerdict, VERDICT_CLASS, MANUAL_REVIEW } from './src/verdict.js';

const expected = {
  'HIGH RELIABILITY': 'v-safe',
  'MODERATE RELIABILITY': 'v-attention',
  'LOW RELIABILITY': 'v-manual',
  'MANUAL REVIEW REQUIRED': 'v-manual',
};
for (const [v, cls] of Object.entries(expected)) {
  assert.equal(verdictClass(v), cls, v);
  assert.equal(verdictLabel(v), v, v);
  assert.equal(isKnownVerdict(v), true, v);
}
assert.deepEqual({ ...VERDICT_CLASS }, expected);

// Missing / malformed / unknown values fail conservatively.
const bad = [undefined, null, '', 'SAFE', 'NEEDS ATTENTION', 'high reliability', ' HIGH RELIABILITY',
  'toString', '__proto__', 'constructor', 0, 1, true, {}, [], ['HIGH RELIABILITY'], NaN];
for (const v of bad) {
  assert.equal(verdictClass(v), 'v-manual', String(v));
  assert.equal(verdictLabel(v), MANUAL_REVIEW, String(v));
  assert.equal(isKnownVerdict(v), false, String(v));
}
console.log('verdict tests passed');
