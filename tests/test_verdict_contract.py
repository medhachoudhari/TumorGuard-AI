import math
import os
import re

import pytest

import tumorguard_core as core

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_verdict_strings_are_canonical():
    assert core.VERDICTS == (
        "HIGH RELIABILITY", "MODERATE RELIABILITY", "LOW RELIABILITY", "MANUAL REVIEW REQUIRED",
    )


def test_verdict_aliases():
    assert core.SUPPORTED_VERDICTS == core.VERDICTS
    assert core.VERDICT_HIGH_RELIABILITY == core.VERDICT_HIGH
    assert core.VERDICT_MODERATE_RELIABILITY == core.VERDICT_MODERATE
    assert core.VERDICT_LOW_RELIABILITY == core.VERDICT_LOW
    assert core.VERDICT_MANUAL_REVIEW == core.VERDICT_MANUAL



@pytest.mark.parametrize("score,expected", [
    (1.0, "HIGH RELIABILITY"), (0.70, "HIGH RELIABILITY"),
    (0.6999999, "MODERATE RELIABILITY"), (0.50, "MODERATE RELIABILITY"),
    (0.4999999, "LOW RELIABILITY"), (0.30, "LOW RELIABILITY"),
    (0.2999999, "MANUAL REVIEW REQUIRED"), (0.0, "MANUAL REVIEW REQUIRED"),
    (-1.0, "MANUAL REVIEW REQUIRED"),
])
def test_threshold_boundaries(score, expected):
    assert core.verdict_from_score(score) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "abc", [], {}])
def test_invalid_score_requires_manual_review(bad):
    assert core.verdict_from_score(bad) == "MANUAL REVIEW REQUIRED"


def test_best_case_and_worst_case_scores():
    score, verdict = core.calculate_reliability_and_verdict(1.0, 0.0, variance=0.0, max_diff=0.0)
    assert math.isclose(score, 1.0) and verdict == "HIGH RELIABILITY"
    score, verdict = core.calculate_reliability_and_verdict(0.0, math.log(2), variance=1.0, max_diff=1.0)
    assert score == 0.0 and verdict == "MANUAL REVIEW REQUIRED"


def test_score_is_always_in_unit_interval_and_verdict_is_canonical():
    for fam in (-1, 0, 0.5, 1, 2):
        for ent in (-1, 0, 0.3, 0.7, 5):
            for var in (0, 0.0005, 1):
                for md in (0, 0.8, 5):
                    s, v = core.calculate_reliability_and_verdict(fam, ent, var, md)
                    assert 0.0 <= s <= 1.0 and v in core.VERDICTS


@pytest.mark.parametrize("args", [
    (float("nan"), 0.1, 0.0005, 0.8), (0.9, float("nan"), 0.0005, 0.8),
    (0.9, 0.1, float("inf"), 0.8), (0.9, 0.1, 0.0005, float("-inf")),
    (None, 0.1, 0.0005, 0.8), ("x", 0.1, 0.0005, 0.8),
])
def test_invalid_evidence_is_never_favourable(args):
    assert core.calculate_reliability_and_verdict(*args) == (0.0, "MANUAL REVIEW REQUIRED")


def test_frontend_mapping_covers_exactly_the_backend_verdicts():
    src = open(os.path.join(ROOT, "frontend", "src", "verdict.js"), encoding="utf-8").read()
    frontend = set(re.findall(r"'([A-Z ]+ (?:RELIABILITY|REQUIRED))'", src))
    assert frontend == set(core.VERDICTS)


def test_server_uses_core_verdict_function():
    src = open(os.path.join(ROOT, "server.py"), encoding="utf-8").read()
    assert "calculate_reliability_and_verdict" in src and '"verdict": verdict' in src


# ==============================================================================
# AUDIT REGRESSION TESTS (Section 5)
# ==============================================================================

def test_policy_high_moderate_low_reliability_profiles():
    """Verify realistic signal combinations achieve HIGH, MODERATE, LOW, and MANUAL verdicts under existing policy."""
    # High: high familiarity, low entropy, low variance, low pass difference
    # 0.40 * 0.90 + 0.20 * 0.95 + 0.20 * 1.0 + 0.20 * 1.0 = 0.36 + 0.19 + 0.20 + 0.20 = 0.95 >= 0.70
    s_high, v_high = core.calculate_reliability_and_verdict(0.90, 0.03, variance=0.0001, max_diff=0.20)
    assert s_high >= 0.70 and v_high == core.VERDICT_HIGH

    # Moderate: moderate familiarity, low entropy, moderate variance, higher pass difference
    # e.g., real BraTS slice 69: fam=0.40, ent=0.03, unc=0.00045, max_diff=0.95 -> ~0.53
    s_mod, v_mod = core.calculate_reliability_and_verdict(0.40, 0.03, variance=0.00045, max_diff=0.90)
    assert 0.50 <= s_mod < 0.70 and v_mod == core.VERDICT_MODERATE

    # Low: low familiarity, moderate variance, high pass difference
    # e.g., fam=0.25, ent=0.03, unc=0.00065, max_diff=0.95 -> ~0.43
    s_low, v_low = core.calculate_reliability_and_verdict(0.25, 0.03, variance=0.00065, max_diff=0.95)
    assert 0.30 <= s_low < 0.50 and v_low == core.VERDICT_LOW

    # Manual review required: out-of-distribution familiarity, high entropy, extreme variance
    s_man, v_man = core.calculate_reliability_and_verdict(0.02, 0.60, variance=0.0012, max_diff=1.00)
    assert s_man < 0.30 and v_man == core.VERDICT_MANUAL


def test_missing_familiarity_reference_graceful_handling():
    """Missing or corrupted familiarity_reference must fall back safely without unhandled exceptions."""
    import torch
    dummy_input = torch.zeros(1, 3, 128, 128)

    # 1. training_stats is None
    with pytest.warns(UserWarning, match="familiarity_reference missing"):
        fam_none = core.calculate_similarity_score(dummy_input, None)
    assert fam_none == 0.5

    # 2. training_stats is empty dict
    with pytest.warns(UserWarning, match="familiarity_reference missing"):
        fam_empty = core.calculate_similarity_score(dummy_input, {})
    assert fam_empty == 0.5

    # 3. training_stats has no familiarity_reference key
    with pytest.warns(UserWarning, match="familiarity_reference missing"):
        fam_missing_key = core.calculate_similarity_score(dummy_input, {"training_patients": 4})
    assert fam_missing_key == 0.5

    # 4. familiarity_reference is None
    with pytest.warns(UserWarning, match="familiarity_reference missing"):
        fam_ref_none = core.calculate_similarity_score(dummy_input, {"familiarity_reference": None})
    assert fam_ref_none == 0.5

    # 5. familiarity_reference missing 'mean' or 'std'
    with pytest.warns(UserWarning, match="familiarity_reference missing"):
        fam_incomplete = core.calculate_similarity_score(dummy_input, {"familiarity_reference": {"mean": [0.0] * 18}})
    assert fam_incomplete == 0.5


def test_non_finite_scores_and_evidence_pipeline():
    """All components of the reliability scoring pipeline reject NaN and Inf safely."""
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        # verdict_from_score must reject
        assert core.verdict_from_score(bad_val) == core.VERDICT_MANUAL

        # calculate_reliability_and_verdict must reject
        s, v = core.calculate_reliability_and_verdict(bad_val, 0.02, 0.0003, 0.7)
        assert s == 0.0 and v == core.VERDICT_MANUAL

        s, v = core.calculate_reliability_and_verdict(0.5, bad_val, 0.0003, 0.7)
        assert s == 0.0 and v == core.VERDICT_MANUAL

        s, v = core.calculate_reliability_and_verdict(0.5, 0.02, bad_val, 0.7)
        assert s == 0.0 and v == core.VERDICT_MANUAL

        s, v = core.calculate_reliability_and_verdict(0.5, 0.02, 0.0003, bad_val)
        assert s == 0.0 and v == core.VERDICT_MANUAL

        # reliability_components must return None for any non-finite input
        assert core.reliability_components(bad_val, 0.02, 0.0003, 0.7) is None
        assert core.reliability_components(0.5, bad_val, 0.0003, 0.7) is None
        assert core.reliability_components(0.5, 0.02, bad_val, 0.7) is None
        assert core.reliability_components(0.5, 0.02, 0.0003, bad_val) is None


def test_malformed_verdict_values_fail_safely():
    """Non-numeric scores in backend and unknown verdict strings in frontend must yield MANUAL REVIEW REQUIRED."""
    import subprocess
    import json

    # 1. Backend score parsing: strings/containers that cannot be parsed as a float
    non_numeric_scores = [
        "SAFE", "NEEDS ATTENTION", "HIGH", "MODERATE", "LOW",
        "high reliability", "HIGH RELIABILITY ", " HIGH RELIABILITY",
        "__proto__", "constructor", "toString",
        None, [], {},
    ]
    for s in non_numeric_scores:
        assert core.verdict_from_score(s) == core.VERDICT_MANUAL

    # 2. Frontend verdict mapper: unknown/malformed verdict strings must fail safely
    fe_test_script = """
    import('./frontend/src/verdict.js').then(m => {
        const tests = [
            'SAFE', 'NEEDS ATTENTION', 'HIGH', 'MODERATE', 'LOW',
            'high reliability', 'HIGH RELIABILITY ', ' HIGH RELIABILITY',
            '__proto__', 'constructor', 'toString', '',
            null, undefined, 0, 1, true, false, {}, []
        ];
        const results = tests.map(t => ({
            input: String(t),
            cls: m.verdictClass(t),
            label: m.verdictLabel(t),
            known: m.isKnownVerdict(t)
        }));
        console.log(JSON.stringify(results));
    });
    """
    res = subprocess.run(["node", "-e", fe_test_script], cwd=ROOT, capture_output=True, text=True, check=True)
    results = json.loads(res.stdout)
    for r in results:
        assert r["cls"] == "v-manual", f"Failed for {r['input']}"
        assert r["label"] == "MANUAL REVIEW REQUIRED", f"Failed for {r['input']}"
        assert r["known"] is False, f"Failed for {r['input']}"


def test_backend_frontend_verdict_parity():
    """Ensure exact bidirectional parity between backend constants and frontend mapping."""
    import subprocess
    import json

    # Read frontend verdict definitions via node execution
    node_cmd = [
        "node", "-e",
        "import('./frontend/src/verdict.js').then(m => console.log(JSON.stringify({"
        "  classes: m.VERDICT_CLASS,"
        "  manualReview: m.MANUAL_REVIEW"
        "})))"
    ]
    res = subprocess.run(node_cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    fe_data = json.loads(res.stdout)

    fe_classes = fe_data["classes"]
    fe_manual = fe_data["manualReview"]

    # 1. Manual review constant matches
    assert fe_manual == core.VERDICT_MANUAL

    # 2. Frontend classes keys match backend VERDICTS exactly
    assert set(fe_classes.keys()) == set(core.VERDICTS)

    # 3. Each canonical verdict maps to an expected CSS class
    assert fe_classes[core.VERDICT_HIGH] == "v-safe"
    assert fe_classes[core.VERDICT_MODERATE] == "v-attention"
    assert fe_classes[core.VERDICT_LOW] == "v-manual"
    assert fe_classes[core.VERDICT_MANUAL] == "v-manual"

