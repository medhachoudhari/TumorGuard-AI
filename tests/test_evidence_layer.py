import math

import pytest

import reliability_evidence as ev
import tumorguard_core as core

OK_INPUT = {"shape": [64, 64, 20], "num_slices": 20, "modalities": ["flair", "t1ce", "t2"]}


def build(**over):
    kw = dict(input_summary=OK_INPUT, familiarity=0.9, familiarity_reference_available=True,
              entropy=0.01, variance=0.0001, max_diff=0.3, mask_pixels=500,
              reliability_score=0.95, verdict=core.VERDICT_HIGH)
    kw.update(over)
    return ev.build_evidence_report(**kw)


def sig(report, sig_id):
    return next(s for s in report["signals"] if s["id"] == sig_id)


def test_clean_case_flags_nothing_but_is_not_called_safe():
    r = build()
    assert r["recommendation"] == ev.NO_ADDITIONAL_REVIEW_FLAGGED
    assert r["recommendation_is_clinical"] is False
    assert any("not evidence that the segmentation is correct" in x["text"] for x in r["reasons"])
    assert r["limitations"] and r["thresholds"]["status"].startswith("experimental")


def test_familiarity_is_never_marked_scientifically_valid():
    assert sig(build(), "training_familiarity")["state"] == ev.NOT_YET_SCIENTIFICALLY_VALIDATED
    assert all(not s["scientifically_validated"] for s in build()["signals"])


@pytest.mark.parametrize("verdict", [core.VERDICT_MODERATE, core.VERDICT_LOW, core.VERDICT_MANUAL])
def test_non_high_verdict_recommends_review(verdict):
    r = build(verdict=verdict, reliability_score=0.4)
    assert r["recommendation"] == ev.HUMAN_REVIEW_RECOMMENDED
    assert any(x["signal"] == "reliability_score" for x in r["reasons"])


def test_empty_mask_recommends_review():
    r = build(mask_pixels=0)
    assert r["recommendation"] == ev.HUMAN_REVIEW_RECOMMENDED
    assert sig(r, "mask_extent")["triggered"]


def test_high_variance_recommends_review():
    r = build(variance=core.VAR_HIGH)
    assert r["recommendation"] == ev.HUMAN_REVIEW_RECOMMENDED and sig(r, "mc_variance")["triggered"]


def test_missing_familiarity_reference_is_unavailable_not_safe():
    r = build(familiarity=0.5, familiarity_reference_available=False)
    assert r["recommendation"] == ev.ASSESSMENT_UNAVAILABLE
    assert sig(r, "training_familiarity")["state"] == ev.MISSING_OR_UNAVAILABLE


@pytest.mark.parametrize("field", ["entropy", "variance", "max_diff"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None])
def test_invalid_mc_signal_makes_assessment_unavailable(field, bad):
    r = build(**{field: bad})
    assert r["recommendation"] == ev.ASSESSMENT_UNAVAILABLE
    state = {"entropy": "predictive_entropy", "variance": "mc_variance", "max_diff": "pass_difference"}[field]
    expected = ev.MISSING_OR_UNAVAILABLE if bad is None else ev.AVAILABLE_BUT_INVALID
    assert sig(r, state)["state"] == expected
    assert sig(r, state)["value"] is None


def test_invalid_familiarity_value():
    r = build(familiarity=float("nan"))
    assert r["recommendation"] == ev.ASSESSMENT_UNAVAILABLE
    assert sig(r, "training_familiarity")["state"] == ev.AVAILABLE_BUT_INVALID


def test_missing_input_summary_is_unavailable():
    r = build(input_summary=None)
    assert r["recommendation"] == ev.ASSESSMENT_UNAVAILABLE
    assert sig(r, "input_validation")["state"] == ev.MISSING_OR_UNAVAILABLE


def test_states_and_recommendations_are_from_fixed_vocabularies():
    for r in (build(), build(mask_pixels=0), build(entropy=float("nan"))):
        assert r["recommendation"] in ev.RECOMMENDATIONS
        assert all(s["state"] in ev.VALIDITY_STATES for s in r["signals"])
        assert sum(r["validity_counts"].values()) == len(r["signals"])


def test_report_does_not_alter_legacy_fields():
    r = build(reliability_score=0.42, verdict=core.VERDICT_LOW)
    assert r["legacy_verdict"] == core.VERDICT_LOW and r["legacy_reliability_score"] == 0.42


def test_report_is_json_serialisable_with_invalid_input():
    import json
    json.dumps(build(entropy=float("nan"), variance=float("inf")), allow_nan=False)


# ---- reliability_components must reproduce the original formula exactly -------------

def legacy(sim, ent, var, md):
    fam = max(0.0, min(float(sim), 1.0))
    me = math.log(2.0)
    ent = max(0.0, min(float(ent), me))
    ec = max(0.0, min((me - ent) / me, 1.0))
    var = max(0.0, float(var))
    vc = max(0.0, min(1.0 - ((var - 0.0003) / (0.0009 - 0.0003)), 1.0))
    md = max(0.0, float(md))
    dc = max(0.0, min(1.0 - ((md - 0.65) / (1.0 - 0.65)), 1.0))
    return max(0.0, min(0.40 * fam + 0.20 * ec + 0.20 * vc + 0.20 * dc, 1.0))


def test_score_matches_original_formula_on_grid():
    for fam in (0, 0.1, 0.5, 0.9, 1):
        for ent in (0, 0.05, 0.3, 0.69):
            for var in (0, 0.0003, 0.0006, 0.0009, 0.01):
                for md in (0, 0.65, 0.8, 1.0):
                    got, _ = core.calculate_reliability_and_verdict(fam, ent, var, md)
                    assert got == pytest.approx(legacy(fam, ent, var, md), abs=1e-12)


def test_components_sum_to_score():
    c = core.reliability_components(0.3, 0.2, 0.0005, 0.8)
    score, _ = core.calculate_reliability_and_verdict(0.3, 0.2, 0.0005, 0.8)
    assert sum(v["contribution"] for v in c.values()) == pytest.approx(score)
    assert core.reliability_components(float("nan"), 0.2, 0.0005, 0.8) is None


def test_structural_cap_without_familiarity():
    """Documents why verdicts rarely reach HIGH: with familiarity ~0 the score is at most 0.60."""
    score, verdict = core.calculate_reliability_and_verdict(0.0, 0.0, 0.0, 0.0)
    assert score == pytest.approx(0.6) and verdict == core.VERDICT_MODERATE
