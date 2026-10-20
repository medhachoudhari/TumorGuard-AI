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
