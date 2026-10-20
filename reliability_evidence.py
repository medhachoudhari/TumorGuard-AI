"""Reliability Evidence Layer.

Builds a structured, conservative review recommendation from signals that the real
PyTorch pipeline already produced. It never changes the predicted mask, the
reliability score or the canonical verdict, and its output is a workflow
recommendation, not a clinical statement.
"""
import math

import tumorguard_core as core

AVAILABLE_AND_VALID = "AVAILABLE_AND_VALID"
AVAILABLE_BUT_INVALID = "AVAILABLE_BUT_INVALID"
MISSING_OR_UNAVAILABLE = "MISSING_OR_UNAVAILABLE"
NOT_YET_SCIENTIFICALLY_VALIDATED = "NOT_YET_SCIENTIFICALLY_VALIDATED"
VALIDITY_STATES = (AVAILABLE_AND_VALID, AVAILABLE_BUT_INVALID, MISSING_OR_UNAVAILABLE,
                   NOT_YET_SCIENTIFICALLY_VALIDATED)

HUMAN_REVIEW_RECOMMENDED = "HUMAN_REVIEW_RECOMMENDED"
NO_ADDITIONAL_REVIEW_FLAGGED = "NO_ADDITIONAL_REVIEW_FLAGGED"
ASSESSMENT_UNAVAILABLE = "ASSESSMENT_UNAVAILABLE"
RECOMMENDATIONS = (HUMAN_REVIEW_RECOMMENDED, NO_ADDITIONAL_REVIEW_FLAGGED, ASSESSMENT_UNAVAILABLE)

SCHEMA_VERSION = "1.0"

# Origins: where a value comes from.
ORIGIN_MODEL = "model_output"
ORIGIN_TRANSFORM = "deterministic_transform_of_model_output"
ORIGIN_HEURISTIC = "heuristic_indicator"
ORIGIN_CHECK = "input_check"

LIMITATIONS = [
    "Research prototype: not clinically validated and not a diagnostic tool.",
    "No signal here has been shown to predict segmentation errors; error-detection experiments have not been run.",
    "The familiarity reference was built from only 4 training patients and 6 of its 18 features sit at the minimum-std floor.",
    "Reliability weights and cut-offs are heuristics that have not been calibrated against segmentation error.",
    "Entropy and variance are averaged over the whole 128x128 slice, including background (D-21).",
    "The pass-difference signal compares only the first and last Monte Carlo pass (D-15).",
    "The reported baseline metrics come from validation patients 031-040, which were also used for model selection.",
    "A missing review flag is not evidence that a segmentation is correct.",
]


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _signal(sig_id, label, value, state, origin, explanation, *, display=None, triggered=False,
            rule=None, scientifically_validated=False):
    return {
        "id": sig_id,
        "label": label,
        "value": float(value) if _finite(value) else None,
        "display": display,
        "state": state,
        "origin": origin,
        "scientifically_validated": scientifically_validated,
        "triggered": bool(triggered),
        "trigger_rule": rule,
        "explanation": explanation,
    }


def build_evidence_report(*, input_summary, familiarity, familiarity_reference_available,
                          entropy, variance, max_diff, mask_pixels, reliability_score, verdict,
                          provenance=None):
    """All numeric arguments are the real pipeline outputs; nothing is simulated here."""
    signals = []
    reasons = []

    # 1. Input validation ------------------------------------------------------
    if input_summary:
        signals.append(_signal(
            "input_validation", "Input modality validation", None, AVAILABLE_AND_VALID, ORIGIN_CHECK,
            "FLAIR, T1CE and T2 volumes were readable, 3D, finite, the same shape and spatially consistent.",
            display=f"{input_summary.get('shape')}"))
    else:
        signals.append(_signal(
            "input_validation", "Input modality validation", None, MISSING_OR_UNAVAILABLE, ORIGIN_CHECK,
            "No input validation summary is available."))

    # 2. Familiarity -----------------------------------------------------------
    if not familiarity_reference_available:
        signals.append(_signal(
            "training_familiarity", "Training-distribution familiarity", None, MISSING_OR_UNAVAILABLE,
            ORIGIN_HEURISTIC,
            "The familiarity reference statistics are missing, so the 0.5 fallback used by the legacy score "
            "carries no information."))
    elif not _finite(familiarity):
        signals.append(_signal(
            "training_familiarity", "Training-distribution familiarity", None, AVAILABLE_BUT_INVALID,
            ORIGIN_HEURISTIC, "The familiarity value is not a finite number."))
    else:
        signals.append(_signal(
            "training_familiarity", "Training-distribution familiarity", familiarity,
            NOT_YET_SCIENTIFICALLY_VALIDATED, ORIGIN_HEURISTIC,
            "exp(-mean |z|) over 18 image statistics against a 4-patient reference. A low value means the "
            "slice looks unlike that small reference; it has not been shown to relate to segmentation error.",
            display=f"{float(familiarity):.4f}"))

    # 3-5. MC Dropout signals --------------------------------------------------
    def mc_signal(sig_id, label, value, explanation, origin, rule_text=None, triggered=False):
        if value is None:
            return _signal(sig_id, label, None, MISSING_OR_UNAVAILABLE, origin, explanation)
        if not _finite(value):
            return _signal(sig_id, label, None, AVAILABLE_BUT_INVALID, origin,
                           "The value is not a finite number.")
        return _signal(sig_id, label, value, AVAILABLE_AND_VALID, origin, explanation,
                       display=f"{float(value):.6g}", triggered=triggered, rule=rule_text)

    var_trig = _finite(variance) and float(variance) >= core.VAR_HIGH
    signals.append(mc_signal(
        "mc_variance", "MC Dropout predictive variance", variance,
        "Mean per-pixel variance of the sigmoid probability across 10 stochastic passes. The value is a correct "
        "measurement; its usefulness as an error predictor is not yet validated.",
        ORIGIN_MODEL,
        f"variance >= {core.VAR_HIGH} (upper normalisation bound of the heuristic score in tumorguard_core.VAR_HIGH)",
        var_trig))
    signals.append(mc_signal(
        "predictive_entropy", "Predictive entropy", entropy,
        "Mean binary entropy of the mean MC probability (nats; maximum ln 2 = 0.693).",
        ORIGIN_TRANSFORM))
    signals.append(mc_signal(
        "pass_difference", "Max pass divergence", max_diff,
        "Maximum absolute probability difference between the first and last MC pass (not all pass pairs).",
        ORIGIN_MODEL))

    # 6. Mask extent -----------------------------------------------------------
    if mask_pixels is None:
        signals.append(_signal("mask_extent", "Predicted mask extent", None, MISSING_OR_UNAVAILABLE,
                               ORIGIN_TRANSFORM, "No mask was available."))
    else:
        empty = int(mask_pixels) == 0
        signals.append(_signal(
            "mask_extent", "Predicted mask extent", mask_pixels, AVAILABLE_AND_VALID, ORIGIN_TRANSFORM,
            "Number of pixels in the thresholded (mean probability > 0.5) mask on this slice.",
            display=f"{int(mask_pixels)} px", triggered=empty,
            rule="predicted mask is empty on the displayed slice"))

    # Recommendation -----------------------------------------------------------
    by_id = {s["id"]: s for s in signals}
    essential = ("input_validation", "mc_variance", "predictive_entropy", "pass_difference")
    unavailable = [by_id[i] for i in essential if by_id[i]["state"] != AVAILABLE_AND_VALID]

    components = core.reliability_components(
        familiarity if familiarity_reference_available else 0.5, entropy, variance, max_diff)

    if unavailable:
        recommendation = ASSESSMENT_UNAVAILABLE
        for s in unavailable:
            reasons.append({"signal": s["id"], "text": f"{s['label']} is {s['state'].replace('_', ' ').lower()}."})
    elif not familiarity_reference_available or not _finite(familiarity):
        recommendation = ASSESSMENT_UNAVAILABLE
        reasons.append({"signal": "training_familiarity",
                        "text": "Familiarity evidence is missing or invalid, so the legacy score is not interpretable."})
    else:
        triggered = False
        if verdict != core.VERDICT_HIGH:
            triggered = True
            reasons.append({"signal": "reliability_score",
                            "text": f"Heuristic reliability verdict is {verdict} (score {float(reliability_score):.3f}; "
                                    f"HIGH requires >= 0.70)."})
        if var_trig:
            triggered = True
            reasons.append({"signal": "mc_variance",
                            "text": "MC Dropout variance is at or above the heuristic upper bound."})
        if by_id["mask_extent"]["triggered"]:
            triggered = True
            reasons.append({"signal": "mask_extent",
                            "text": "The model predicted no tumour pixels on this slice; a missed lesion cannot be excluded."})
        recommendation = HUMAN_REVIEW_RECOMMENDED if triggered else NO_ADDITIONAL_REVIEW_FLAGGED
        if not triggered:
            reasons.append({"signal": None,
                            "text": "No experimental review rule fired. This is not evidence that the segmentation is correct."})

    breakdown = None
    if components is not None:
        breakdown = {k: {"confidence": v["confidence"], "weight": v["weight"], "contribution": v["contribution"]}
                     for k, v in components.items()}

    counts = {s: sum(1 for x in signals if x["state"] == s) for s in VALIDITY_STATES}
    return {
        "schema_version": SCHEMA_VERSION,
        "recommendation": recommendation,
        "recommendation_is_clinical": False,
        "reasons": reasons,
        "signals": signals,
        "validity_counts": counts,
        "score_breakdown": breakdown,
        "legacy_verdict": verdict,
        "legacy_reliability_score": float(reliability_score) if _finite(reliability_score) else None,
        "thresholds": {
            "status": "experimental heuristic; not calibrated or externally validated",
            "verdict_cutoffs": {"HIGH": 0.70, "MODERATE": 0.50, "LOW": 0.30},
            "variance_bounds": [core.VAR_LOW, core.VAR_HIGH],
            "pass_difference_bounds": [core.DIFF_LOW, core.DIFF_HIGH],
            "weights": core.RELIABILITY_WEIGHTS,
            "source": "tumorguard_core.py",
        },
        "limitations": LIMITATIONS,
        "provenance": provenance or {},
    }
