"""Summary and soft warnings for already-validated modality volumes.

Hard validation (3D, finite, non-empty, matching shape / voxel spacing / affine) is
tumorguard_core.validate_modality_volumes (Phase 2B). This module only derives a summary
for the API response and non-blocking warnings.
"""


def error_code_for(message):
    """Stable machine-readable code for a SpatialValidationError message."""
    m = str(message)
    if m.startswith("Mismatched volume dimensions") or m.startswith("Incompatible spatial affines") \
            or m.startswith("Mismatched voxel spacing"):
        return "SPATIAL_MISMATCH"
    if "non-finite" in m:
        return "NON_FINITE_DATA"
    if "only zero or empty" in m:
        return "EMPTY_VOLUME"
    if "must be a 3D volume" in m or "invalid dimension" in m:
        return "INVALID_SHAPE"
    if m.startswith("Missing required modality"):
        return "MISSING_MODALITY"
    return "INVALID_NIFTI"


def summarize_volumes(images):
    """`images` maps modality name -> nibabel image (as returned by validate_modality_volumes)."""
    names = list(images)
    shape = tuple(int(d) for d in images[names[0]].shape)
    warnings = []
    if shape[2] >= max(shape[0], shape[1]):
        warnings.append(
            "The third array axis is not smaller than the in-plane axes; the volume may not be stored "
            "as axial slices (x, y, z). Slices would then be resized with a distorted aspect ratio.")
    if max(shape[0], shape[1]) / max(min(shape[0], shape[1]), 1) > 1.25:
        warnings.append("In-plane dimensions are far from square; resizing to 128x128 will distort the image.")
    return {
        "shape": list(shape),
        "num_slices": int(shape[2]),
        "modalities": names,
        "nonfinite_fraction": 0.0,
        "affine_consistent": True,
        "warnings": warnings,
    }
