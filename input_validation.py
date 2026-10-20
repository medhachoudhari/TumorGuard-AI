"""Server-side validation of uploaded NIfTI volumes.

All error messages are safe to return to clients: they name the modality, never a
filesystem path or exception text.
"""
import numpy as np
import nibabel as nib

MIN_INPLANE = 32
MAX_DIM = 1024
MAX_NONFINITE_FRACTION = 1e-3
AFFINE_ATOL = 1e-3


class InputValidationError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _load(name, path):
    try:
        img = nib.load(path)
        shape = tuple(int(d) for d in img.shape)
    except Exception:
        raise InputValidationError("INVALID_NIFTI", f"'{name}' could not be read as a NIfTI volume.")
    if len(shape) != 3:
        raise InputValidationError(
            "INVALID_SHAPE", f"'{name}' must be a 3D volume (found {len(shape)} dimensions).")
    if shape[0] < MIN_INPLANE or shape[1] < MIN_INPLANE or shape[2] < 1:
        raise InputValidationError("INVALID_SHAPE", f"'{name}' volume is too small ({shape}).")
    if max(shape) > MAX_DIM:
        raise InputValidationError("INVALID_SHAPE", f"'{name}' volume is too large ({shape}).")
    return img, shape


def inspect_volumes(paths):
    """
    Validate the three modality volumes. Returns a summary dict, or raises
    InputValidationError. Checks: readable, 3D, plausible size, identical shape and
    affine across modalities, finite values, not all zero.
    """
    loaded = {name: _load(name, p) for name, p in paths.items()}
    shapes = {name: shape for name, (_, shape) in loaded.items()}
    if len(set(shapes.values())) != 1:
        raise InputValidationError(
            "SPATIAL_MISMATCH", "Modalities have different volume shapes; they must be co-registered.")

    names = list(loaded)
    ref_affine = loaded[names[0]][0].affine
    for name in names[1:]:
        if not np.allclose(loaded[name][0].affine, ref_affine, atol=AFFINE_ATOL):
            raise InputValidationError(
                "SPATIAL_MISMATCH", "Modalities are not spatially aligned (different affine transforms).")

    nonfinite = 0
    total = 0
    for name, (img, _) in loaded.items():
        try:
            data = np.asarray(img.get_fdata(dtype=np.float32))
        except Exception:
            raise InputValidationError("INVALID_NIFTI", f"'{name}' voxel data could not be read.")
        finite = np.isfinite(data)
        nonfinite += int(data.size - finite.sum())
        total += int(data.size)
        if not np.any(data[finite] != 0):
            raise InputValidationError("EMPTY_VOLUME", f"'{name}' contains no non-zero voxels.")
    frac = nonfinite / max(total, 1)
    if frac > MAX_NONFINITE_FRACTION:
        raise InputValidationError(
            "NON_FINITE_DATA", "Input contains too many non-finite (NaN/Inf) voxels.")

    shape = shapes[names[0]]
    warnings = []
    if shape[2] >= max(shape[0], shape[1]):
        warnings.append(
            "The third array axis is not smaller than the in-plane axes; the volume may not be stored "
            "as axial slices (x, y, z). Slices would then be resized with a distorted aspect ratio.")
    if max(shape[0], shape[1]) / min(shape[0], shape[1]) > 1.25:
        warnings.append("In-plane dimensions are far from square; resizing to 128x128 will distort the image.")
    return {
        "shape": list(shape),
        "num_slices": int(shape[2]),
        "modalities": names,
        "nonfinite_fraction": frac,
        "affine_consistent": True,
        "warnings": warnings,
    }
