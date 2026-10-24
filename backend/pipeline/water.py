"""Optical and SAR water classification."""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute a normalized difference while safely handling zero denominators."""
    a_f, b_f = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = a_f + b_f
    return np.divide(a_f - b_f, denom, out=np.zeros_like(denom), where=denom != 0)


def classify_optical_water(
    green: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    valid: np.ndarray | None = None,
    agreement: str = "both",
) -> tuple[np.ndarray, dict[str, float]]:
    """Classify water from NDWI/MNDWI using independently derived Otsu thresholds."""
    if agreement not in {"mndwi", "ndwi", "both"}:
        raise ValueError("agreement must be one of: mndwi, ndwi, both")
    ndwi = normalized_difference(green, nir)
    mndwi = normalized_difference(green, swir)
    valid_mask = np.ones(ndwi.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid_mask.shape != ndwi.shape or mndwi.shape != ndwi.shape:
        raise ValueError("bands and valid mask must have identical shapes")
    if not valid_mask.any():
        raise ValueError("No valid pixels available for water classification")

    def otsu(values: np.ndarray) -> float:
        sample = values[valid_mask]
        return float(threshold_otsu(sample)) if np.ptp(sample) else float(sample[0])

    ndwi_threshold, mndwi_threshold = otsu(ndwi), otsu(mndwi)
    ndwi_water = ndwi > ndwi_threshold
    mndwi_water = mndwi > mndwi_threshold
    water = {
        "ndwi": ndwi_water,
        "mndwi": mndwi_water,
        "both": ndwi_water & mndwi_water,
    }[agreement] & valid_mask
    return water, {"ndwi": ndwi_threshold, "mndwi": mndwi_threshold}


def classify_sar_water(
    vv_db: np.ndarray, valid: np.ndarray | None = None, filter_size: int = 3
) -> tuple[np.ndarray, float]:
    """Median-filter SAR backscatter and use Otsu to identify dark water pixels."""
    from scipy.ndimage import median_filter

    values = np.asarray(vv_db, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("VV input must be a 2D array")
    valid_mask = np.ones(values.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid_mask.shape != values.shape or not valid_mask.any():
        raise ValueError("valid mask must match VV and include at least one pixel")
    filtered = median_filter(values, size=filter_size)
    threshold = float(threshold_otsu(filtered[valid_mask]))
    return (filtered < threshold) & valid_mask, threshold
