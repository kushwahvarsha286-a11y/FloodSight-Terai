"""Flood-change detection, cleanup, and terrain filtering."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects


def detect_new_flood(
    water_pre: np.ndarray,
    water_post: np.ndarray,
    permanent_water: np.ndarray | None = None,
) -> np.ndarray:
    """Return pixels wet after but not before, excluding known permanent water."""
    pre, post = np.asarray(water_pre, dtype=bool), np.asarray(water_post, dtype=bool)
    if pre.shape != post.shape:
        raise ValueError("pre/post water arrays must have identical shapes")
    flood = post & ~pre
    if permanent_water is not None:
        permanent = np.asarray(permanent_water, dtype=bool)
        if permanent.shape != flood.shape:
            raise ValueError("permanent-water mask must match water arrays")
        flood &= ~permanent
    return flood


def clean_flood(
    new_flood: np.ndarray,
    min_object_pixels: int = 2,
    slope_degrees: np.ndarray | None = None,
    max_slope_degrees: float = 5.0,
) -> np.ndarray:
    """Remove speckles, fill small gaps, and reject steep-terrain detections."""
    flood = np.asarray(new_flood, dtype=bool)
    if flood.ndim != 2:
        raise ValueError("flood mask must be two-dimensional")
    cleaned = ndimage.binary_opening(flood, structure=np.ones((3, 3), dtype=bool))
    cleaned = ndimage.binary_closing(cleaned, structure=np.ones((3, 3), dtype=bool))
    cleaned = remove_small_objects(cleaned, min_size=max(1, min_object_pixels))
    if slope_degrees is not None:
        slope = np.asarray(slope_degrees)
        if slope.shape != flood.shape:
            raise ValueError("slope raster must match flood mask")
        cleaned &= slope <= max_slope_degrees
    return cleaned.astype(bool)
