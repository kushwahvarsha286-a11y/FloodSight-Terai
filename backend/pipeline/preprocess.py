"""Raster masking and common-grid preparation helpers."""

from __future__ import annotations

import numpy as np


def scl_valid_mask(scl: np.ndarray) -> np.ndarray:
    """Return valid Sentinel-2 pixels, excluding cloud, shadow, cirrus, and snow classes."""
    classes = np.asarray(scl)
    return ~np.isin(classes, [0, 1, 3, 8, 9, 10, 11])


def resample_to_grid(
    source: np.ndarray,
    source_transform: object,
    source_crs: object,
    destination_shape: tuple[int, int],
    destination_transform: object,
    destination_crs: object,
    categorical: bool = False,
) -> np.ndarray:
    """Reproject one band onto a shared destination grid using nearest/bilinear resampling."""
    from rasterio.warp import Resampling, reproject

    destination = np.empty(destination_shape, dtype=np.asarray(source).dtype)
    reproject(
        source=np.asarray(source),
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        resampling=Resampling.nearest if categorical else Resampling.bilinear,
    )
    return destination
