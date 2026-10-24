"""Reference-mask alignment and binary classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def calculate_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict:
    """Compute IoU, F1, precision, recall, and a [[TN, FP], [FN, TP]] matrix."""
    pred, ref = np.asarray(predicted, dtype=bool), np.asarray(reference, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError("predicted and reference masks must have identical shapes")
    intersection = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    return {
        "iou": intersection / union if union else 1.0,
        "f1": float(f1_score(ref.ravel(), pred.ravel(), zero_division=0)),
        "precision": float(precision_score(ref.ravel(), pred.ravel(), zero_division=0)),
        "recall": float(recall_score(ref.ravel(), pred.ravel(), zero_division=0)),
        "confusion_matrix": confusion_matrix(
            ref.ravel(), pred.ravel(), labels=[False, True]
        ).tolist(),
    }


def align_reference_mask(path: str, shape: tuple[int, int], transform, crs) -> np.ndarray:
    """Read and nearest-neighbor align a GeoTIFF reference flood mask."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    aligned = np.zeros(shape, dtype=np.uint8)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return aligned.astype(bool)


def geojson_reference_mask(reference: dict, shape: tuple[int, int], transform, crs) -> np.ndarray:
    """Rasterize a GeoJSON polygon reference mask onto the analysis grid."""
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.geometry import shape as make_shape
    from shapely.ops import transform as transform_geometry

    collection = reference.get("features") if reference.get("type") == "FeatureCollection" else [reference]
    if not collection:
        raise ValueError("Reference GeoJSON contains no features")
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    geometries = []
    for feature in collection:
        geometry = feature.get("geometry") if feature.get("type") == "Feature" else feature
        if geometry:
            polygon = transform_geometry(to_grid.transform, make_shape(geometry))
            geometries.append((polygon, 1))
    if not geometries:
        raise ValueError("Reference GeoJSON contains no geometries")
    return rasterize(geometries, out_shape=shape, transform=transform, fill=0).astype(bool)
