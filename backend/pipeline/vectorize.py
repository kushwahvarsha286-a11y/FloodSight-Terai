"""Conversion from flood raster masks into GeoJSON-ready polygons."""

from __future__ import annotations

import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape


def mask_to_features(
    mask: np.ndarray,
    transform: object,
    crs: str = "EPSG:4326",
    min_area_ha: float = 0.0,
) -> list[dict]:
    """Polygonize a boolean mask and return GeoJSON features with area in hectares."""
    from pyproj import CRS, Transformer
    from shapely.ops import transform as transform_geom

    raster = np.asarray(mask, dtype=np.uint8)
    target = CRS.from_user_input(crs)
    to_metric = Transformer.from_crs(target, target if target.is_projected else "EPSG:6933", always_xy=True)
    features = []
    for geom, value in shapes(raster, mask=raster.astype(bool), transform=transform):
        if not value:
            continue
        polygon = shape(geom)
        metric = transform_geom(to_metric.transform, polygon)
        area_ha = metric.area / 10_000.0
        if area_ha < min_area_ha:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {"area_ha": round(area_ha, 3)},
            }
        )
    return features
