"""Generate the deterministic, illustrative offline demonstration scene."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping

from .settlements import assess_settlement
from .vectorize import mask_to_features
from .water import classify_optical_water

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
BOUNDS = (87.27, 26.4278, 87.29, 26.4422)
HEIGHT, WIDTH = 160, 200


def create_sample_data() -> None:
    """Write reproducible optical rasters and example result layers for offline demo."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(*BOUNDS, WIDTH, HEIGHT)
    rows, cols = np.indices((HEIGHT, WIDTH))
    river = ((cols - (75 + rows * 0.35)) ** 2 < 80) | ((cols - (135 - rows * 0.15)) ** 2 < 18)
    added_water = ((rows > 48) & (rows < 116) & (cols > 84) & (cols < 120)) | (
        (rows > 78) & (rows < 128) & (cols > 122) & (cols < 154)
    )
    pre_water = river
    post_water = river | added_water
    green_pre = np.where(pre_water, 0.40, 0.16).astype("float32")
    nir_pre = np.where(pre_water, 0.09, 0.48).astype("float32")
    swir_pre = np.where(pre_water, 0.08, 0.37).astype("float32")
    green_post = np.where(post_water, 0.42, 0.16).astype("float32")
    nir_post = np.where(post_water, 0.08, 0.48).astype("float32")
    swir_post = np.where(post_water, 0.07, 0.37).astype("float32")
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    water_pre, _ = classify_optical_water(green_pre, nir_pre, swir_pre, valid)
    water_post, _ = classify_optical_water(green_post, nir_post, swir_post, valid)
    # The synthetic truth is deliberate and stable; the classifier arrays remain available for review.
    new_flood = added_water & ~river

    profile = {
        "driver": "GTiff",
        "height": HEIGHT,
        "width": WIDTH,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
    }
    layers = {
        "water_pre.tif": water_pre,
        "water_post.tif": water_post,
        "new_flood.tif": new_flood,
        "permanent_water.tif": river,
        "pre_green.tif": green_pre,
        "pre_nir.tif": nir_pre,
        "pre_swir.tif": swir_pre,
        "post_green.tif": green_post,
        "post_nir.tif": nir_post,
        "post_swir.tif": swir_post,
    }
    for filename, array in layers.items():
        is_float = np.issubdtype(array.dtype, np.floating)
        with rasterio.open(
            SAMPLE_DIR / filename,
            "w",
            **{**profile, "dtype": "float32" if is_float else "uint8"},
        ) as dst:
            dst.write(array.astype("float32" if is_float else "uint8"), 1)

    flood_features = mask_to_features(new_flood, transform, min_area_ha=0.5)
    (SAMPLE_DIR / "flood_polygons.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": flood_features}),
        encoding="utf-8",
    )
    settlements = [
        ("Biratnagar East", 61.4, 34, box(87.274, 26.429, 87.281, 26.434)),
        ("Katahari", 38.2, 17, box(87.273, 26.435, 87.278, 26.439)),
        ("Jahada", 18.7, 8, box(87.282, 26.4285, 87.288, 26.432)),
        ("Rangeli North", 7.4, 0, box(87.284, 26.436, 87.289, 26.441)),
        ("Duhabi", 0.0, 0, box(87.271, 26.433, 87.274, 26.437)),
        ("Tankisinuwari", 24.1, 11, box(87.279, 26.438, 87.285, 26.442)),
    ]
    settlement_features, rows_csv = [], []
    for name, pct, buildings, geometry in settlements:
        impact = assess_settlement(name, pct, buildings)
        properties = {
            "name": name,
            "area_flooded_pct": pct,
            "flooded_buildings": buildings,
            "people_exposed_est": impact.people_exposed_est,
            "population_basis": "Estimate: flooded buildings x 5 persons/building",
            "severity": impact.severity,
            "affected": impact.affected,
        }
        settlement_features.append(
            {"type": "Feature", "geometry": mapping(geometry), "properties": properties}
        )
        rows_csv.append(properties)
    (SAMPLE_DIR / "settlements.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": settlement_features}),
        encoding="utf-8",
    )
    with (SAMPLE_DIR / "affected_settlements.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)

    flooded_area_ha = sum(feature["properties"]["area_ha"] for feature in flood_features)
    summary = {
        "sample_data": True,
        "label": "SAMPLE DATA - illustrative",
        "aoi_name": "Koshi Terai demonstration sector",
        "pre_date": "2024-07-01",
        "post_date": "2024-07-06",
        "sensor_used": "Synthetic optical (Sentinel-2-like)",
        "sar_fallback_used": False,
        "cloud_cover_pct": 12.0,
        "flooded_area_ha": round(flooded_area_ha, 1),
        "settlements_total": len(settlements),
        "settlements_affected": sum(int(r["affected"]) for r in rows_csv),
        "people_exposed_est": sum(int(r["people_exposed_est"]) for r in rows_csv),
        "population_basis": "Estimate: flooded buildings x 5 persons/building",
        "metrics": None,
        "metrics_message": "No reference mask provided",
        "bounds": [[BOUNDS[0], BOUNDS[1]], [BOUNDS[2], BOUNDS[3]]],
        "crs": "EPSG:4326",
        "resolution_m": 10,
        "classification_note": "Illustrative synthetic rasters; not observed satellite data.",
    }
    (SAMPLE_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (SAMPLE_DIR / "aoi.geojson").write_text(
        json.dumps(
            {
                "type": "Feature",
                "geometry": mapping(box(*BOUNDS)),
                "properties": {"name": summary["aoi_name"]},
            }
        ),
        encoding="utf-8",
    )
