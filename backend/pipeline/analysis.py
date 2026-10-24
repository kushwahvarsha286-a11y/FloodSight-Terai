"""Orchestration for real Sentinel-2/Sentinel-1 AOI analyses and deliverables."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry

from .change import clean_flood, detect_new_flood
from .config import DEFAULTS
from .data_fetch import fetch_dem, fetch_optical_pair, fetch_sar_pair
from .preprocess import scl_valid_mask
from .settlements import assess_settlement, fetch_osm_settlements
from .vectorize import mask_to_features
from .water import classify_optical_water, classify_sar_water
from .validate import align_reference_mask, calculate_metrics, geojson_reference_mask


def run_real_analysis(
    aoi_geojson: dict,
    pre_date: date,
    post_date: date,
    sensor: str,
    index: str,
    min_area_ha: float,
    flag_threshold_pct: float,
    output_dir: Path,
    progress,
    reference_mask: dict | str | None = None,
) -> dict:
    """Fetch real imagery, derive new flood extent, and write the requested result files."""
    if post_date <= pre_date:
        raise ValueError("post_date must be later than pre_date.")
    aoi = shape(aoi_geojson.get("geometry", aoi_geojson))
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(1, 5, "Searching Planetary Computer for date-matched imagery.")
    selected_sensor = sensor
    pair = None
    cloud_cover = None
    if sensor in {"auto", "optical"}:
        pair = fetch_optical_pair(mapping(aoi), pre_date, post_date)
        cloud_cover = pair["cloud_cover_pct"]
        if cloud_cover > DEFAULTS["cloud_limit_pct"]:
            if sensor == "optical":
                raise RuntimeError(
                    f"Optical cloud cover over the AOI is {cloud_cover:.1f}% (>40%). "
                    "Choose Auto or SAR to use cloud-proof Sentinel-1."
                )
            selected_sensor = "sar"
            pair = None
    if selected_sensor in {"sar", "auto"}:
        if pair is None:
            pair = fetch_sar_pair(mapping(aoi), pre_date, post_date)
            selected_sensor = "sar"
    if selected_sensor == "sar":
        pre_water, _ = classify_sar_water(pair["pre"]["vv_db"])
        post_water, _ = classify_sar_water(pair["post"]["vv_db"])
        profile_source = pair["post"]
        cloud_cover = cloud_cover if sensor == "auto" else None
    else:
        pre, post = pair["pre"], pair["post"]
        if pre["crs"] != post["crs"] or pre["green"].shape != post["green"].shape:
            raise RuntimeError("Selected scenes do not share an analysis grid; retry with dates closer together.")
        pre_water, _ = classify_optical_water(
            pre["green"], pre["nir"], pre["swir"], scl_valid_mask(pre["scl"]), index
        )
        post_water, _ = classify_optical_water(
            post["green"], post["nir"], post["swir"], scl_valid_mask(post["scl"]), index
        )
        profile_source = post
    progress(2, 22, f"Loaded {selected_sensor.upper()} scene pair: {', '.join(pair['scene_ids'])}.")
    progress(3, 36, "Applied scene-specific Otsu water classification.")

    transform, crs = profile_source["transform"], profile_source["crs"]
    to_analysis_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    aoi_projected = transform_geometry(to_analysis_crs.transform, aoi)
    grid = pre_water.shape
    resolution = int(round(abs(transform.a)))
    slope = fetch_dem(mapping(aoi), crs, max(1, resolution), grid)
    progress(4, 46, "Calculated Copernicus DEM slope on the analysis grid.")
    raw_flood = detect_new_flood(pre_water, post_water)
    min_pixels = max(1, int(np.ceil(min_area_ha * 10_000 / (resolution * resolution))))
    flood = clean_flood(
        raw_flood,
        min_object_pixels=max(DEFAULTS["min_object_pixels"], min_pixels),
        slope_degrees=slope,
        max_slope_degrees=DEFAULTS["slope_limit_degrees"],
    )
    metrics = None
    reference = None
    if isinstance(reference_mask, dict):
        reference = geojson_reference_mask(reference_mask, grid, transform, crs)
    elif isinstance(reference_mask, str):
        reference = align_reference_mask(reference_mask, grid, transform, crs)
    if reference is not None:
        metrics = calculate_metrics(flood, reference)
    progress(5, 56, f"Cleaned flood pixels using {DEFAULTS['slope_limit_degrees']}° slope limit.")
    flood_features = mask_to_features(flood, transform, crs, min_area_ha=min_area_ha)
    progress(6, 66, f"Created {len(flood_features)} flood polygons.")
    with rasterio.open(
        output_dir / "flood_extent.tif",
        "w",
        driver="GTiff",
        height=grid[0],
        width=grid[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as destination:
        destination.write(flood.astype(np.uint8), 1)
    for filename, mask in (("water_pre.tif", pre_water), ("water_post.tif", post_water)):
        with rasterio.open(
            output_dir / filename,
            "w",
            driver="GTiff",
            height=grid[0],
            width=grid[1],
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
            compress="deflate",
        ) as destination:
            destination.write(mask.astype(np.uint8), 1)
    if selected_sensor == "optical":
        for prefix, scene in (("pre", pair["pre"]), ("post", pair["post"])):
            bands = np.stack([scene["red"], scene["green"], scene["blue"]]).astype(np.float32)
            with rasterio.open(
                output_dir / f"{prefix}_rgb.tif",
                "w",
                driver="GTiff",
                height=grid[0],
                width=grid[1],
                count=3,
                dtype="float32",
                crs=crs,
                transform=transform,
                compress="deflate",
            ) as destination:
                destination.write(bands)
    flood_collection = {"type": "FeatureCollection", "features": flood_features}
    (output_dir / "flood_polygons.geojson").write_text(json.dumps(flood_collection), encoding="utf-8")

    progress(7, 76, "Looking up OpenStreetMap settlements and buildings.")
    osm_key = hashlib.sha256(aoi.wkb).hexdigest()[:16]
    osm_path = output_dir.parent / "osm_cache" / f"osm_{osm_key}.geojson"
    osm = fetch_osm_settlements(aoi, str(osm_path))
    if osm.crs is None:
        raise RuntimeError("OSM features are missing a coordinate reference system.")
    osm = osm.to_crs(crs)
    places = osm[osm["place"].notna()] if "place" in osm else osm.iloc[0:0]
    uses_landuse_fallback = places.empty and "landuse" in osm
    if uses_landuse_fallback:
        places = osm[osm["landuse"].isin(["residential", "village_green"])]
    buildings = osm[osm["building"].notna()] if "building" in osm else osm.iloc[0:0]
    settlements = []
    predicted_flags, reference_flags = [], []
    for place_number, (_, place) in enumerate(places.iterrows(), start=1):
        polygon = place.geometry if place.geometry.geom_type in {"Polygon", "MultiPolygon"} else place.geometry.buffer(500)
        polygon = polygon.intersection(aoi_projected)
        if polygon.is_empty:
            continue
        inside = geometry_mask([mapping(polygon)], out_shape=grid, transform=transform, invert=True)
        if not inside.any():
            continue
        area_pct = 100.0 * int((inside & flood).sum()) / int(inside.sum())
        flooded_buildings = 0
        for building in buildings.geometry:
            if building is None or not polygon.intersects(building):
                continue
            x, y = building.centroid.coords[0]
            r, c = rowcol(transform, x, y)
            if 0 <= r < grid[0] and 0 <= c < grid[1] and flood[r, c]:
                flooded_buildings += 1
        raw_name = place.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            name = raw_name.strip()
        else:
            area_type = place.get("landuse")
            if not isinstance(area_type, str) or not area_type.strip():
                area_type = place.get("place")
            if not isinstance(area_type, str) or not area_type.strip():
                area_type = "Settlement"
            name = f"{area_type.replace('_', ' ').title()} area {place_number}"
        impact = assess_settlement(name, area_pct, flooded_buildings, DEFAULTS["persons_per_building"], flag_threshold_pct)
        settlements.append(
            {
                "type": "Feature",
                "geometry": mapping(
                    transform_geometry(Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform, polygon)
                ),
                "properties": {
                    "name": name,
                    "area_flooded_pct": round(area_pct, 2),
                    "flooded_buildings": flooded_buildings,
                    "people_exposed_est": impact.people_exposed_est,
                    "population_basis": f"Estimate: flooded buildings x {DEFAULTS['persons_per_building']} persons/building",
                    "severity": impact.severity,
                    "affected": impact.affected,
                },
            }
        )
        if reference is not None:
            reference_pct = 100.0 * int((inside & reference).sum()) / int(inside.sum())
            predicted_flags.append(impact.affected)
            reference_flags.append(flooded_buildings > 0 or reference_pct >= flag_threshold_pct)
    if metrics is not None and predicted_flags:
        metrics["settlement_flag_agreement"] = sum(
            predicted == expected for predicted, expected in zip(predicted_flags, reference_flags)
        ) / len(predicted_flags)
    settlement_collection = {"type": "FeatureCollection", "features": settlements}
    (output_dir / "settlements.geojson").write_text(json.dumps(settlement_collection), encoding="utf-8")
    fields = list(settlements[0]["properties"]) if settlements else [
        "name", "area_flooded_pct", "flooded_buildings", "people_exposed_est",
        "population_basis", "severity", "affected",
    ]
    with (output_dir / "affected_settlements.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(item["properties"] for item in settlements)

    area_ha = float(flood.sum() * resolution * resolution / 10_000)
    people = sum(int(item["properties"]["people_exposed_est"]) for item in settlements)
    affected = sum(bool(item["properties"]["affected"]) for item in settlements)
    bounds = transform_bounds(crs, "EPSG:4326", *rasterio.transform.array_bounds(grid[0], grid[1], transform))
    summary = {
        "sample_data": False,
        "label": "REAL DATA",
        "aoi_name": aoi_geojson.get("properties", {}).get("name", "Selected area"),
        "pre_date": pre_date.isoformat(),
        "post_date": post_date.isoformat(),
        "sensor_used": "Sentinel-1 RTC (SAR fallback)" if selected_sensor == "sar" else "Sentinel-2 L2A",
        "sar_fallback_used": selected_sensor == "sar" and sensor == "auto",
        "cloud_cover_pct": cloud_cover,
        "flooded_area_ha": round(area_ha, 2),
        "settlements_total": len(settlements),
        "settlements_affected": affected,
        "people_exposed_est": people,
        "population_basis": f"Estimate: flooded buildings x {DEFAULTS['persons_per_building']} persons/building; OSM coverage may be incomplete.",
        "metrics": metrics,
        "metrics_message": "No reference mask provided" if metrics is None else None,
        "bounds": [[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
        "crs": crs,
        "resolution_m": resolution,
        "scene_ids": pair["scene_ids"],
        "acquisition_dates": pair["acquisition_dates"],
        "osm_features": int(len(osm)),
        "osm_buildings": int(buildings.geometry.notna().sum()),
        "settlement_area_source": (
            "OpenStreetMap residential landuse" if uses_landuse_fallback else "OpenStreetMap place features"
        ),
        "classification_note": "Computed from selected Planetary Computer satellite scenes; not independently validated.",
        "imagery_note": (
            "Pre/post map layers and GeoTIFF downloads are Sentinel-2 true-color composites (B04/B03/B02)."
            if selected_sensor == "optical"
            else "Sentinel-1 VV detections are shown; no optical RGB imagery was available for this SAR run."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_html_map(output_dir / "flood_map.html", flood_collection, settlement_collection, summary)
    _write_pdf(output_dir / "report.pdf", summary)
    progress(8, 95, "Wrote GeoTIFF, GeoJSON, CSV, HTML map, and PDF report.")
    return {
        "summary": summary,
        "flood_geojson": flood_collection,
        "settlements_geojson": settlement_collection,
        "metrics": metrics,
        "files": {
            "geotiff": "flood_extent.tif",
            "geojson": "flood_polygons.geojson",
            "csv": "affected_settlements.csv",
            "html": "flood_map.html",
            "pdf": "report.pdf",
        },
        "output_dir": str(output_dir),
    }


def run_uploaded_pair(
    pre_path: Path,
    post_path: Path,
    pre_date: date,
    post_date: date,
    output_dir: Path,
    progress,
    green_band: int = 2,
    nir_band: int = 4,
    swir_band: int = 5,
    red_band: int = 3,
    blue_band: int = 1,
    index: str = "both",
    min_area_ha: float = DEFAULTS["min_area_ha"],
) -> dict:
    """Analyze two aligned, projected Sentinel-2 GeoTIFF band stacks."""
    from pyproj import CRS
    from shapely.geometry import box

    if post_date <= pre_date:
        raise ValueError("Post-flood date must be later than pre-flood date.")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(1, 8, "Validating uploaded GeoTIFF grids and spectral bands.")
    with rasterio.open(pre_path) as pre_ds, rasterio.open(post_path) as post_ds:
        if pre_ds.crs is None or post_ds.crs is None or not CRS.from_user_input(pre_ds.crs).is_projected:
            raise ValueError("Both uploads must have a projected coordinate reference system (CRS).")
        if (
            pre_ds.crs != post_ds.crs
            or pre_ds.transform != post_ds.transform
            or (pre_ds.width, pre_ds.height) != (post_ds.width, post_ds.height)
        ):
            raise ValueError("Pre- and post-flood GeoTIFFs must use the same CRS, transform, and pixel dimensions.")
        required_band = max(green_band, nir_band, swir_band, red_band, blue_band)
        if min(green_band, nir_band, swir_band, red_band, blue_band) < 1:
            raise ValueError("Band numbers start at 1.")
        if pre_ds.count < required_band or post_ds.count < required_band:
            raise ValueError(f"Both GeoTIFFs must contain at least {required_band} bands for the selected mapping.")
        if pre_ds.width * pre_ds.height > 25_000_000:
            raise ValueError("Each uploaded raster is limited to 25 million pixels.")

        selected = [green_band, nir_band, swir_band]
        pre_bands = pre_ds.read(selected).astype(np.float32)
        post_bands = post_ds.read(selected).astype(np.float32)
        pre_valid = np.all(pre_ds.read_masks(selected) > 0, axis=0) & np.all(np.isfinite(pre_bands), axis=0)
        post_valid = np.all(post_ds.read_masks(selected) > 0, axis=0) & np.all(np.isfinite(post_bands), axis=0)
        valid = pre_valid & post_valid
        if not valid.any():
            raise ValueError("The uploaded images do not contain any shared valid pixels.")
        pre_water, _ = classify_optical_water(*pre_bands, valid=pre_valid, agreement=index)
        post_water, _ = classify_optical_water(*post_bands, valid=post_valid, agreement=index)
        flood = detect_new_flood(pre_water, post_water) & valid

        transform, crs = post_ds.transform, post_ds.crs
        crs_info = CRS.from_user_input(crs)
        unit_factor = crs_info.axis_info[0].unit_conversion_factor
        pixel_area_m2 = abs(transform.a * transform.e) * unit_factor**2
        min_pixels = max(1, int(np.ceil(min_area_ha * 10_000 / pixel_area_m2)))
        flood = clean_flood(flood, min_object_pixels=max(DEFAULTS["min_object_pixels"], min_pixels))
        grid = flood.shape
        rgb_bands = [red_band, green_band, blue_band]
        pre_rgb = pre_ds.read(rgb_bands)
        post_rgb = post_ds.read(rgb_bands)
        profile = post_ds.profile.copy()
        profile.update(count=1, dtype="uint8", compress="deflate", nodata=0)
        with rasterio.open(output_dir / "flood_extent.tif", "w", **profile) as destination:
            destination.write(flood.astype(np.uint8), 1)
        for filename, mask in (("water_pre.tif", pre_water), ("water_post.tif", post_water)):
            with rasterio.open(output_dir / filename, "w", **profile) as destination:
                destination.write(mask.astype(np.uint8), 1)
        rgb_profile = post_ds.profile.copy()
        rgb_profile.update(count=3, compress="deflate")
        for filename, rgb in (("pre_rgb.tif", pre_rgb), ("post_rgb.tif", post_rgb)):
            with rasterio.open(output_dir / filename, "w", **rgb_profile) as destination:
                destination.write(rgb)

    progress(2, 38, "Classified water and isolated newly flooded pixels.")
    flood_features = mask_to_features(flood, transform, crs, min_area_ha=min_area_ha)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    for feature in flood_features:
        feature["geometry"] = mapping(transform_geometry(to_wgs84.transform, shape(feature["geometry"])))
    flood_collection = {"type": "FeatureCollection", "features": flood_features}

    west, south, east, north = transform_bounds(crs, "EPSG:4326", *rasterio.transform.array_bounds(*grid, transform))
    aoi = box(west, south, east, north)
    settlements = []
    osm_features = 0
    progress(3, 50, "Preparing the OpenStreetMap impact layer.")
    try:
        osm_key = hashlib.sha256(aoi.wkb).hexdigest()[:16]
        osm = fetch_osm_settlements(aoi, str(output_dir.parent / "osm_cache" / f"osm_{osm_key}.geojson"))
        osm_features = len(osm)
        if osm.crs is not None:
            osm = osm.to_crs(crs)
            places = osm[osm["place"].notna()] if "place" in osm else osm.iloc[0:0]
            if places.empty and "landuse" in osm:
                places = osm[osm["landuse"].isin(["residential", "village_green"])]
            buildings = osm[osm["building"].notna()] if "building" in osm else osm.iloc[0:0]
            for number, (_, place) in enumerate(places.iterrows(), start=1):
                polygon = place.geometry
                if polygon is None or polygon.is_empty:
                    continue
                inside = geometry_mask([mapping(polygon)], out_shape=grid, transform=transform, invert=True)
                if not inside.any():
                    continue
                area_pct = 100 * int((inside & flood).sum()) / int(inside.sum())
                flooded_buildings = 0
                for building in buildings.geometry:
                    if building is None or not polygon.intersects(building):
                        continue
                    row, col = rowcol(transform, *building.centroid.coords[0])
                    if 0 <= row < grid[0] and 0 <= col < grid[1] and flood[row, col]:
                        flooded_buildings += 1
                name = place.get("name")
                if not isinstance(name, str) or not name.strip():
                    name = f"Settlement area {number}"
                impact = assess_settlement(name, area_pct, flooded_buildings, DEFAULTS["persons_per_building"])
                settlements.append({
                    "type": "Feature",
                    "geometry": mapping(transform_geometry(to_wgs84.transform, polygon)),
                    "properties": {
                        "name": name,
                        "area_flooded_pct": round(area_pct, 2),
                        "flooded_buildings": flooded_buildings,
                        "people_exposed_est": impact.people_exposed_est,
                        "severity": impact.severity,
                        "affected": impact.affected,
                    },
                })
    except Exception:
        osm_features = 0
    settlement_collection = {"type": "FeatureCollection", "features": settlements}
    (output_dir / "flood_polygons.geojson").write_text(json.dumps(flood_collection), encoding="utf-8")
    (output_dir / "settlements.geojson").write_text(json.dumps(settlement_collection), encoding="utf-8")
    fields = ["name", "area_flooded_pct", "flooded_buildings", "people_exposed_est", "severity", "affected"]
    with (output_dir / "affected_settlements.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(item["properties"] for item in settlements)

    resolution_m = int(round(np.sqrt(pixel_area_m2)))
    summary = {
        "sample_data": False,
        "label": "UPLOADED DATA",
        "aoi_name": "Uploaded GeoTIFF footprint",
        "pre_date": pre_date.isoformat(),
        "post_date": post_date.isoformat(),
        "sensor_used": "Uploaded Sentinel-2 GeoTIFF pair",
        "cloud_cover_pct": None,
        "flooded_area_ha": round(float(flood.sum() * pixel_area_m2 / 10_000), 2),
        "settlements_total": len(settlements),
        "settlements_affected": sum(item["properties"]["affected"] for item in settlements),
        "people_exposed_est": sum(item["properties"]["people_exposed_est"] for item in settlements),
        "population_basis": "Estimated from OpenStreetMap buildings when available; coverage may be incomplete.",
        "metrics": None,
        "metrics_message": "No reference mask provided",
        "bounds": [[west, south], [east, north]],
        "resolution_m": resolution_m,
        "osm_features": osm_features,
        "osm_buildings": 0,
        "classification_note": "NDWI/MNDWI change from uploaded, georeferenced optical rasters; not independently validated.",
        "imagery_note": "Uploaded true-color bands are shown with the new-flood overlay.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_html_map(output_dir / "flood_map.html", flood_collection, settlement_collection, summary)
    _write_pdf(output_dir / "report.pdf", summary)
    progress(8, 95, "Wrote GeoTIFF, GeoJSON, CSV, HTML map, and PDF report.")
    return {
        "summary": summary,
        "flood_geojson": flood_collection,
        "settlements_geojson": settlement_collection,
        "metrics": None,
        "files": {"geotiff": "flood_extent.tif", "geojson": "flood_polygons.geojson", "csv": "affected_settlements.csv", "html": "flood_map.html", "pdf": "report.pdf"},
        "output_dir": str(output_dir),
    }


def _write_html_map(path: Path, flood: dict, settlements: dict, summary: dict) -> None:
    import folium

    center = [
        (summary["bounds"][0][1] + summary["bounds"][1][1]) / 2,
        (summary["bounds"][0][0] + summary["bounds"][1][0]) / 2,
    ]
    map_view = folium.Map(
        location=center,
        zoom_start=11,
        tiles="OpenStreetMap",
        attr="© OpenStreetMap contributors",
    )
    folium.GeoJson(flood, name="New flood", style_function=lambda _: {"color": "#c77800", "fillColor": "#f18f01", "fillOpacity": 0.55}).add_to(map_view)
    if settlements.get("features"):
        folium.GeoJson(
            settlements,
            name="Settlements",
            tooltip=folium.GeoJsonTooltip(fields=["name", "severity", "people_exposed_est"]),
        ).add_to(map_view)
    folium.LayerControl().add_to(map_view)
    map_view.save(path)


def _write_pdf(path: Path, summary: dict) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.setTitle("FloodSight Terai flood impact report")
    pdf.drawString(64, 742, "FloodSight Terai — flood impact assessment")
    pdf.drawString(64, 718, f"AOI: {summary['aoi_name']}")
    pdf.drawString(64, 696, f"Dates: {summary['pre_date']} to {summary['post_date']}")
    pdf.drawString(64, 674, f"Sensor: {summary['sensor_used']}")
    pdf.drawString(64, 652, f"Flood extent: {summary['flooded_area_ha']} hectares")
    pdf.drawString(64, 630, f"Settlements affected: {summary['settlements_affected']} / {summary['settlements_total']}")
    pdf.drawString(64, 608, f"People exposed (estimate): {summary['people_exposed_est']}")
    pdf.drawString(64, 578, summary["population_basis"][:105])
    pdf.drawString(64, 548, "No reference mask provided — accuracy metrics are not available.")
    pdf.save()
