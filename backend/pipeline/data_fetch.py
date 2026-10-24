"""Planetary Computer STAC scene discovery and Sentinel-2 loading."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import numpy as np


def _search_period(
    target: date, window_days: int, direction: Literal["before", "after", "nearest"]
) -> str:
    """Return a STAC date interval that preserves the requested pre/post ordering."""
    start, end = target - timedelta(days=window_days), target + timedelta(days=window_days)
    if direction == "before":
        end = target
    elif direction == "after":
        start = target
    elif direction != "nearest":
        raise ValueError(f"Unsupported acquisition direction: {direction}")
    return f"{start}T00:00:00Z/{end}T23:59:59Z"


def search_least_cloudy(
    collection: str,
    aoi_geojson: dict,
    target: date,
    window_days: int = 15,
    direction: Literal["before", "after", "nearest"] = "nearest",
):
    """Find the least-cloudy Planetary Computer item in the target's date window."""
    import planetary_computer
    from pystac_client import Client

    client = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = client.search(
        collections=[collection],
        intersects=aoi_geojson,
        datetime=_search_period(target, window_days, direction),
        query={"eo:cloud_cover": {"lt": 100}},
        max_items=100,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No {collection} scenes found within +/-{window_days} days of {target}")
    return min(
        items,
        key=lambda item: (
            item.properties.get("eo:cloud_cover", 100),
            abs((item.datetime.date() - target).days),
        ),
    )


def load_sentinel2(item, aoi_geojson: dict, resolution: int = 10) -> tuple[dict, float]:
    """Load Sentinel-2 index bands, SCL, and true-color bands clipped to the AOI."""
    import odc.stac
    from shapely.geometry import shape

    aoi = shape(aoi_geojson)
    data = odc.stac.load(
        [item],
        bands=["B02", "B03", "B04", "B08", "B11", "SCL"],
        crs="EPSG:32645",
        resolution=resolution,
        geopolygon=aoi,
        resampling={"SCL": "nearest"},
    )
    arrays = {
        "blue": np.asarray(data["B02"].squeeze().compute()),
        "green": np.asarray(data["B03"].squeeze().compute()),
        "red": np.asarray(data["B04"].squeeze().compute()),
        "nir": np.asarray(data["B08"].squeeze().compute()),
        "swir": np.asarray(data["B11"].squeeze().compute()),
        "scl": np.asarray(data["SCL"].squeeze().compute()),
        "transform": data.odc.geobox.transform,
        "crs": str(data.odc.geobox.crs),
    }
    cloud = float(item.properties.get("eo:cloud_cover", 100.0))
    return arrays, cloud


def fetch_optical_pair(aoi_geojson: dict, pre_date: date, post_date: date) -> dict:
    """Select and load the least-cloudy Sentinel-2 scene for each date."""
    before = search_least_cloudy(
        "sentinel-2-l2a", aoi_geojson, pre_date, direction="before"
    )
    after = search_least_cloudy(
        "sentinel-2-l2a", aoi_geojson, post_date, direction="after"
    )
    pre, pre_cloud = load_sentinel2(before, aoi_geojson)
    post, post_cloud = load_sentinel2(after, aoi_geojson)
    return {
        "pre": pre,
        "post": post,
        "cloud_cover_pct": max(pre_cloud, post_cloud),
        "scene_ids": [before.id, after.id],
        "acquisition_dates": [
            before.datetime.date().isoformat(),
            after.datetime.date().isoformat(),
        ],
    }


def fetch_sar_pair(aoi_geojson: dict, pre_date: date, post_date: date) -> dict:
    """Load least-cloud-independent Sentinel-1 RTC VV acquisitions for both dates."""
    import odc.stac
    import planetary_computer
    from pystac_client import Client
    from shapely.geometry import shape

    client = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = []
    for target, direction in ((pre_date, "before"), (post_date, "after")):
        search = client.search(
            collections=["sentinel-1-rtc"],
            intersects=aoi_geojson,
            datetime=_search_period(target, 15, direction),
            max_items=100,
        )
        candidates = list(search.items())
        if not candidates:
            raise RuntimeError(f"No Sentinel-1 RTC scenes found within +/-15 days of {target}")
        items.append(min(candidates, key=lambda item: abs((item.datetime.date() - target).days)))

    aoi = shape(aoi_geojson)
    loaded = []
    for item in items:
        data = odc.stac.load(
            [item],
            bands=["vv"],
            crs="EPSG:32645",
            resolution=10,
            geopolygon=aoi,
        )
        values = np.asarray(data["vv"].squeeze().compute(), dtype=np.float32)
        # Planetary Computer RTC VV assets are linear power; convert to dB for Otsu.
        vv_db = 10.0 * np.log10(np.maximum(values, 1e-8))
        loaded.append(
            {
                "vv_db": vv_db,
                "transform": data.odc.geobox.transform,
                "crs": str(data.odc.geobox.crs),
            }
        )
    return {
        "pre": loaded[0],
        "post": loaded[1],
        "scene_ids": [item.id for item in items],
        "acquisition_dates": [item.datetime.date().isoformat() for item in items],
    }


def fetch_dem(aoi_geojson: dict, crs: str, resolution: int, shape_yx: tuple[int, int]) -> np.ndarray:
    """Load Copernicus DEM and calculate slope in degrees on the analysis grid."""
    import odc.stac
    import planetary_computer
    from pystac_client import Client
    from scipy.ndimage import zoom
    from shapely.geometry import shape

    client = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = client.search(
        collections=["cop-dem-glo-30"],
        intersects=aoi_geojson,
        max_items=20,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No Copernicus DEM tiles found for the selected AOI")
    data = odc.stac.load(
        items,
        bands=["data"],
        crs=crs,
        resolution=resolution,
        geopolygon=shape(aoi_geojson),
    )
    elevation = np.asarray(data["data"].squeeze().compute(), dtype=np.float32)
    if elevation.shape != shape_yx:
        elevation = zoom(elevation, (shape_yx[0] / elevation.shape[0], shape_yx[1] / elevation.shape[1]))
        elevation = elevation[: shape_yx[0], : shape_yx[1]]
    resolution_xy = data.odc.geobox.resolution
    xres, yres = resolution_xy.x, resolution_xy.y
    dy, dx = np.gradient(elevation, abs(yres), abs(xres))
    return np.degrees(np.arctan(np.hypot(dx, dy)))
