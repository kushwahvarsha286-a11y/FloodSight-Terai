"""Settlement exposure estimation and severity flags."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SettlementImpact:
    """Impact metrics for one settlement."""

    name: str
    area_flooded_pct: float
    flooded_buildings: int
    people_exposed_est: int
    severity: str
    affected: bool


def assess_settlement(
    name: str,
    area_flooded_pct: float,
    flooded_buildings: int,
    persons_per_building: int = 5,
    flag_threshold_pct: float = 10.0,
) -> SettlementImpact:
    """Apply the documented building-or-area flag and area-based severity rules."""
    pct = min(100.0, max(0.0, float(area_flooded_pct)))
    count = max(0, int(flooded_buildings))
    affected = count > 0 or pct >= flag_threshold_pct
    severity = (
        "Not affected"
        if not affected
        else "High"
        if pct >= 50.0
        else "Medium"
        if pct >= 20.0
        else "Low"
    )
    return SettlementImpact(
        name=name,
        area_flooded_pct=pct,
        flooded_buildings=count,
        people_exposed_est=count * persons_per_building,
        severity=severity,
        affected=affected,
    )


def fetch_osm_settlements(aoi, cache_path: str):
    """Fetch OSM settlement/building features, retaining a GeoJSON cache for offline use."""
    import geopandas as gpd
    import osmnx as ox
    from pathlib import Path

    cache = Path(cache_path)
    if cache.exists():
        return gpd.read_file(cache)
    try:
        features = ox.features_from_polygon(
            aoi,
            tags={
                "place": ["city", "town", "village", "hamlet"],
                "building": True,
                "landuse": ["residential", "village_green"],
            },
        )
    except Exception as exc:
        raise RuntimeError(
            "OpenStreetMap lookup failed and no cached settlement data is available: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    cache.parent.mkdir(parents=True, exist_ok=True)
    features.to_file(cache, driver="GeoJSON")
    return features
