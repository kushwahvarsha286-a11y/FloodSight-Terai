import numpy as np
import pytest

from pipeline.change import detect_new_flood
from pipeline.data_fetch import _search_period
from pipeline.settlements import assess_settlement
from pipeline.validate import calculate_metrics
from pipeline.validate import geojson_reference_mask
from pipeline.water import classify_optical_water, normalized_difference
from rasterio.transform import from_origin


def test_normalized_difference_and_zero_denominator():
    assert normalized_difference(np.array([3, 0]), np.array([1, 0])).tolist() == [0.5, 0.0]


def test_optical_indices_detect_water_agreement():
    green = np.array([[0.7, 0.1], [0.8, 0.15]], dtype=np.float32)
    nir = np.array([[0.1, 0.7], [0.15, 0.8]], dtype=np.float32)
    swir = nir.copy()
    water, thresholds = classify_optical_water(green, nir, swir, agreement="both")
    assert water[0, 0] and water[1, 0]
    assert not water[0, 1] and not water[1, 1]
    assert set(thresholds) == {"ndwi", "mndwi"}


def test_invalid_water_index_and_empty_mask():
    with pytest.raises(ValueError):
        classify_optical_water(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), agreement="bad")
    with pytest.raises(ValueError):
        classify_optical_water(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), np.zeros((2, 2), bool))


def test_new_flood_excludes_preexisting_and_permanent_water():
    pre = np.array([[1, 0], [0, 0]], dtype=bool)
    post = np.array([[1, 1], [0, 1]], dtype=bool)
    permanent = np.array([[0, 0], [0, 1]], dtype=bool)
    assert detect_new_flood(pre, post, permanent).tolist() == [[False, True], [False, False]]


@pytest.mark.parametrize(
    ("pct", "buildings", "affected", "severity"),
    [(0, 0, False, "Not affected"), (9, 1, True, "Low"), (20, 0, True, "Medium"), (50, 0, True, "High")],
)
def test_settlement_flag_rule(pct, buildings, affected, severity):
    impact = assess_settlement("Sample", pct, buildings)
    assert (impact.affected, impact.severity) == (affected, severity)
    assert impact.people_exposed_est == buildings * 5


def test_metrics_are_calculated_from_masks():
    expected = np.array([[0, 1], [1, 0]], dtype=bool)
    actual = np.array([[0, 1], [0, 0]], dtype=bool)
    metrics = calculate_metrics(actual, expected)
    assert metrics["iou"] == pytest.approx(0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]


def test_geojson_reference_is_rasterized_to_analysis_grid():
    reference = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [0.002, 0], [0.002, 0.002], [0, 0.002], [0, 0]]],
        },
        "properties": {},
    }
    mask = geojson_reference_mask(reference, (2, 2), from_origin(0, 0.002, 0.001, 0.001), "EPSG:4326")
    assert mask.tolist() == [[True, True], [True, True]]
def test_uploaded_pair_detects_new_water_and_writes_outputs(tmp_path, monkeypatch):
    import rasterio
    from datetime import date
    from rasterio.transform import from_origin

    from pipeline import analysis

    pre_path, post_path = tmp_path / "pre.tif", tmp_path / "post.tif"
    profile = {
        "driver": "GTiff",
        "height": 20,
        "width": 20,
        "count": 5,
        "dtype": "float32",
        "crs": "EPSG:32645",
        "transform": from_origin(500000, 3000000, 10, 10),
    }
    pre = np.zeros((5, 20, 20), dtype=np.float32)
    pre[1], pre[3], pre[4] = 0.1, 0.8, 0.8
    post = pre.copy()
    post[1, 5:15, 5:15], post[3, 5:15, 5:15], post[4, 5:15, 5:15] = 0.8, 0.1, 0.1
    for path, data in ((pre_path, pre), (post_path, post)):
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(data)

    def fail_osm(*_args):
        raise RuntimeError("offline")

    monkeypatch.setattr(analysis, "fetch_osm_settlements", fail_osm)
    result = analysis.run_uploaded_pair(
        pre_path,
        post_path,
        date(2024, 7, 1),
        date(2024, 7, 6),
        tmp_path / "results",
        lambda *_: None,
        min_area_ha=0,
    )

    assert result["summary"]["flooded_area_ha"] > 0
    assert result["summary"]["sensor_used"] == "Uploaded Sentinel-2 GeoTIFF pair"
    assert (tmp_path / "results" / "flood_extent.tif").exists()
    assert (tmp_path / "results" / "flood_map.html").exists()
    assert (tmp_path / "results" / "report.pdf").exists()


def test_satellite_date_windows_keep_pre_and_post_acquisitions_ordered():
    from datetime import date

    target = date(2024, 7, 6)
    assert _search_period(target, 15, "before") == "2024-06-21T00:00:00Z/2024-07-06T23:59:59Z"
    assert _search_period(target, 15, "after") == "2024-07-06T00:00:00Z/2024-07-21T23:59:59Z"
    assert _search_period(target, 15, "nearest") == "2024-06-21T00:00:00Z/2024-07-21T23:59:59Z"
