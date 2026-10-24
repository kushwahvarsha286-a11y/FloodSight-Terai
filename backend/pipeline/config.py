"""Central defaults for the FloodSight analysis pipeline."""

DEFAULTS = {
    "cloud_limit_pct": 40.0,
    "date_window_days": 15,
    "min_area_ha": 2.0,
    "flag_threshold_pct": 10.0,
    "persons_per_building": 5,
    "water_index": "both",
    "slope_limit_degrees": 5.0,
    "speckle_size": 3,
    "min_object_pixels": 2,
    "permanent_water_exclusion": True,
}

PIPELINE_STEPS = [
    "Input",
    "Preprocess",
    "Water index",
    "Change detect",
    "Clean",
    "Vectorize",
    "Settlements",
    "Outputs",
]
