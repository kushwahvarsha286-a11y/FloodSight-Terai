"""Command-line entry point for running an AOI flood analysis."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="FloodSight Terai flood extent analysis")
    parser.add_argument("--aoi", required=True, help="AOI GeoJSON polygon")
    parser.add_argument("--pre", required=True, type=date.fromisoformat, help="Pre-flood date")
    parser.add_argument("--post", required=True, type=date.fromisoformat, help="Post-flood date")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--sensor", choices=["auto", "optical", "sar"], default="auto")
    parser.add_argument("--index", choices=["mndwi", "ndwi", "both"], default="both")
    parser.add_argument("--min-area-ha", type=float, default=2.0)
    parser.add_argument("--flag-threshold-pct", type=float, default=10.0)
    parser.add_argument("--reference", help="Optional reference mask GeoTIFF or GeoJSON")
    args = parser.parse_args()
    aoi = json.loads(Path(args.aoi).read_text(encoding="utf-8"))
    from .analysis import run_real_analysis

    def report_progress(step: int, percent: int, message: str) -> None:
        print(f"[{percent:3d}%] Step {step}/8 — {message}")

    result = run_real_analysis(
        aoi,
        args.pre,
        args.post,
        args.sensor,
        args.index,
        args.min_area_ha,
        args.flag_threshold_pct,
        Path(args.out),
        report_progress,
        json.loads(Path(args.reference).read_text(encoding="utf-8"))
        if args.reference and args.reference.lower().endswith((".json", ".geojson"))
        else args.reference,
    )
    print(f"Analysis complete: {result['summary']['flooded_area_ha']} ha flooded.")
    print(f"Output files: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
