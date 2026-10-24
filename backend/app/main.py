"""FastAPI service for FloodSight analysis jobs and offline demonstration results."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator
from rasterio.transform import array_bounds
from starlette.concurrency import run_in_threadpool

from pipeline.config import DEFAULTS, PIPELINE_STEPS
from pipeline.sample import BOUNDS, SAMPLE_DIR, create_sample_data
from pipeline.settlements import assess_settlement
from pipeline.water import classify_optical_water
from pipeline.change import detect_new_flood

app = FastAPI(title="FloodSight Terai API", version="1.0.0")
REAL_SAMPLE_DIR = SAMPLE_DIR / "real_sentinel"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    """Validated analysis request."""

    aoi_geojson: dict
    pre_date: date
    post_date: date
    sensor: Literal["auto", "optical", "sar"] = "auto"
    index: Literal["mndwi", "ndwi", "both"] = "both"
    min_area_ha: float = Field(default=DEFAULTS["min_area_ha"], ge=0)
    flag_threshold_pct: float = Field(default=DEFAULTS["flag_threshold_pct"], ge=0, le=100)
    reference_mask_geojson: dict | None = None

    @field_validator("post_date")
    @classmethod
    def post_after_pre(cls, value: date, info):
        pre = info.data.get("pre_date")
        if pre is not None and value <= pre:
            raise ValueError("post_date must be later than pre_date")
        return value


jobs: dict[str, dict] = {}
presets = [
    {"id": "koshi", "name": "Koshi Terai", "basin": "Koshi", "aoi_geojson": json.loads((SAMPLE_DIR / "aoi.geojson").read_text()) if (SAMPLE_DIR / "aoi.geojson").exists() else None},
    {
        "id": "bagmati",
        "name": "Bagmati Terai",
        "basin": "Bagmati",
        "aoi_geojson": {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[85.15, 26.7], [85.35, 26.7], [85.35, 27.0], [85.15, 27.0], [85.15, 26.7]]],
            },
            "properties": {"name": "Bagmati Terai (approximate preset)"},
        },
    },
    {
        "id": "gandaki",
        "name": "Gandaki Terai",
        "basin": "Gandaki",
        "aoi_geojson": {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[83.45, 27.3], [83.7, 27.3], [83.7, 27.55], [83.45, 27.55], [83.45, 27.3]]],
            },
            "properties": {"name": "Gandaki Terai (approximate preset)"},
        },
    },
]


def ensure_sample() -> None:
    if not (SAMPLE_DIR / "summary.json").exists():
        create_sample_data()


@app.on_event("startup")
async def startup() -> None:
    await run_in_threadpool(ensure_sample)


async def run_uploaded_analysis(job_id: str, pre_path: Path, post_path: Path, options: dict) -> None:
    job = jobs[job_id]
    job.update(status="running", current_step=1, progress=3)
    job["logs"].append("Uploaded GeoTIFF pair accepted; validating raster grids.")
    try:
        from pipeline.analysis import run_uploaded_pair

        def progress(step: int, percent: int, message: str) -> None:
            job.update(current_step=step, progress=percent, step=PIPELINE_STEPS[min(step - 1, 7)])
            job["logs"].append(message)

        output_dir = Path(__file__).resolve().parents[2] / "results" / job_id
        result = await run_in_threadpool(
            run_uploaded_pair,
            pre_path,
            post_path,
            options["pre_date"],
            options["post_date"],
            output_dir,
            progress,
            options["green_band"],
            options["nir_band"],
            options["swir_band"],
            options["red_band"],
            options["blue_band"],
            options["index"],
            options["min_area_ha"],
        )
        result["files"] = {
            key: f"/api/results/{job_id}/download/{filename}"
            for key, filename in {
                "geotiff": "flood_extent.tif",
                "geojson": "flood_polygons.geojson",
                "csv": "affected_settlements.csv",
                "html": "flood_map.html",
                "pdf": "report.pdf",
                "pre_image": "pre_rgb.tif",
                "post_image": "post_rgb.tif",
            }.items()
        }
        result["output_dir"] = str(output_dir)
        job.update(status="completed", current_step=8, progress=100, result=result, output_dir=str(output_dir))
        job["logs"].append("Uploaded-image comparison and result files are ready.")
    except Exception as exc:
        job.update(status="failed", error=str(exc))
        job["logs"].append(f"Uploaded-image analysis failed: {exc}")
    finally:
        pre_path.unlink(missing_ok=True)
        post_path.unlink(missing_ok=True)


async def _save_geotiff_upload(upload: UploadFile, destination: Path) -> None:
    if Path(upload.filename or "").suffix.lower() not in {".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Upload georeferenced .tif or .tiff files only.")
    total = 0
    with destination.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > 200 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Each GeoTIFF upload must be 200 MB or smaller.")
            target.write(chunk)
    if not total:
        raise HTTPException(status_code=400, detail="Uploaded GeoTIFF is empty.")


@app.post("/api/analyze-upload", status_code=202)
async def analyze_upload(
    background_tasks: BackgroundTasks,
    pre_image: UploadFile = File(...),
    post_image: UploadFile = File(...),
    pre_date: date = Form(...),
    post_date: date = Form(...),
    index: Literal["mndwi", "ndwi", "both"] = Form("both"),
    min_area_ha: float = Form(DEFAULTS["min_area_ha"], ge=0),
    green_band: int = Form(2, ge=1),
    nir_band: int = Form(4, ge=1),
    swir_band: int = Form(5, ge=1),
    red_band: int = Form(3, ge=1),
    blue_band: int = Form(1, ge=1),
):
    if post_date <= pre_date:
        raise HTTPException(status_code=422, detail="Post-flood date must be later than pre-flood date.")
    job_id = str(uuid.uuid4())
    input_dir = Path(__file__).resolve().parents[2] / "results" / job_id / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    pre_path, post_path = input_dir / "pre_upload.tif", input_dir / "post_upload.tif"
    try:
        await _save_geotiff_upload(pre_image, pre_path)
        await _save_geotiff_upload(post_image, post_path)
    except Exception:
        pre_path.unlink(missing_ok=True)
        post_path.unlink(missing_ok=True)
        raise
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "current_step": 0,
        "step": "Input",
        "progress": 0,
        "logs": ["Uploaded raster pair queued for analysis."],
        "error": None,
    }
    options = {
        "pre_date": pre_date,
        "post_date": post_date,
        "index": index,
        "min_area_ha": min_area_ha,
        "green_band": green_band,
        "nir_band": nir_band,
        "swir_band": swir_band,
        "red_band": red_band,
        "blue_band": blue_band,
    }
    background_tasks.add_task(run_uploaded_analysis, job_id, pre_path, post_path, options)
    return {"job_id": job_id}


async def run_analysis(job_id: str, request: AnalyzeRequest) -> None:
    """Run real imagery processing in a worker thread and expose progress/errors."""
    job = jobs[job_id]
    job.update(status="running", current_step=1, progress=3)
    job["logs"].append("AOI and dates validated.")
    try:
        from pipeline.analysis import run_real_analysis

        def progress(step: int, percent: int, message: str) -> None:
            job.update(current_step=step, progress=percent, step=PIPELINE_STEPS[min(step - 1, 7)])
            job["logs"].append(message)

        result = await run_in_threadpool(
            run_real_analysis,
            request.aoi_geojson,
            request.pre_date,
            request.post_date,
            request.sensor,
            request.index,
            request.min_area_ha,
            request.flag_threshold_pct,
            Path(__file__).resolve().parents[2] / "results" / job_id,
            progress,
            request.reference_mask_geojson,
        )
        result["files"] = {
            key: f"/api/results/{job_id}/download/{filename}"
            for key, filename in {
                "geotiff": "flood_extent.tif",
                "geojson": "flood_polygons.geojson",
                "csv": "affected_settlements.csv",
                "html": "flood_map.html",
                "pdf": "report.pdf",
            }.items()
        }
        if result["summary"]["sensor_used"] == "Sentinel-2 L2A":
            result["files"].update(
                pre_image=f"/api/results/{job_id}/download/pre_rgb.tif",
                post_image=f"/api/results/{job_id}/download/post_rgb.tif",
            )
        job.update(
            status="completed",
            current_step=8,
            progress=100,
            result=result,
            output_dir=result.pop("output_dir", None),
        )
        job["logs"].append("All requested output files were generated.")
    except Exception as exc:
        job.update(status="failed", error=str(exc))
        job["logs"].append(f"Analysis failed: {exc}")
    job = jobs[job_id]
    job.update(status="running", current_step=1, progress=3)
    job["logs"].append("AOI and dates validated.")
    try:
        from pipeline.analysis import run_real_analysis

        def progress(step: int, percent: int, message: str) -> None:
            job.update(current_step=step, progress=percent, step=PIPELINE_STEPS[min(step - 1, 7)])
            job["logs"].append(message)

        result = await run_in_threadpool(
            run_real_analysis,
            request.aoi_geojson,
            request.pre_date,
            request.post_date,
            request.sensor,
            request.index,
            request.min_area_ha,
            request.flag_threshold_pct,
            Path(__file__).resolve().parents[2] / "results" / job_id,
            progress,
            request.reference_mask_geojson,
        )
        result["files"] = {
            key: f"/api/results/{job_id}/download/{filename}"
            for key, filename in {
                "geotiff": "flood_extent.tif",
                "geojson": "flood_polygons.geojson",
                "csv": "affected_settlements.csv",
                "html": "flood_map.html",
                "pdf": "report.pdf",
            }.items()
        }
        if result["summary"]["sensor_used"] == "Sentinel-2 L2A":
            result["files"].update(
                pre_image=f"/api/results/{job_id}/download/pre_rgb.tif",
                post_image=f"/api/results/{job_id}/download/post_rgb.tif",
            )
        job.update(
            status="completed",
            current_step=8,
            progress=100,
            result=result,
            output_dir=result.pop("output_dir", None),
        )
        job["logs"].append("All requested output files were generated.")
    except Exception as exc:
        job.update(status="failed", error=str(exc))
        job["logs"].append(f"Analysis failed: {exc}")


@app.post("/api/analyze", status_code=202)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "current_step": 0,
        "step": "Input",
        "progress": 0,
        "logs": ["Job accepted; validating analysis input."],
        "error": None,
    }
    background_tasks.add_task(run_analysis, job_id, request)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    item = dict(jobs[job_id])
    if item["status"] == "running":
        index = min(max(item["current_step"] - 1, 0), len(PIPELINE_STEPS) - 1)
        item["step"] = PIPELINE_STEPS[index]
    return item


@app.get("/api/presets")
async def get_presets():
    ensure_sample()
    values = [dict(item) for item in presets]
    values[0]["aoi_geojson"] = json.loads((SAMPLE_DIR / "aoi.geojson").read_text(encoding="utf-8"))
    return {"presets": values}


@app.get("/api/demo")
async def get_demo():
    ensure_sample()
    return _read_result()


@app.get("/api/real-demo")
async def get_real_demo():
    if not (REAL_SAMPLE_DIR / "summary.json").exists():
        raise HTTPException(status_code=404, detail="The cached real Sentinel-2 example is not available")
    return _read_real_demo()


def _read_result() -> dict:
    summary = json.loads((SAMPLE_DIR / "summary.json").read_text(encoding="utf-8"))
    floods = json.loads((SAMPLE_DIR / "flood_polygons.geojson").read_text(encoding="utf-8"))
    settlements = json.loads((SAMPLE_DIR / "settlements.geojson").read_text(encoding="utf-8"))
    return {
        "summary": summary,
        "flood_geojson": floods,
        "settlements_geojson": settlements,
        "metrics": None,
        "files": {
            "geotiff": "/api/results/demo/download/flood_extent.tif",
            "geojson": "/api/results/demo/download/flood_polygons.geojson",
            "csv": "/api/results/demo/download/affected_settlements.csv",
            "html": "/api/results/demo/download/flood_map.html",
            "pdf": "/api/results/demo/download/report.pdf",
        },
    }


def _read_real_demo() -> dict:
    summary = json.loads((REAL_SAMPLE_DIR / "summary.json").read_text(encoding="utf-8"))
    floods = json.loads((REAL_SAMPLE_DIR / "flood_polygons.geojson").read_text(encoding="utf-8"))
    settlements = json.loads((REAL_SAMPLE_DIR / "settlements.geojson").read_text(encoding="utf-8"))
    return {
        "summary": summary,
        "flood_geojson": floods,
        "settlements_geojson": settlements,
        "metrics": None,
        "files": {
            key: f"/api/results/real-demo/download/{filename}"
            for key, filename in {
                "geotiff": "flood_extent.tif",
                "geojson": "flood_polygons.geojson",
                "csv": "affected_settlements.csv",
                "html": "flood_map.html",
                "pdf": "report.pdf",
                "pre_image": "pre_rgb.tif",
                "post_image": "post_rgb.tif",
            }.items()
        },
    }


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    if job_id == "demo":
        ensure_sample()
        return _read_result()
    if job_id == "real-demo":
        return await get_real_demo()
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=job.get("error") or "Analysis is not complete")
    return job["result"]


@app.get("/api/results/{job_id}/layers/{layer_name}")
async def get_layer(job_id: str, layer_name: str):
    layer_files = {
        "water_pre": "water_pre.tif",
        "water_post": "water_post.tif",
        "permanent_water": "permanent_water.tif",
        "new_flood": "new_flood.tif",
    }
    if job_id == "real-demo":
        real_layers = {
            "water_pre": "water_pre.tif",
            "water_post": "water_post.tif",
            "new_flood": "flood_extent.tif",
            "pre_rgb": "pre_rgb.tif",
            "post_rgb": "post_rgb.tif",
        }
        if layer_name not in real_layers or not (REAL_SAMPLE_DIR / real_layers[layer_name]).exists():
            raise HTTPException(status_code=404, detail="Requested layer is not available for this example")
        return _render_raster_layer(REAL_SAMPLE_DIR / real_layers[layer_name], layer_name)
    if job_id != "demo":
        job = jobs.get(job_id)
        output_dir = Path(job["output_dir"]) if job and job.get("output_dir") else None
        real_layers = {
            "water_pre": "water_pre.tif",
            "water_post": "water_post.tif",
            "new_flood": "flood_extent.tif",
            "pre_rgb": "pre_rgb.tif",
            "post_rgb": "post_rgb.tif",
        }
        if output_dir is None or layer_name not in real_layers or not (output_dir / real_layers[layer_name]).exists():
            raise HTTPException(status_code=404, detail="Requested layer is not available for this analysis")
        return _render_raster_layer(output_dir / real_layers[layer_name], layer_name)
    if layer_name not in {*layer_files, "pre_rgb", "post_rgb"}:
        raise HTTPException(status_code=404, detail="Unknown layer name")
    ensure_sample()
    if layer_name in layer_files:
        with rasterio.open(SAMPLE_DIR / layer_files[layer_name]) as dataset:
            raster = dataset.read(1)
            bounds = dataset.bounds
        palette = {
            "water_pre": (46, 134, 171),
            "water_post": (46, 134, 171),
            "permanent_water": (46, 134, 171),
            "new_flood": (241, 143, 1),
        }[layer_name]
        rgba = np.zeros((*raster.shape, 4), dtype=np.uint8)
        rgba[raster > 0] = (*palette, 190)
    else:
        prefix = "pre" if layer_name == "pre_rgb" else "post"
        channels = []
        for band in ("green", "nir", "swir"):
            with rasterio.open(SAMPLE_DIR / f"{prefix}_{band}.tif") as dataset:
                channels.append(dataset.read(1))
                bounds = dataset.bounds
        rgb = np.stack([channels[1], channels[0], channels[2]], axis=-1)
        low, high = np.percentile(rgb, [2, 98])
        scaled = np.clip((rgb - low) / max(float(high - low), 1e-6) * 255, 0, 255).astype(np.uint8)
        rgba = np.dstack((scaled, np.full(scaled.shape[:2], 255, dtype=np.uint8)))
    return _png_response(rgba, bounds, "EPSG:4326")


def _render_raster_layer(path: Path, layer_name: str) -> Response:
    import numpy as np
    from PIL import Image
    import io

    with rasterio.open(path) as dataset:
        bounds = dataset.bounds
        if dataset.count == 1:
            pixels = dataset.read(1)
            color = (241, 143, 1) if layer_name == "new_flood" else (46, 134, 171)
            rgba = np.zeros((*pixels.shape, 4), dtype=np.uint8)
            rgba[pixels > 0] = (*color, 190)
        else:
            channels = dataset.read()
            rgb = np.moveaxis(channels, 0, -1)
            low, high = np.percentile(rgb, [2, 98])
            scaled = np.clip((rgb - low) / max(float(high - low), 1e-6) * 255, 0, 255).astype(np.uint8)
            rgba = np.dstack((scaled, np.full(scaled.shape[:2], 255, dtype=np.uint8)))
    return _png_response(rgba, bounds, dataset.crs)


def _png_response(rgba: np.ndarray, bounds, crs="EPSG:4326") -> Response:
    import io
    from PIL import Image
    from rasterio.warp import transform_bounds

    west, south, east, north = bounds
    try:
        west, south, east, north = transform_bounds(crs, "EPSG:4326", west, south, east, north)
    except Exception:
        pass
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"X-Bounds": json.dumps([[west, south], [east, north]])},
    )


@app.get("/api/results/{job_id}/download/{filename}")
async def download_result(job_id: str, filename: str):
    if job_id == "real-demo":
        allowed = {
            "flood_extent.tif",
            "flood_polygons.geojson",
            "affected_settlements.csv",
            "flood_map.html",
            "report.pdf",
            "pre_rgb.tif",
            "post_rgb.tif",
        }
        if filename not in allowed or not (REAL_SAMPLE_DIR / filename).exists():
            raise HTTPException(status_code=404, detail="Download not found")
        return FileResponse(REAL_SAMPLE_DIR / filename, filename=filename)
    if job_id != "demo":
        job = jobs.get(job_id)
        output_dir = Path(job["output_dir"]) if job and job.get("output_dir") else None
        allowed = {
            "flood_extent.tif",
            "flood_polygons.geojson",
            "affected_settlements.csv",
            "flood_map.html",
            "report.pdf",
            "pre_rgb.tif",
            "post_rgb.tif",
        }
        if output_dir is None or filename not in allowed or not (output_dir / filename).exists():
            raise HTTPException(status_code=404, detail="Download not found")
        return FileResponse(output_dir / filename, filename=filename)
    ensure_sample()
    names = {
        "flood_extent.tif": "new_flood.tif",
        "flood_polygons.geojson": "flood_polygons.geojson",
        "affected_settlements.csv": "affected_settlements.csv",
    }
    if filename in names:
        return FileResponse(SAMPLE_DIR / names[filename], filename=filename)
    if filename == "flood_map.html":
        return Response(
            "<!doctype html><title>FloodSight Terai - illustrative sample</title>"
            "<h1>SAMPLE DATA - illustrative</h1><p>Open this result in FloodSight Terai to view map layers.</p>",
            media_type="text/html",
        )
    if filename == "report.pdf":
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        import io

        data = _read_result()["summary"]
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setTitle("FloodSight Terai illustrative sample report")
        pdf.drawString(72, 730, "FloodSight Terai - SAMPLE DATA (illustrative)")
        pdf.drawString(72, 700, f"Flooded area: {data['flooded_area_ha']} ha")
        pdf.drawString(72, 680, f"Affected settlements: {data['settlements_affected']}")
        pdf.drawString(72, 660, f"People exposed (estimate): {data['people_exposed_est']}")
        pdf.drawString(72, 630, "Synthetic data only. No observed satellite imagery or accuracy metrics.")
        pdf.save()
        buffer.seek(0)
        return Response(buffer.getvalue(), media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="report.pdf"'})
    raise HTTPException(status_code=404, detail="Unknown result file")
