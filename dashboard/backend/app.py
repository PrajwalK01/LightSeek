"""
LightSeek — FastAPI Backend (W5)
Author: Team Integral X
"""

import sys
import os
import json
import numpy as np
import lightkurve as lk

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from pipeline import detect, challenge, analyze, judge

app = FastAPI(title="LightSeek API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND = os.path.join(os.path.dirname(__file__),
                         "..", "frontend", "templates")


class StarRequest(BaseModel):
    target_id: str
    mission:   str = "kepler"
    quarter:   int = 1


def run_full_pipeline(time, flux):
    detect_report   = detect.run(time, flux)
    challenge_report = challenge.run(time, flux, detect_report)
    analyze_report  = analyze.run(time, flux)
    verdict         = judge.run(detect_report, challenge_report,
                                analyze_report)
    return {
        "agents": {
            "detect":    detect_report,
            "challenge": challenge_report,
            "analyze":   analyze_report
        },
        "verdict": verdict
    }


@app.get("/")
def index():
    return FileResponse(
        os.path.join(FRONTEND, "index.html")
    )


@app.get("/health")
def health():
    return {"status": "ok", "system": "LightSeek v1.0"}


@app.post("/predict/id")
async def predict_by_id(body: StarRequest):
    try:
        print(f"Analyzing {body.target_id}...")
        result = lk.search_lightcurve(
            body.target_id,
            mission=body.mission,
            quarter=body.quarter
        )
        if len(result) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No light curve found for {body.target_id}"
            )

        lc   = result[0].download()
        lc   = lc.remove_nans().remove_outliers().normalize()
        time = lc.time.value
        flux = lc.flux.value
        flux = np.nan_to_num(flux, nan=1.0)

        # Light curve data for plotting
        lc_data = {
            "time": time.tolist()[:500],
            "flux": flux.tolist()[:500]
        }

        pipeline_result = run_full_pipeline(time, flux)
        pipeline_result["target"]  = body.target_id
        pipeline_result["mission"] = body.mission
        pipeline_result["lc_data"] = lc_data

        return JSONResponse(content=pipeline_result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/file")
async def predict_by_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        tmp_path = f"tmp_{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(contents)

        lc   = lk.read(tmp_path)
        lc   = lc.remove_nans().remove_outliers().normalize()
        time = lc.time.value
        flux = lc.flux.value
        flux = np.nan_to_num(flux, nan=1.0)
        os.remove(tmp_path)

        lc_data = {
            "time": time.tolist()[:500],
            "flux": flux.tolist()[:500]
        }

        pipeline_result = run_full_pipeline(time, flux)
        pipeline_result["target"]  = file.filename
        pipeline_result["mission"] = "uploaded"
        pipeline_result["lc_data"] = lc_data

        return JSONResponse(content=pipeline_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample")
def get_sample():
    """Return pre-computed Kepler-17b sample for demo."""
    sample_path = "results/sample_verdicts/kepler17_report.json"
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            data = json.load(f)
        time = np.linspace(131, 165, 500).tolist()
        flux = (1.0 + 0.002 * np.sin(
            2 * np.pi * np.array(time) / 11.4
        )).tolist()
        data["lc_data"] = {"time": time, "flux": flux}
        return JSONResponse(content=data)
    return {"error": "Sample not found — run pipeline/run.py first"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)