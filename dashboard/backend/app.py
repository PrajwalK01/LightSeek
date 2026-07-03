"""
LightSeek — FastAPI Backend
Author: Team Integral X
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json, os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

app = FastAPI(title="LightSeek API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictByID(BaseModel):
    target_id: str
    mission: str = "kepler"


@app.get("/health")
def health():
    return {"status": "ok", "system": "LightSeek v1.0"}


@app.post("/predict/id")
async def predict_by_id(body: PredictByID):
    """Analyze a star by its TIC/KIC ID."""
    # TODO Week 6: Connect to pipeline
    return {
        "target": body.target_id,
        "status": "processing",
        "message": "Pipeline not yet connected"
    }


@app.post("/predict/file")
async def predict_by_file(file: UploadFile = File(...)):
    """Analyze a star by uploading a FITS file."""
    # TODO Week 6: Save file + run pipeline
    return {
        "filename": file.filename,
        "status": "processing",
        "message": "Pipeline not yet connected"
    }


@app.get("/sample")
def get_sample():
    """Return a sample vetting report for demo."""
    return {
        "target": "Kepler-17b",
        "mission": "kepler",
        "agents": {
            "detect": {
                "verdict": "PASS",
                "period_days": 1.486,
                "depth": 0.014,
                "snr": 28.4,
                "n_transits": 12
            },
            "challenge": {
                "verdict": "PASS",
                "secondary_eclipse_detected": False,
                "centroid_shift_px": 0.002,
                "transit_shape": "flat-bottomed"
            },
            "analyze": {
                "verdict": "PASS",
                "flares_detected": 0,
                "stellar_quiet": True,
                "oot_scatter_ppm": 145
            }
        },
        "verdict": {
            "verdict": "PLANET CANDIDATE",
            "confidence": 0.94,
            "reasoning": "All three agents agree. Strong periodic signal with flat-bottomed transit shape, no secondary eclipse, quiet host star. Consistent with a hot Jupiter in close orbit.",
            "recommended_followup": "Radial velocity confirmation to measure planetary mass."
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
