"""SecureID Production FastAPI Server.
MasterCard CSB-02 & ISO/IEC 30107-3 compliant verification endpoints.
Serves the dark glassmorphic KYC onboarding web application.
"""

import os
import json
import base64
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.pipeline import SecureIDPipeline
from core.hardware import inspect_hardware_integrity

app = FastAPI(
    title="SecureID Biometric Identity Verification",
    description="Multimodal Late Fusion KYC Verification with Hardware Gatekeeper",
    version="2.0.0"
)

# Enable CORS for cross-origin or local test clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web assets
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize Pipeline Singleton
pipeline = SecureIDPipeline()


class VerifyRequest(BaseModel):
    device_label: Optional[str] = None
    frames: List[str] # List of base64 data URIs or base64 strings
    flash_events: Optional[List[Dict[str, Any]]] = None


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main KYC verification onboarding dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>SecureID Dashboard Initializing...</h2>")


@app.get("/api/hardware-status")
async def get_hardware_status(device_label: Optional[str] = None):
    """Layer 1: Device Attestation inspection endpoint."""
    res = inspect_hardware_integrity(device_label)
    return {
        "s_hw": res.s_hw,
        "is_clean_hardware": res.is_clean_hardware,
        "device_name": res.device_name,
        "details": res.details,
        "virtual_driver_detected": res.virtual_driver_detected,
    }


@app.get("/api/model-info")
async def get_model_info():
    """Returns model architecture, Meta DFDC dataset reference, and formula weights."""
    return {
        "architecture": "PyTorchVideo X3D-S Spatio-Temporal CNN",
        "training_dataset": "Meta DFDC (DeepFake Detection Challenge)",
        "dataset_url": "https://ai.meta.com/datasets/dfdc/",
        "formula": "Final Verification Score = S_hw * (0.35 * S_Flash + 0.45 * S_X3D + 0.20 * S_FFT) * 100",
        "weights": {
            "S_Flash": 0.35,
            "S_X3D": 0.45,
            "S_FFT": 0.20
        },
        "thresholds": {
            "Verified": ">= 85% (Instant Auto-Approval)",
            "Flagged": "60% - 84% (Step-up check / Retry)",
            "Rejected": "< 60% or S_hw = 0 (Block & Alert Fraud Team)"
        },
        "compliance": ["MasterCard CSB-02", "ISO/IEC 30107-3 Level 2"]
    }


@app.post("/api/verify")
async def verify_biometrics(payload: VerifyRequest):
    """Processes captured video frames, checks hardware integrity, runs late fusion."""
    try:
        # Decode base64 frames to BGR numpy arrays
        frames_bgr: List[np.ndarray] = []
        for f_data in payload.frames:
            if "," in f_data:
                f_data = f_data.split(",", 1)[1]
            img_bytes = base64.b64decode(f_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                frames_bgr.append(img)

        # Run Verification Pipeline
        report = pipeline.verify_live_stream(
            frames_bgr=frames_bgr,
            client_device_label=payload.device_label,
            flash_events=payload.flash_events
        )

        return {
            "s_hw": report.s_hw,
            "s_flash": report.s_flash,
            "s_x3d": report.s_x3d,
            "s_fft": report.s_fft,
            "final_score": report.final_score,
            "kyc_status": report.kyc_status,
            "action_taken": report.action_taken,
            "reason_code": report.reason_code,
            "details": report.details,
            "hardware_details": report.hardware_details,
            "flash_details": report.flash_details,
            "temporal_details": report.temporal_details,
            "frequency_details": report.frequency_details,
            "heatmap_data_uri": report.heatmap_data_uri,
            "iso_compliance": report.iso_compliance,
            "mastercard_tier": report.mastercard_tier
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification Pipeline Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n[SecureID] Launching production server on http://127.0.0.1:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
