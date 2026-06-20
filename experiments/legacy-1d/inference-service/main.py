"""
PINN Inference API.

Endpoints
─────────
GET  /health        — liveness check + device info
GET  /materials     — list material presets with properties
POST /predict       — single-point temperature query
POST /heatmap       — full (x, t) grid for UI visualisation

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

With a custom checkpoint:
    MODEL_PATH=./checkpoints/model_final.pt uvicorn main:app --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predictor import load_predictor, get_predictor
from schemas import (
    MATERIALS,
    HealthResponse,
    HeatmapRequest,
    HeatmapResponse,
    MaterialInfo,
    MaterialsResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_predictor()      # loads model into memory once; raises if file missing
    log.info("Predictor ready.")
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PINN Heat Equation Inference",
    description=(
        "Physics-Informed Neural Network that predicts temperature distribution "
        "in a 1D pipe with a moving gas burner.\n\n"
        "Governing PDE: ∂T/∂t = α·∂²T/∂x² + (i/ρc)·δ(x−x_b(t))"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    p = get_predictor()
    return HealthResponse(
        status="ok",
        device=str(p.device),
        model_loaded=True,
    )


@app.get("/materials", response_model=MaterialsResponse, tags=["meta"])
def materials() -> MaterialsResponse:
    items = [
        MaterialInfo(
            name=name,
            alpha=props["alpha"],
            rho_c=props["rho_c"],
            thermal_conductivity_approx=props["alpha"] * props["rho_c"],
        )
        for name, props in MATERIALS.items()
    ]
    return MaterialsResponse(materials=items)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict temperature at a single (x, t) point for the given pipe parameters.
    Also returns the analytical reference for comparison.
    """
    try:
        result = get_predictor().predict_point(
            material=req.material,
            length=req.length,
            intensity=req.intensity,
            x0=req.x0,
            velocity=req.velocity,
            t_total=req.t_total,
            x_query=req.x_query,
            t_query=req.t_query,
        )
    except Exception as exc:
        log.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(**result)


@app.post("/heatmap", response_model=HeatmapResponse, tags=["inference"])
def heatmap(req: HeatmapRequest) -> HeatmapResponse:
    """
    Evaluate PINN and analytical solution on an (nt × nx) grid.
    Used by the UI to render the temperature heatmap and the burner animation.
    """
    try:
        result = get_predictor().predict_heatmap(
            material=req.material,
            length=req.length,
            intensity=req.intensity,
            x0=req.x0,
            velocity=req.velocity,
            t_total=req.t_total,
            nx=req.nx,
            nt=req.nt,
        )
    except Exception as exc:
        log.exception("heatmap failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return HeatmapResponse(**result)
