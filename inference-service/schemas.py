"""
Pydantic schemas for all API endpoints.
"""

from typing import Literal
from pydantic import BaseModel, Field, model_validator

# ── Material presets ──────────────────────────────────────────────────────────

MATERIALS: dict[str, dict[str, float]] = {
    "steel":    {"alpha": 1.20e-5, "rho_c": 3.90e6},
    "aluminum": {"alpha": 9.70e-5, "rho_c": 2.43e6},
    "copper":   {"alpha": 1.17e-4, "rho_c": 3.45e6},
    "titanium": {"alpha": 2.90e-6, "rho_c": 2.34e6},
}

MaterialName = Literal["steel", "aluminum", "copper", "titanium"]


# ── /predict ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    material:  MaterialName = Field(..., description="Pipe material")
    length:    float = Field(..., gt=0,   description="Pipe length  [m]")
    intensity: float = Field(..., gt=0,   description="Burner intensity  [W/m]")
    x0:        float = Field(..., ge=0,   description="Burner start position  [m]")
    velocity:  float = Field(..., gt=0,   description="Burner velocity  [m/s]")
    t_total:   float = Field(..., gt=0,   description="Total simulation time  [s]")
    x_query:   float = Field(..., ge=0,   description="Query position along pipe  [m]")
    t_query:   float = Field(..., ge=0,   description="Query time  [s]")

    @model_validator(mode="after")
    def check_bounds(self) -> "PredictRequest":
        if self.x0 >= self.length:
            raise ValueError("x0 must be less than pipe length")
        if self.x0 + self.velocity * self.t_total > self.length * 1.001:
            raise ValueError(
                "Burner exits pipe: x0 + velocity * t_total must be ≤ length"
            )
        if self.x_query > self.length:
            raise ValueError("x_query must be ≤ pipe length")
        if self.t_query > self.t_total:
            raise ValueError("t_query must be ≤ t_total")
        return self


class PredictResponse(BaseModel):
    temperature_K:        float = Field(..., description="PINN predicted temperature  [K]")
    temperature_C:        float = Field(..., description="PINN predicted temperature  [°C]")
    delta_T:              float = Field(..., description="Temperature rise above ambient  [K]")
    analytical_K:         float = Field(..., description="Analytical reference temperature  [K]")
    analytical_delta_T:   float = Field(..., description="Analytical temperature rise  [K]")
    relative_error:       float = Field(..., description="|PINN − ref| / |ref|")
    T_amb:                float = Field(..., description="Ambient temperature  [K]")
    device:               str   = Field(..., description="Inference device (cuda/mps/cpu)")


# ── /heatmap ─────────────────────────────────────────────────────────────────

class HeatmapRequest(BaseModel):
    material:  MaterialName
    length:    float = Field(..., gt=0)
    intensity: float = Field(..., gt=0)
    x0:        float = Field(..., ge=0)
    velocity:  float = Field(..., gt=0)
    t_total:   float = Field(..., gt=0)
    nx:        int   = Field(60,  ge=10, le=200, description="Grid points along x")
    nt:        int   = Field(60,  ge=10, le=200, description="Grid points along t")

    @model_validator(mode="after")
    def check_bounds(self) -> "HeatmapRequest":
        if self.x0 >= self.length:
            raise ValueError("x0 must be less than pipe length")
        if self.x0 + self.velocity * self.t_total > self.length * 1.001:
            raise ValueError("Burner exits pipe: reduce velocity or t_total")
        return self


class HeatmapResponse(BaseModel):
    x_grid:             list[float]       # (nx,)  positions [m]
    t_grid:             list[float]       # (nt,)  times [s]
    delta_T_pinn:       list[list[float]] # (nt, nx)  PINN ΔT [K]
    delta_T_analytical: list[list[float]] # (nt, nx)  reference ΔT [K]
    burner_positions:   list[float]       # (nt,)  x_b(t) [m]
    T_amb:              float


# ── /materials ────────────────────────────────────────────────────────────────

class MaterialInfo(BaseModel):
    name:              str
    alpha:             float = Field(..., description="Thermal diffusivity  [m²/s]")
    rho_c:             float = Field(..., description="Volumetric heat capacity  [J/(m³·K)]")
    thermal_conductivity_approx: float = Field(..., description="k ≈ α·ρc  [W/(m·K)]")


class MaterialsResponse(BaseModel):
    materials: list[MaterialInfo]


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:      str
    device:      str
    model_loaded: bool
