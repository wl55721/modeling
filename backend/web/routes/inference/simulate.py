from __future__ import annotations

from fastapi import APIRouter

from .schemas import SimulateRequest, SimulateMultiResponse
from .services.simulation import SimulationService

router = APIRouter(prefix="/api", tags=["simulate"])


@router.post("/simulate", response_model=SimulateMultiResponse)
def simulate(req: SimulateRequest):
    return SimulationService().simulate(req)
