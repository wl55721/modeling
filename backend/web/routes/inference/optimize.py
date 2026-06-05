from __future__ import annotations

from fastapi import APIRouter

from .schemas import OptimizeRequest, OptimizeResponse
from .services.optimizer import OptimizerService

router = APIRouter(prefix="/api", tags=["optimize"])


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    return OptimizerService().optimize(req)
