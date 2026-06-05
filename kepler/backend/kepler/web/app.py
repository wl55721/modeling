from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.simulate import router as simulate_router
from .routes.library import router as library_router
from .routes.optimize import router as optimize_router

app = FastAPI(title="Kepler", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulate_router)
app.include_router(library_router)
app.include_router(optimize_router)
