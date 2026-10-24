"""FastAPI application entry point for NoctTwin Decision Support."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router

app = FastAPI(
    title="NoctTwin Clinical Decision Support API",
    description=(
        "Bedtime nocturnal hypoglycemia risk estimation and digital twin what-if "
        "countermeasure simulation for hospitalized T2DM patients."
    ),
    version="1.0.0",
)

# Enable CORS for local Vite development and static frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Mount static figures if folder exists
if os.path.exists("web/public/data/figures"):
    app.mount("/static/figures", StaticFiles(directory="web/public/data/figures"), name="figures")


@app.get("/")
def root():
    return {
        "message": "NoctTwin Clinical API is running. Access interactive documentation at /docs",
        "health": "/api/health",
        "patients": "/api/patients",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
