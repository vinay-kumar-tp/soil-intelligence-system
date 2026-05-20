"""Phase 3E - FastAPI Production Server.

The entry point for the Soil Intelligence AI API.
Wraps the inference engine in an async-safe, production-ready server.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.logging import RequestTracingMiddleware
from api.routes.inference import router as inference_router
from api.routes.image_inference import router as image_router
from inference.loaders import registry_cache
import logging

# Ensure logs go to inference log file
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(_PROJECT_ROOT / "logs" / "api.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
logger.addHandler(file_handler)


def create_app() -> FastAPI:
    """Factory pattern for FastAPI initialization."""
    app = FastAPI(
        title="Soil Intelligence System API",
        version="3.0.0",
        description="Phase 3 Production Inference System & AI Orchestration",
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestTracingMiddleware)

    # Routes
    app.include_router(inference_router, prefix="/api/v1")
    app.include_router(image_router, prefix="/api/v1")

    @app.on_event("startup")
    async def startup_event():
        """Pre-warm the model cache during startup if desired.
        
        Note: The prompt specified 'DO NOT load all models at startup',
        so we strictly rely on lazy loading upon first request. 
        We just initialize the registry index here.
        """
        logger.info("Initializing API Startup Sequence...")
        # Just loads the JSON index into memory, not the huge artifacts
        registry_cache._load_registry()
        logger.info("Registry index pre-loaded. Models will load lazily.")

    @app.get("/")
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
