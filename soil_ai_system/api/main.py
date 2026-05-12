from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import SoilInput, PredictionResponse
from api.predictor import run_prediction
from utils.logger import get_logger
import time

logger = get_logger("api", "api.log")

app = FastAPI(
    title="Soil Intelligence API",
    description="AI-powered soil quality prediction, crop recommendation, and explainable AI system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request path, status, and latency for each HTTP call.

    Args:
        request (fastapi.Request): Incoming request object.
        call_next (callable): Next handler in the middleware chain.

    Returns:
        fastapi.Response: Response from downstream handler.

    Side Effects:
        - Writes request logs to the API logger.
    """
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {elapsed}ms")
    return response


@app.get("/")
def root():
    """Return a basic API status message.

    Args:
        None

    Returns:
        dict: Service status payload.
    """
    return {"message": "Soil Intelligence System API v2.0 is running", "status": "healthy"}


@app.get("/health")
def health():
    """Return health and model version information.

    Args:
        None

    Returns:
        dict: Health status payload.
    """
    return {"status": "healthy", "model_version": "v1"}


@app.post("/predict", response_model=PredictionResponse)
def predict(soil_data: SoilInput):
    """Run inference for a soil input payload.

    Args:
        soil_data (SoilInput): Request payload validated by Pydantic.

    Returns:
        PredictionResponse: Inference response payload.

    Side Effects:
        - Logs prediction errors when exceptions occur.
    """
    try:
        return run_prediction(soil_data.dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Prediction error: {str(exc)}")
        raise HTTPException(status_code=500, detail="Internal prediction error")
