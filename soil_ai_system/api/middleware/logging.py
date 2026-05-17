"""Phase 3F - Robustness & Failure Handling (Middleware).

Implements structured logging, request tracing, and latency tracking.
Provides global exception handling to ensure the API never crashes completely.
"""

import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api.middleware")

from api.metrics import global_metrics

class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Unique trace ID for the request (mocked via timestamp here for simplicity)
        trace_id = str(int(time.time() * 1000))
        logger.info("--> Incoming %s %s [Trace: %s]", request.method, request.url.path, trace_id)
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            latency_ms = process_time * 1000
            
            response.headers["X-Process-Time-Ms"] = str(round(latency_ms, 2))
            response.headers["X-Trace-Id"] = trace_id
            
            # Record Success Metrics
            if "/api/v1/predict" in request.url.path:
                global_metrics.record_request(latency_ms, error=(response.status_code >= 400))
            
            logger.info(
                "<-- Completed %s %s with status %s in %.2fms [Trace: %s]",
                request.method, request.url.path, response.status_code, latency_ms, trace_id
            )
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            latency_ms = process_time * 1000
            
            if "/api/v1/predict" in request.url.path:
                global_metrics.record_request(latency_ms, error=True)
                
            logger.error(
                "<-- Failed %s %s after %.2fms [Trace: %s]. Error: %s",
                request.method, request.url.path, latency_ms, trace_id, str(e),
                exc_info=True
            )
            raise e
