"""Alizia AI Backend - Main Application Entry Point"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers

from core.config import settings
from api.router import router as api_router
from api.router import auth_router
from api.router import model_router
from api.router import conversation_router
from api.router import agent_router
from api.router import file_router
from api.router import embedding_router
from api.router import rag_router
from api.router import tool_router
from api.router import safety_router
from api.router import billing_router
from api.router import rate_limit_router
from api.router import observability_router
from middleware.rate_limit import RateLimitMiddleware, rate_limiter


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="Frontier multimodal AI and agent platform backend",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
    )
    
    # Security middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS,
    )
    
    # Rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        rate=settings.RATE_LIMIT_DEFAULT,
        burst=settings.RATE_LIMIT_BURST,
    )
    
    # Custom exception handler middleware
    app.add_middleware(
        BaseHTTPMiddleware,
        dispatch=lambda request, call_next: process_request(request, call_next),
    )
    
    # Include routers
    app.include_router(api_router, prefix="/v1")
    app.include_router(auth_router, prefix="/v1")
    app.include_router(model_router, prefix="/v1")
    app.include_router(conversation_router, prefix="/v1")
    app.include_router(agent_router, prefix="/v1")
    app.include_router(file_router, prefix="/v1")
    app.include_router(embedding_router, prefix="/v1")
    app.include_router(rag_router, prefix="/v1")
    app.include_router(tool_router, prefix="/v1")
    app.include_router(safety_router, prefix="/v1")
    app.include_router(billing_router, prefix="/v1")
    app.include_router(rate_limit_router, prefix="/v1")
    app.include_router(observability_router, prefix="/v1")
    
    # Global middleware that processes all requests
    @app.middleware("http")
    async def add_process_headers(request: Request, call_next):
        """Add processing headers and telemetry."""
        start_time = __import__('time').perf_counter()
        response: Response = await call_next(request)
        process_time = __import__('time').perf_counter() - start_time
        response.headers["x-processing-time-ms"] = str(int(process_time * 1000))
        response.headers["x-request-id"] = request.headers.get("x-request-id", "")
        return response
    
    # Health check route
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint."""
        return {
            "name": settings.APP_NAME,
            "version": settings.VERSION,
            "status": "operational",
            "environment": settings.ENVIRONMENT,
        }
    
    # Health check route
    @app.get("/health", include_in_schema=False)
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "alizia-backend",
            "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        }
    
    # Ready check
    @app.get("/ready", include_in_schema=False)
    async def ready_check():
        """Readiness check for Kubernetes."""
        return {"ready": True}
    
    # Liveness check
    @app.get("/live", include_in_schema=False)
    async def live_check():
        """Liveness check for Kubernetes."""
        return {"alive": True}
    
    return app


def process_request(request: Request, call_next):
    """Process incoming request with telemetry."""
    import time
    start_time = time.perf_counter()
    
    # Note: in production, use proper telemetry
    
    try:
        response = call_next(request)
        return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "code": "internal_server_error",
                    "message": "An internal error occurred",
                    "request_id": getattr(request.state, 'request_id', 'unknown'),
                }
            },
        )
    finally:
        process_time = time.perf_counter() - start_time


# Create the app instance
app = create_app()