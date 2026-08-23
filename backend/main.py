from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.config import settings
from routers.router import api_router
from services.mbert_service import mbert_service

# Configure root application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager:
    Preloads mBERT model weights into memory at startup to eliminate cold-start latency.
    """
    logger.info("Starting up FastAPI application...")
    try:
        logger.info("Pre-loading mBERT model weights into memory...")
        mbert_service.load_model()
    except Exception as exc:
        logger.warning("Could not pre-load mBERT model at startup (will load on first request): %s", exc)

    yield
    logger.info("Shutting down FastAPI application...")


# Initialize FastAPI Application instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Cross-Lingual Job Matching System Engine based on Multilingual BERT (mBERT) and Content-Based Filtering.",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS Middleware for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Health Check"])
def root() -> dict:
    """Root health-check and metadata endpoint."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Health Check"])
def health_check() -> dict:
    """Standard liveness probe."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
