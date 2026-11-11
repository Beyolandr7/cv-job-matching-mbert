from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.config import settings
from routers.auth import router as auth_router
from routers.matching import router as matching_router
from services.mbert_service import mbert_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        mbert_service.load_model()
    except Exception as exc:
        logger.warning("Could not pre-load mBERT model at startup: %s", exc)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Cross-Lingual Job Matching Engine (NER + Siamese mBERT + Cosine Similarity)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(matching_router, prefix=f"{settings.API_V1_STR}/matching", tags=["Matching"])


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    return {"status": "ok", "service": settings.PROJECT_NAME}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
