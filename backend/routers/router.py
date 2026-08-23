from fastapi import APIRouter

from routers import auth, matching

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(matching.router, prefix="/matching", tags=["Job Matching Engine"])
