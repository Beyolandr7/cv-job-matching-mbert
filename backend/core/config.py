import os
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Job Matching System (mBERT)"
    API_V1_STR: str = "/api/v1"

    # Security & JWT
    SECRET_KEY: str = "SUPER_SECRET_KEY_CHANGE_THIS_IN_PRODUCTION_JWT_SECRET_KEY_12345"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/job_matching_db"
    )

    # CORS settings (Frontend Next.js)
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # mBERT NLP Configuration
    MODEL_NAME: str = "bert-base-multilingual-cased"
    MAX_SEQ_LENGTH: int = 512
    CHUNK_STRIDE: int = 256

    # Decision Thresholds (Referenced from Khatri et al., 2025)
    THRESHOLD_FIT: float = 0.80
    THRESHOLD_CLOSE_FIT: float = 0.60

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
