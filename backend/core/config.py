import os
from typing import List

from pydantic_settings import BaseSettings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings(BaseSettings):
    PROJECT_NAME: str = "Job Matching System (mBERT)"
    API_V1_STR: str = "/api/v1"

    # Security & JWT (Hardcoded Admin Credentials as per Supervisor Whiteboard)
    SECRET_KEY: str = "SUPER_SECRET_KEY_CHANGE_THIS_IN_PRODUCTION_JWT_SECRET_KEY_12345"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")

    # File-Based Storage (No External Database Required)
    DATASET_CSV_PATH: str = os.getenv(
        "DATASET_CSV_PATH",
        os.path.join(BASE_DIR, "data", "dataset_cv_job_matching_3class_1500.csv"),
    )
    HISTORY_JSON_PATH: str = os.getenv(
        "HISTORY_JSON_PATH",
        os.path.join(BASE_DIR, "data", "analysis_history.json"),
    )

    # CORS settings (Frontend Next.js)
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]

    # mBERT NLP Configuration
    MODEL_NAME: str = "bert-base-multilingual-cased"
    FINE_TUNED_MODEL_PATH: str = os.getenv(
        "FINE_TUNED_MODEL_PATH",
        os.path.join(
            BASE_DIR,
            "models",
            "Skripsi_mBERT_Models",
            "best_siamese_mbert_cosine.pt",
        ),
    )
    MAX_SEQ_LENGTH: int = 256
    CHUNK_STRIDE: int = 128

    # Decision Thresholds (Referenced from Khatri et al., 2025 & Thesis Proposal)
    THRESHOLD_FIT: float = 0.80
    THRESHOLD_CLOSE_FIT: float = 0.60

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
