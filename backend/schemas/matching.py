import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MatchingAnalysisRequest(BaseModel):
    job_title: str = Field(..., description="Target Job Title")
    job_description: str = Field(..., description="Job qualification and responsibilities text")


class ExtractedEntities(BaseModel):
    skills: List[str] = []
    education: List[str] = []
    experience_years_detected: Optional[int] = None


class MatchingAnalysisResponse(BaseModel):
    id: Optional[uuid.UUID] = None
    cv_filename: str
    job_title: str
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0.0000 - 1.0000)"
    )
    similarity_percentage: float = Field(
        ...,
        description="Cosine similarity percentage (0.0% - 100.0%)"
    )
    status_category: str = Field(
        ...,
        description="'Fit', 'Close Fit', or 'Not Fit'"
    )
    extracted_entities: Dict[str, Any]
    cv_text_preview: str
    job_description_preview: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MatchingHistoryItem(BaseModel):
    id: uuid.UUID
    cv_filename: str
    job_title: str
    similarity_score: float
    status_category: str
    extracted_entities: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


class MatchingHistoryResponse(BaseModel):
    total: int
    items: List[MatchingHistoryItem]
