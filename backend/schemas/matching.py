from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MatchingAnalysisRequest(BaseModel):
    job_title: str = Field(..., description="Target Job Title")
    job_description: str = Field(..., description="Job qualification and responsibilities text")


class MatchingAnalysisResponse(BaseModel):
    id: Optional[str] = None
    cv_filename: str
    job_title: str
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0.0000 - 1.0000)",
    )
    similarity_percentage: float = Field(
        ...,
        description="Cosine similarity percentage (0.0% - 100.0%)",
    )
    status_category: str = Field(
        ...,
        description="'Fit', 'Close Fit', or 'Not Fit'",
    )
    extracted_entities: Dict[str, Any]
    cv_text_preview: str
    job_description_preview: str
    cv_text_full: Optional[str] = None
    job_text_full: Optional[str] = None
    is_validated: bool = False
    created_at: Optional[str] = None


class ValidationRequest(BaseModel):
    analysis_id: Optional[str] = None
    cv_filename: Optional[str] = "cv_document.pdf"
    cv_text: str = Field(..., description="Full extracted CV text to append to training CSV")
    job_title: str = Field(..., description="Target Job Title")
    job_description: str = Field(..., description="Target Job Description")
    validated_category: str = Field(..., description="Validated label from Admin combobox: 'Fit', 'Close Fit', or 'Not Fit'")
    validated_percentage: float = Field(..., ge=0.0, le=100.0, description="Validated cosine percentage from numeric up-down (0 - 100)")
    trigger_retrain: bool = Field(default=True, description="Whether to immediately re-train the Siamese mBERT model")


class RetrainRequest(BaseModel):
    epochs: int = Field(default=4, ge=1, le=20)


class MatchingHistoryItem(BaseModel):
    id: str
    cv_filename: str
    job_title: str
    similarity_score: float
    similarity_percentage: float
    status_category: str
    extracted_entities: Dict[str, Any]
    is_validated: bool = False
    created_at: str


class MatchingHistoryResponse(BaseModel):
    total: int
    items: List[MatchingHistoryItem]
