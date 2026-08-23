import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from db.session import get_db
from models.matching_history import MatchingHistory
from models.user import User
from routers.deps import get_current_user, get_current_user_optional
from schemas.matching import MatchingAnalysisResponse, MatchingHistoryResponse
from services.mbert_service import mbert_service
from services.pdf_service import pdf_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=MatchingAnalysisResponse)
async def analyze_matching(
    file: UploadFile = File(..., description="PDF file of the candidate CV/Resume"),
    job_title: str = Form(..., description="Target Job Title"),
    job_description: str = Form(..., description="Job requirements and responsibilities text"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> dict:
    """
    Main Matching Engine Endpoint:
    1. Extracts and cleans text from the in-memory uploaded PDF CV.
    2. Generates 768-d mBERT embeddings using Sliding Window Mean Pooling.
    3. Calculates Cosine Similarity and classifies status ('Fit', 'Close Fit', 'Not Fit').
    4. Extracts skill and education highlights.
    5. Saves matching history to PostgreSQL (if user is authenticated).
    """
    filename = file.filename or "cv_document.pdf"
    if not filename.lower().endswith(".pdf") and file.content_type not in ["application/pdf", "application/octet-stream"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Berkas yang diunggah harus berformat PDF (.pdf)."
        )

    # 1. Ingestion: Extract text from PDF in-memory stream
    try:
        pdf_bytes = await file.read()
        cv_text = pdf_service.extract_text_from_pdf_stream(pdf_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Gagal memproses berkas PDF: {str(exc)}"
        ) from exc

    if not cv_text or len(cv_text.strip()) < 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Teks dalam PDF tidak terbaca atau terlalu pendek. Pastikan PDF bukan scan gambar kosong."
        )

    # 2. NLP Semantic Matching via mBERT
    match_result = mbert_service.match_cv_with_job(
        cv_text=cv_text,
        job_title=job_title,
        job_description=job_description
    )

    history_record_id = None
    created_timestamp = None

    # 3. Persist record to PostgreSQL if user is authenticated
    if current_user:
        try:
            history_record = MatchingHistory(
                user_id=current_user.id,
                cv_filename=filename,
                cv_text=cv_text,
                job_title=job_title,
                job_description=job_description,
                extracted_entities=match_result["extracted_entities"],
                similarity_score=match_result["similarity_score"],
                status_category=match_result["status_category"]
            )
            db.add(history_record)
            db.commit()
            db.refresh(history_record)
            history_record_id = history_record.id
            created_timestamp = history_record.created_at
        except Exception as db_err:
            db.rollback()
            logger.warning("Failed to save history to database: %s", db_err)

    return {
        "id": history_record_id,
        "cv_filename": filename,
        "job_title": job_title,
        "similarity_score": match_result["similarity_score"],
        "similarity_percentage": match_result["similarity_percentage"],
        "status_category": match_result["status_category"],
        "extracted_entities": match_result["extracted_entities"],
        "cv_text_preview": match_result["cv_text_preview"],
        "job_description_preview": match_result["job_description_preview"],
        "created_at": created_timestamp
    }


@router.get("/history", response_model=MatchingHistoryResponse)
def get_user_matching_history(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Retrieves paginated matching histories for the authenticated user."""
    query = db.query(MatchingHistory).filter(MatchingHistory.user_id == current_user.id)
    total = query.count()
    items = query.order_by(MatchingHistory.created_at.desc()).offset(skip).limit(limit).all()

    return {
        "total": total,
        "items": items
    }


@router.delete("/history/{history_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history_item(
    history_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> None:
    """Deletes a specific matching history record."""
    item = db.query(MatchingHistory).filter(
        MatchingHistory.id == history_id,
        MatchingHistory.user_id == current_user.id
    ).first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Riwayat pencocokan tidak ditemukan."
        )

    db.delete(item)
    db.commit()
    return None
