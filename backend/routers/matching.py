import csv
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
import pandas as pd

from core.config import settings
from routers.deps import get_current_user, get_current_user_optional
from schemas.auth import UserResponse
from schemas.matching import (
    MatchingAnalysisResponse,
    MatchingHistoryResponse,
    RetrainRequest,
    ValidationRequest,
)
from services.mbert_service import mbert_service
from services.pdf_service import pdf_service

logger = logging.getLogger(__name__)
router = APIRouter()

PRESET_JOBS: List[Dict[str, str]] = [
    {
        "id": "banking-ops-manager",
        "domain": "BANKING / FINANCE",
        "job_title": "Consumer Banking Operations Manager",
        "job_description": (
            "Seeking an experienced Banking Operations Manager & Default Management Supervisor with a "
            "Bachelor of Science in Finance or Business Administration. Must have 10+ years of experience "
            "in customer service management, call center operations, cash deposits, account resolution, "
            "staffing, scheduling, budgeting, payroll, and process improvement."
        ),
    },
    {
        "id": "financial-credit-analyst",
        "domain": "FINANCE / ACCOUNTING",
        "job_title": "Financial Analyst & Credit Analyst",
        "job_description": (
            "Seeking a Financial Analyst and Credit Analyst with 3+ years of experience to conduct "
            "corporate financial modeling, credit risk assessment, variance analysis, billing review, "
            "and quarterly forecasting."
        ),
    },
    {
        "id": "senior-software-engineer",
        "domain": "INFORMATION-TECHNOLOGY",
        "job_title": "Senior Full Stack Software Engineer",
        "job_description": (
            "Seeking a Senior Software Engineer with a Bachelor of Computer Science or S1 Informatika, "
            "proficient in Python, Golang, C#, React, FastAPI, Docker, Kubernetes, and PostgreSQL to "
            "build distributed cloud microservices."
        ),
    },
    {
        "id": "it-helpdesk-support",
        "domain": "INFORMATION-TECHNOLOGY",
        "job_title": "IT Help Desk & Technical Support Specialist",
        "job_description": (
            "Seeking an IT Help Desk and Technical Support Specialist to troubleshoot hardware, network "
            "connectivity, database administration, and end-user enterprise software issues."
        ),
    },
    {
        "id": "senior-accountant-cpa",
        "domain": "ACCOUNTANT",
        "job_title": "Senior Financial Accountant (CPA)",
        "job_description": (
            "Looking for a Senior Staff Accountant with CPA certification and Bachelor in Accounting. "
            "Responsible for Financial Accounting, General Ledger reconciliation, GAAP compliance, "
            "Accounts Payable/Receivable, Tax filings, and Financial Reporting."
        ),
    },
    {
        "id": "registered-nurse-icu",
        "domain": "HEALTHCARE",
        "job_title": "Registered Nurse (RN) - Clinical Patient Care",
        "job_description": (
            "Seeking a licensed Registered Nurse (BSN) with 3+ years of clinical hospital experience "
            "in patient care, triage, medication administration, EMR documentation, and HIPAA compliance."
        ),
    },
    {
        "id": "executive-chef",
        "domain": "CHEF / CULINARY",
        "job_title": "Executive Chef & Culinary Manager",
        "job_description": (
            "Seeking an Executive Chef skilled in Culinary Arts, Pastry, Menu Planning, Food Safety "
            "(HACCP), kitchen staff supervision, inventory control, and high-volume banquet catering."
        ),
    },
    {
        "id": "classroom-teacher",
        "domain": "TEACHER / EDUCATION",
        "job_title": "Classroom Teacher & Curriculum Educator",
        "job_description": (
            "Seeking a dedicated Classroom Teacher with a Bachelor of Education to lead lesson planning, "
            "curriculum development, student assessment, and interactive classroom management."
        ),
    },
    {
        "id": "commercial-legal-counsel",
        "domain": "ADVOCATE / LEGAL",
        "job_title": "Commercial Litigation Attorney & Legal Counsel",
        "job_description": (
            "Seeking a licensed Attorney / Legal Counsel (J.D. or S1 Hukum) experienced in civil litigation, "
            "legal research, contract management, corporate law, and regulatory compliance."
        ),
    },
]


def _load_history() -> List[Dict[str, Any]]:
    if not os.path.exists(settings.HISTORY_JSON_PATH):
        return []
    try:
        with open(settings.HISTORY_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Failed to read analysis_history.json: %s", exc)
        return []


def _save_history(items: List[Dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(settings.HISTORY_JSON_PATH), exist_ok=True)
        with open(settings.HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to write analysis_history.json: %s", exc)


@router.get("/preset-jobs")
def get_preset_jobs() -> dict:
    """Returns curated preset job postings across domains for quick selection in the UI."""
    return {"items": PRESET_JOBS}


@router.post("/analyze", response_model=MatchingAnalysisResponse)
async def analyze_matching(
    file: UploadFile = File(..., description="PDF file of the candidate CV/Resume"),
    job_title: str = Form(..., description="Target Job Title"),
    job_description: str = Form(..., description="Job requirements and responsibilities text"),
    current_user: Optional[UserResponse] = Depends(get_current_user_optional),
) -> dict:
    """
    Main Matching Engine Endpoint (No External Database Required):
    1. Extracts and cleans text from the uploaded PDF CV.
    2. Extracts NER entities (EDU, ORG, SKILL, ROLE, DUR) and encodes 768-d Siamese mBERT vectors.
    3. Calculates Scikit-Learn Cosine Similarity and classifies status ('Fit', 'Close Fit', 'Not Fit').
    4. Saves record to local `data/analysis_history.json` for Admin validation & summary charts.
    """
    filename = file.filename or "cv_document.pdf"
    if not filename.lower().endswith(".pdf") and file.content_type not in [
        "application/pdf",
        "application/octet-stream",
    ]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Berkas yang diunggah harus berformat PDF (.pdf).",
        )

    try:
        pdf_bytes = await file.read()
        cv_text = pdf_service.extract_text_from_pdf_stream(pdf_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Gagal memproses berkas PDF: {str(exc)}",
        ) from exc

    if not cv_text or len(cv_text.strip()) < 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Teks dalam PDF tidak terbaca atau terlalu pendek. Pastikan PDF bukan hasil scan gambar kosong.",
        )

    match_result = mbert_service.match_cv_with_job(
        cv_text=cv_text,
        job_title=job_title,
        job_description=job_description,
    )

    record_id = str(uuid.uuid4())
    created_at_iso = datetime.now(timezone.utc).isoformat()

    history_item = {
        "id": record_id,
        "cv_filename": filename,
        "job_title": job_title,
        "job_description": job_description,
        "cv_text": cv_text,
        "similarity_score": match_result["similarity_score"],
        "similarity_percentage": match_result["similarity_percentage"],
        "status_category": match_result["status_category"],
        "extracted_entities": match_result["extracted_entities"],
        "cv_text_preview": match_result["cv_text_preview"],
        "job_description_preview": match_result["job_description_preview"],
        "is_validated": False,
        "analyzed_by": current_user.username if current_user else "user_biasa",
        "created_at": created_at_iso,
    }

    history = _load_history()
    history.insert(0, history_item)
    _save_history(history[:300])

    return {
        "id": record_id,
        "cv_filename": filename,
        "job_title": job_title,
        "similarity_score": match_result["similarity_score"],
        "similarity_percentage": match_result["similarity_percentage"],
        "status_category": match_result["status_category"],
        "extracted_entities": match_result["extracted_entities"],
        "cv_text_preview": match_result["cv_text_preview"],
        "job_description_preview": match_result["job_description_preview"],
        "cv_text_full": match_result["cv_text_full"],
        "job_text_full": match_result["job_text_full"],
        "is_validated": False,
        "created_at": created_at_iso,
    }


@router.post("/validate")
def validate_and_retrain(
    req: ValidationRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> dict:
    """
    Admin Validation & Model Re-training Endpoint (Whiteboard Flow):
    1. Receives Admin-validated category ('Fit', 'Close Fit', 'Not Fit' via Combobox)
       and cosine similarity percentage (via Numeric Up-Down).
    2. Appends the validated (job, cv, kategori) pair to `data/dataset_cv_job_matching_3class_1500.csv`.
    3. Updates `data/analysis_history.json` status and score.
    4. Runs incremental fine-tuning on `SiameseMBERTCosineExtractor` and updates `best_siamese_mbert_cosine.pt`.
    """
    valid_categories = {"fit": "Fit", "close fit": "Close Fit", "not fit": "Not Fit"}
    cat_key = req.validated_category.strip().lower()
    if cat_key not in valid_categories:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Kategori validasi harus salah satu dari: 'Fit', 'Close Fit', atau 'Not Fit'.",
        )

    canonical_category = valid_categories[cat_key]
    csv_category_label = cat_key  # 'fit', 'close fit', 'not fit' matching CSV convention
    full_job_text = f"Position: {req.job_title.strip()}. Job Description: {req.job_description.strip()}"
    clean_cv_text = req.cv_text.strip()

    # 1. Append to dataset_cv_job_matching_3class_1500.csv
    os.makedirs(os.path.dirname(settings.DATASET_CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(settings.DATASET_CSV_PATH)
    try:
        with open(settings.DATASET_CSV_PATH, "a", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(["job", "cv", "kategori"])
            writer.writerow([full_job_text, clean_cv_text, csv_category_label])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal menyimpan data baru ke CSV dataset: {str(exc)}",
        ) from exc

    # 2. Update history record in analysis_history.json
    history = _load_history()
    updated_score = round(req.validated_percentage / 100.0, 4)
    updated_pct = round(req.validated_percentage, 2)
    matched_history = False

    for item in history:
        if req.analysis_id and item.get("id") == req.analysis_id:
            item["status_category"] = canonical_category
            item["similarity_score"] = updated_score
            item["similarity_percentage"] = updated_pct
            item["is_validated"] = True
            item["validated_by"] = current_user.username
            item["validated_at"] = datetime.now(timezone.utc).isoformat()
            matched_history = True
            break

    if not matched_history:
        history.insert(
            0,
            {
                "id": req.analysis_id or str(uuid.uuid4()),
                "cv_filename": req.cv_filename or "validated_cv.pdf",
                "job_title": req.job_title,
                "job_description": req.job_description,
                "cv_text": clean_cv_text,
                "similarity_score": updated_score,
                "similarity_percentage": updated_pct,
                "status_category": canonical_category,
                "extracted_entities": mbert_service.extract_entities_hybrid(clean_cv_text),
                "cv_text_preview": clean_cv_text[:300],
                "job_description_preview": req.job_description[:300],
                "is_validated": True,
                "validated_by": current_user.username,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    _save_history(history[:300])

    # 3. Trigger incremental re-training on SiameseMBERTCosineExtractor
    retrain_metrics = None
    if req.trigger_retrain:
        retrain_metrics = mbert_service.retrain_on_validated_pair(
            cv_text=clean_cv_text,
            job_text=full_job_text,
            validated_category=canonical_category,
            validated_percentage=updated_pct,
            epochs=4,
        )

    return {
        "status": "success",
        "message": (
            f"Data CV berhasil divalidasi sebagai '{canonical_category}' ({updated_pct}%), "
            f"ditambahkan ke file {os.path.basename(settings.DATASET_CSV_PATH)}, dan model mBERT telah di-retrain."
        ),
        "validated_category": canonical_category,
        "validated_percentage": updated_pct,
        "validated_score": updated_score,
        "dataset_csv_path": os.path.basename(settings.DATASET_CSV_PATH),
        "retrain_metrics": retrain_metrics,
    }


@router.post("/retrain")
def trigger_standalone_retrain(
    req: RetrainRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> dict:
    """
    Allows Admin to explicitly trigger a model re-training pass on the latest validated dataset CSV rows.
    """
    metrics = mbert_service.retrain_on_validated_pair(
        cv_text=None,
        job_text=None,
        validated_category="Fit",
        validated_percentage=85.0,
        epochs=req.epochs,
    )
    return {
        "status": "success",
        "message": f"Re-training model selesai ({metrics['epochs_trained']} epoch pada {metrics['samples_trained']} sampel).",
        "retrain_metrics": metrics,
    }


@router.get("/stats")
def get_processing_statistics(
    current_user: UserResponse = Depends(get_current_user),
) -> dict:
    """
    Returns summary statistics for the Admin Dashboard Charts:
    1. Processing History Summary (`analysis_history.json`): % and count of Fit, Close Fit, Not Fit.
    2. Training Dataset CSV Summary (`dataset_cv_job_matching_3class_1500.csv`): % and count of Fit, Close Fit, Not Fit.
    """
    history = _load_history()
    hist_counts = {"Fit": 0, "Close Fit": 0, "Not Fit": 0}
    validated_count = 0

    for item in history:
        cat = item.get("status_category", "Not Fit")
        if cat in hist_counts:
            hist_counts[cat] += 1
        if item.get("is_validated"):
            validated_count += 1

    total_hist = sum(hist_counts.values())
    hist_percentages = {
        k: round((v / total_hist) * 100.0, 2) if total_hist > 0 else 0.0
        for k, v in hist_counts.items()
    }

    # Dataset CSV statistics
    csv_counts = {"Fit": 0, "Close Fit": 0, "Not Fit": 0}
    total_csv = 0
    if os.path.exists(settings.DATASET_CSV_PATH):
        try:
            df = pd.read_csv(settings.DATASET_CSV_PATH)
            total_csv = len(df)
            val_map = df["kategori"].astype(str).str.strip().str.lower().value_counts().to_dict()
            csv_counts["Fit"] = int(val_map.get("fit", 0))
            csv_counts["Close Fit"] = int(val_map.get("close fit", 0))
            csv_counts["Not Fit"] = int(val_map.get("not fit", 0))
        except Exception as exc:
            logger.warning("Could not read dataset CSV for stats: %s", exc)

    csv_percentages = {
        k: round((v / total_csv) * 100.0, 2) if total_csv > 0 else 0.0
        for k, v in csv_counts.items()
    }
    newly_added_rows = max(0, total_csv - 1500)

    return {
        "history_summary": {
            "total_processed": total_hist,
            "validated_count": validated_count,
            "counts": hist_counts,
            "percentages": hist_percentages,
        },
        "dataset_summary": {
            "dataset_filename": os.path.basename(settings.DATASET_CSV_PATH),
            "total_rows": total_csv,
            "initial_rows": 1500,
            "newly_validated_rows": newly_added_rows,
            "counts": csv_counts,
            "percentages": csv_percentages,
        },
        "recent_history": [
            {
                "id": h.get("id"),
                "cv_filename": h.get("cv_filename"),
                "job_title": h.get("job_title"),
                "similarity_score": h.get("similarity_score"),
                "similarity_percentage": h.get("similarity_percentage"),
                "status_category": h.get("status_category"),
                "is_validated": h.get("is_validated", False),
                "created_at": h.get("created_at"),
            }
            for h in history[:25]
        ],
    }


@router.get("/history", response_model=MatchingHistoryResponse)
def get_user_matching_history(
    skip: int = 0,
    limit: int = 25,
    current_user: UserResponse = Depends(get_current_user),
) -> dict:
    """Retrieves paginated matching histories from local storage for the authenticated Admin."""
    history = _load_history()
    sliced = history[skip : skip + limit]
    items = [
        {
            "id": str(h.get("id", "")),
            "cv_filename": h.get("cv_filename", "cv.pdf"),
            "job_title": h.get("job_title", ""),
            "similarity_score": float(h.get("similarity_score", 0.0)),
            "similarity_percentage": float(h.get("similarity_percentage", 0.0)),
            "status_category": h.get("status_category", "Not Fit"),
            "extracted_entities": h.get("extracted_entities", {}),
            "is_validated": bool(h.get("is_validated", False)),
            "created_at": str(h.get("created_at", "")),
        }
        for h in sliced
    ]
    return {"total": len(history), "items": items}


@router.delete("/history/{history_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history_item(
    history_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> None:
    """Deletes a specific matching history record from local storage."""
    history = _load_history()
    new_history = [h for h in history if str(h.get("id")) != history_id]
    if len(new_history) == len(history):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Riwayat pencocokan tidak ditemukan.",
        )
    _save_history(new_history)
    return None
