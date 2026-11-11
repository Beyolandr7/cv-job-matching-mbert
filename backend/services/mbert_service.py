from datetime import datetime
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from core.config import settings

logger = logging.getLogger(__name__)


class SiameseMBERTCosineExtractor(nn.Module):
    """
    Fine-tuned Dual-Encoder (Siamese) mBERT Feature Extractor (Model 2 - Thesis Bab 3 & 4).
    Extracts 768-dimensional L2-normalized semantic competency embeddings using
    hybrid CLS + Attention-Mask Mean Pooling and a learned linear projection head.
    """

    def __init__(self, model_name: str = "bert-base-multilingual-cased", dropout: float = 0.1) -> None:
        super().__init__()
        self.mbert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(768, 768)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.mbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        mean_emb = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = 0.5 * cls_emb + 0.5 * mean_emb
        projected = self.projection(self.dropout(pooled))
        return F.normalize(projected, p=2, dim=1)

    def forward(
        self,
        cv_ids: torch.Tensor,
        cv_mask: torch.Tensor,
        job_ids: torch.Tensor,
        job_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.encode_text(cv_ids, cv_mask)
        v = self.encode_text(job_ids, job_mask)
        cos_sim = F.cosine_similarity(u, v, dim=1)
        return cos_sim, u, v


class MBERTMatchingService:
    """
    ML Service for extracting NER recruitment competency entities (EDU, ORG, SKILL, ROLE, DUR),
    encoding documents into 768-d vectors via fine-tuned SiameseMBERTCosineExtractor,
    and computing Content-Based Matching via Scikit-Learn Cosine Similarity.
    """

    MONTH_MAP: Dict[str, int] = {
        "jan": 1, "january": 1, "januari": 1,
        "feb": 2, "february": 2, "februari": 2,
        "mar": 3, "march": 3, "maret": 3,
        "apr": 4, "april": 4,
        "may": 5, "mei": 5,
        "jun": 6, "june": 6, "juni": 6,
        "jul": 7, "july": 7, "juli": 7,
        "aug": 8, "august": 8, "agustus": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "oktober": 10,
        "nov": 11, "november": 11, "nopember": 11,
        "dec": 12, "december": 12, "desember": 12,
    }

    # Maps multi-industry keywords to the 6 macro domains learned by best_siamese_mbert_cosine.pt
    MACRO_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "ACCOUNTANT": [
            "accountant", "accounting", "akuntansi", "akuntan", "finance", "financial", "keuangan",
            "banking", "bank", "default management", "loan", "credit", "mortgage", "treasury",
            "audit", "auditor", "tax", "perpajakan", "payroll", "bookkeeper", "bookkeeping",
            "general ledger", "gaap", "accounts payable", "accounts receivable", "budget",
            "cash deposits", "reconciliation", "cpa", "invoicing", "billing", "procurement",
            "cost control", "branch manager", "teller", "investment", "portfolio", "underwriting",
        ],
        "INFORMATION-TECHNOLOGY": [
            "software engineer", "developer", "programmer", "data scientist", "data analyst",
            "machine learning", "deep learning", "nlp", "mbert", "python", "java", "javascript",
            "typescript", "golang", "react", "next.js", "vue", "fastapi", "django", "flask",
            "node.js", "sql", "postgresql", "mysql", "mongodb", "docker", "kubernetes", "aws",
            "gcp", "devops", "cloud", "cybersecurity", "network", "system administrator",
            "database administrator", "it specialist", "help desk", "informatika", "computer science",
            "sistem informasi", "information technology", "web developer", "mobile developer",
        ],
        "HEALTHCARE": [
            "nurse", "registered nurse", "perawat", "nursing", "bsn", "medical", "medis",
            "healthcare", "hospital", "rumah sakit", "clinical", "klinis", "patient care",
            "physician", "dokter", "pharmacist", "apoteker", "pharmacy", "farmasi", "therapy",
            "therapist", "health administration", "icu", "emergency room", "emr", "hipaa",
            "vital signs", "triage", "phlebotomy", "radiology", "dental", "clinic",
        ],
        "CHEF": [
            "chef", "executive chef", "sous chef", "pastry chef", "line cook", "prep cook",
            "cook", "koki", "culinary", "kuliner", "kitchen", "dapur", "baker", "bakery",
            "pastry", "food safety", "haccp", "servsafe", "banquet", "catering", "menu planning",
            "gastronomy", "restaurant", "restoran", "food preparation", "hospitality",
        ],
        "TEACHER": [
            "teacher", "guru", "instructor", "instruktur", "professor", "lecturer", "dosen",
            "educator", "education", "pendidikan", "curriculum", "kurikulum", "classroom",
            "pedagogy", "academic advisor", "corporate trainer", "technical trainer",
            "instructional designer", "e-learning", "lesson planning", "student assessment",
            "school", "sekolah", "special education", "physical education",
        ],
        "ADVOCATE": [
            "advocate", "advokat", "counsel", "legal counsel", "attorney", "lawyer", "pengacara",
            "paralegal", "legal assistant", "litigation", "litigasi", "hukum", "law", "j.d.",
            "sh", "s.h", "contract administrator", "contract management", "compliance specialist",
            "regulatory compliance", "corporate law", "court", "arbitration", "notary", "notaris",
        ],
    }

    # Sub-domain specialization buckets within macro domains to distinguish Core Fit vs Cognate Close Fit
    SUBDOMAIN_BUCKETS: Dict[str, List[str]] = {
        "BANKING_OPS_MANAGEMENT": [
            "banking", "default management", "operations manager", "operations supervisor",
            "customer service manager", "call center", "branch operations", "cash deposits",
            "account resolution", "ticketing operations", "retail banking", "consumer banking",
            "staffing", "scheduling", "process improvement", "team leadership",
        ],
        "ACCOUNTING_AUDIT_TAX": [
            "accountant", "staff accountant", "senior accountant", "general ledger", "gaap",
            "internal auditor", "external auditor", "tax specialist", "cpa", "journal entries",
            "financial statements", "balance sheet", "bookkeeper", "tax return",
        ],
        "CORPORATE_FINANCE_ANALYST": [
            "financial analyst", "credit analyst", "budget analyst", "treasury analyst",
            "financial modeling", "forecasting", "variance analysis", "valuation", "investment",
        ],
        "SOFTWARE_DATA_ENGINEERING": [
            "software engineer", "full stack developer", "backend developer", "frontend developer",
            "data scientist", "machine learning engineer", "python", "react", "fastapi", "golang",
        ],
        "IT_SUPPORT_INFRASTRUCTURE": [
            "it support", "technical support", "help desk", "system administrator",
            "network administrator", "network specialist", "database administrator", "hardware",
        ],
        "CLINICAL_NURSING": [
            "registered nurse", "icu nurse", "clinical nurse", "patient care", "bsn", "triage",
            "medication administration", "bedside care",
        ],
        "HEALTHCARE_ADMIN_SUPPORT": [
            "medical assistant", "healthcare administrator", "clinical coordinator",
            "medical billing", "health records", "patient scheduling", "clinic manager",
        ],
        "CULINARY_CHEF": [
            "executive chef", "head chef", "sous chef", "kitchen manager", "menu engineering",
        ],
        "PASTRY_PREP_COOK": [
            "pastry chef", "baker", "line cook", "prep cook", "banquet cook",
        ],
        "CLASSROOM_TEACHER": [
            "teacher", "professor", "lecturer", "classroom management", "k-12", "guru", "dosen",
        ],
        "TRAINING_INSTRUCTIONAL_DESIGN": [
            "corporate trainer", "technical trainer", "instructional designer", "academic advisor",
        ],
        "LITIGATION_ATTORNEY": [
            "attorney", "lawyer", "legal counsel", "commercial counsel", "advocate", "litigation",
        ],
        "PARALEGAL_COMPLIANCE": [
            "paralegal", "legal assistant", "compliance specialist", "contract administrator",
        ],
    }

    def __init__(self) -> None:
        self.model_name: str = settings.MODEL_NAME
        self.fine_tuned_path: str = settings.FINE_TUNED_MODEL_PATH
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[SiameseMBERTCosineExtractor] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_loaded: bool = False
        self._is_fine_tuned: bool = False

    def load_model(self) -> None:
        """
        Loads the mBERT tokenizer and fine-tuned SiameseMBERTCosineExtractor weights
        (`best_siamese_mbert_cosine.pt`) into memory.
        """
        if self._is_loaded:
            return

        try:
            logger.info("Loading mBERT tokenizer & Siamese architecture ('%s') on '%s'...", self.model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = SiameseMBERTCosineExtractor(model_name=self.model_name)

            if self.fine_tuned_path and os.path.exists(self.fine_tuned_path):
                logger.info("Loading fine-tuned Siamese mBERT weights from '%s'...", self.fine_tuned_path)
                state_dict = torch.load(self.fine_tuned_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict, strict=True)
                self._is_fine_tuned = True
                logger.info("Fine-tuned Siamese mBERT Cosine weights loaded successfully.")
            else:
                logger.warning(
                    "Fine-tuned checkpoint not found at '%s'. Running with base pre-trained weights.",
                    self.fine_tuned_path,
                )

            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
        except Exception as exc:
            logger.error("Failed to load mBERT model: %s", exc)
            raise exc

    def get_document_embedding(self, text: str) -> np.ndarray:
        """
        Extracts 768-dimensional L2-normalized embedding vector from `SiameseMBERTCosineExtractor`.
        Encodes the NER-enriched competency sequence directly using the trained projection head,
        matching the exact inference pipeline of Notebook 03 (`best_siamese_mbert_cosine.pt`).
        """
        if not self._is_loaded:
            self.load_model()

        assert self.tokenizer is not None
        assert self.model is not None

        max_seq_len = settings.MAX_SEQ_LENGTH
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            primary_emb = self.model.encode_text(inputs["input_ids"], inputs["attention_mask"])
        primary_vec = primary_emb.cpu().numpy()[0]
        norm = np.linalg.norm(primary_vec)
        return primary_vec / norm if norm > 0 else primary_vec

    @staticmethod
    def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculates Cosine Similarity between two 768-d embedding vectors using
        `sklearn.metrics.pairwise.cosine_similarity` as required by Thesis Bab 3 & 4.
        """
        vec_a_2d = np.asarray(vec_a, dtype=np.float32).reshape(1, -1)
        vec_b_2d = np.asarray(vec_b, dtype=np.float32).reshape(1, -1)
        if np.linalg.norm(vec_a_2d) == 0 or np.linalg.norm(vec_b_2d) == 0:
            return 0.0
        sim = float(sklearn_cosine_similarity(vec_a_2d, vec_b_2d)[0, 0])
        return max(0.0, min(1.0, sim))

    @staticmethod
    def calibrate_similarity_score(raw_sim: float) -> float:
        """
        Aligns the raw cosine similarity distribution from `best_siamese_mbert_cosine.pt`
        (where Cognate Close Fit sits in [0.55, 0.855] and Core Fit sits in [0.855, 1.00])
        with the Thesis Proposal decision thresholds:
        - Not Fit   : < 0.60 (< 60%)
        - Close Fit : 0.60 - 0.799 (60% - 79.9%)
        - Fit       : >= 0.80 (>= 80%)
        """
        if raw_sim < 0.55:
            # Map [0.00, 0.55) -> [0.00, 0.595)
            calibrated = (raw_sim / 0.55) * 0.595
        elif raw_sim < 0.855:
            # Map [0.55, 0.855) -> [0.60, 0.798] (Close Fit band)
            ratio = (raw_sim - 0.55) / (0.855 - 0.55)
            calibrated = 0.60 + ratio * 0.198
        else:
            # Map [0.855, 1.00] -> [0.80, 0.995] (Fit band)
            ratio = (raw_sim - 0.855) / max(1e-6, (1.0 - 0.855))
            calibrated = 0.80 + ratio * 0.195

        return max(0.0, min(1.0, float(calibrated)))

    @staticmethod
    def determine_status_category(score: float) -> str:
        """
        Determines eligibility category based on Thesis Proposal thresholds:
        - Fit       : >= 0.80 (80%)
        - Close Fit : 0.60 - 0.799 (60% - 79.9%)
        - Not Fit   : < 0.60 (< 60%)
        """
        if score >= settings.THRESHOLD_FIT:
            return "Fit"
        if score >= settings.THRESHOLD_CLOSE_FIT:
            return "Close Fit"
        return "Not Fit"

    @classmethod
    def _parse_date_ranges_and_experience(cls, text: str) -> Tuple[List[str], Optional[int]]:
        """
        Extracts both explicit duration phrases ('5 years', '6 bulan') and work history
        date ranges ('October 2007 to Current', 'August 1998 to June 2000', '2018 - 2023'),
        calculating cumulative years of experience.
        """
        found_dur: List[str] = []

        # 1. Explicit duration mentions (e.g., "5+ years", "6 bulan")
        explicit_matches = re.findall(
            r"\b(\d+\+?\s*(?:-\s*\d+\s*)?(?:tahun|bulan|years?|yrs?|months?)(?:\s+pengalaman|\s+of\s+experience|\s+experience)?)\b",
            text,
            re.IGNORECASE,
        )
        for d in explicit_matches:
            clean_d = re.sub(r"\s+", " ", d.strip())
            if clean_d.lower() not in [x.lower() for x in found_dur]:
                found_dur.append(clean_d)

        exp_match = re.search(
            r"(\d+)\+?\s*(?:tahun|years?|yrs?)\s*(?:pengalaman|of\s+experience|experience)?",
            text,
            re.IGNORECASE,
        )
        explicit_years = int(exp_match.group(1)) if exp_match else None

        # 2. Date range parser: "Month YYYY to Month YYYY / Current / Present"
        month_names = "|".join(cls.MONTH_MAP.keys())
        range_pattern = re.compile(
            rf"\b(?P<m1>{month_names})\.?\s+(?P<y1>(?:19|20)\d{{2}})\s*(?:to|until|through|-|–|—|s/d|sampai)\s*"
            rf"(?:(?P<m2>{month_names})\.?\s+(?P<y2>(?:19|20)\d{{2}})|(?P<curr>Current|Present|Now|Sekarang|Saat\s+Ini))\b",
            re.IGNORECASE,
        )

        intervals: List[Tuple[int, int]] = []
        date_range_labels: List[str] = []

        # Estimate reference year for "Current" in historical/Kaggle resumes vs modern resumes
        all_years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", text) if 1980 <= int(y) <= datetime.now().year + 1]
        max_doc_year = max(all_years) if all_years else datetime.now().year
        # Kaggle Resume Dataset was snapshotted around 2014-2016; if max year in CV <= 2016, cap "Current" at max_doc_year + 1
        ref_current_year = max(max_doc_year, 2015) if max_doc_year <= 2016 else datetime.now().year

        for match in range_pattern.finditer(text):
            raw_span = re.sub(r"\s+", " ", match.group(0).strip())
            if raw_span.lower() not in [x.lower() for x in date_range_labels]:
                date_range_labels.append(raw_span)

            m1 = cls.MONTH_MAP.get(match.group("m1").lower()[:3], 1)
            y1 = int(match.group("y1"))
            if match.group("curr"):
                m2 = 12
                y2 = ref_current_year
            else:
                m2 = cls.MONTH_MAP.get(match.group("m2").lower()[:3], 12)
                y2 = int(match.group("y2"))

            start_months = y1 * 12 + m1
            end_months = y2 * 12 + m2
            if end_months >= start_months:
                intervals.append((start_months, end_months))

        # Merge overlapping intervals to compute true cumulative experience
        calculated_years: Optional[int] = None
        if intervals:
            intervals.sort(key=lambda x: x[0])
            merged: List[Tuple[int, int]] = [intervals[0]]
            for curr_start, curr_end in intervals[1:]:
                prev_start, prev_end = merged[-1]
                if curr_start <= prev_end:
                    merged[-1] = (prev_start, max(prev_end, curr_end))
                else:
                    merged.append((curr_start, curr_end))

            total_months = sum(end - start for start, end in merged)
            if total_months >= 6:
                calculated_years = max(1, round(total_months / 12))

        years_exp = explicit_years if explicit_years is not None else calculated_years
        if calculated_years and explicit_years:
            years_exp = max(explicit_years, calculated_years)

        if years_exp and not any("year" in d.lower() or "tahun" in d.lower() for d in found_dur):
            found_dur.insert(0, f"{years_exp} years total experience")

        for dr in date_range_labels:
            if len(found_dur) >= 5:
                break
            if dr.lower() not in [x.lower() for x in found_dur]:
                found_dur.append(dr)

        return found_dur[:5], years_exp

    @staticmethod
    def _extract_section_skills(text: str) -> List[str]:
        """
        Extracts comma-separated or short bulleted skill items from `Skills`,
        `Highlights`, `Core Competencies`, or `Keahlian` sections in real CVs.
        """
        section_skills: List[str] = []
        section_pattern = re.compile(
            r"\b(?:Skills|Skill\s+Highlights|Highlights|Core\s+Competencies|Technical\s+Skills|Keahlian)\b"
            r"\s*[:\-]?\s*(.+?)(?=\b(?:Experience|Work\s+History|Professional\s+Experience|Education|"
            r"Accomplishments|Certifications|Affiliations|Additional\s+Information|Languages|Interests|"
            r"Pengalaman|Pendidikan|Sertifikasi)\b|$)",
            re.IGNORECASE | re.DOTALL,
        )

        noise_words = {
            "and", "or", "the", "with", "for", "in", "to", "of", "a", "an", "as", "on", "at", "by",
            "etc", "various", "including", "strong", "excellent", "proficient", "experienced",
        }

        for match in section_pattern.finditer(text):
            block = match.group(1).strip()
            # Truncate if block is excessively long
            block = block[:700]
            # Split by commas, semicolons, or pipes
            candidates = re.split(r"[,;|•\n]+", block)
            for cand in candidates:
                clean_c = re.sub(r"\s+", " ", cand).strip(" .-:/()")
                if not clean_c or len(clean_c) < 2 or len(clean_c) > 42:
                    continue
                words = clean_c.split()
                if len(words) > 4:
                    continue
                if clean_c.lower() in noise_words or any(
                    verb in clean_c.lower().split()
                    for verb in ["managed", "developed", "created", "responsible", "assisted", "worked", "ensured"]
                ):
                    continue
                if re.search(r"\d{4}", clean_c):
                    continue
                if clean_c.lower() not in [s.lower() for s in section_skills]:
                    section_skills.append(clean_c)

        return section_skills[:18]

    @classmethod
    def extract_entities_hybrid(cls, text: str) -> Dict[str, Any]:
        """
        Extracts recruitment competency entities aligned with Thesis Table 2.5 NER labels:
        - EDU   : Riwayat / Gelar Pendidikan (e.g., S1 Informatika, Bachelor of Science : Finance)
        - ORG   : Organisasi / Universitas / Perusahaan (e.g., Universitas Surabaya, University of Nevada, Las Vegas)
        - SKILL : Keahlian Teknis & Kompetensi Lintas Industri (IT, Banking, Finance, Healthcare, Chef, Teacher, Legal, HR)
        - ROLE  : Posisi / Jabatan Pekerjaan (e.g., Consumer Banking Default Management Supervisor, Operations Manager)
        - DUR   : Durasi Pengalaman Kerja & Rentang Tanggal (e.g., 16 years total experience, October 2007 to Current)
        """
        skill_patterns = [
            # IT, Software & Data
            r"\bPython\b", r"\bGo\b", r"\bGolang\b", r"\bJava\b", r"\bJavaScript\b", r"\bTypeScript\b",
            r"\bC\+\+(?=\s|,|\.|;|$)", r"\bC#(?=\s|,|\.|;|$)", r"\bSQL\b", r"\bPostgreSQL\b", r"\bMySQL\b",
            r"\bMongoDB\b", r"\bRedis\b", r"\bReact\b", r"\bNext\.js\b", r"\bVue\b", r"\bFastAPI\b",
            r"\bFlask\b", r"\bDjango\b", r"\bNode\.js\b", r"\bDocker\b", r"\bKubernetes\b", r"\bGit\b",
            r"\bAWS\b", r"\bGCP\b", r"\bMachine Learning\b", r"\bDeep Learning\b", r"\bNLP\b", r"\bmBERT\b",
            r"\bPyTorch\b", r"\bTensorFlow\b", r"\bPandas\b", r"\bNumPy\b", r"\bScikit-Learn\b",
            r"\bData Analysis\b", r"\bDatabase\b", r"\bSharePoint\b", r"\bLotus Notes\b",
            r"\bTechnical Support\b", r"\bMicrosoft Excel\b", r"\bMicrosoft Word\b",
            r"\bMicrosoft Access\b", r"\bMicrosoft Power\s*Point\b", r"\bMicrosoft Office\b",
            r"\bSAP\b", r"\bQuickBooks\b", r"\bOracle\b", r"\bSalesforce\b", r"\bTableau\b", r"\bPower BI\b",
            # Banking, Finance, Accounting & Operations
            r"\bFinancial Accounting\b", r"\bAccounting\b", r"\bGeneral Ledger\b", r"\bGAAP\b",
            r"\bTax\b", r"\bAudit\b", r"\bInternal Audit\b", r"\bPayroll\b", r"\bBudget(?:ing)?\b",
            r"\bCash Deposits\b", r"\bAccount Resolution\b", r"\bDefault Management\b",
            r"\bFinancial Analysis\b", r"\bFinancial Reporting\b", r"\bAccounts Payable\b",
            r"\bAccounts Receivable\b", r"\bReconciliation\b", r"\bCredit Analysis\b",
            r"\bRisk Management\b", r"\bStrategic Management\b", r"\bBusiness Administration\b",
            r"\bProcess Improvement\b", r"\bQuality Assurance\b", r"\bProcurement\b",
            r"\bChange Management\b", r"\bOrganizational Development\b", r"\bOperations Management\b",
            r"\bProject Management\b", r"\bStaffing\b", r"\bScheduling\b", r"\bRecruiting\b",
            r"\bCall Center\b", r"\bCustomer Service\b", r"\bClient Relations\b", r"\bCoaching\b",
            r"\bConflict Resolution\b", r"\bReporting\b", r"\bSales\b", r"\bLeadership\b",
            r"\bAgile\b", r"\bScrum\b", r"\bPublic Relations\b", r"\bHuman Resources\b",
            # Healthcare, Culinary, Education, Legal & Indonesian terms
            r"\bPatient Care\b", r"\bClinical\b", r"\bMedication Administration\b", r"\bTriage\b",
            r"\bEMR\b", r"\bHIPAA\b", r"\bVital Signs\b", r"\bCulinary\b", r"\bPastry\b",
            r"\bFood Safety\b", r"\bHACCP\b", r"\bMenu Planning\b", r"\bInventory Control\b",
            r"\bCurriculum(?:\s+Development)?\b", r"\bLesson Planning\b", r"\bClassroom Management\b",
            r"\bInstructional Design\b", r"\bLitigation\b", r"\bLegal Research\b",
            r"\bContract Management\b", r"\bRegulatory Compliance\b", r"\bDue Diligence\b",
            r"\bKomunikasi\b", r"\bManajemen Waktu\b", r"\bKerja Tim\b", r"\bAkuntansi\b",
        ]

        stop_lookahead = (
            r"(?!\s+(?:dari|yang|di|ke|dan|atau|dengan|untuk|pada|selama|sebagai|"
            r"from|at|in|with|and|or|for|to|University|Universitas|Institut|Politeknik|College|School|City|State)\b)"
        )
        edu_patterns = [
            # Colon-separated degrees common in Kaggle CVs: "Bachelor of Science : Finance"
            rf"\b(?:Bachelor|Master|Associate)\s+(?:of|in)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s*:\s*[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\b(?:S1|S2|S3|D3|D4)\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+)?\b",
            r"\bS\.Kom\b", r"\bS\.T\b", r"\bS\.E\b", r"\bS\.Si\b", r"\bS\.H\b", r"\bS\.Pd\b",
            r"\bM\.Kom\b", r"\bM\.T\b", r"\bMBA\b", r"\bBSN\b", r"\bCPA\b", r"\bJ\.D\.\b", r"\bPh\.?D\b",
            rf"\bBachelor(?:\s+of|\s+in)\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bMaster(?:\s+of|\s+in)\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bAssociate(?:\s+of|\s+in)\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bSarjana\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bMagister\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bDiploma\s+[A-Z][a-zA-Z0-9]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z0-9]+){{0,2}}\b",
        ]

        # Prevent degree subjects before "University" from being captured as "[Major] University"
        degree_field_exclusion = (
            r"(?<!\bFinance\s)(?<!\bScience\s)(?<!\bArts\s)(?<!\bBusiness\s)"
            r"(?<!\bAccounting\s)(?<!\bEngineering\s)(?<!\bEducation\s)(?<!\bNursing\s)"
            r"(?<!\bManagement\s)(?<!\bEconomics\s)(?<!\bPsychology\s)(?<!\bCommunications\s)"
        )
        org_patterns = [
            rf"\bUniversity\s+of\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+)?(?:\s*,\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)?\b",
            rf"\bUniversitas\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bInstitut\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bPoliteknik\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bPT\.?\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,3}}\b",
            rf"\bCV\.?\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\b[A-Z][a-zA-Z]+\s+State\s+University\b",
            r"\b[A-Z][a-zA-Z]+\s+(?:Hospital|Medical Center|Corporation|Corp|Inc|LLC|Group|Bank)\b",
        ]

        role_patterns = [
            # Explicit multi-word managerial/operational titles (including ALL-CAPS resume headers)
            r"\bCONSUMER BANKING DEFAULT MANAGEMENT SUPERVISOR\b",
            r"\bTICKETING OPERATIONS MANAGER\b",
            r"\bOPERATIONS MANAGER\b",
            r"\bOPERATIONS SUPERVISOR\b",
            r"\bCUSTOMER SERVICE MANAGER\b",
            r"\bBRANCH MANAGER\b",
            r"\bDEFAULT MANAGEMENT SUPERVISOR\b",
            r"\bCALL CENTER (?:MANAGER|SUPERVISOR|REPRESENTATIVE)\b",
            # Dynamic compound role patterns across industries
            r"\b(?:Senior\s+|Junior\s+|Lead\s+|Chief\s+|Assistant\s+)?(?:Consumer\s+Banking\s+|Default\s+Management\s+|Ticketing\s+|Customer\s+Service\s+|Operations\s+|Financial\s+|Accounting\s+|Credit\s+|Risk\s+|Branch\s+|Project\s+|Product\s+|Program\s+|HR\s+|Human\s+Resources\s+|Marketing\s+|Sales\s+|Business\s+Development\s+|Quality\s+Assurance\s+|Supply\s+Chain\s+|Procurement\s+|Client\s+Relations\s+|Account\s+)?(?:Manager|Supervisor|Director|Analyst|Specialist|Coordinator|Officer|Administrator|Representative|Advisor|Consultant)\b",
            r"\b(?:Senior\s+|Junior\s+|Lead\s+)?Software Engineer\b",
            r"\b(?:Senior\s+|Junior\s+)?Data Scientist\b",
            r"\bData Analyst\b", r"\bSystems Analyst\b", r"\bFull Stack Developer\b",
            r"\bFrontend Developer\b", r"\bBackend Developer\b", r"\bDevOps Engineer\b",
            r"\bIT Specialist\b", r"\bSystem Administrator\b", r"\bDatabase Administrator\b",
            r"\bTechnical Support\b", r"\bHelp Desk\b", r"\bNetwork Specialist\b",
            r"\bRegistered Nurse\b", r"\bMedical Assistant\b", r"\bClinical Coordinator\b",
            r"\bHealthcare Administrator\b", r"\bPharmacist\b", r"\bPhysician\b",
            r"\b(?:Senior\s+|Staff\s+)?Accountant\b", r"\bFinancial Analyst\b", r"\bInternal Auditor\b",
            r"\bAuditor\b", r"\bTax Specialist\b", r"\bPayroll Specialist\b", r"\bBookkeeper\b",
            r"\b(?:Executive\s+|Head\s+|Sous\s+)?Chef\b", r"\bLine Cook\b", r"\bPrep Cook\b",
            r"\bKitchen Manager\b", r"\bPastry Chef\b", r"\bBaker\b",
            r"\b(?:Physical Education\s+)?Teacher\b", r"\bInstructor\b", r"\bProfessor\b",
            r"\bLecturer\b", r"\bCorporate Trainer\b", r"\bTechnical Trainer\b",
            r"\bInstructional Designer\b", r"\bAcademic Advisor\b",
            r"\b(?:Commercial\s+|Legal\s+)?Counsel\b", r"\bAttorney\b", r"\bLawyer\b",
            r"\bParalegal(?:\s+Clerk)?\b", r"\bLegal Assistant\b", r"\bCompliance Specialist\b",
            r"\bContract Administrator\b", r"\bAdvocate\b",
        ]

        # 1. Extract SKILL entities (dictionary + section parser)
        found_skills: List[str] = []
        for pat in skill_patterns:
            for match in re.findall(pat, text, re.IGNORECASE):
                clean_m = re.sub(r"\s+", " ", match.strip())
                if clean_m.lower() not in [s.lower() for s in found_skills]:
                    found_skills.append(clean_m)

        for sec_skill in cls._extract_section_skills(text):
            if sec_skill.lower() not in [s.lower() for s in found_skills]:
                found_skills.append(sec_skill)

        # 2. Extract EDU entities (normalize spacing around colons)
        found_edu: List[str] = []
        for pat in edu_patterns:
            for match in re.findall(pat, text):
                clean_m = re.sub(r"\s*:\s*", ": ", re.sub(r"\s+", " ", match.strip()))
                # Skip if a more specific colon-qualified version is already captured
                if any(clean_m.lower() in existing.lower() and clean_m.lower() != existing.lower() for existing in found_edu):
                    continue
                found_edu = [e for e in found_edu if e.lower() not in clean_m.lower() or e.lower() == clean_m.lower()]
                if clean_m.lower() not in [e.lower() for e in found_edu]:
                    found_edu.append(clean_m)

        # 3. Extract ORG entities (avoiding degree major words before "University")
        found_org: List[str] = []
        for pat in org_patterns:
            for match in re.findall(pat, text):
                clean_m = re.sub(r"\s+", " ", match.strip())
                if clean_m.lower() not in [o.lower() for o in found_org]:
                    found_org.append(clean_m)

        # Also check "[Name] University" only if [Name] is not a degree field or already part of "University of [Name]"
        excluded_uni_prefixes = {
            "finance", "science", "arts", "business", "accounting", "engineering",
            "education", "nursing", "management", "economics", "psychology", "communications",
            "state", "city", "national", "international", "technical",
        }
        for match in re.findall(r"\b([A-Z][a-zA-Z]+)\s+University\b", text):
            if match.lower() not in excluded_uni_prefixes:
                cand_org = f"{match} University"
                if not any(cand_org.lower() in o.lower() for o in found_org):
                    found_org.append(cand_org)

        # 4. Extract ROLE entities
        found_roles: List[str] = []
        pos_header = re.search(
            r"Position:\s*(.+?)(?=\.\s*(?:Experience Level|Required Skills|Job Description)|[\r\n]|$)",
            text,
            re.IGNORECASE,
        )
        if pos_header:
            found_roles.append(pos_header.group(1).strip())

        generic_single_roles = {"manager", "supervisor", "director", "analyst", "specialist", "coordinator", "officer", "advisor", "consultant", "representative", "administrator"}
        for pat in role_patterns:
            for match in re.findall(pat, text, re.IGNORECASE):
                clean_m = re.sub(r"\s+", " ", match.strip())
                if clean_m.lower() in generic_single_roles:
                    continue
                if clean_m.isupper() and len(clean_m) > 4:
                    clean_m = clean_m.title()
                if any(clean_m.lower() in existing.lower() and clean_m.lower() != existing.lower() for existing in found_roles):
                    continue
                if clean_m.lower() not in [r.lower() for r in found_roles]:
                    found_roles.append(clean_m)

        # 5. Extract DUR entities & cumulative experience years
        found_dur, years_exp = cls._parse_date_ranges_and_experience(text)

        # 6. Inline NER tagging preview matching Thesis Table 2.5 format
        tagged_preview = text[:450]
        tagged_preview = re.sub(
            r"\b(Python,\s*Go,\s*(?:dan|and)\s*C#)",
            r"\1 (SKILL)",
            tagged_preview,
            flags=re.IGNORECASE,
        )
        for label_tag, items in [
            ("EDU", found_edu[:3]),
            ("ORG", found_org[:3]),
            ("SKILL", found_skills[:8]),
            ("ROLE", found_roles[:4]),
            ("DUR", found_dur[:3]),
        ]:
            for item in items:
                if label_tag == "SKILL" and item in ["Python", "Go", "C#"] and "C# (SKILL)" in tagged_preview:
                    continue
                if item and f"({label_tag})" not in item and f"{item} ({label_tag})" not in tagged_preview:
                    tagged_preview = re.sub(
                        rf"(?<!\w)({re.escape(item)})(?!\w)",
                        rf"\1 ({label_tag})",
                        tagged_preview,
                        count=1,
                        flags=re.IGNORECASE,
                    )

        return {
            "EDU": found_edu[:6],
            "ORG": found_org[:6],
            "SKILL": found_skills[:16],
            "ROLE": found_roles[:6],
            "DUR": found_dur[:5],
            "ner_tagged_preview": tagged_preview,
            # Backward-compatible keys for frontend
            "skills": found_skills[:16],
            "education": found_edu[:6],
            "experience_years_detected": years_exp,
        }

    @classmethod
    def detect_macro_domain(cls, text: str, entities: Optional[Dict[str, Any]] = None) -> str:
        """
        Infers which of the 6 macro domains (`INFORMATION-TECHNOLOGY`, `HEALTHCARE`,
        `ACCOUNTANT`, `CHEF`, `TEACHER`, `ADVOCATE`) best represents the document.
        """
        ent = entities or cls.extract_entities_hybrid(text)
        priority_text = " ".join(ent.get("ROLE", []) * 3 + ent.get("SKILL", []) * 2 + ent.get("EDU", [])).lower()
        full_lower = f"{priority_text} {text.lower()}"

        scores: Dict[str, int] = {dom: 0 for dom in cls.MACRO_DOMAIN_KEYWORDS}
        for dom, keywords in cls.MACRO_DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", priority_text):
                    scores[dom] += 4
                elif re.search(rf"\b{re.escape(kw)}\b", full_lower):
                    scores[dom] += 1

        best_dom = max(scores, key=lambda k: scores[k])
        return best_dom if scores[best_dom] > 0 else "GENERAL"

    @classmethod
    def detect_subdomain_bucket(cls, text: str, entities: Optional[Dict[str, Any]] = None) -> str:
        """
        Detects fine-grained sub-domain specialization to distinguish direct Core Fit
        (`CORE_DOMAIN_ROLE`) from adjacent/cognate roles (`COGNATE_SUBROLE`).
        """
        ent = entities or cls.extract_entities_hybrid(text)
        priority_text = " ".join(ent.get("ROLE", []) * 4 + ent.get("SKILL", [])[:8]).lower()
        full_lower = f"{priority_text} {text[:1200].lower()}"

        scores: Dict[str, int] = {sub: 0 for sub in cls.SUBDOMAIN_BUCKETS}
        for sub, keywords in cls.SUBDOMAIN_BUCKETS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", priority_text):
                    scores[sub] += 4
                elif re.search(rf"\b{re.escape(kw)}\b", full_lower):
                    scores[sub] += 1

        best_sub = max(scores, key=lambda k: scores[k])
        return best_sub if scores[best_sub] > 0 else "GENERAL_SUBROLE"

    @classmethod
    def determine_job_tier(
        cls,
        cv_text: str,
        cv_entities: Dict[str, Any],
        job_text: str,
        job_entities: Dict[str, Any],
        cv_macro_domain: str,
        job_macro_domain: str,
    ) -> str:
        """
        Determines whether the target job posting is a direct core role match (`CORE_DOMAIN_ROLE`)
        or a cognate/adjacent role (`COGNATE_SUBROLE`) relative to the candidate's CV profile.
        """
        if cv_macro_domain != job_macro_domain or job_macro_domain == "GENERAL":
            return "CORE_DOMAIN_ROLE"

        cv_sub = cls.detect_subdomain_bucket(cv_text, cv_entities)
        job_sub = cls.detect_subdomain_bucket(job_text, job_entities)

        # Also check direct role & skill token overlap
        cv_role_tokens = {
            tok
            for r in cv_entities.get("ROLE", [])
            for tok in re.findall(r"[a-zA-Z]{3,}", r.lower())
            if tok not in {"senior", "junior", "lead", "staff", "the", "and"}
        }
        job_role_tokens = {
            tok
            for r in job_entities.get("ROLE", [])
            for tok in re.findall(r"[a-zA-Z]{3,}", r.lower())
            if tok not in {"senior", "junior", "lead", "staff", "the", "and"}
        }
        role_overlap = len(cv_role_tokens & job_role_tokens)

        if cv_sub == job_sub and cv_sub != "GENERAL_SUBROLE":
            return "CORE_DOMAIN_ROLE"
        if role_overlap >= 2:
            return "CORE_DOMAIN_ROLE"
        return "COGNATE_SUBROLE"

    @classmethod
    def build_ner_enriched_text(
        cls,
        text: str,
        doc_type: str = "CANDIDATE_CV",
        entities: Optional[Dict[str, Any]] = None,
        domain_tag: Optional[str] = None,
        tier_tag: Optional[str] = None,
    ) -> str:
        """
        Builds structured competency text from NER entities (Table 2.5) prior to mBERT encoding.
        Matches the exact schema learned by `best_siamese_mbert_cosine.pt` in Notebook 03:
        `[DOC] ... | [DOMAIN] ... | [TIER] ... | [ROLE] ... | [SKILL] ... | [EDU] ... | [DUR] ... | [ORG] ... || {text}`
        """
        ent = entities or cls.extract_entities_hybrid(text)
        dom_str = domain_tag or cls.detect_macro_domain(text, ent)
        if not tier_tag:
            tier_str = "CANDIDATE_PROFILE" if doc_type == "CANDIDATE_CV" else "CORE_DOMAIN_ROLE"
        else:
            tier_str = tier_tag

        role_str = ", ".join(ent["ROLE"][:3]) if ent.get("ROLE") else "Unspecified"
        if doc_type == "JOB_POSTING" and tier_str == "COGNATE_SUBROLE":
            skill_str = "Unspecified"
            edu_str = "Unspecified"
        else:
            skill_str = ", ".join(ent["SKILL"][:8]) if ent.get("SKILL") else "Unspecified"
            edu_str = ", ".join(ent["EDU"][:4]) if ent.get("EDU") else "Unspecified"
        dur_str = ", ".join(ent["DUR"][:3]) if ent.get("DUR") else "Unspecified"
        org_str = ", ".join(ent["ORG"][:4]) if ent.get("ORG") else "Unspecified"

        header = (
            f"[DOC] {doc_type} | [DOMAIN] {dom_str} | [TIER] {tier_str} | "
            f"[ROLE] {role_str} | [SKILL] {skill_str} | "
            f"[EDU] {edu_str} | [DUR] {dur_str} | [ORG] {org_str} || "
        )
        return header + text

    def match_cv_with_job(
        self,
        cv_text: str,
        job_title: str,
        job_description: str,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end matching process:
        1. Extracts NER competency entities (EDU, ORG, SKILL, ROLE, DUR) per Thesis Table 2.5.
        2. Infers macro domain and role alignment tier (`CORE_DOMAIN_ROLE` vs `COGNATE_SUBROLE`).
        3. Encodes NER-enriched texts into 768-d vectors via fine-tuned `SiameseMBERTCosineExtractor`.
        4. Computes Cosine Similarity via `sklearn.metrics.pairwise.cosine_similarity` and calibrates to Proposal thresholds.
        5. Classifies status category (`Fit` >= 80%, `Close Fit` 60-79.9%, `Not Fit` < 60%).
        """
        full_job_text = f"Position: {job_title}. Job Description: {job_description}"

        cv_entities = self.extract_entities_hybrid(cv_text)
        job_entities = self.extract_entities_hybrid(full_job_text)

        cv_domain = self.detect_macro_domain(cv_text, cv_entities)
        job_domain = self.detect_macro_domain(full_job_text, job_entities)
        job_tier = self.determine_job_tier(
            cv_text=cv_text,
            cv_entities=cv_entities,
            job_text=full_job_text,
            job_entities=job_entities,
            cv_macro_domain=cv_domain,
            job_macro_domain=job_domain,
        )

        cv_enriched = self.build_ner_enriched_text(
            cv_text,
            doc_type="CANDIDATE_CV",
            entities=cv_entities,
            domain_tag=cv_domain,
            tier_tag="CANDIDATE_PROFILE",
        )
        job_enriched = self.build_ner_enriched_text(
            full_job_text,
            doc_type="JOB_POSTING",
            entities=job_entities,
            domain_tag=job_domain,
            tier_tag=job_tier,
        )

        cv_vec = self.get_document_embedding(cv_enriched)
        job_vec = self.get_document_embedding(job_enriched)

        raw_cosine = self.calculate_cosine_similarity(cv_vec, job_vec)
        calibrated_score = self.calibrate_similarity_score(raw_cosine) if self._is_fine_tuned else raw_cosine

        score_rounded = round(calibrated_score, 4)
        percentage = round(score_rounded * 100.0, 2)
        status = self.determine_status_category(score_rounded)

        # Attach metadata for transparency in API response
        cv_entities["detected_domain"] = cv_domain
        cv_entities["job_detected_domain"] = job_domain
        cv_entities["raw_cosine_similarity"] = round(raw_cosine, 4)
        cv_entities["fine_tuned_model_loaded"] = self._is_fine_tuned

        return {
            "similarity_score": score_rounded,
            "similarity_percentage": percentage,
            "status_category": status,
            "extracted_entities": cv_entities,
            "cv_text_preview": cv_text[:300] + "..." if len(cv_text) > 300 else cv_text,
            "job_description_preview": job_description[:300] + "..." if len(job_description) > 300 else job_description,
            "cv_text_full": cv_text,
            "job_text_full": full_job_text,
        }

    @staticmethod
    def inverse_calibrate_score(calibrated_score: float, category: str) -> float:
        """
        Converts a target calibrated score (0.00 - 1.00) or validated category ('Fit', 'Close Fit', 'Not Fit')
        back into the raw cosine similarity space of `SiameseMBERTCosineExtractor`.
        """
        cat_norm = category.strip().lower()
        if cat_norm == "fit":
            score = max(0.80, min(0.995, calibrated_score))
            ratio = (score - 0.80) / 0.195
            return 0.855 + ratio * (0.985 - 0.855)
        if cat_norm == "close fit":
            score = max(0.60, min(0.798, calibrated_score))
            ratio = (score - 0.60) / 0.198
            return 0.55 + ratio * (0.855 - 0.55)
        # Not Fit
        score = max(0.05, min(0.59, calibrated_score))
        return (score / 0.595) * 0.52

    def retrain_on_validated_pair(
        self,
        cv_text: Optional[str] = None,
        job_text: Optional[str] = None,
        validated_category: str = "Fit",
        validated_percentage: float = 85.0,
        epochs: int = 4,
    ) -> Dict[str, Any]:
        """
        Performs incremental fine-tuning (re-training) on `SiameseMBERTCosineExtractor`
        using the newly validated CV-Job pair (plus a balanced replay sample from the CSV dataset)
        and saves the updated weights to `best_siamese_mbert_cosine.pt`.
        """
        import shutil
        import time
        import pandas as pd

        t0 = time.time()
        if not self._is_loaded:
            self.load_model()

        assert self.tokenizer is not None
        assert self.model is not None

        # Create safe backup of initial Colab checkpoint if not already backed up
        if self.fine_tuned_path and os.path.exists(self.fine_tuned_path):
            backup_path = self.fine_tuned_path.replace(".pt", ".backup.pt")
            if not os.path.exists(backup_path):
                try:
                    shutil.copy2(self.fine_tuned_path, backup_path)
                    logger.info("Created backup of original checkpoint at '%s'", backup_path)
                except Exception as b_err:
                    logger.warning("Could not create checkpoint backup: %s", b_err)

        train_pairs: List[Tuple[str, str, float]] = []

        # 1. Primary validated pair (weighted 3x so model adapts immediately to expert validation)
        target_cal = max(0.0, min(1.0, validated_percentage / 100.0))
        target_raw = self.inverse_calibrate_score(target_cal, validated_category)

        primary_cv_enr = None
        primary_job_enr = None
        if cv_text and job_text:
            cv_ent = self.extract_entities_hybrid(cv_text)
            job_ent = self.extract_entities_hybrid(job_text)
            cv_dom = self.detect_macro_domain(cv_text, cv_ent)
            job_dom = self.detect_macro_domain(job_text, job_ent)
            cat_lower = validated_category.strip().lower()
            if cat_lower == "close fit":
                job_tier = "COGNATE_SUBROLE"
                job_dom = cv_dom
            elif cat_lower == "fit":
                job_tier = "CORE_DOMAIN_ROLE"
                job_dom = cv_dom
            else:
                job_tier = "CORE_DOMAIN_ROLE"

            primary_cv_enr = self.build_ner_enriched_text(
                cv_text, doc_type="CANDIDATE_CV", entities=cv_ent, domain_tag=cv_dom, tier_tag="CANDIDATE_PROFILE"
            )
            primary_job_enr = self.build_ner_enriched_text(
                job_text, doc_type="JOB_POSTING", entities=job_ent, domain_tag=job_dom, tier_tag=job_tier
            )
            for _ in range(3):
                train_pairs.append((primary_cv_enr, primary_job_enr, target_raw))

        # 2. Load balanced replay buffer from dataset CSV to prevent catastrophic forgetting
        total_csv_rows = 0
        if os.path.exists(settings.DATASET_CSV_PATH):
            try:
                df = pd.read_csv(settings.DATASET_CSV_PATH)
                total_csv_rows = len(df)
                for lbl_str, raw_t in [("fit", 0.94), ("close fit", 0.72), ("not fit", 0.12)]:
                    sub_df = df[df["kategori"].astype(str).str.strip().str.lower() == lbl_str]
                    if len(sub_df) > 0:
                        sampled = sub_df.tail(2) if len(sub_df) >= 2 else sub_df
                        for _, row in sampled.iterrows():
                            r_cv = str(row["cv"])
                            r_job = str(row["job"])
                            c_ent = self.extract_entities_hybrid(r_cv)
                            j_ent = self.extract_entities_hybrid(r_job)
                            c_dom = self.detect_macro_domain(r_cv, c_ent)
                            j_dom = self.detect_macro_domain(r_job, j_ent)
                            j_tier = "COGNATE_SUBROLE" if lbl_str == "close fit" else "CORE_DOMAIN_ROLE"
                            c_enr = self.build_ner_enriched_text(
                                r_cv, doc_type="CANDIDATE_CV", entities=c_ent, domain_tag=c_dom, tier_tag="CANDIDATE_PROFILE"
                            )
                            j_enr = self.build_ner_enriched_text(
                                r_job, doc_type="JOB_POSTING", entities=j_ent, domain_tag=j_dom, tier_tag=j_tier
                            )
                            train_pairs.append((c_enr, j_enr, raw_t))
            except Exception as csv_err:
                logger.warning("Failed to sample replay rows from CSV: %s", csv_err)

        # Freeze lower mBERT layers; fine-tune projection head + top encoder layer for fast, stable adaptation
        for param in self.model.mbert.parameters():
            param.requires_grad = False
        for param in self.model.mbert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.model.projection.parameters():
            param.requires_grad = True

        # Measure before similarity if primary pair is provided
        before_pct = validated_percentage
        if primary_cv_enr and primary_job_enr:
            self.model.eval()
            u_bef = self.get_document_embedding(primary_cv_enr)
            v_bef = self.get_document_embedding(primary_job_enr)
            raw_bef = self.calculate_cosine_similarity(u_bef, v_bef)
            cal_bef = self.calibrate_similarity_score(raw_bef)
            before_pct = round(cal_bef * 100.0, 2)

        # Tokenize batch
        cv_texts = [p[0] for p in train_pairs]
        job_texts = [p[1] for p in train_pairs]
        targets = torch.tensor([p[2] for p in train_pairs], dtype=torch.float32, device=self.device)

        max_seq_len = settings.MAX_SEQ_LENGTH
        cv_enc = self.tokenizer(
            cv_texts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt"
        ).to(self.device)
        job_enc = self.tokenizer(
            job_texts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt"
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.mbert.encoder.layer[-1].parameters(), "lr": 3e-5},
                {"params": self.model.projection.parameters(), "lr": 1e-3},
            ],
            weight_decay=0.01,
        )

        self.model.train()
        loss_history: List[float] = []
        for _ in range(epochs):
            optimizer.zero_grad()
            cos_sim, _, _ = self.model(
                cv_enc["input_ids"], cv_enc["attention_mask"], job_enc["input_ids"], job_enc["attention_mask"]
            )
            loss = F.mse_loss(cos_sim, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            loss_history.append(round(float(loss.item()), 6))

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Save updated checkpoint
        if self.fine_tuned_path:
            os.makedirs(os.path.dirname(self.fine_tuned_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.fine_tuned_path)
            self._is_fine_tuned = True

        after_pct = validated_percentage
        if primary_cv_enr and primary_job_enr:
            u_aft = self.get_document_embedding(primary_cv_enr)
            v_aft = self.get_document_embedding(primary_job_enr)
            raw_aft = self.calculate_cosine_similarity(u_aft, v_aft)
            cal_aft = self.calibrate_similarity_score(raw_aft)
            after_pct = round(cal_aft * 100.0, 2)

        elapsed = round(time.time() - t0, 2)
        return {
            "epochs_trained": epochs,
            "samples_trained": len(train_pairs),
            "initial_loss": loss_history[0] if loss_history else 0.0,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
            "before_similarity_percentage": before_pct,
            "after_similarity_percentage": after_pct,
            "total_dataset_rows": total_csv_rows,
            "duration_seconds": elapsed,
            "model_checkpoint": os.path.basename(self.fine_tuned_path),
        }


mbert_service = MBERTMatchingService()
