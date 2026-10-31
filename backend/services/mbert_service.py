import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from core.config import settings

logger = logging.getLogger(__name__)


class MBERTMatchingService:
    """
    ML Service for extracting mBERT embeddings using Sliding Window Mean Pooling
    and performing Content-Based Matching via Cosine Similarity.
    """

    def __init__(self) -> None:
        self.model_name: str = settings.MODEL_NAME
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_loaded: bool = False

    def load_model(self) -> None:
        """
        Loads the pre-trained mBERT model and tokenizer into memory.
        Preloading at application startup eliminates cold-start latency.
        """
        if self._is_loaded:
            return

        try:
            logger.info("Loading mBERT model '%s' on device '%s'...", self.model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("mBERT model '%s' loaded successfully.", self.model_name)
        except Exception as exc:
            logger.error("Failed to load mBERT model: %s", exc)
            raise exc

    def _mean_pooling(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes mean pooling of token representations considering the attention mask.
        """
        token_embeddings = model_output.last_hidden_state  # Shape: [batch_size, seq_len, 768]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_document_embedding(self, text: str) -> np.ndarray:
        """
        Extracts 768-dimensional mBERT embedding vector for arbitrary-length text
        using Sliding Window Chunking + Mean Pooling.
        """
        if not self._is_loaded:
            self.load_model()

        assert self.tokenizer is not None
        assert self.model is not None

        # Tokenize entire text without truncation
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        max_len = settings.MAX_SEQ_LENGTH - 2  # Space for [CLS] and [SEP]
        stride = settings.CHUNK_STRIDE

        # Case 1: Short document fits in a single window
        if len(tokens) <= max_len:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=settings.MAX_SEQ_LENGTH,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = self._mean_pooling(outputs, inputs["attention_mask"])

            vec = embedding.cpu().numpy()[0]
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec

        # Case 2: Long document (Sliding Window Chunking)
        chunk_embeddings: List[np.ndarray] = []
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + max_len]
            if not chunk_tokens:
                break

            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            inputs = self.tokenizer(
                chunk_text,
                padding=True,
                truncation=True,
                max_length=settings.MAX_SEQ_LENGTH,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = self._mean_pooling(outputs, inputs["attention_mask"])

            chunk_embeddings.append(emb.cpu().numpy()[0])

            if i + max_len >= len(tokens):
                break

        avg_vec = np.mean(chunk_embeddings, axis=0)
        norm = np.linalg.norm(avg_vec)
        return avg_vec / norm if norm > 0 else avg_vec

    @staticmethod
    def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculates normalized Cosine Similarity between two embedding vectors.
        Returns float clamped between 0.0000 and 1.0000.
        """
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0

        sim = float(dot / (norm_a * norm_b))
        return max(0.0, min(1.0, sim))

    @staticmethod
    def determine_status_category(score: float) -> str:
        """
        Determines eligibility category based purely on Cosine Similarity score.
        """
        if score >= settings.THRESHOLD_FIT:
            return "Fit"
        if score >= settings.THRESHOLD_CLOSE_FIT:
            return "Close Fit"
        return "Not Fit"

    @staticmethod
    def extract_entities_hybrid(text: str) -> Dict[str, Any]:
        """
        Extracts recruitment competency entities aligned with Thesis Table 2.5 NER labels:
        - EDU   : Riwayat / Gelar Pendidikan (e.g., S1 Informatika, Bachelor of Computer Science)
        - ORG   : Organisasi / Universitas / Perusahaan (e.g., Universitas Surabaya, PT Telkom)
        - SKILL : Keahlian Teknis & Kompetensi (e.g., Python, Go, C#, SQL, Machine Learning)
        - ROLE  : Posisi / Jabatan Pekerjaan (e.g., Software Engineer, Data Scientist)
        - DUR   : Durasi Pengalaman Kerja (e.g., 6 bulan, 2+ years)
        """
        skill_patterns = [
            r"\bPython\b", r"\bGo\b", r"\bGolang\b", r"\bJava\b", r"\bJavaScript\b", r"\bTypeScript\b",
            r"\bC\+\+(?=\s|,|\.|;|$)", r"\bC#(?=\s|,|\.|;|$)", r"\bSQL\b", r"\bPostgreSQL\b", r"\bMySQL\b",
            r"\bMongoDB\b", r"\bRedis\b", r"\bReact\b", r"\bNext\.js\b", r"\bVue\b", r"\bFastAPI\b",
            r"\bFlask\b", r"\bDjango\b", r"\bNode\.js\b", r"\bDocker\b", r"\bKubernetes\b", r"\bGit\b",
            r"\bAWS\b", r"\bGCP\b", r"\bMachine Learning\b", r"\bDeep Learning\b", r"\bNLP\b", r"\bmBERT\b",
            r"\bPyTorch\b", r"\bTensorFlow\b", r"\bPandas\b", r"\bNumPy\b", r"\bScikit-Learn\b",
            r"\bData Analysis\b", r"\bProject Management\b", r"\bAgile\b", r"\bScrum\b", r"\bLeadership\b",
            r"\bPublic Relations\b", r"\bFinancial Accounting\b", r"\bGeneral Ledger\b", r"\bGAAP\b",
            r"\bTax\b", r"\bAudit\b", r"\bPatient Care\b", r"\bClinical\b", r"\bCulinary\b", r"\bPastry\b",
            r"\bCurriculum\b", r"\bLitigation\b", r"\bLegal Research\b", r"\bContract Management\b",
            r"\bHuman Resources\b", r"\bKomunikasi\b", r"\bManajemen Waktu\b", r"\bKerja Tim\b"
        ]

        stop_lookahead = r"(?!\s+(?:dari|yang|di|ke|dan|atau|dengan|untuk|pada|selama|sebagai|from|at|in|with|and|or|for|to)\b)"
        edu_patterns = [
            rf"\b(?:S1|S2|S3|D3|D4)\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+)?\b",
            r"\bS\.Kom\b", r"\bS\.T\b", r"\bS\.E\b", r"\bS\.Si\b", r"\bS\.H\b", r"\bS\.Pd\b",
            r"\bM\.Kom\b", r"\bM\.T\b", r"\bMBA\b", r"\bBSN\b", r"\bCPA\b", r"\bJ\.D\.\b",
            rf"\bBachelor(?:\s+of|\s+in)\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bMaster(?:\s+of|\s+in)\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bSarjana\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bMagister\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bDiploma\s+[A-Z][a-zA-Z0-9]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z0-9]+){{0,2}}\b"
        ]

        org_patterns = [
            rf"\bUniversitas\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bInstitut\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bPoliteknik\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bPT\.?\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,3}}\b",
            rf"\bCV\.?\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            r"\b[A-Z][a-zA-Z]+\s+University\b",
            rf"\bUniversity\s+of\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+)?\b",
            r"\b[A-Z][a-zA-Z]+\s+(?:Hospital|Medical Center|Corporation|Corp|Inc|LLC|Group|Bank)\b"
        ]

        role_patterns = [
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
            r"\bContract Administrator\b", r"\bAdvocate\b"
        ]

        found_skills: List[str] = []
        for pat in skill_patterns:
            for match in re.findall(pat, text, re.IGNORECASE):
                clean_m = match.strip()
                if clean_m.lower() not in [s.lower() for s in found_skills]:
                    found_skills.append(clean_m)

        found_edu: List[str] = []
        for pat in edu_patterns:
            for match in re.findall(pat, text):
                clean_m = match.strip()
                if clean_m.lower() not in [e.lower() for e in found_edu]:
                    found_edu.append(clean_m)

        found_org: List[str] = []
        for pat in org_patterns:
            for match in re.findall(pat, text):
                clean_m = match.strip()
                if clean_m.lower() not in [o.lower() for o in found_org]:
                    found_org.append(clean_m)

        found_roles: List[str] = []
        pos_header = re.search(
            r"Position:\s*(.+?)(?=\.\s*(?:Experience Level|Required Skills|Job Description)|[\r\n]|$)",
            text,
            re.IGNORECASE
        )
        if pos_header:
            found_roles.append(pos_header.group(1).strip())
        for pat in role_patterns:
            for match in re.findall(pat, text, re.IGNORECASE):
                clean_m = match.strip()
                if clean_m.lower() not in [r.lower() for r in found_roles]:
                    found_roles.append(clean_m)

        dur_matches = re.findall(
            r"\b(\d+\+?\s*(?:-\s*\d+\s*)?(?:tahun|bulan|years?|yrs?|months?)(?:\s+pengalaman|\s+of\s+experience|\s+experience)?)\b",
            text,
            re.IGNORECASE
        )
        found_dur = [d.strip() for d in dur_matches[:4]]

        exp_match = re.search(
            r"(\d+)\+?\s*(?:tahun|years?|yrs?)\s*(?:pengalaman|of\s+experience|experience)?",
            text,
            re.IGNORECASE
        )
        years_exp = int(exp_match.group(1)) if exp_match else None

        # Inline NER tagging preview matching Thesis Table 2.5 format
        tagged_preview = text[:350]
        # Group consecutive skill lists like 'Python, Go, dan C#' if present
        tagged_preview = re.sub(
            r"\b(Python,\s*Go,\s*(?:dan|and)\s*C#)",
            r"\1 (SKILL)",
            tagged_preview,
            flags=re.IGNORECASE
        )
        for label_tag, items in [
            ("EDU", found_edu[:3]),
            ("ORG", found_org[:3]),
            ("SKILL", found_skills[:6]),
            ("ROLE", found_roles[:3]),
            ("DUR", found_dur[:2]),
        ]:
            for item in items:
                if label_tag == "SKILL" and item in ["Python", "Go", "C#"] and "C# (SKILL)" in tagged_preview:
                    continue
                if item and f"{item} ({label_tag})" not in tagged_preview:
                    tagged_preview = re.sub(
                        rf"(?<!\w)({re.escape(item)})(?!\w)",
                        rf"\1 ({label_tag})",
                        tagged_preview,
                        count=1,
                        flags=re.IGNORECASE
                    )

        return {
            "EDU": found_edu[:5],
            "ORG": found_org[:5],
            "SKILL": found_skills[:12],
            "ROLE": found_roles[:5],
            "DUR": found_dur[:4],
            "ner_tagged_preview": tagged_preview,
            # Backward-compatible keys for frontend
            "skills": found_skills,
            "education": found_edu[:5],
            "experience_years_detected": years_exp,
        }

    @classmethod
    def build_ner_enriched_text(cls, text: str, doc_type: str = "CV") -> str:
        """
        Builds structured competency text from NER entities (Table 2.5) prior to mBERT encoding.
        Normalizes varying CV layouts so core competency tokens appear in the priority window.
        """
        ent = cls.extract_entities_hybrid(text)
        role_str = ", ".join(ent["ROLE"]) if ent["ROLE"] else "General"
        skill_str = ", ".join(ent["SKILL"]) if ent["SKILL"] else "General"
        edu_str = ", ".join(ent["EDU"]) if ent["EDU"] else "General"
        dur_str = ", ".join(ent["DUR"]) if ent["DUR"] else "Unspecified"
        org_str = ", ".join(ent["ORG"]) if ent["ORG"] else "General"
        header = (
            f"[DOC] {doc_type} | [ROLE] {role_str} | [SKILL] {skill_str} | "
            f"[EDU] {edu_str} | [DUR] {dur_str} | [ORG] {org_str} || "
        )
        return header + text

    def match_cv_with_job(
        self,
        cv_text: str,
        job_title: str,
        job_description: str
    ) -> Dict[str, Any]:
        """
        Executes end-to-end matching process:
        1. Extracts NER competency entities (EDU, ORG, SKILL, ROLE, DUR) per Table 2.5.
        2. Constructs NER-enriched structured competency representation and encodes into 768-d vectors.
        3. Computes Cosine Similarity score.
        4. Classifies status category.
        """
        full_job_text = f"Position: {job_title}. Job Description: {job_description}"

        entities = self.extract_entities_hybrid(cv_text)
        cv_enriched = self.build_ner_enriched_text(cv_text, doc_type="CANDIDATE_CV")
        job_enriched = self.build_ner_enriched_text(full_job_text, doc_type="JOB_POSTING")

        cv_vec = self.get_document_embedding(cv_enriched)
        job_vec = self.get_document_embedding(job_enriched)

        score = self.calculate_cosine_similarity(cv_vec, job_vec)
        score_rounded = round(score, 4)
        percentage = round(score_rounded * 100.0, 2)
        status = self.determine_status_category(score_rounded)

        return {
            "similarity_score": score_rounded,
            "similarity_percentage": percentage,
            "status_category": status,
            "extracted_entities": entities,
            "cv_text_preview": cv_text[:300] + "..." if len(cv_text) > 300 else cv_text,
            "job_description_preview": job_description[:300] + "..." if len(job_description) > 300 else job_description,
        }


mbert_service = MBERTMatchingService()
