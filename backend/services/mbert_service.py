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


NER_TAGS: List[str] = [
    "O",
    "B-EDU", "I-EDU",
    "B-ORG", "I-ORG",
    "B-SKILL", "I-SKILL",
    "B-ROLE", "I-ROLE",
    "B-DUR", "I-DUR",
]
TAG2ID: Dict[str, int] = {t: i for i, t in enumerate(NER_TAGS)}
ID2TAG: Dict[int, str] = {i: t for i, t in enumerate(NER_TAGS)}


class SiameseMBERTCosineExtractor(nn.Module):
    """
    Unified Deep Learning NER & Siamese mBERT Feature Extractor (Model 2 - Thesis Bab 2.4, 3 & 4).
    - Branch 1 (`self.ner_head`): Token-level classifier (768 -> 11 BIO tags) predicting
      `EDU`, `ORG`, `SKILL`, `ROLE`, `DUR` on both CV and Job Description tokens.
    - Branch 2 (`self.projection`): NER-Guided Entity-Weighted Pooling + Linear Projection (768 -> 768)
      + L2 Normalization producing 768-d semantic competency vectors (u, v) for Scikit-Learn Cosine Similarity.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        dropout: float = 0.1,
        num_ner_labels: int = 11,
    ) -> None:
        super().__init__()
        self.mbert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.ner_head = nn.Linear(768, num_ner_labels)
        self.projection = nn.Linear(768, 768)
        self.use_ner_pooling: bool = False

    def encode_with_ner(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.mbert(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = outputs.last_hidden_state  # [B, L, 768]
        ner_logits = self.ner_head(self.dropout(token_embs))  # [B, L, 11]

        cls_emb = token_embs[:, 0, :]
        mask_f = attention_mask.float()

        if self.use_ner_pooling:
            ner_probs = torch.softmax(ner_logits, dim=-1)
            entity_prob = 1.0 - ner_probs[:, :, 0]  # Probability token is a competency entity (not 'O')
            weights = (1.0 + 0.5 * entity_prob.detach()) * mask_f
        else:
            weights = mask_f

        weights_exp = weights.unsqueeze(-1).expand(token_embs.size())
        mean_emb = torch.sum(token_embs * weights_exp, dim=1) / torch.clamp(weights_exp.sum(dim=1), min=1e-9)
        pooled = 0.5 * cls_emb + 0.5 * mean_emb
        projected = self.projection(self.dropout(pooled))
        vec_768 = F.normalize(projected, p=2, dim=1)
        return vec_768, ner_logits

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        vec_768, _ = self.encode_with_ner(input_ids, attention_mask)
        return vec_768

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

    # Exact Notebook 03 patterns for [DOMAIN] and [TIER] header consistency with best_siamese_mbert_cosine.pt
    CV_HEADER_PATTERNS: Dict[str, str] = {
        "INFORMATION-TECHNOLOGY": r"\b(?:INFORMATION TECHNOLOGY|IT\b|SOFTWARE|SYSTEMS?|NETWORK|DEVELOPER|TECHNOLOGY|DATA SCIENTIST|BACKEND|FRONTEND|DEVOPS)",
        "HEALTHCARE": r"\b(?:HEALTHCARE|HEALTH CARE|CLINICAL|MEDICAL|NURSE|NURSING|PATIENT|PHARMACY|THERAPIST|HOSPITAL|PHYSICIAN)",
        "ACCOUNTANT": r"\b(?:ACCOUNTANT|ACCOUNTING|CPA|AUDITOR|FINANCE|TAX|BOOKKEEPER|PAYROLL)",
        "CHEF": r"\b(?:CHEF|CULINARY|COOK|PASTRY|BAKER|KITCHEN|FOOD|CATERING)",
        "TEACHER": r"\b(?:TEACHER|TEACHING|EDUCATOR|INSTRUCTOR|FACULTY|PROFESSOR|TUTOR)",
        "ADVOCATE": r"\b(?:ADVOCATE|ATTORNEY|LEGAL|LAWYER|COUNSEL|PARALEGAL|COURT|CASE MANAGER)",
    }

    CORE_ROLE_PATTERNS: Dict[str, str] = {
        "INFORMATION-TECHNOLOGY": r"\b(?:Senior\s+|Junior\s+|Lead\s+)?(?:Software Engineer|Software Developer|Full Stack(?:\s+Developer)?|DevOps(?:\s+Engineer)?|System Administrator|Frontend(?:\s+Developer)?|Backend(?:\s+Developer)?|Data Scientist|Programmer|Solutions Architect|Provisioning Engineer|Information Technology(?:\s+Manager|\s+Director|\s+Specialist)?)\b",
        "HEALTHCARE": r"\b(?:Registered Nurse|Nurse Practitioner|Physician|Doctor|Pharmacist|Physical Therapist|Occupational Therapist|Clinical Nurse|RN|Licensed Practical Nurse|Nursing Assistant)\b",
        "ACCOUNTANT": r"\b(?:Senior\s+|Staff\s+|Tax\s+|Financial\s+|Cost\s+)?(?:Accountant|CPA|Bookkeeper|Accounting Manager|Senior Accounting Manager|Tax Professional|Consumer Banking(?:\s+Default Management)?\s+Supervisor|Operations Manager|Banking Operations Manager)\b",
        "CHEF": r"\b(?:Executive\s+|Head\s+|Sous\s+|Executive\s+Sous\s+|Banquet\s+|Pastry\s+)?(?:Chef|Chef de Cuisine|Culinary|Kitchen Associate)\b",
        "TEACHER": r"\b(?:Physical Education\s+|Classroom\s+|Special Education\s+|Assistant\s+|Preschool\s+|Substitute\s+)?(?:Teacher|Professor|Educator|Faculty|Lecturer|CDL Instructor)\b",
        "ADVOCATE": r"\b(?:Sr\.\s+|Senior\s+|Commercial\s+|General\s+|Legal\s+|Associate\s+|Litigation\s+|Trial\s+)?(?:Attorney|Lawyer|Litigation Associate|Litigation|Counsel|Probate Paralegal|Advocate)\b",
    }

    COGNATE_ROLE_PATTERNS: Dict[str, str] = {
        "INFORMATION-TECHNOLOGY": r"\b(?:Data Analyst|Help Desk|Technical Support|Systems Analyst|Technical Writer|QA|Quality Assurance|Database Administrator|Network Specialist|Network Administrator|IT Coordinator|Business Analyst)\b",
        "HEALTHCARE": r"\b(?:Medical Assistant|Clinical Coordinator|Healthcare Administrator|Health Care Coordinator|Case Manager|Medical Records|Patient Access(?:\s+Representative)?|Health Coach|Pharmacy Technician|Dental Assistant)\b",
        "ACCOUNTANT": r"\b(?:Financial Analyst|Senior Financial Analyst|Internal Auditor|Auditor|Accounts Payable(?:\s+Specialist)?|Accounts Receivable(?:\s+Specialist)?|Payroll Specialist|Billing Specialist|Credit Analyst|Tax Advisor)\b",
        "CHEF": r"\b(?:Line Cook|Prep Cook|Kitchen Manager|Food Service|Baker|Dietary Aide|Catering|Barista|Restaurant Manager)\b",
        "TEACHER": r"\b(?:Corporate Trainer|Technical Trainer|Instructional Designer|Curriculum|Education Coordinator|Academic Advisor|Tutor|Training Specialist|Learning Specialist)\b",
        "ADVOCATE": r"\b(?:Compliance Specialist|Contract Administrator|Legal Assistant|Paralegal|Regulatory Affairs|Risk Analyst|Contract Specialist|Contracts Manager)\b",
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
                has_ner_head = "ner_head.weight" in state_dict
                self.model.load_state_dict(state_dict, strict=False)
                self.model.use_ner_pooling = has_ner_head
                self._is_fine_tuned = True
                logger.info(
                    "Fine-tuned Siamese mBERT Cosine weights loaded (Unified NER head in checkpoint: %s).",
                    has_ner_head,
                )
            else:
                has_ner_head = False
                logger.warning(
                    "Fine-tuned checkpoint not found at '%s'. Running with base pre-trained weights.",
                    self.fine_tuned_path,
                )

            self.model.to(self.device)
            if not has_ner_head:
                self._warm_train_ner_head()
            self.model.eval()
            self._is_loaded = True
        except Exception as exc:
            logger.error("Failed to load mBERT model: %s", exc)
            raise exc

    def _warm_train_ner_head(self) -> None:
        """
        Warm-trains `self.model.ner_head` on the bilingual BIO corpus (Table 2.5 + 6 domain sentences)
        while keeping `self.model.mbert` and `self.model.projection` frozen when loading a pre-unified checkpoint.
        """
        assert self.tokenizer is not None and self.model is not None
        corpus = [
            (
                "Lulusan S1 Informatika dari Universitas Surabaya yang menguasai Python, Go, dan C# dan memiliki pengalaman magang sebagai Software Engineer di PT Telekomunikasi Indonesia selama 6 bulan.",
                [("S1 Informatika", "EDU"), ("Universitas Surabaya", "ORG"), ("Python, Go, dan C#", "SKILL"), ("Software Engineer", "ROLE"), ("PT Telekomunikasi Indonesia", "ORG"), ("6 bulan", "DUR")],
            ),
            (
                "Graduate with Bachelor of Science : Finance from University of Nevada, Las Vegas skilled in Accounting, Payroll, Budget, and Microsoft Excel working as Consumer Banking Operations Manager for 10 years.",
                [("Bachelor of Science : Finance", "EDU"), ("University of Nevada, Las Vegas", "ORG"), ("Accounting, Payroll, Budget, and Microsoft Excel", "SKILL"), ("Consumer Banking Operations Manager", "ROLE"), ("10 years", "DUR")],
            ),
            (
                "Senior Accountant with Bachelor of Accounting from California State University proficient in General Ledger, GAAP, QuickBooks, and Financial Reporting at Deloitte Corp for 5 years.",
                [("Senior Accountant", "ROLE"), ("Bachelor of Accounting", "EDU"), ("California State University", "ORG"), ("General Ledger, GAAP, QuickBooks, and Financial Reporting", "SKILL"), ("Deloitte Corp", "ORG"), ("5 years", "DUR")],
            ),
            (
                "Registered Nurse holding BSN in Nursing from Hartford Hospital experienced in Patient Care, Clinical Triage, EMR, and HIPAA compliance for 4 years.",
                [("Registered Nurse", "ROLE"), ("BSN", "EDU"), ("Hartford Hospital", "ORG"), ("Patient Care, Clinical Triage, EMR, and HIPAA", "SKILL"), ("4 years", "DUR")],
            ),
            (
                "Executive Sous Chef with Culinary Arts Diploma from Culinary Institute skilled in Pastry, Menu Planning, Food Safety, and Inventory Control for 7 years.",
                [("Executive Sous Chef", "ROLE"), ("Culinary Arts", "EDU"), ("Culinary Institute", "ORG"), ("Pastry, Menu Planning, Food Safety, and Inventory Control", "SKILL"), ("7 years", "DUR")],
            ),
            (
                "Commercial Litigation Attorney holding J.D. from Temple University skilled in Legal Research, Contract Management, and Regulatory Compliance at Law Firm for 6 years.",
                [("Commercial Litigation Attorney", "ROLE"), ("J.D.", "EDU"), ("Temple University", "ORG"), ("Legal Research, Contract Management, and Regulatory Compliance", "SKILL"), ("Law Firm", "ORG"), ("6 years", "DUR")],
            ),
        ]
        input_ids_list, mask_list, label_ids_list = [], [], []
        for text, spans in corpus * 4:
            char_labels = ["O"] * len(text)
            for phrase, ent_type in spans:
                idx = text.find(phrase)
                if idx != -1:
                    for c_i in range(idx, idx + len(phrase)):
                        char_labels[c_i] = ent_type
            enc = self.tokenizer(
                text, max_length=128, padding="max_length", truncation=True, return_offsets_mapping=True, return_tensors="pt"
            )
            offsets = enc["offset_mapping"][0].tolist()
            tok_labels = []
            prev_ent = "O"
            for s_c, e_c in offsets:
                if s_c == 0 and e_c == 0:
                    tok_labels.append(-100)
                    prev_ent = "O"
                    continue
                votes = [c for c in char_labels[s_c:e_c] if c != "O"]
                if votes:
                    curr_ent = votes[0]
                    prefix = "I-" if prev_ent == curr_ent else "B-"
                    tok_labels.append(TAG2ID[f"{prefix}{curr_ent}"])
                    prev_ent = curr_ent
                else:
                    tok_labels.append(TAG2ID["O"])
                    prev_ent = "O"
            input_ids_list.append(enc["input_ids"][0])
            mask_list.append(enc["attention_mask"][0])
            label_ids_list.append(torch.tensor(tok_labels, dtype=torch.long))

        b_ids = torch.stack(input_ids_list).to(self.device)
        b_mask = torch.stack(mask_list).to(self.device)
        b_lbl = torch.stack(label_ids_list).to(self.device)

        with torch.no_grad():
            hidden = self.model.mbert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state.detach()

        for p in self.model.ner_head.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(self.model.ner_head.parameters(), lr=1e-2, weight_decay=0.01)
        ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.model.ner_head.train()
        for _ in range(18):
            opt.zero_grad()
            logits = self.model.ner_head(hidden)
            loss = ce(logits.view(-1, len(NER_TAGS)), b_lbl.view(-1))
            loss.backward()
            opt.step()
        for p in self.model.ner_head.parameters():
            p.requires_grad = False
        self.model.eval()

    def predict_ner_spans_dl(self, text: str) -> Dict[str, List[str]]:
        """
        Runs token-level BIO inference using `self.model.ner_head` inside `SiameseMBERTCosineExtractor`.
        """
        empty: Dict[str, List[str]] = {"EDU": [], "ORG": [], "SKILL": [], "ROLE": [], "DUR": []}
        if not self._is_loaded or self.tokenizer is None or self.model is None or not text.strip():
            return empty
        try:
            enc = self.tokenizer(
                text[:600],
                return_offsets_mapping=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                _, ner_logits = self.model.encode_with_ner(enc["input_ids"], enc["attention_mask"])
            pred_ids = torch.argmax(ner_logits[0], dim=-1).cpu().tolist()
            curr_type, curr_start, curr_end = None, 0, 0
            for p_id, (s_c, e_c) in zip(pred_ids, offsets):
                if s_c == 0 and e_c == 0:
                    continue
                tag = ID2TAG.get(p_id, "O")
                if tag.startswith("B-"):
                    if curr_type and curr_end > curr_start:
                        span = text[curr_start:curr_end].strip(" ,.;:()")
                        if len(span) >= 2 and span.lower() not in [x.lower() for x in empty[curr_type]]:
                            empty[curr_type].append(span)
                    curr_type = tag[2:]
                    curr_start, curr_end = s_c, e_c
                elif tag.startswith("I-") and curr_type == tag[2:]:
                    curr_end = e_c
                else:
                    if curr_type and curr_end > curr_start:
                        span = text[curr_start:curr_end].strip(" ,.;:()")
                        if len(span) >= 2 and span.lower() not in [x.lower() for x in empty[curr_type]]:
                            empty[curr_type].append(span)
                    curr_type = None
            if curr_type and curr_end > curr_start:
                span = text[curr_start:curr_end].strip(" ,.;:()")
                if len(span) >= 2 and span.lower() not in [x.lower() for x in empty[curr_type]]:
                    empty[curr_type].append(span)
        except Exception as exc:
            logger.debug("DL NER span prediction skipped: %s", exc)
        return empty

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
    def regex_standardize_text(text: str) -> str:
        """Applies the exact cased Regex standardization used in Notebook 03 prior to NER & tokenization."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"[^\w\s.,;:()/\-+&#%]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _encode_bio_labels(self, text: str, entities: Dict[str, Any], max_len: int) -> List[int]:
        """Aligns character-level entity spans to WordPiece BIO tag IDs (Notebook 03 Cell 3)."""
        assert self.tokenizer is not None
        char_labels = ["O"] * len(text)
        for tag in ["ROLE", "EDU", "ORG", "DUR", "SKILL"]:
            for phrase in entities.get(tag, []):
                if phrase and len(phrase) >= 2:
                    for m in re.finditer(re.escape(phrase), text, re.IGNORECASE):
                        for idx in range(m.start(), min(m.end(), len(text))):
                            char_labels[idx] = tag
        enc = self.tokenizer(text, max_length=max_len, padding="max_length", truncation=True, return_offsets_mapping=True)
        token_labels: List[int] = []
        prev_ent = "O"
        for s_c, e_c in enc["offset_mapping"]:
            if s_c == 0 and e_c == 0:
                token_labels.append(-100)
                prev_ent = "O"
                continue
            votes = [c for c in char_labels[s_c:min(e_c, len(text))] if c != "O"]
            if votes:
                curr_ent = votes[0]
                prefix = "I-" if prev_ent == curr_ent else "B-"
                token_labels.append(TAG2ID.get(f"{prefix}{curr_ent}", 0))
                prev_ent = curr_ent
            else:
                token_labels.append(TAG2ID["O"])
                prev_ent = "O"
        return token_labels

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
            # IT, Software & Data (including Dataset-Mined Keywords)
            r"\bPython\b", r"\bGo\b", r"\bGolang\b", r"\bJava\b", r"\bJavaScript\b", r"\bTypeScript\b",
            r"\bC\+\+(?=\s|,|\.|;|$)", r"\bC#(?=\s|,|\.|;|$)", r"\bSQL\b", r"\bPostgreSQL\b", r"\bMySQL\b",
            r"\bMongoDB\b", r"\bRedis\b", r"\bReact\b", r"\bNext\.js\b", r"\bVue\b", r"\bFastAPI\b",
            r"\bFlask\b", r"\bDjango\b", r"\bNode\.js\b", r"\bDocker\b", r"\bKubernetes\b", r"\bGit\b",
            r"\bAWS\b", r"\bGCP\b", r"\bMachine Learning\b", r"\bDeep Learning\b", r"\bNLP\b", r"\bmBERT\b",
            r"\bPyTorch\b", r"\bTensorFlow\b", r"\bPandas\b", r"\bNumPy\b", r"\bScikit-Learn\b",
            r"\bData Analysis\b", r"\bDatabase(?:\s+Administration)?\b", r"\bSharePoint\b", r"\bLotus Notes\b",
            r"\bTechnical Support\b", r"\b(?:Microsoft\s+)?Excel\b", r"\b(?:Microsoft\s+)?Word\b",
            r"\b(?:Microsoft\s+)?Access\b", r"\b(?:Microsoft\s+)?Power\s*Point\b", r"\b(?:Microsoft\s+)?Outlook\b",
            r"\bMicrosoft Office\b", r"\bActive Directory\b", r"\bLinux\b", r"\bWindows Server\b",
            r"\bVeeva Vault\b", r"\bAS400\b", r"\bSAP\b", r"\bQuickBooks\b", r"\bOracle\b",
            r"\bSalesforce\b", r"\bTableau\b", r"\bPower BI\b", r"\bTroubleshooting\b",
            # Banking, Finance, Accounting & Operations (Dataset-Mined Keywords)
            r"\bFinancial Accounting\b", r"\bAccounting\b", r"\bGeneral Ledger\b", r"\bGAAP\b",
            r"\bFinancial Statements\b", r"\bTax(?:\s+Preparation|\s+Compliance)?\b", r"\bAudit(?:ing)?\b",
            r"\bInternal Audit\b", r"\bPayroll\b", r"\bBudget(?:ing)?\b", r"\bBilling\b",
            r"\bCash Deposits\b", r"\bAccount Resolution\b", r"\bDefault Management\b",
            r"\bFinancial Analysis\b", r"\bFinancial Reporting\b", r"\bAccounts Payable\b",
            r"\bAccounts Receivable\b", r"\bReconciliation\b", r"\bCredit(?:\s+Analysis)?\b",
            r"\bClosing\b", r"\bForecasting\b", r"\bVariance Analysis\b", r"\bInvoicing\b",
            r"\bRisk Management\b", r"\bStrategic Management\b", r"\bBusiness Administration\b",
            r"\bProcess Improvement\b", r"\bQuality(?:\s+Assurance|\s+Control)?\b", r"\bProcurement\b",
            r"\bChange Management\b", r"\bOrganizational Development\b", r"\bOperations Management\b",
            r"\bProject Management\b", r"\bStaffing\b", r"\bScheduling\b", r"\bRecruiting\b",
            r"\bCall Center\b", r"\bCustomer Service\b", r"\bCustomer Satisfaction\b",
            r"\bClient Relations\b", r"\bCoaching\b", r"\bConflict Resolution\b", r"\bReporting\b",
            r"\bMarketing\b", r"\bSales\b", r"\bLeadership\b", r"\bSafety\b", r"\bInsurance\b",
            r"\bDocumentation\b", r"\bCritical Thinking\b", r"\bTime Management\b",
            r"\bAgile\b", r"\bScrum\b", r"\bPublic Relations\b", r"\bHuman Resources\b",
            # Healthcare, Culinary, Education, Legal & Indonesian terms (Dataset-Mined Keywords)
            r"\bPatient Care\b", r"\bClinical\b", r"\bMedication Administration\b", r"\bTriage\b",
            r"\bEMR\b", r"\bHIPAA\b", r"\bVital Signs\b", r"\bMedical Terminology\b", r"\bCPR\b",
            r"\bPhlebotomy\b", r"\bElectronic Health Records\b", r"\bCulinary(?:\s+Arts)?\b",
            r"\bPastry\b", r"\bFood Safety\b", r"\bHACCP\b", r"\bMenu Planning\b",
            r"\bFood Preparation\b", r"\bKitchen Management\b", r"\bBanquet\b", r"\bCatering\b",
            r"\bSanitation\b", r"\bInventory(?:\s+Control|\s+Management)?\b", r"\bCost Control\b",
            r"\bCurriculum(?:\s+Development)?\b", r"\bLesson Planning\b", r"\bClassroom Management\b",
            r"\bInstructional Design\b", r"\bSpecial Education\b", r"\bTutoring\b",
            r"\bStudent Assessment\b", r"\bLitigation\b", r"\bLegal Research\b", r"\bLegal Drafting\b",
            r"\bContracts?\b", r"\bContract Management\b", r"\bContract Negotiation\b",
            r"\bRegulatory Compliance\b", r"\bDue Diligence\b", r"\bTrial Preparation\b",
            r"\bMediation\b", r"\bArbitration\b", r"\bCorporate Law\b",
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
            r"\bHigh School Diploma\b", r"\bGED\b", r"\bAssociate['’]?s Degree\b",
            r"\bBachelor['’]?s Degree(?:\s+in\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)?\b",
            r"\bMaster['’]?s Degree(?:\s+in\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)?\b",
            rf"\bBachelor(?:\s+of|\s+in)\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bMaster(?:\s+of|\s+in)\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bAssociate(?:\s+of|\s+in)\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bSarjana\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bMagister\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bDiploma\s+[A-Z][a-zA-Z0-9]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z0-9]+){{0,2}}\b",
        ]

        org_patterns = [
            rf"\bUniversity\s+of\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+)?(?:\s*,\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)?\b",
            rf"\bUniversitas\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bInstitut\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bPoliteknik\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\bPT\.?\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,3}}\b",
            rf"\bCV\.?\s+[A-Z][a-zA-Z]+(?:{stop_lookahead}\s+[A-Z][a-zA-Z]+){{0,2}}\b",
            rf"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s+State\s+University\b",
            r"\b(?:Culinary Institute(?:\s+of\s+America)?|Community College|Technical College|Technical Institute|City College|State College|Law Firm|School District)\b",
            r"\b[A-Z][a-zA-Z]+\s+(?:Hospital|Medical Center|Clinic|Healthcare|College|Academy|Institute|Law School|Corporation|Corp|Inc|LLC|Group|Bank)\b",
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
            r"\b(?:Senior\s+|Junior\s+|Lead\s+|Chief\s+|Assistant\s+|Executive\s+)?(?:Consumer\s+Banking\s+|Default\s+Management\s+|Ticketing\s+|Customer\s+Service\s+|Operations\s+|Financial\s+|Accounting\s+|Accounts\s+Payable\s+|Accounts\s+Receivable\s+|Credit\s+|Risk\s+|Branch\s+|Project\s+|Product\s+|Program\s+|HR\s+|Human\s+Resources\s+|Marketing\s+|Sales\s+|Business\s+Development\s+|Quality\s+Assurance\s+|Supply\s+Chain\s+|Procurement\s+|Client\s+Relations\s+|Account\s+|Training\s+|Patient\s+Access\s+|Restaurant\s+|Kitchen\s+)?(?:Manager|Supervisor|Director|Analyst|Specialist|Coordinator|Officer|Administrator|Representative|Advisor|Consultant)\b",
            r"\b(?:Senior\s+|Junior\s+|Lead\s+)?Software Engineer\b",
            r"\b(?:Senior\s+|Junior\s+)?Data Scientist\b",
            r"\bData Analyst\b", r"\bSystems Analyst\b", r"\bFull Stack Developer\b",
            r"\bFrontend Developer\b", r"\bBackend Developer\b", r"\bDevOps Engineer\b",
            r"\bIT Specialist\b", r"\bSystem Administrator\b", r"\bDatabase Administrator\b",
            r"\bTechnical Support(?:\s+Specialist|\s+Engineer)?\b", r"\bHelp Desk(?:\s+Analyst|\s+Specialist)?\b",
            r"\bNetwork (?:Specialist|Engineer|Administrator)\b",
            r"\bRegistered Nurse\b", r"\bMedical Assistant\b", r"\bDental Assistant\b",
            r"\bClinical Coordinator\b", r"\bHealthcare Administrator\b", r"\bPharmacist\b",
            r"\bPhysician\b", r"\bNursing Assistant\b", r"\bLicensed Practical Nurse\b",
            r"\b(?:Senior\s+|Staff\s+|Cost\s+|Tax\s+)?Accountant\b", r"\bFinancial Analyst\b",
            r"\bInternal Auditor\b", r"\bAuditor\b", r"\bTax Specialist\b",
            r"\bPayroll Specialist\b", r"\bBookkeeper\b",
            r"\b(?:Executive\s+|Head\s+|Sous\s+|Executive\s+Sous\s+|Banquet\s+|Pastry\s+)?Chef\b",
            r"\bChef de Cuisine\b", r"\bLine Cook\b", r"\bPrep Cook\b",
            r"\bKitchen Manager\b", r"\bRestaurant Manager\b", r"\bBaker\b",
            r"\b(?:Physical Education\s+|Classroom\s+|Special Education\s+|Assistant\s+|Preschool\s+|Substitute\s+)?Teacher\b",
            r"\bInstructor\b", r"\bProfessor\b", r"\bLecturer\b",
            r"\bCorporate Trainer\b", r"\bTechnical Trainer\b",
            r"\bInstructional Designer\b", r"\bAcademic Advisor\b", r"\bCurriculum Educator\b",
            r"\b(?:Commercial\s+|Legal\s+|General\s+|Corporate\s+)?Counsel\b",
            r"\b(?:Associate\s+|Litigation\s+|Senior\s+|Trial\s+)?Attorney\b",
            r"\bLitigation Associate\b", r"\bLawyer\b",
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

    def extract_entities_with_dl(self, text: str) -> Dict[str, Any]:
        """
        Unified Deep Learning + Dataset-Mined Hybrid NER Extraction:
        Runs the fine-tuned `SiameseMBERTCosineExtractor.ner_head` token classification layer
        (11 BIO labels) alongside the dataset-mined competency dictionary, merging spans
        cleanly for both CV and Job Description documents.
        """
        base_ent = self.extract_entities_hybrid(text)
        dl_spans = self.predict_ner_spans_dl(text)
        noise_spans = {
            "and", "or", "the", "in", "of", "to", "for", "with", "on", "at", "by",
            "position", "job", "description", "summary", "experience", "education", "skills",
        }
        limits = {"EDU": 6, "ORG": 6, "SKILL": 16, "ROLE": 6, "DUR": 5}
        for tag, max_k in limits.items():
            existing = list(base_ent.get(tag, []))
            for span in dl_spans.get(tag, []):
                clean_s = re.sub(r"\s+", " ", span.strip(" ,.;:()[]/-"))
                if len(clean_s) < 3 or clean_s.lower() in noise_spans:
                    continue
                # Avoid partial single-word fragments if already covered by a fuller entity span
                if any(clean_s.lower() in ex.lower() or ex.lower() in clean_s.lower() for ex in existing):
                    continue
                existing.append(clean_s)
            base_ent[tag] = existing[:max_k]

        base_ent["skills"] = base_ent["SKILL"]
        base_ent["education"] = base_ent["EDU"]
        return base_ent

    @staticmethod
    def compute_matched_entities(
        cv_entities: Dict[str, Any], job_entities: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Finds matching or overlapping NER competency entities between the Candidate CV
        and the Job Description across all 5 Thesis Table 2.5 categories.
        """
        matched: Dict[str, List[str]] = {"ROLE": [], "SKILL": [], "EDU": [], "DUR": [], "ORG": []}
        stop_tokens = {"and", "the", "for", "with", "of", "in", "to", "senior", "junior", "lead", "staff", "associate"}
        for tag in ["ROLE", "SKILL", "EDU", "DUR", "ORG"]:
            cv_list = cv_entities.get(tag, [])
            job_list = job_entities.get(tag, [])
            for c_item in cv_list:
                c_low = c_item.lower()
                c_toks = {
                    t for t in re.findall(r"[a-zA-Z0-9+#]{2,}", c_low) if t not in stop_tokens
                }
                is_match = False
                for j_item in job_list:
                    j_low = j_item.lower()
                    if c_low == j_low or c_low in j_low or j_low in c_low:
                        is_match = True
                        break
                    j_toks = {
                        t for t in re.findall(r"[a-zA-Z0-9+#]{2,}", j_low) if t not in stop_tokens
                    }
                    if c_toks and j_toks and (c_toks & j_toks):
                        # Require at least 1 strong shared domain/skill/degree/role token
                        is_match = True
                        break
                if is_match and c_item not in matched[tag]:
                    matched[tag].append(c_item)
        return matched

    @classmethod
    def detect_macro_domain(cls, text: str, entities: Optional[Dict[str, Any]] = None) -> str:
        """
        Infers which of the 6 macro domains (`INFORMATION-TECHNOLOGY`, `HEALTHCARE`,
        `ACCOUNTANT`, `CHEF`, `TEACHER`, `ADVOCATE`) best represents the document.
        """
        domain, _ = cls.detect_domain_and_tier(text, doc_type="CANDIDATE_CV", entities=entities)
        return domain

    @classmethod
    def detect_domain_and_tier(
        cls,
        text: str,
        doc_type: str = "CANDIDATE_CV",
        entities: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """
        Infers `[DOMAIN]` and `[TIER]` using the exact Notebook 03 symmetric logic so
        runtime embeddings match `best_siamese_mbert_cosine.pt`.
        """
        ent = entities or cls.extract_entities_hybrid(text)
        skill_text = " ".join(ent.get("SKILL", [])).lower()
        scores: Dict[str, int] = {}
        for dom, keywords in cls.MACRO_DOMAIN_KEYWORDS.items():
            scores[dom] = sum(
                1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", skill_text + " " + text.lower())
            )
        top_skill_dom = max(scores, key=lambda k: scores[k]) if any(scores.values()) else "GENERAL"

        if doc_type == "JOB_POSTING":
            pos_m = re.search(r"Position:\s*(.+?)(?=\.\s*(?:Experience Level|Required Skills|Job Description)|[\r\n]|$)", text, re.IGNORECASE)
            target = pos_m.group(1).strip() if pos_m else text[:400]
            core_doms = [d for d, p in cls.CORE_ROLE_PATTERNS.items() if re.search(p, target, re.IGNORECASE)]
            cog_doms = [d for d, p in cls.COGNATE_ROLE_PATTERNS.items() if re.search(p, target, re.IGNORECASE)]
            if cog_doms:
                return ", ".join(cog_doms), "COGNATE_SUBROLE"
            if core_doms:
                return ", ".join(core_doms), "CORE_DOMAIN_ROLE"
            return top_skill_dom, "CORE_DOMAIN_ROLE"

        head85 = text[:85]
        header_matches = [
            (m.start(), dom)
            for dom, pat in cls.CV_HEADER_PATTERNS.items()
            for m in [re.search(pat, head85, re.IGNORECASE)]
            if m
        ]
        if header_matches:
            header_matches.sort(key=lambda x: x[0])
            return header_matches[0][1], "CANDIDATE_PROFILE"
        return top_skill_dom, "CANDIDATE_PROFILE"

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
        Builds structured competency text from NER entities matching Notebook 03 Cell 3:
        `[DOC] ... | [DOMAIN] ... | [TIER] ... | [ROLE] ... | [SKILL] ... | [EDU] ... | [DUR] ... | [ORG] ... || {clean_text}`
        """
        clean_text = cls.regex_standardize_text(text)
        ent = entities or cls.extract_entities_hybrid(clean_text)
        inferred_dom, inferred_tier = cls.detect_domain_and_tier(clean_text, doc_type=doc_type, entities=ent)
        dom_str = domain_tag or inferred_dom
        tier_str = tier_tag or inferred_tier

        role_str = ", ".join(ent["ROLE"][:3]) if ent.get("ROLE") else "Unspecified"
        skill_str = ", ".join(ent["SKILL"][:8]) if ent.get("SKILL") else "Unspecified"
        edu_str = ", ".join(ent["EDU"][:4]) if ent.get("EDU") else "Unspecified"
        dur_str = ", ".join(ent["DUR"][:3]) if ent.get("DUR") else "Unspecified"
        org_str = ", ".join(ent["ORG"][:4]) if ent.get("ORG") else "Unspecified"

        header = (
            f"[DOC] {doc_type} | [DOMAIN] {dom_str} | [TIER] {tier_str} | "
            f"[ROLE] {role_str} | [SKILL] {skill_str} | "
            f"[EDU] {edu_str} | [DUR] {dur_str} | [ORG] {org_str} || "
        )
        return header + clean_text

    def match_cv_with_job(
        self,
        cv_text: str,
        job_title: str,
        job_description: str,
    ) -> Dict[str, Any]:
        """
        End-to-end matching aligned with Notebook 03:
        1. Extracts NER entities symmetrically on CV & Job Description (DL NER + hybrid dictionary) for Web UI.
        2. Encodes standardized NER-enriched texts at max_length=128 via `SiameseMBERTCosineExtractor`.
        3. Computes raw `sklearn` Cosine Similarity and classifies directly against Thesis thresholds (`80%`, `60%`).
        """
        job_desc_clean = job_description.strip()
        if re.match(r"^\s*Position\s*:", job_desc_clean, re.IGNORECASE) or not job_title.strip():
            full_job_text = job_desc_clean
        else:
            full_job_text = f"Position: {job_title.strip()}. Job Description: {job_desc_clean}"

        cv_entities = self.extract_entities_with_dl(cv_text)
        job_entities = self.extract_entities_with_dl(full_job_text)
        matched_entities = self.compute_matched_entities(cv_entities, job_entities)

        cv_clean = self.regex_standardize_text(cv_text)
        job_clean = self.regex_standardize_text(full_job_text)
        cv_domain, _ = self.detect_domain_and_tier(cv_clean, doc_type="CANDIDATE_CV")
        job_domain, _ = self.detect_domain_and_tier(job_clean, doc_type="JOB_POSTING")

        cv_enriched = self.build_ner_enriched_text(cv_clean, doc_type="CANDIDATE_CV")
        job_enriched = self.build_ner_enriched_text(job_clean, doc_type="JOB_POSTING")

        cv_vec = self.get_document_embedding(cv_enriched)
        job_vec = self.get_document_embedding(job_enriched)

        # Model 2 is trained directly with PureCosineMarginLoss -> raw sklearn cosine is natively calibrated
        raw_cosine = self.calculate_cosine_similarity(cv_vec, job_vec)
        score_rounded = round(raw_cosine, 4)
        percentage = round(score_rounded * 100.0, 2)
        status = self.determine_status_category(score_rounded)

        cv_entities["detected_domain"] = cv_domain
        cv_entities["job_detected_domain"] = job_domain
        cv_entities["raw_cosine_similarity"] = score_rounded
        cv_entities["fine_tuned_model_loaded"] = self._is_fine_tuned
        cv_entities["job_entities"] = {
            "EDU": job_entities.get("EDU", []),
            "ORG": job_entities.get("ORG", []),
            "SKILL": job_entities.get("SKILL", []),
            "ROLE": job_entities.get("ROLE", []),
            "DUR": job_entities.get("DUR", []),
            "ner_tagged_preview": job_entities.get("ner_tagged_preview", ""),
            "experience_years_detected": job_entities.get("experience_years_detected", 0),
        }
        cv_entities["matched_entities"] = matched_entities

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

    def retrain_on_validated_pair(
        self,
        cv_text: Optional[str] = None,
        job_text: Optional[str] = None,
        validated_category: str = "Fit",
        validated_percentage: float = 85.0,
        epochs: int = 4,
    ) -> Dict[str, Any]:
        """
        Incremental fine-tuning on `SiameseMBERTCosineExtractor` using the validated pair
        plus a balanced replay sample aligned directly to Notebook 03 cosine targets (0.90, 0.70, 0.15).
        """
        import shutil
        import time
        import pandas as pd

        t0 = time.time()
        if not self._is_loaded:
            self.load_model()

        assert self.tokenizer is not None
        assert self.model is not None

        if self.fine_tuned_path and os.path.exists(self.fine_tuned_path):
            backup_path = self.fine_tuned_path.replace(".pt", ".backup.pt")
            if not os.path.exists(backup_path):
                try:
                    shutil.copy2(self.fine_tuned_path, backup_path)
                    logger.info("Created backup of original checkpoint at '%s'", backup_path)
                except Exception as b_err:
                    logger.warning("Could not create checkpoint backup: %s", b_err)

        train_pairs: List[Tuple[str, str, float]] = []
        target_score = max(0.0, min(1.0, validated_percentage / 100.0))

        primary_cv_enr = None
        primary_job_enr = None
        if cv_text and job_text:
            primary_cv_enr = self.build_ner_enriched_text(cv_text, doc_type="CANDIDATE_CV")
            primary_job_enr = self.build_ner_enriched_text(job_text, doc_type="JOB_POSTING")
            for _ in range(3):
                train_pairs.append((primary_cv_enr, primary_job_enr, target_score))

        total_csv_rows = 0
        if os.path.exists(settings.DATASET_CSV_PATH):
            try:
                df = pd.read_csv(settings.DATASET_CSV_PATH)
                total_csv_rows = len(df)
                for lbl_str, target_val in [("fit", 0.90), ("close fit", 0.70), ("not fit", 0.15)]:
                    sub_df = df[df["kategori"].astype(str).str.strip().str.lower() == lbl_str]
                    if len(sub_df) > 0:
                        sampled = sub_df.tail(2) if len(sub_df) >= 2 else sub_df
                        for _, row in sampled.iterrows():
                            c_enr = self.build_ner_enriched_text(str(row["cv"]), doc_type="CANDIDATE_CV")
                            j_enr = self.build_ner_enriched_text(str(row["job"]), doc_type="JOB_POSTING")
                            train_pairs.append((c_enr, j_enr, target_val))
            except Exception as csv_err:
                logger.warning("Failed to sample replay rows from CSV: %s", csv_err)

        for param in self.model.mbert.parameters():
            param.requires_grad = False
        for param in self.model.mbert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.model.ner_head.parameters():
            param.requires_grad = True
        for param in self.model.projection.parameters():
            param.requires_grad = True

        before_pct = validated_percentage
        if primary_cv_enr and primary_job_enr:
            self.model.eval()
            u_bef = self.get_document_embedding(primary_cv_enr)
            v_bef = self.get_document_embedding(primary_job_enr)
            before_pct = round(self.calculate_cosine_similarity(u_bef, v_bef) * 100.0, 2)

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

        cv_bio = torch.tensor(
            [self._encode_bio_labels(t, self.extract_entities_hybrid(t), max_seq_len) for t in cv_texts],
            dtype=torch.long,
            device=self.device,
        )
        job_bio = torch.tensor(
            [self._encode_bio_labels(t, self.extract_entities_hybrid(t), max_seq_len) for t in job_texts],
            dtype=torch.long,
            device=self.device,
        )
        ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.mbert.encoder.layer[-1].parameters(), "lr": 2e-5},
                {"params": self.model.ner_head.parameters(), "lr": 2e-4},
                {"params": self.model.projection.parameters(), "lr": 2e-4},
            ],
            weight_decay=0.01,
        )

        self.model.train()
        loss_history: List[float] = []
        for _ in range(epochs):
            optimizer.zero_grad()
            u, cv_ner_logits = self.model.encode_with_ner(cv_enc["input_ids"], cv_enc["attention_mask"])
            v, job_ner_logits = self.model.encode_with_ner(job_enc["input_ids"], job_enc["attention_mask"])
            cos_sim = F.cosine_similarity(u, v, dim=1)
            loss_cos = F.mse_loss(cos_sim, targets) * 10.0
            loss_ner = 0.5 * (
                ce_loss_fn(cv_ner_logits.view(-1, len(NER_TAGS)), cv_bio.view(-1))
                + ce_loss_fn(job_ner_logits.view(-1, len(NER_TAGS)), job_bio.view(-1))
            )
            loss = loss_cos + 0.20 * loss_ner
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            loss_history.append(round(float(loss.item()), 6))

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if self.fine_tuned_path:
            os.makedirs(os.path.dirname(self.fine_tuned_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.fine_tuned_path)
            self._is_fine_tuned = True

        after_pct = validated_percentage
        if primary_cv_enr and primary_job_enr:
            u_aft = self.get_document_embedding(primary_cv_enr)
            v_aft = self.get_document_embedding(primary_job_enr)
            after_pct = round(self.calculate_cosine_similarity(u_aft, v_aft) * 100.0, 2)

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
