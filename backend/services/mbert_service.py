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
        Extracts skill, education, and experience markers from text
        using hybrid rule-based pattern matching for Indonesian & English CVs.
        """
        skill_patterns = [
            r"\bPython\b", r"\bJava\b", r"\bJavaScript\b", r"\bTypeScript\b", r"\bC\+\+\b", r"\bC#\b",
            r"\bSQL\b", r"\bPostgreSQL\b", r"\bMySQL\b", r"\bMongoDB\b", r"\bRedis\b",
            r"\bReact\b", r"\bNext\.js\b", r"\bVue\b", r"\bFastAPI\b", r"\bFlask\b", r"\bDjango\b",
            r"\bNode\.js\b", r"\bDocker\b", r"\bKubernetes\b", r"\bGit\b", r"\bAWS\b", r"\bGCP\b",
            r"\bMachine Learning\b", r"\bDeep Learning\b", r"\bNLP\b", r"\bmBERT\b", r"\bPyTorch\b",
            r"\bTensorFlow\b", r"\bPandas\b", r"\bNumPy\b", r"\bScikit-Learn\b", r"\bData Analysis\b",
            r"\bProject Management\b", r"\bAgile\b", r"\bScrum\b", r"\bLeadership\b",
            r"\bPublic Relations\b", r"\bFinancial Accounting\b", r"\bHuman Resources\b",
            r"\bKomunikasi\b", r"\bManajemen Waktu\b", r"\bKerja Tim\b"
        ]

        edu_patterns = [
            r"\bS\.Kom\b", r"\bS\.T\b", r"\bS\.E\b", r"\bS\.Si\b", r"\bM\.Kom\b", r"\bM\.T\b", r"\bMBA\b",
            r"\bBachelor of [A-Za-z ]+\b", r"\bMaster of [A-Za-z ]+\b", r"\bDoctor of [A-Za-z ]+\b",
            r"\bSarjana [A-Za-z ]+\b", r"\bMagister [A-Za-z ]+\b", r"\bDiploma [A-Za-z0-9 ]+\b",
            r"\bUniversitas [A-Za-z ]+\b", r"\bInstitut [A-Za-z ]+\b", r"\bUniversity\b"
        ]

        found_skills = set()
        for pat in skill_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for match in matches:
                found_skills.add(match)

        found_edu = set()
        for pat in edu_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for match in matches:
                found_edu.add(match.strip())

        exp_match = re.search(
            r"(\d+)\+?\s*(?:tahun|years?)\s*(?:pengalaman|experience)",
            text,
            re.IGNORECASE
        )
        years_exp = int(exp_match.group(1)) if exp_match else None

        return {
            "skills": sorted(list(found_skills)),
            "education": sorted(list(found_edu))[:5],
            "experience_years_detected": years_exp,
        }

    def match_cv_with_job(
        self,
        cv_text: str,
        job_title: str,
        job_description: str
    ) -> Dict[str, Any]:
        """
        Executes end-to-end matching process:
        1. Encodes CV and Job Description into 768-d vectors.
        2. Computes Cosine Similarity score.
        3. Classifies status category.
        4. Extracts NER entity highlights.
        """
        full_job_text = f"Job Title: {job_title}\n\nDescription:\n{job_description}"

        cv_vec = self.get_document_embedding(cv_text)
        job_vec = self.get_document_embedding(full_job_text)

        score = self.calculate_cosine_similarity(cv_vec, job_vec)
        score_rounded = round(score, 4)
        percentage = round(score_rounded * 100.0, 2)
        status = self.determine_status_category(score_rounded)
        entities = self.extract_entities_hybrid(cv_text)

        return {
            "similarity_score": score_rounded,
            "similarity_percentage": percentage,
            "status_category": status,
            "extracted_entities": entities,
            "cv_text_preview": cv_text[:300] + "..." if len(cv_text) > 300 else cv_text,
            "job_description_preview": job_description[:300] + "..." if len(job_description) > 300 else job_description,
        }


mbert_service = MBERTMatchingService()
