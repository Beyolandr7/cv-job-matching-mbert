import html
import io
import re
from typing import BinaryIO, Union

from pypdf import PdfReader


class PDFService:
    """Service for extracting and cleaning text from in-memory PDF files."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans extracted text specifically for mBERT tokenization:
        1. Unescapes HTML entities (&amp; -> &, &quot; -> ", etc.).
        2. Strips HTML tags and URLs.
        3. Normalizes visual escape characters and non-printable unicode whitespace.
        4. Normalizes decorative bullets to clean space.
        5. Preserves CAPITALIZATION strictly for NER and cased mBERT embeddings.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Unescape HTML entities
        cleaned = html.unescape(text)

        # 2. Strip HTML tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        # 3. Strip URLs
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

        # 4. Normalize escape characters and invisible unicode characters
        cleaned = re.sub(r"[\r\n\t\xa0\u200b\u200c\u200d\u200e\u200f\ufeff\u3000]", " ", cleaned)

        # 5. Normalize visual bullets and special decorative symbols to spaces
        cleaned = re.sub(r"[\u2022\u2023\u25e6\u2043\u2219\u25cf\u25aa\u25ab\u25c6\u25c7\u2713\u2714\u2013\u2014]", " ", cleaned)

        # 6. Normalize multiple consecutive spaces into a single space (PRESERVE CASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    @classmethod
    def extract_text_from_pdf_stream(cls, file_stream: Union[BinaryIO, bytes]) -> str:
        """
        Extracts raw text from an in-memory PDF file stream and applies text cleaning.
        """
        try:
            if isinstance(file_stream, bytes):
                reader = PdfReader(io.BytesIO(file_stream))
            else:
                reader = PdfReader(file_stream)

            raw_text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    raw_text_parts.append(page_text)

            full_raw_text = "\n".join(raw_text_parts)
            return cls.clean_text(full_raw_text)
        except Exception as exc:
            raise ValueError(f"Gagal membaca berkas PDF: {str(exc)}") from exc


pdf_service = PDFService()
