# biokg/pdf/pdf_reader.py

from pathlib import Path
from typing import List

import pdfplumber


class PDFReader:
    """
    Reads biomedical PDFs and returns paragraphs.
    """

    def __init__(self):
        pass

    def extract_paragraphs(self, pdf_path: str) -> List[str]:
        """
        Extract paragraphs from a PDF file.
        Very simple paragraph detection: split pages by double newlines.
        """
        path = Path(pdf_path)
        paragraphs: List[str] = []

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                # Normalize whitespace and split into paragraphs
                for para in text.split("\n\n"):
                    cleaned = " ".join(para.split())
                    if cleaned:
                        paragraphs.append(cleaned)

        return paragraphs