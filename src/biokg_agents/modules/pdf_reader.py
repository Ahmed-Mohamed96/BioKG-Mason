from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import pdfplumber

@dataclass
class DocSegment:
    kind: str                  # "paragraph" or "table"
    text: str                  # text used as input to the LLM
    page: int
    table_index: Optional[int] = None

class PDFReader:
    """
    Reads biomedical PDFs and returns paragraphs or mixed segments (paragraphs + tables).
    """

    def __init__(self):
        pass

    def extract_paragraphs(self, pdf_path: str) -> List[str]:
        """
        Original method (kept for backward compatibility).
        Extract paragraphs from a PDF file.
        Very simple paragraph detection: split pages by double newlines.
        """
        path = Path(pdf_path)
        paragraphs: List[str] = []

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for para in text.split("\n\n"):
                    cleaned = " ".join(para.split())
                    if cleaned:
                        paragraphs.append(cleaned)

        return paragraphs

    def extract_segments(self, pdf_path: str) -> List[DocSegment]:
        """
        New method: returns both paragraphs and tables as DocSegments.
        """
        segments: List[DocSegment] = []
        path = Path(pdf_path)

        with pdfplumber.open(path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # 1) Paragraphs
                text = page.extract_text() or ""
                for para in text.split("\n\n"):
                    cleaned = " ".join(para.split())
                    if cleaned:
                        segments.append(
                            DocSegment(
                                kind="paragraph",
                                text=cleaned,
                                page=page_idx,
                            )
                        )

                # 2) Tables
                tables = page.extract_tables() or []
                for t_idx, table in enumerate(tables):
                    table_text = self._table_to_text(table, page_idx, t_idx)
                    if table_text.strip():
                        segments.append(
                            DocSegment(
                                kind="table",
                                text=table_text,
                                page=page_idx,
                                table_index=t_idx,
                            )
                        )

        return segments

    @staticmethod
    def _table_to_text(table, page: int, idx: int) -> str:
        """
        Convert a 2D list `table` -> text that the generic TripletExtractor can understand.
        Assumes first row is header.
        """
        if not table or len(table) < 2:
            return ""

        header = [h if h is not None else "" for h in table[0]]
        rows = table[1:]

        lines = []
        for row in rows:
            cells = []
            for col_name, value in zip(header, row):
                col_name = (col_name or "").strip()
                value = (value or "").strip()
                if col_name or value:
                    cells.append(f"{col_name}: {value}")
            if cells:
                lines.append("; ".join(cells))

        return (
            f"Table {idx + 1} on page {page}. Each line is a row; "
            f"each 'column: value' is a cell.\n"
            + "\n".join(lines)

        )