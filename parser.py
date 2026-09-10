"""Document parser supporting PDF, DOCX, and TXT files for resumes and job descriptions."""

import io
import re
from pathlib import Path
from typing import Union

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None


class DocumentParser:
    """Extracts clean text from various document formats."""

    @staticmethod
    def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
        """Extracts text from raw bytes based on the file extension."""
        extension = Path(filename).suffix.lower()

        if extension == ".pdf":
            return DocumentParser._parse_pdf(file_bytes)
        elif extension in [".docx", ".doc"]:
            return DocumentParser._parse_docx(file_bytes)
        elif extension in [".txt", ".md"]:
            return DocumentParser._parse_text(file_bytes)
        else:
            # Fallback to UTF-8 text decoding
            return DocumentParser._parse_text(file_bytes)

    @staticmethod
    def extract_text_from_file(file_path: Union[str, Path]) -> str:
        """Extracts text directly from a local file path."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, "rb") as f:
            return DocumentParser.extract_text_from_bytes(f.read(), path.name)

    @staticmethod
    def _parse_pdf(file_bytes: bytes) -> str:
        """Extract text from PDF bytes using pypdf."""
        if PdfReader is None:
            raise ImportError("pypdf is not installed. Please install pypdf to process PDF files.")
        
        reader = PdfReader(io.BytesIO(file_bytes))
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text)
        
        raw_text = "\n\n".join(extracted_pages)
        return DocumentParser._clean_text(raw_text)

    @staticmethod
    def _parse_docx(file_bytes: bytes) -> str:
        """Extract text from DOCX bytes using python-docx."""
        if docx is None:
            raise ImportError("python-docx is not installed. Please install python-docx to process DOCX files.")
        
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        
        raw_text = "\n".join(paragraphs)
        return DocumentParser._clean_text(raw_text)

    @staticmethod
    def _parse_text(file_bytes: bytes) -> str:
        """Extract text from plain text/markdown bytes."""
        try:
            raw_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = file_bytes.decode("latin-1", errors="replace")
        return DocumentParser._clean_text(raw_text)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize line breaks and clean excessive whitespace while retaining structure."""
        if not text:
            return ""
        # Standardize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove consecutive blank lines (more than 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove unusual whitespace characters
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
