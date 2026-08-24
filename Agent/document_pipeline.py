"""
Phase 1 MVP — File -> Extract -> LLM pipeline.

Upload File -> Detect File Type -> Extract Content -> Convert to common text
-> User Question -> LLM -> Answer + source

Supported: PDF, DOCX, TXT, CSV, XLSX.

This module does NOT import or modify anything from dynamic_langgraph_backend.py
except the already-constructed `llm` instance, so the orchestrator graph is untouched.
"""
import io
import os
import uuid
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from dynamic_langgraph_backend import llm

MAX_CHARS_FOR_CONTEXT = 12000  # MVP: whole-document context, no chunking/retrieval yet

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx"}


class UnsupportedFileType(Exception):
    pass


def detect_file_type(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Unsupported file type: '{ext or 'unknown'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return ext


# ---------- extractors (one per type, common text out) ----------

def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[Page {i}]\n{text}")
    return "\n\n".join(pages)


def _extract_docx(data: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _extract_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _extract_csv(data: bytes) -> str:
    import pandas as pd
    df = pd.read_csv(io.BytesIO(data))
    return df.to_csv(index=False)


def _extract_xlsx(data: bytes) -> str:
    import pandas as pd
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    parts = [f"[Sheet: {name}]\n{df.to_csv(index=False)}" for name, df in sheets.items()]
    return "\n\n".join(parts)


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
    ".csv": _extract_csv,
    ".xlsx": _extract_xlsx,
}


def extract_text(filename: str, data: bytes) -> str:
    """Detect file type, extract, and convert to a common plain-text format."""
    ext = detect_file_type(filename)
    text = _EXTRACTORS[ext](data).strip()
    if not text:
        raise ValueError("No extractable text found in this file.")
    return text


# ---------- storage (in-memory, per-process — Phase 1 only) ----------

class DocumentStore:
    def __init__(self):
        self._docs: dict[str, dict] = {}

    def add(self, filename: str, text: str) -> str:
        doc_id = str(uuid.uuid4())
        self._docs[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "text": text,
            "char_count": len(text),
            "uploaded_at": datetime.now().isoformat(),
        }
        return doc_id

    def get(self, doc_id: str) -> dict | None:
        return self._docs.get(doc_id)

    def list(self) -> list[dict]:
        return [
            {k: v for k, v in d.items() if k != "text"}
            for d in sorted(self._docs.values(), key=lambda d: d["uploaded_at"], reverse=True)
        ]


document_store = DocumentStore()


# ---------- Q&A ----------

def ask_document(doc_id: str, question: str) -> dict:
    """Answer a question grounded in a single uploaded document's extracted text."""
    doc = document_store.get(doc_id)
    if doc is None:
        raise KeyError(f"No document found with id: {doc_id}")

    context = doc["text"][:MAX_CHARS_FOR_CONTEXT]
    truncated = len(doc["text"]) > MAX_CHARS_FOR_CONTEXT

    system_prompt = (
        "You answer questions using ONLY the document content provided below. "
        "If the answer isn't contained in the document, say so clearly instead of guessing.\n\n"
        f"--- DOCUMENT: {doc['filename']} ---\n{context}\n--- END DOCUMENT ---"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])

    return {
        "answer": response.content,
        "source": {
            "filename": doc["filename"],
            "doc_id": doc_id,
            "truncated": truncated,
        },
    }
