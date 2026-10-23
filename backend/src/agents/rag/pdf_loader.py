
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_PAGES = 500

def load_pdf(file_path: str | Path) -> list[Document]:
    """Extract PDF text and split it into page aware chunks."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError("PDF file was not found.")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    if path.stat().st_size > MAX_PDF_SIZE_BYTES:
        raise ValueError("PDF exceeds the 20 MB size limit.")

    reader = PdfReader(str(path), strict=True)

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    if len(reader.pages) == 0:
        raise ValueError("PDF contains no pages.")

    if len(reader.pages) > MAX_PAGES:
        raise ValueError("PDF exceeds the 500-page limit.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )

    chunks: list[Document] = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue

        page_document = Document(
            page_content=text,
            metadata={
                "source": path.name,
                "page": page_index + 1,
            },
        )

        page_chunks = splitter.split_documents([page_document])
        chunks.extend(page_chunks)

    if not chunks:
        raise ValueError(
            "No extractable text found. The PDF may contain scanned images."
        )

    return chunks

