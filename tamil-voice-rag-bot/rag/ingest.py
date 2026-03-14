from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import chromadb
import yaml
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    """Text chunk and source metadata ready for vector ingestion."""

    doc_id: str
    source: str
    file_type: str
    text: str
    index: int


def load_config(config_path: str) -> dict:
    """Load YAML config from disk."""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_text_from_txt(path: Path) -> str:
    """Extract text from a UTF-8 text file."""

    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf."""

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""

    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def iter_supported_files(literature_dir: Path) -> List[Path]:
    """Return all supported document files under the literature folder."""

    supported_suffixes = {".txt", ".pdf", ".docx"}
    files = [path for path in literature_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported_suffixes]
    return sorted(files)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping character chunks."""

    text = " ".join(text.split())
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def build_chunks(literature_dir: Path, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    """Extract text from supported files and build chunks with metadata."""

    extractors: dict[str, Callable[[Path], str]] = {
        ".txt": extract_text_from_txt,
        ".pdf": extract_text_from_pdf,
        ".docx": extract_text_from_docx,
    }

    all_chunks: List[Chunk] = []
    for path in iter_supported_files(literature_dir):
        file_type = path.suffix.lower().lstrip(".")
        extractor = extractors.get(path.suffix.lower())
        if extractor is None:
            print(f"Skipping unsupported file: {path}")
            continue

        try:
            text = extractor(path)
        except Exception as exc:
            print(f"Skipping unreadable file: {path} ({exc})")
            continue

        raw_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not raw_chunks:
            print(f"Skipping empty file: {path}")
            continue

        source = str(path.relative_to(literature_dir))
        doc_id = hashlib.md5(source.encode("utf-8")).hexdigest()
        for idx, chunk in enumerate(raw_chunks):
            all_chunks.append(Chunk(doc_id=doc_id, source=source, file_type=file_type, text=chunk, index=idx))
    return all_chunks


def ingest(config_path: str) -> None:
    """Build embeddings from literature files and store them in Chroma."""

    config = load_config(config_path)
    literature_dir = Path(config["paths"]["literature_dir"])
    vector_store_dir = Path(config["paths"]["vector_store_dir"])
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = int(config["retrieval"]["chunk_size"])
    chunk_overlap = int(config["retrieval"]["chunk_overlap"])
    collection_name = str(config["retrieval"]["collection_name"])
    embedding_model_name = str(config["embeddings"]["model_name"])

    chunks = build_chunks(literature_dir, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError(f"No supported non-empty documents found in {literature_dir}")

    embedder = SentenceTransformer(embedding_model_name)
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(vector_store_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)

    ids = [f"{chunk.doc_id}_{chunk.index}" for chunk in chunks]
    metadatas = [
        {"source": chunk.source, "file_type": chunk.file_type, "chunk_index": chunk.index}
        for chunk in chunks
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    manifest = {
        "collection_name": collection_name,
        "document_count": len({c.source for c in chunks}),
        "chunk_count": len(chunks),
        "embedding_model": embedding_model_name,
    }
    (vector_store_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Chroma vector store from Tamil text docs.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    ingest(args.config)
