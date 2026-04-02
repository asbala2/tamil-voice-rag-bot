from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import chromadb
import yaml
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    """Container for a single text chunk and its metadata."""

    doc_id: str
    source: str
    text: str
    index: int


@dataclass
class FileIngestionResult:
    """Diagnostics for a single source file during ingestion."""

    filename: str
    file_type: str
    extraction_succeeded: bool
    extracted_char_count: int
    chunk_count: int
    final_status: str
    extraction_error: str | None = None


def load_config(config_path: str) -> dict:
    """Load YAML configuration using UTF-8."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping character chunks for retrieval."""
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned_text):
        end = min(len(cleaned_text), start + chunk_size)
        chunks.append(cleaned_text[start:end])
        if end == len(cleaned_text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def build_chunks(
    literature_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[Chunk], list[FileIngestionResult]]:
    """Build chunk objects and diagnostics for files in the source directory."""
    all_chunks: list[Chunk] = []
    diagnostics: list[FileIngestionResult] = []

    for path in sorted(p for p in literature_dir.iterdir() if p.is_file()):
        file_type = path.suffix.lower() or "(no extension)"
        try:
            text = extract_text(path)
        except Exception as error:
            diagnostics.append(
                FileIngestionResult(
                    filename=path.name,
                    file_type=file_type,
                    extraction_succeeded=False,
                    extracted_char_count=0,
                    chunk_count=0,
                    final_status="skipped",
                    extraction_error=str(error),
                )
            )
            continue

        extracted_char_count = len(text)
        pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not pieces:
            diagnostics.append(
                FileIngestionResult(
                    filename=path.name,
                    file_type=file_type,
                    extraction_succeeded=True,
                    extracted_char_count=extracted_char_count,
                    chunk_count=0,
                    final_status="skipped",
                )
            )
            continue

        doc_id = hashlib.md5(path.name.encode("utf-8")).hexdigest()
        for idx, piece in enumerate(pieces):
            all_chunks.append(Chunk(doc_id=doc_id, source=path.name, text=piece, index=idx))

        diagnostics.append(
            FileIngestionResult(
                filename=path.name,
                file_type=file_type,
                extraction_succeeded=True,
                extracted_char_count=extracted_char_count,
                chunk_count=len(pieces),
                final_status="ingested",
            )
        )

    return all_chunks, diagnostics


def extract_text_from_txt(path: Path) -> str:
    """Read UTF-8 text from plain text files."""
    return path.read_text(encoding="utf-8", errors="strict")


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from text-searchable PDF files."""
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def extract_text_from_docx(path: Path) -> str:
    """Extract text content from DOCX files without external DOCX dependencies."""
    with ZipFile(path) as archive:
        xml_content = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_content)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        paragraph_text = "".join(texts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)
    return "\n".join(paragraphs)


def extract_text(path: Path) -> str:
    """Extract source text for supported document types."""
    file_type = path.suffix.lower()
    if file_type == ".txt":
        return extract_text_from_txt(path)
    if file_type == ".pdf":
        return extract_text_from_pdf(path)
    if file_type == ".docx":
        return extract_text_from_docx(path)
    raise ValueError(f"unsupported type: {file_type or '(no extension)'}")


def print_ingestion_summary(results: list[FileIngestionResult]) -> None:
    """Print a human-readable per-file ingestion summary."""
    print("\n=== Ingestion File Summary ===")
    if not results:
        print("No files found in literature directory.")
        return

    for result in results:
        error_suffix = f" | error={result.extraction_error}" if result.extraction_error else ""
        print(
            " | ".join(
                [
                    f"filename={result.filename}",
                    f"type={result.file_type}",
                    f"extraction_succeeded={result.extraction_succeeded}",
                    f"chars={result.extracted_char_count}",
                    f"chunks={result.chunk_count}",
                    f"status={result.final_status}",
                ]
            )
            + error_suffix
        )


def ingest(config_path: str, rebuild_store: bool = False) -> None:
    """Ingest Tamil literature files into a Chroma vector database."""
    config = load_config(config_path)

    literature_dir = Path(config["paths"]["literature_dir"])
    vector_store_dir = Path(config["paths"]["vector_store_dir"])
    if rebuild_store and vector_store_dir.exists():
        shutil.rmtree(vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = int(config["retrieval"]["chunk_size"])
    chunk_overlap = int(config["retrieval"]["chunk_overlap"])
    collection_name = str(config["retrieval"]["collection_name"])
    embedding_model_name = str(config["embeddings"]["model_name"])

    chunks, diagnostics = build_chunks(literature_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print_ingestion_summary(diagnostics)
    if not chunks:
        raise ValueError(f"No supported documents produced chunks in {literature_dir}")

    embedder = SentenceTransformer(embedding_model_name, device="cpu")
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    client = chromadb.PersistentClient(path=str(vector_store_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)

    ids = [f"{chunk.doc_id}_{chunk.index}" for chunk in chunks]
    metadatas = [{"source": chunk.source, "chunk_index": chunk.index} for chunk in chunks]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    manifest = {
        "collection_name": collection_name,
        "document_count": len({chunk.source for chunk in chunks}),
        "chunk_count": len(chunks),
        "embedding_model": embedding_model_name,
    }
    (vector_store_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    """CLI entry point for Tamil literature ingestion."""
    parser = argparse.ArgumentParser(
        description="Read Tamil UTF-8 text/PDF/DOCX files and index them into Chroma.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--rebuild-store",
        action="store_true",
        help="Delete the existing vector DB directory before ingesting.",
    )
    args = parser.parse_args()
    ingest(args.config, rebuild_store=args.rebuild_store)


if __name__ == "__main__":
    main()
