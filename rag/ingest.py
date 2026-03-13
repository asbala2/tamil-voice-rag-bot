from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import yaml
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    """Container for a single text chunk and its metadata."""

    doc_id: str
    source: str
    text: str
    index: int


def load_config(config_path: str) -> dict:
    """Load YAML configuration using UTF-8."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_text_files(literature_dir: Path) -> Iterable[tuple[str, str]]:
    """Read UTF-8 Tamil text files from the configured literature directory."""
    for path in sorted(literature_dir.glob("*.txt")):
        yield path.name, path.read_text(encoding="utf-8", errors="strict")


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


def build_chunks(literature_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Build chunk objects for every Tamil text file in the source directory."""
    all_chunks: list[Chunk] = []
    for filename, text in read_text_files(literature_dir):
        doc_id = hashlib.md5(filename.encode("utf-8")).hexdigest()
        for idx, piece in enumerate(chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
            all_chunks.append(Chunk(doc_id=doc_id, source=filename, text=piece, index=idx))
    return all_chunks


def ingest(config_path: str) -> None:
    """Ingest Tamil literature files into a Chroma vector database."""
    config = load_config(config_path)

    literature_dir = Path(config["paths"]["literature_dir"])
    vector_store_dir = Path(config["paths"]["vector_store_dir"])
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = int(config["retrieval"]["chunk_size"])
    chunk_overlap = int(config["retrieval"]["chunk_overlap"])
    collection_name = str(config["retrieval"]["collection_name"])
    embedding_model_name = str(config["embeddings"]["model_name"])

    chunks = build_chunks(literature_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError(f"No UTF-8 .txt documents found in {literature_dir}")

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
        description="Read Tamil UTF-8 text files and index them into Chroma.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    ingest(args.config)


if __name__ == "__main__":
    main()
