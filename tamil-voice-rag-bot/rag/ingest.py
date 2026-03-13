from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import chromadb
import yaml
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    doc_id: str
    source: str
    text: str
    index: int


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text_files(literature_dir: Path) -> Iterable[tuple[str, str]]:
    for path in sorted(literature_dir.glob("*.txt")):
        yield path.name, path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
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
    all_chunks: List[Chunk] = []
    for filename, text in read_text_files(literature_dir):
        raw_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc_id = hashlib.md5(filename.encode("utf-8")).hexdigest()
        for idx, chunk in enumerate(raw_chunks):
            all_chunks.append(Chunk(doc_id=doc_id, source=filename, text=chunk, index=idx))
    return all_chunks


def ingest(config_path: str) -> None:
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
        raise ValueError(f"No .txt documents found in {literature_dir}")

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
    metadatas = [{"source": chunk.source, "chunk_index": chunk.index} for chunk in chunks]

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
