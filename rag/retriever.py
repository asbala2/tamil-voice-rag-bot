from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import chromadb
import yaml
from sentence_transformers import SentenceTransformer


class TamilRetriever:
    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.vector_store_dir = Path(self.config["paths"]["vector_store_dir"])
        self.collection_name = str(self.config["retrieval"]["collection_name"])
        self.top_k = int(self.config["retrieval"]["top_k"])
        self.embedder = SentenceTransformer(self.config["embeddings"]["model_name"])
        self.client = chromadb.PersistentClient(path=str(self.vector_store_dir))
        self.collection = self.client.get_collection(self.collection_name)

    def retrieve(self, question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        question_embedding = self.embedder.encode([question], normalize_embeddings=True).tolist()[0]
        results = self.collection.query(query_embeddings=[question_embedding], n_results=k)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved = []
        for doc, meta, distance in zip(docs, metas, distances):
            retrieved.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", -1),
                    "distance": float(distance) if distance is not None else None,
                }
            )
        return retrieved


def format_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return ""
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"[ஆதாரம் {idx}] {chunk['source']} / பகுதி {chunk['chunk_index']}\n{chunk['text']}")
    return "\n\n".join(lines)
