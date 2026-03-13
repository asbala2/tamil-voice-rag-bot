from rag.ingest import chunk_text


def test_chunk_text_basic() -> None:
    text = "அ" * 1200
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)
