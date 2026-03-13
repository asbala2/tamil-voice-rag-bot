from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Tamil RAG demo."""
    parser = argparse.ArgumentParser(description="Run Tamil RAG retrieval + answering demo")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--question", help="Tamil text question")
    parser.add_argument("--audio", help="Path to Tamil audio file")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k retrieval count")
    return parser.parse_args()


def run_text_qa(config_path: str, question: str, top_k: int | None = None) -> None:
    """Retrieve context from Chroma and print a Tamil answer from Ollama."""
    from llm.ollama_client import OllamaTamilQA
    from rag.retriever import TamilRetriever

    retriever = TamilRetriever(config_path=config_path)
    qa_client = OllamaTamilQA(config_path=config_path)

    retrieved_chunks = retriever.retrieve(question, top_k=top_k)
    answer = qa_client.answer(question, retrieved_chunks)

    print("\n=== Question ===")
    print(question)
    print("\n=== Retrieved Chunks ===")
    print(json.dumps(retrieved_chunks, ensure_ascii=False, indent=2))
    print("\n=== Answer ===")
    print(answer)


def run_audio_qa(config_path: str, audio_path: str, top_k: int | None = None) -> None:
    """Transcribe audio and run retrieval + Tamil answering from the transcript."""
    from speech.whisper_transcribe import TamilWhisperTranscriber

    transcriber = TamilWhisperTranscriber(config_path=config_path)
    transcription = transcriber.transcribe_file(audio_path)
    question_text = transcription["text"].strip()

    if not question_text:
        raise SystemExit("Unable to transcribe a Tamil question from the given audio file")

    run_text_qa(config_path=config_path, question=question_text, top_k=top_k)


def main() -> None:
    """Run Tamil QA from question text or from an audio file transcription."""
    args = parse_args()

    if not args.audio and not args.question:
        raise SystemExit("Provide either --question or --audio")

    if args.question:
        run_text_qa(config_path=args.config, question=args.question, top_k=args.top_k)
        return

    run_audio_qa(config_path=args.config, audio_path=args.audio, top_k=args.top_k)


if __name__ == "__main__":
    main()
