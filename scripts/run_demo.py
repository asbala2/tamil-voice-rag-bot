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
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Synthesize the final Tamil answer into speech using Piper TTS",
    )
    return parser.parse_args()


def maybe_speak_answer(config_path: str, answer: str, speak: bool) -> str | None:
    """Optionally synthesize Tamil answer audio and return the written path."""
    if not speak:
        return None

    from tts.piper_speak import PiperSpeaker

    speaker = PiperSpeaker(config_path=config_path)

    if not speaker.enabled:
        print("\n=== Audio Reply Skipped ===")
        print("TTS is disabled in config.yaml (tts.enabled: false).")
        return None

    try:
        audio_path = speaker.synthesize(answer)
    except RuntimeError as exc:
        print("\n=== Audio Reply Failed ===")
        print(str(exc))
        return None

    print("\n=== Audio Reply Saved ===")
    print(audio_path)
    return audio_path


def print_retrieved_sources(retrieved_chunks: list[dict]) -> list[str]:
    """Print unique retrieved source filenames in ranked order."""
    seen = set()
    ordered_sources: list[str] = []
    for chunk in retrieved_chunks:
        source = str(chunk.get("source", "unknown"))
        if source not in seen:
            seen.add(source)
            ordered_sources.append(source)

    print("\n=== Retrieved Sources ===")
    if not ordered_sources:
        print("(No sources retrieved)")
    else:
        for idx, source in enumerate(ordered_sources, start=1):
            print(f"{idx}. {source}")
    return ordered_sources


def run_text_qa(
    config_path: str,
    question: str,
    top_k: int | None = None,
    speak: bool = False,
) -> list[str]:
    """Retrieve context from Chroma, print Tamil answer, and optionally generate voice output."""
    from llm.ollama_client import OllamaTamilQA
    from rag.query_router import QueryRouter
    from rag.retriever import TamilRetriever

    router = QueryRouter()
    route_result = router.route(question)
    qa_client = OllamaTamilQA(config_path=config_path)

    retrieved_chunks: list[dict] = []
    if route_result.route == "document_related":
        print("Route: RAG")
        retriever = TamilRetriever(config_path=config_path)
        retrieved_chunks = retriever.retrieve(question, top_k=top_k)
        answer = qa_client.answer(question, retrieved_chunks)
    else:
        print("Route: General LLM")
        answer = qa_client.answer_general(question)

    print("\n=== Question ===")
    print(question)
    sources = print_retrieved_sources(retrieved_chunks)
    print("\n=== Retrieved Chunks ===")
    print(json.dumps(retrieved_chunks, ensure_ascii=False, indent=2))
    print("\n=== Answer ===")
    print(answer)

    maybe_speak_answer(config_path=config_path, answer=answer, speak=speak)
    return sources


def run_audio_qa(
    config_path: str,
    audio_path: str,
    top_k: int | None = None,
    speak: bool = False,
) -> None:
    """Transcribe audio and run retrieval + Tamil answering from the transcript."""
    from speech.whisper_transcribe import TamilWhisperTranscriber

    transcriber = TamilWhisperTranscriber(config_path=config_path)
    transcription = transcriber.transcribe_file(audio_path)
    question_text = transcription["text"].strip()

    if not question_text:
        raise SystemExit("Unable to transcribe a Tamil question from the given audio file")

    run_text_qa(config_path=config_path, question=question_text, top_k=top_k, speak=speak)


def main() -> None:
    """Run Tamil QA from question text or from an audio file transcription."""
    args = parse_args()

    if not args.audio and not args.question:
        raise SystemExit("Provide either --question or --audio")

    if args.question:
        run_text_qa(config_path=args.config, question=args.question, top_k=args.top_k, speak=args.speak)
        return

    run_audio_qa(config_path=args.config, audio_path=args.audio, top_k=args.top_k, speak=args.speak)


if __name__ == "__main__":
    main()
