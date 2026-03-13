from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.voice_pipeline import TamilVoiceRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tamil Voice RAG Bot demo")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--audio", help="Path to Tamil audio file")
    parser.add_argument("--question", help="Tamil text question")
    parser.add_argument("--voice-reply", action="store_true", help="Generate XTTS voice reply")
    args = parser.parse_args()

    if not args.audio and not args.question:
        raise SystemExit("Provide either --audio or --question")

    pipeline = TamilVoiceRAGPipeline(config_path=args.config)

    if args.audio:
        result = pipeline.run_with_audio_file(args.audio, make_voice_reply=args.voice_reply)
    else:
        result = pipeline.run_with_text(args.question, make_voice_reply=args.voice_reply)

    print("\n=== Question ===")
    print(result.question_text)
    print("\n=== Retrieved Chunks ===")
    print(json.dumps(result.retrieved_chunks, ensure_ascii=False, indent=2))
    print("\n=== Answer ===")
    print(result.answer_text)
    if result.audio_reply_path:
        print("\n=== Voice Reply ===")
        print(result.audio_reply_path)


if __name__ == "__main__":
    main()
