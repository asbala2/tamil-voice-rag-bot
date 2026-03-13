from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml
from faster_whisper import WhisperModel


class TamilWhisperTranscriber:
    """Tamil audio transcription wrapper around Faster-Whisper."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load speech settings from config and initialize a CPU-friendly model."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config["speech"]["whisper_model"]
        self.language = self.config["speech"]["language"]
        self.beam_size = int(self.config["speech"]["beam_size"])
        self.vad_filter = bool(self.config["speech"]["vad_filter"])

        self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe a local audio file and return text and metadata."""
        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        segments, info = self.model.transcribe(
            str(audio),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )

        text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        text = " ".join(text_parts)
        return {
            "text": text,
            "language": getattr(info, "language", self.language),
            "duration": getattr(info, "duration", None),
        }


def build_parser() -> argparse.ArgumentParser:
    """Build a CLI parser for direct Tamil audio transcription."""
    parser = argparse.ArgumentParser(description="Transcribe a Tamil audio file using Faster-Whisper")
    parser.add_argument("--input", required=True, help="Path to audio file (e.g. sample.wav)")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full transcription output as JSON instead of only text",
    )
    return parser


def main() -> None:
    """Run CLI entry point for Tamil audio-file transcription."""
    args = build_parser().parse_args()
    transcriber = TamilWhisperTranscriber(config_path=args.config)
    result = transcriber.transcribe_file(args.input)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["text"])


if __name__ == "__main__":
    main()
