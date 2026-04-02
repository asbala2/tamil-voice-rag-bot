from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from faster_whisper import WhisperModel


class TamilWhisperTranscriber:
    """Tamil audio transcription wrapper around Faster-Whisper."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load speech settings from config and initialize a CPU-friendly model."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        speech_config = self.config["speech"]
        self.model_name = speech_config.get("whisper_model_size") or speech_config["whisper_model"]
        self.language = self._parse_language_mode(speech_config.get("language", "ta"))
        self.beam_size = int(speech_config.get("beam_size", 8))
        self.best_of = int(speech_config.get("best_of", 5))
        self.temperature = float(speech_config.get("temperature", 0.0))
        self.vad_filter = bool(speech_config.get("vad_filter", True))

        self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")


    @staticmethod
    def _parse_language_mode(language_value: Optional[str]) -> Optional[str]:
        """Return a Faster-Whisper language code or ``None`` for auto-detection."""
        if language_value is None:
            return None

        normalized = str(language_value).strip().lower()
        if normalized in {"", "auto", "none"}:
            return None
        return normalized

    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe a local audio file and return text and metadata."""
        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        transcription_language = self._parse_language_mode(language) if language is not None else self.language
        segments, info = self.model.transcribe(
            str(audio),
            language=transcription_language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            vad_filter=self.vad_filter,
        )

        text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        text = " ".join(text_parts)
        return {
            "text": text,
            "language": getattr(info, "language", transcription_language),
            "language_mode": transcription_language or "auto",
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
