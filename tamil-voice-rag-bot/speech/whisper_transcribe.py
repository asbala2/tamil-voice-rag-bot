from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import yaml
from faster_whisper import WhisperModel


class TamilWhisperTranscriber:
    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config["speech"]["whisper_model"]
        self.language = self.config["speech"]["language"]
        self.beam_size = int(self.config["speech"]["beam_size"])
        self.vad_filter = bool(self.config["speech"]["vad_filter"])

        self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
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
