from __future__ import annotations

from pathlib import Path

import yaml
from TTS.api import TTS


class XTTSSpeaker:
    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.enabled = bool(self.config["tts"]["enabled"])
        self.model_name = self.config["tts"]["model_name"]
        self.language = self.config["tts"]["language"]
        self.speaker_wav = self.config["tts"]["speaker_wav"]
        self.output_file = Path(self.config["tts"]["output_file"])
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._tts = None

    def _ensure_model(self) -> None:
        if self._tts is None:
            self._tts = TTS(self.model_name)

    def synthesize(self, text: str, output_path: str | None = None) -> str:
        if not self.enabled:
            raise RuntimeError("TTS is disabled in config.yaml. Set tts.enabled: true to use XTTS.")

        self._ensure_model()
        target = Path(output_path) if output_path else self.output_file
        target.parent.mkdir(parents=True, exist_ok=True)

        kwargs = {
            "text": text,
            "file_path": str(target),
            "language": self.language,
        }
        if self.speaker_wav:
            kwargs["speaker_wav"] = self.speaker_wav

        self._tts.tts_to_file(**kwargs)
        return str(target)
