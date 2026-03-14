from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml
from TTS.api import TTS


class XTTSSpeaker:
    """Tamil speech synthesis using Coqui TTS with optional model fallback."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load TTS settings from config and prepare lazy model initialization."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.enabled = bool(self.config["tts"]["enabled"])
        self.model_name = self.config["tts"]["model_name"]
        self.fallback_model_name = self.config["tts"].get("fallback_model_name")
        self.language = self.config["tts"]["language"]
        self.speaker_wav = self.config["tts"]["speaker_wav"]
        self.output_file = Path(self.config["tts"]["output_file"])
        self.auto_play = bool(self.config["tts"].get("auto_play", True))
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._tts = None
        self._active_model_name: str | None = None

    def _ensure_model(self) -> None:
        """Load the preferred Coqui model, then fall back to a lighter option if needed."""
        if self._tts is None:
            model_candidates: list[str] = [self.model_name]
            if self.fallback_model_name and self.fallback_model_name != self.model_name:
                model_candidates.append(self.fallback_model_name)

            load_errors: list[str] = []
            for model_name in model_candidates:
                try:
                    self._tts = TTS(model_name)
                    self._active_model_name = model_name
                    return
                except Exception as exc:  # noqa: BLE001
                    load_errors.append(f"{model_name}: {exc}")

            joined_errors = " | ".join(load_errors)
            raise RuntimeError(f"Unable to load configured Coqui TTS model(s). {joined_errors}")

    def _build_tts_kwargs(self, text: str, target_path: Path) -> dict[str, Any]:
        """Build kwargs safely for selected model families."""
        kwargs: dict[str, Any] = {
            "text": text,
            "file_path": str(target_path),
        }

        if self._active_model_name and "multilingual" in self._active_model_name:
            kwargs["language"] = self.language

        if self.speaker_wav:
            kwargs["speaker_wav"] = self.speaker_wav

        return kwargs

    def _try_play_audio(self, audio_path: Path) -> bool:
        """Try to play synthesized audio on common platforms without extra dependencies."""
        if sys.platform.startswith("win"):
            try:
                import winsound

                winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
                return True
            except Exception:  # noqa: BLE001
                return False

        commands = [["afplay", str(audio_path)], ["aplay", str(audio_path)]]
        for command in commands:
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (FileNotFoundError, subprocess.SubprocessError):
                continue
        return False

    def synthesize(self, text: str, output_path: str | None = None) -> str:
        """Synthesize Tamil speech to a wav file and optionally play it."""
        if not self.enabled:
            raise RuntimeError("TTS is disabled in config.yaml. Set tts.enabled: true to use XTTS.")

        self._ensure_model()
        target = Path(output_path) if output_path else self.output_file
        target.parent.mkdir(parents=True, exist_ok=True)

        kwargs = self._build_tts_kwargs(text=text, target_path=target)
        self._tts.tts_to_file(**kwargs)

        if self.auto_play:
            self._try_play_audio(target)

        return str(target)
