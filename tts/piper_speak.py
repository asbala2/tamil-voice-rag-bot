from __future__ import annotations

from pathlib import Path
import string
import subprocess
import sys
import wave

import yaml


class PiperSpeaker:
    """Tamil speech synthesis using Piper's Python API and local ONNX voice files."""

    _ALLOWED_PUNCTUATION = set(string.punctuation) | {"…", "“", "”", "‘", "’", "-", "–", "—"}

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load Piper settings from config.yaml."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        tts_cfg = self.config["tts"]
        self.enabled = bool(tts_cfg["enabled"])
        # Kept for backward compatibility with existing config structure.
        self.piper_binary = tts_cfg.get("piper_binary", "piper")
        self.model_path = Path(tts_cfg["model_path"])
        self.config_path = Path(tts_cfg["config_path"])
        self.output_file = Path(tts_cfg.get("output_file", "data/output/output_answer.wav"))
        self.auto_play = bool(tts_cfg.get("auto_play", True))

    def _play_audio(self, audio_path: Path) -> bool:
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

    def _validate_paths(self) -> None:
        """Ensure required Piper model files exist before synthesis."""
        missing = [str(p) for p in [self.model_path, self.config_path] if not p.exists()]
        if missing:
            joined = ", ".join(missing)
            raise FileNotFoundError(
                "Missing Piper voice file(s): "
                f"{joined}. Download a Tamil Piper model and set tts.model_path/tts.config_path."
            )

    def _sanitize_text(self, text: str) -> str:
        """Keep Tamil/script-safe characters and remove invalid Unicode before synthesis."""

        def is_tamil_char(ch: str) -> bool:
            codepoint = ord(ch)
            return 0x0B80 <= codepoint <= 0x0BFF

        sanitized_chars: list[str] = []
        for char in text:
            codepoint = ord(char)

            # Strip surrogate code points that break UTF-8 encoding paths on some platforms.
            if 0xD800 <= codepoint <= 0xDFFF:
                continue

            if is_tamil_char(char) or char.isdigit() or char.isspace() or char in self._ALLOWED_PUNCTUATION:
                sanitized_chars.append(char)

        normalized = " ".join("".join(sanitized_chars).split())
        return normalized or "பதில் கிடைக்கவில்லை."

    def synthesize(self, text: str, output_path: str | None = None) -> str:
        """Generate Tamil speech via Piper Python API and return wav output path."""
        if not self.enabled:
            raise RuntimeError("TTS is disabled in config.yaml. Set tts.enabled: true to use Piper.")

        self._validate_paths()
        target = Path(output_path) if output_path else self.output_file
        target.parent.mkdir(parents=True, exist_ok=True)

        safe_text = self._sanitize_text(text)

        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise RuntimeError(
                "Piper Python package is not installed. Install dependencies with `pip install -r requirements.txt` "
                "and ensure `piper-tts` is available in this environment."
            ) from exc

        try:
            voice = PiperVoice.load(str(self.model_path), config_path=str(self.config_path), use_cuda=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load Piper voice model with Python API. "
                f"model_path={self.model_path!s}, config_path={self.config_path!s}. "
                "Ensure the ONNX and JSON files match and are readable."
            ) from exc

        try:
            with wave.open(str(target), "wb") as wav_file:
                voice.synthesize(safe_text, wav_file)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Piper synthesis failed via Python API. "
                f"Output path: {target!s}. "
                f"Sanitized text preview: {safe_text[:200]!r}\n"
                f"Original error: {exc}"
            ) from exc

        if self.auto_play:
            self._play_audio(target)

        return str(target)
