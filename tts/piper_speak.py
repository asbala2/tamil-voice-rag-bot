from __future__ import annotations

import array
import json
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

    def _resolve_wav_params(self, voice: object) -> tuple[int, int, int]:
        """Resolve WAV metadata from Piper voice/model configuration."""

        def _attr_path(obj: object, path: str) -> object | None:
            current: object | None = obj
            for part in path.split("."):
                if current is None:
                    return None
                current = getattr(current, part, None)
            return current

        sample_rate = None
        for attr_name in (
            "config.sample_rate",
            "config.audio.sample_rate",
            "sample_rate",
            "audio_sample_rate",
        ):
            value = _attr_path(voice, attr_name)
            if isinstance(value, int) and value > 0:
                sample_rate = value
                break

        channels = 1
        sample_width = 2

        try:
            with open(self.config_path, "r", encoding="utf-8") as cfg_file:
                model_cfg = json.load(cfg_file)

            audio_cfg = model_cfg.get("audio", {}) if isinstance(model_cfg, dict) else {}
            if sample_rate is None:
                for value in (audio_cfg.get("sample_rate"), model_cfg.get("sample_rate")):
                    if isinstance(value, int) and value > 0:
                        sample_rate = value
                        break

            ch = audio_cfg.get("channels", audio_cfg.get("num_channels"))
            if isinstance(ch, int) and ch > 0:
                channels = ch

            bits = audio_cfg.get("bits_per_sample", audio_cfg.get("bit_depth"))
            if isinstance(bits, int) and bits in (8, 16, 24, 32):
                sample_width = bits // 8
        except Exception:  # noqa: BLE001
            # Keep sensible defaults if JSON metadata cannot be parsed.
            pass

        if sample_rate is None:
            raise RuntimeError(
                "Unable to determine Piper sample rate from voice/model metadata. "
                f"Check config file: {self.config_path!s}."
            )

        return channels, sample_width, sample_rate

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

        channels, sample_width, sample_rate = self._resolve_wav_params(voice)

        chunk_count = 0
        try:
            with wave.open(str(target), "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)

                if hasattr(voice, "synthesize_stream_raw"):
                    for raw_chunk in voice.synthesize_stream_raw(safe_text):
                        if raw_chunk is None:
                            continue

                        if isinstance(raw_chunk, (bytes, bytearray, memoryview)):
                            chunk_bytes = bytes(raw_chunk)
                        elif hasattr(raw_chunk, "tobytes"):
                            chunk_bytes = raw_chunk.tobytes()
                        else:
                            try:
                                chunk_bytes = bytes(raw_chunk)
                            except Exception:  # noqa: BLE001
                                chunk_bytes = array.array("h", raw_chunk).tobytes()

                        if not chunk_bytes:
                            continue

                        wav_file.writeframesraw(chunk_bytes)
                        chunk_count += 1
                else:
                    voice.synthesize(safe_text, wav_file)
                    chunk_count = 1

                if chunk_count == 0:
                    raise RuntimeError(
                        "Piper returned zero audio chunks; no WAV audio frames were generated."
                    )
        except Exception as exc:  # noqa: BLE001
            if target.exists():
                target.unlink(missing_ok=True)

            zero_chunks = "yes" if chunk_count == 0 else "no"
            raise RuntimeError(
                "Piper synthesis failed via Python API. "
                f"Output path: {target!s}. "
                f"Zero chunks returned: {zero_chunks}. "
                f"Sanitized text preview: {safe_text[:200]!r}\n"
                f"Original error: {exc}"
            ) from exc

        if self.auto_play:
            self._play_audio(target)

        return str(target)
