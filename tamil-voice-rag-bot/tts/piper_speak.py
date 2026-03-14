from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


class PiperSpeaker:
    """Tamil speech synthesis using Piper CLI and local ONNX voice files."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load Piper settings from config.yaml."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        tts_cfg = self.config["tts"]
        self.enabled = bool(tts_cfg["enabled"])
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

    def synthesize(self, text: str, output_path: str | None = None) -> str:
        """Generate Tamil speech via Piper and return wav output path."""
        if not self.enabled:
            raise RuntimeError("TTS is disabled in config.yaml. Set tts.enabled: true to use Piper.")

        self._validate_paths()
        target = Path(output_path) if output_path else self.output_file
        target.parent.mkdir(parents=True, exist_ok=True)

        command = [
            self.piper_binary,
            "--model",
            str(self.model_path),
            "--config",
            str(self.config_path),
            "--output_file",
            str(target),
        ]

        try:
            subprocess.run(
                command,
                input=text,
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                encoding="utf-8",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Piper executable not found. Install Piper and ensure `piper` is on PATH, "
                "or set tts.piper_binary in config.yaml."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"Piper synthesis failed: {stderr}") from exc

        if self.auto_play:
            self._play_audio(target)

        return str(target)
