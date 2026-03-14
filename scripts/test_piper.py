from __future__ import annotations

import argparse
from pathlib import Path
import string
import wave

import yaml


PHRASE = "வணக்கம்"
DEFAULT_CONFIG = "config.yaml"
OUTPUT_WAV = Path("data/output/piper_test.wav")


def _sanitize_text(text: str) -> str:
    """Keep Tamil/script-safe characters and remove invalid Unicode before synthesis."""
    allowed_punctuation = set(string.punctuation) | {"…", "“", "”", "‘", "’", "-", "–", "—"}

    def is_tamil_char(ch: str) -> bool:
        codepoint = ord(ch)
        return 0x0B80 <= codepoint <= 0x0BFF

    sanitized_chars: list[str] = []
    for char in text:
        codepoint = ord(char)
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        if is_tamil_char(char) or char.isdigit() or char.isspace() or char in allowed_punctuation:
            sanitized_chars.append(char)

    normalized = " ".join("".join(sanitized_chars).split())
    return normalized or "பதில் கிடைக்கவில்லை."


class TrackingWaveWriter:
    """Track WAV writer usage so synthesis failures can be diagnosed precisely."""

    def __init__(self, wav_file: wave.Wave_write) -> None:
        self._wav_file = wav_file
        self.params_initialized = False
        self.write_calls = 0
        self.written_bytes = 0

    def __getattr__(self, name: str):
        return getattr(self._wav_file, name)

    def setparams(self, params) -> None:
        self.params_initialized = True
        self._wav_file.setparams(params)

    def setframerate(self, framerate: int) -> None:
        self.params_initialized = True
        self._wav_file.setframerate(framerate)

    def setnchannels(self, nchannels: int) -> None:
        self.params_initialized = True
        self._wav_file.setnchannels(nchannels)

    def setsampwidth(self, sampwidth: int) -> None:
        self.params_initialized = True
        self._wav_file.setsampwidth(sampwidth)

    def writeframes(self, data: bytes) -> None:
        self.write_calls += 1
        self.written_bytes += len(data)
        self._wav_file.writeframes(data)

    def writeframesraw(self, data: bytes) -> None:
        self.write_calls += 1
        self.written_bytes += len(data)
        self._wav_file.writeframesraw(data)


def main(config_file: str = DEFAULT_CONFIG, model_override: str | None = None) -> None:
    """Load Piper model directly, synthesize a Tamil sample, and print diagnostics."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tts_cfg = config["tts"]
    model_path = Path(model_override) if model_override else Path(tts_cfg["model_path"])
    config_path = Path(tts_cfg["config_path"])
    output_wav = Path(tts_cfg.get("output_file", str(OUTPUT_WAV)))

    print(f"Model path: {model_path}")
    print(f"Config path: {config_path}")
    print(f"Output path: {output_wav}")

    if not model_path.exists() or not config_path.exists():
        missing = [str(p) for p in (model_path, config_path) if not p.exists()]
        raise FileNotFoundError(f"Missing Piper file(s): {', '.join(missing)}")

    from piper.voice import PiperVoice

    voice = PiperVoice.load(str(model_path), config_path=str(config_path), use_cuda=False)
    print("Model loaded successfully: True")

    sample_rate = getattr(getattr(voice, "config", None), "sample_rate", None)
    speaker_id_map = getattr(getattr(voice, "config", None), "speaker_id_map", None)
    num_speakers = len(speaker_id_map) if isinstance(speaker_id_map, dict) else None
    print(f"Config sample rate: {sample_rate}")
    print(f"Speaker info: num_speakers={num_speakers}, speaker_id_map={speaker_id_map}")

    safe_text = _sanitize_text(PHRASE)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    synthesize_entered = False
    synthesize_error: Exception | None = None
    close_error: Exception | None = None
    writer_state: TrackingWaveWriter | None = None

    wav_file = wave.open(str(output_wav), "w")
    writer_state = TrackingWaveWriter(wav_file)
    try:
        synthesize_entered = True
        voice.synthesize(safe_text, writer_state)
    except Exception as exc:  # noqa: BLE001
        synthesize_error = exc
    finally:
        try:
            wav_file.close()
        except Exception as exc:  # noqa: BLE001
            close_error = exc

    print(f"synthesize() entered: {synthesize_entered}")
    if writer_state is not None:
        print(f"WAV header initialized: {writer_state.params_initialized}")
        print(f"Audio chunks/frames written: {writer_state.write_calls}")
        print(f"Audio bytes written: {writer_state.written_bytes}")

    if synthesize_error is not None:
        if output_wav.exists():
            output_wav.unlink(missing_ok=True)
        close_note = f" Close error: {close_error}" if close_error is not None else ""
        raise RuntimeError(
            "Piper synthesis failed before producing a valid WAV stream in diagnostic test. "
            f"Output path: {output_wav!s}. "
            f"Sanitized text: {safe_text!r}. "
            f"Original synthesis error: {synthesize_error}."
            f"{close_note}"
        ) from synthesize_error

    if close_error is not None:
        if output_wav.exists():
            output_wav.unlink(missing_ok=True)
        raise RuntimeError(
            "Piper synthesis completed but WAV finalization failed in diagnostic test. "
            f"Output path: {output_wav!s}. Close error: {close_error}"
        ) from close_error

    if not output_wav.exists():
        raise RuntimeError(f"Piper did not produce an output WAV file: {output_wav!s}")

    file_size = output_wav.stat().st_size
    try:
        with wave.open(str(output_wav), "r") as wav_file:
            frame_count = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
    except Exception as exc:  # noqa: BLE001
        output_wav.unlink(missing_ok=True)
        raise RuntimeError(
            "Piper produced an invalid WAV file in diagnostic test. "
            f"Output path: {output_wav!s}. Original error: {exc}"
        ) from exc

    if frame_count <= 0 or file_size <= 44:
        output_wav.unlink(missing_ok=True)
        raise RuntimeError(
            "Piper loaded successfully but this model/runtime combination produced no audio. "
            f"Output path: {output_wav!s}. Frames: {frame_count}, bytes: {file_size}"
        )

    print(f"Frames: {frame_count}")
    print(f"Sample rate: {sample_rate}")
    print(f"Channels: {channels}")
    print(f"Output bytes: {file_size}")
    print(f"Saved diagnostic WAV: {output_wav}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Piper synthesis diagnostic")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override Piper model path without editing config.yaml",
    )
    args = parser.parse_args()
    main(config_file=args.config, model_override=args.model_path)
