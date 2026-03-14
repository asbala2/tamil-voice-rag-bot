from __future__ import annotations

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


def main(config_file: str = DEFAULT_CONFIG) -> None:
    """Load Piper model directly, synthesize a Tamil sample, and print diagnostics."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tts_cfg = config["tts"]
    model_path = Path(tts_cfg["model_path"])
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

    safe_text = _sanitize_text(PHRASE)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    try:
        with wave.open(str(output_wav), "w") as wav_file:
            voice.synthesize(safe_text, wav_file)
    except Exception as exc:  # noqa: BLE001
        if output_wav.exists():
            output_wav.unlink(missing_ok=True)
        raise RuntimeError(
            "Piper synthesis failed in diagnostic test. "
            f"Output path: {output_wav!s}. "
            f"Sanitized text: {safe_text!r}. "
            f"Original error: {exc}"
        ) from exc

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
            "Piper produced an empty WAV file in diagnostic test. "
            f"Output path: {output_wav!s}. Frames: {frame_count}, bytes: {file_size}"
        )

    print(f"Frames: {frame_count}")
    print(f"Sample rate: {sample_rate}")
    print(f"Channels: {channels}")
    print(f"Output bytes: {file_size}")
    print(f"Saved diagnostic WAV: {output_wav}")


if __name__ == "__main__":
    main()
