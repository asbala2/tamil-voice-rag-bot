from __future__ import annotations

import json
from pathlib import Path
import wave

import yaml


PHRASE = "வணக்கம்"
DEFAULT_CONFIG = "config.yaml"
OUTPUT_WAV = Path("data/output/piper_test.wav")


def _resolve_sample_rate(config_path: Path, voice: object) -> int:
    """Resolve Piper sample rate from voice metadata first, then model config JSON."""

    def _attr_path(obj: object, path: str) -> object | None:
        current: object | None = obj
        for part in path.split("."):
            if current is None:
                return None
            current = getattr(current, part, None)
        return current

    for attr_name in ("config.sample_rate", "config.audio.sample_rate", "sample_rate", "audio_sample_rate"):
        value = _attr_path(voice, attr_name)
        if isinstance(value, int) and value > 0:
            return value

    try:
        with config_path.open("r", encoding="utf-8") as cfg_file:
            model_cfg = json.load(cfg_file)
        if isinstance(model_cfg, dict):
            audio_cfg = model_cfg.get("audio", {})
            for value in (audio_cfg.get("sample_rate"), model_cfg.get("sample_rate")):
                if isinstance(value, int) and value > 0:
                    return value
    except Exception:  # noqa: BLE001
        pass

    raise RuntimeError(f"Unable to determine Piper sample rate from config: {config_path!s}")


def main(config_file: str = DEFAULT_CONFIG) -> None:
    """Load Piper model directly, synthesize a Tamil sample, and print diagnostics."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tts_cfg = config["tts"]
    model_path = Path(tts_cfg["model_path"])
    config_path = Path(tts_cfg["config_path"])

    print(f"Model path: {model_path}")
    print(f"Config path: {config_path}")

    if not model_path.exists() or not config_path.exists():
        missing = [str(p) for p in (model_path, config_path) if not p.exists()]
        raise FileNotFoundError(f"Missing Piper file(s): {', '.join(missing)}")

    from piper.voice import PiperVoice

    loaded_ok = False
    voice = None
    try:
        voice = PiperVoice.load(str(model_path), config_path=str(config_path), use_cuda=False)
        loaded_ok = True
    finally:
        print(f"Model loaded successfully: {loaded_ok}")

    if voice is None:
        raise RuntimeError("Failed to initialize Piper voice object")

    sample_rate = _resolve_sample_rate(config_path=config_path, voice=voice)

    chunks: list[bytes] = []
    if hasattr(voice, "synthesize_stream_raw"):
        for raw_chunk in voice.synthesize_stream_raw(PHRASE):
            if raw_chunk is None:
                continue
            if isinstance(raw_chunk, (bytes, bytearray, memoryview)):
                chunk_bytes = bytes(raw_chunk)
            elif hasattr(raw_chunk, "tobytes"):
                chunk_bytes = raw_chunk.tobytes()
            else:
                chunk_bytes = bytes(raw_chunk)

            if chunk_bytes:
                chunks.append(chunk_bytes)
    else:
        raise RuntimeError("Installed Piper version does not support synthesize_stream_raw")

    chunk_count = len(chunks)
    total_audio_bytes = sum(len(chunk) for chunk in chunks)

    print(f"Sample rate: {sample_rate}")
    print(f"Number of chunks returned: {chunk_count}")
    print(f"Total audio bytes returned: {total_audio_bytes}")

    if chunk_count == 0:
        raise RuntimeError("Piper produced no audio")

    OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUTPUT_WAV), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for chunk in chunks:
            wav_file.writeframesraw(chunk)

    print(f"Saved diagnostic WAV: {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
