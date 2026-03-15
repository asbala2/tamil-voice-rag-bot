from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import sounddevice as sd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_demo import run_text_qa
from speech.whisper_transcribe import TamilWhisperTranscriber


DEFAULT_SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the microphone voice assistant."""
    parser = argparse.ArgumentParser(
        description="Record Tamil speech from microphone, transcribe, and run RAG QA"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Recording duration per turn in seconds (default: 6.0)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    parser.add_argument(
        "--speak",
        action="store_true",
        default=True,
        help="Synthesize answer audio with Piper (enabled by default to match run_demo.py)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Microphone sample rate in Hz (default: 16000)",
    )
    return parser.parse_args()


def record_microphone_audio(duration_seconds: float, sample_rate: int):
    """Record mono microphone audio and return samples with the sample rate."""
    frames = max(1, int(duration_seconds * sample_rate))
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio, sample_rate


def transcribe_recording(transcriber: TamilWhisperTranscriber, audio, sample_rate: int) -> str:
    """Save in-memory recording to a temporary WAV file and transcribe it."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)

    try:
        sf.write(temp_path, audio, sample_rate)
        result = transcriber.transcribe_file(str(temp_path))
        return result["text"].strip()
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    """Run a loop that records speech, transcribes it, and answers until Ctrl+C."""
    args = parse_args()

    if args.sample_rate != DEFAULT_SAMPLE_RATE:
        print(
            f"Warning: requested sample rate {args.sample_rate} Hz; default recommended is {DEFAULT_SAMPLE_RATE} Hz."
        )

    transcriber = TamilWhisperTranscriber(config_path=args.config)

    print("Tamil voice assistant started.")
    print("Press Enter to record a question, then wait for recording to finish.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            input("Press Enter to start recording...")
            print(f"Recording for {args.duration:.1f} seconds at {args.sample_rate} Hz...")
            audio, sample_rate = record_microphone_audio(
                duration_seconds=args.duration,
                sample_rate=args.sample_rate,
            )

            question_text = transcribe_recording(transcriber=transcriber, audio=audio, sample_rate=sample_rate)
            print("\n=== Recognized Text ===")
            print(question_text if question_text else "(No speech recognized)")

            if not question_text:
                print("Skipping answer generation because no text was recognized.\n")
                continue

            run_text_qa(
                config_path=args.config,
                question=question_text,
                top_k=args.top_k,
                speak=args.speak,
            )
            print("\n--- Ready for next turn ---\n")
    except KeyboardInterrupt:
        print("\nExiting voice assistant. Goodbye!")


if __name__ == "__main__":
    main()
