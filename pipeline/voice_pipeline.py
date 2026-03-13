from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

from llm.ollama_client import OllamaTamilQA
from rag.retriever import TamilRetriever
from speech.whisper_transcribe import TamilWhisperTranscriber
from tts.xtts_speak import XTTSSpeaker


@dataclass
class PipelineResult:
    question_text: str
    retrieved_chunks: list[dict[str, Any]]
    answer_text: str
    audio_reply_path: str | None


class TamilVoiceRAGPipeline:
    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.retriever = TamilRetriever(config_path)
        self.qa = OllamaTamilQA(config_path)
        self.tts = XTTSSpeaker(config_path)
        self.transcriber = TamilWhisperTranscriber(config_path)

    def run_with_text(self, question_text: str, make_voice_reply: bool = False) -> PipelineResult:
        chunks = self.retriever.retrieve(question_text)
        answer = self.qa.answer(question_text, chunks)
        audio_path = None
        if make_voice_reply:
            audio_path = self.tts.synthesize(answer)
        return PipelineResult(
            question_text=question_text,
            retrieved_chunks=chunks,
            answer_text=answer,
            audio_reply_path=audio_path,
        )

    def run_with_audio_file(self, audio_path: str, make_voice_reply: bool = False) -> PipelineResult:
        transcribed = self.transcriber.transcribe_file(audio_path)
        return self.run_with_text(transcribed["text"], make_voice_reply=make_voice_reply)


def save_result_json(result: PipelineResult, output_file: str) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(result), allow_unicode=True, sort_keys=False), encoding="utf-8")
