from __future__ import annotations

import os
from typing import Dict, Any

import requests
import yaml

from rag.retriever import format_context


class OllamaTamilQA:
    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = self.config["llm"]["model"]
        self.temperature = float(self.config["llm"]["temperature"])
        self.num_predict = int(self.config["llm"]["num_predict"])
        self.system_prompt = self.config["llm"]["system_prompt"]

    def build_prompt(self, question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
        context = format_context(retrieved_chunks)
        return f"""{self.system_prompt}

பயனர் கேள்வி:
{question}

கிடைத்த தகவல்:
{context if context else 'பொருத்தமான தகவல் எதுவும் இல்லை.'}

வழிமுறைகள்:
- பதிலை தமிழில் எழுதவும்.
- "கிடைத்த தகவல்" பகுதியை முதன்மை ஆதாரமாக பயன்படுத்தவும்; ஆனால் அது முழுமையற்றால் பொதுஅறிவை துணையாக பயன்படுத்தலாம்.
- பதிலில் தெளிவான இரண்டு பகுதிகள் இருக்கட்டும்:
  1) "ஆதார தகவல் (RAG)" - கிடைத்த தகவலிலிருந்து உறுதிப்படுத்தப்பட்ட பகுதி
  2) "பொது விளக்கம்" - மாதிரி பொது அறிவு/பொருள் புரிதலின் அடிப்படையிலான பகுதி
- "தகவல் இல்லை" போன்ற வாக்கியம், கிடைத்த தகவலும் பொது அறிவும் இரண்டும் போதாதபோது மட்டும் பயன்படுத்தவும்.
- முடிந்தால் ஆதாரப் பகுதிகளை சுருக்கமாக குறிப்பிடவும்.
"""

    def answer(self, question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
        prompt = self.build_prompt(question, retrieved_chunks)
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "").strip()
