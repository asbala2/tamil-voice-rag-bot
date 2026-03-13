# Tamil Voice RAG Bot

A local-first voice assistant pipeline for Tamil question answering:

Audio / Mic -> Whisper -> Question Text -> RAG over Tamil documents -> Ollama answer

## What this starter repo includes

- `speech/whisper_transcribe.py` - transcribe Tamil audio with Faster-Whisper
- `rag/ingest.py` - chunk Tamil text documents and build a Chroma vector store
- `rag/retriever.py` - retrieve top-k context chunks for a question
- `llm/ollama_client.py` - send prompt + retrieved context to Ollama
- `scripts/run_demo.py` - simple CLI entry point

## Suggested first MVP

1. Put a few Tamil `.txt` files into `data/literature/`
2. Build the vector DB
3. Run the pipeline with an audio file
4. Get a Tamil text answer

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml
├── data/
│   └── literature/
├── rag/
│   ├── ingest.py
│   └── retriever.py
├── speech/
│   └── whisper_transcribe.py
├── llm/
│   └── ollama_client.py
├── tts/
│   └── xtts_speak.py
├── pipeline/
│   └── voice_pipeline.py
├── scripts/
│   └── run_demo.py
└── tests/
```

## Quick start

### 1) Create environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Make sure Ollama is running

Example:

```bash
ollama serve
ollama pull gemma3:4b
```

### 4) Add Tamil source text

Place one or more UTF-8 Tamil `.txt` files in:

```text
data/literature/
```

A sample file is already included.

### 5) Build the vector DB

```bash
python rag/ingest.py --config config.yaml
```

### 6) Run end-to-end with an audio file

This command transcribes audio with Whisper, retrieves relevant Tamil chunks from Chroma, and asks Ollama for a final Tamil answer.

```bash
python scripts/run_demo.py --config config.yaml --audio path/to/question.wav
```

### 6a) Transcribe an audio file only

```bash
python speech/whisper_transcribe.py --input sample.wav
```

### 7) Run retrieval + answering with text input

```bash
python scripts/run_demo.py --config config.yaml --question "திருக்குறளில் தலைமை பற்றி என்ன சொல்லப்பட்டுள்ளது?"
```

Optional: override retrieval depth.

```bash
python scripts/run_demo.py --question "திருக்குறளில் தலைமை பற்றி என்ன சொல்லப்பட்டுள்ளது?" --top-k 5
```

## CLI commands

```bash
python rag/ingest.py --config config.yaml
python scripts/run_demo.py --question "திருக்குறளில் தலைமை பற்றி என்ன சொல்லப்பட்டுள்ளது?"
```

## Notes

- Start with text output first.
- Add XTTS once retrieval + answer quality is acceptable.
- On a CPU-only PC, keep models small initially.
- Use clean Tamil text files and clear audio for best results.

## Next recommended improvements

- Add microphone streaming input
- Add Tamil literary glossary for post-transcription correction
- Add source citations in answers
- Add simple web UI with Streamlit
- Add conversation memory
