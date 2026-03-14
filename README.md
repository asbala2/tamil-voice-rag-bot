# Tamil Voice RAG Bot

A local-first voice assistant pipeline for Tamil question answering:

Audio / Mic -> Whisper -> Question Text -> RAG over Tamil documents -> Ollama answer -> Optional Piper TTS Tamil voice reply

## What this starter repo includes

- `speech/whisper_transcribe.py` - transcribe Tamil audio with Faster-Whisper
- `rag/ingest.py` - chunk Tamil text documents and build a Chroma vector store
- `rag/retriever.py` - retrieve top-k context chunks for a question
- `llm/ollama_client.py` - send prompt + retrieved context to Ollama
- `scripts/run_demo.py` - simple CLI entry point with optional speech reply (`--speak`)
- `tts/piper_speak.py` - Piper-based Tamil speech synthesis

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
│   ├── piper_speak.py
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

Optional: add `--speak` to synthesize the final Tamil answer to `data/output/output_answer.wav` and attempt automatic playback.

```bash
python scripts/run_demo.py --config config.yaml --audio path/to/question.wav --speak
```

### 6a) Transcribe an audio file only

```bash
python speech/whisper_transcribe.py --input sample.wav
```

### 7) Run retrieval + answering with text input

```bash
python scripts/run_demo.py --config config.yaml --question "திருக்குறளில் தலைமை பற்றி என்ன சொல்லப்பட்டுள்ளது?"
```

Optional voice reply:

```bash
python scripts/run_demo.py --config config.yaml --question "திருக்குறளில் தலைமை பற்றி என்ன சொல்லப்படுகிறது?" --speak
```

Optional: override retrieval depth.

```bash
python scripts/run_demo.py --question "திருக்குறளில் தலைமை பற்றி என்ன சொல்லப்பட்டுள்ளது?" --top-k 5
```

## CLI commands

```bash
python rag/ingest.py --config config.yaml
python scripts/run_demo.py --question "திருக்குறளில் தலைமை பற்றி என்ன சொல்லப்பட்டுள்ளது?"
python scripts/run_demo.py --config config.yaml --question "திருக்குறளில் தலைமை பற்றி என்ன சொல்லப்படுகிறது?" --speak
python scripts/run_demo.py --config config.yaml --audio path/to/question.wav --speak
```


## Piper Tamil voice setup

By default, speech output expects these files:

- `data/models/piper/ta_IN-kani-medium.onnx`
- `data/models/piper/ta_IN-kani-medium.onnx.json`

If they are not present, download a Tamil Piper voice model and its matching `.json` config from the Piper voices release page:

- https://github.com/rhasspy/piper/blob/master/VOICES.md

Then place both files under `data/models/piper/` (or any path you prefer) and update `config.yaml`:

```yaml
tts:
  enabled: true
  piper_binary: piper
  model_path: data/models/piper/ta_IN-kani-medium.onnx
  config_path: data/models/piper/ta_IN-kani-medium.onnx.json
  output_file: data/output/output_answer.wav
  auto_play: true
```

Windows note: if `piper` is not on PATH, set `tts.piper_binary` to the full path of `piper.exe`.

Troubleshooting (Windows Unicode): if Piper reports `surrogates not allowed`, the app now sanitizes reply text before synthesis (removes invalid surrogate code points and unsupported symbols) and sends UTF-8 bytes directly to Piper stdin.

## Notes

- Text output remains the default behavior; speech reply is optional via `--speak`.
- Piper TTS is used when `--speak` is passed and `tts.enabled: true`.
- On a CPU-only PC, choose a `medium` or `low` Piper voice for faster synthesis.
- Use clean Tamil text files and clear audio for best results.

## Next recommended improvements

- Add microphone streaming input
- Add Tamil literary glossary for post-transcription correction
- Add source citations in answers
- Add simple web UI with Streamlit
- Add conversation memory
