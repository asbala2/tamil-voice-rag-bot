# Tamil Voice RAG Bot

A local-first voice assistant pipeline for Tamil question answering:

Audio / Mic -> Whisper -> Question Text -> RAG over Tamil documents -> Ollama answer -> Optional Piper TTS Tamil voice reply

## What this starter repo includes

- `speech/whisper_transcribe.py` - transcribe Tamil audio with Faster-Whisper
- `rag/ingest.py` - chunk Tamil text documents and build a Chroma vector store
- `rag/retriever.py` - retrieve top-k context chunks for a question
- `llm/ollama_client.py` - send prompt + retrieved context to Ollama
- `scripts/run_demo.py` - simple CLI entry point with optional speech reply (`--speak`)
- `tts/piper_speak.py` - Piper Python API-based Tamil speech synthesis

## Suggested first MVP

1. Put Tamil source files (`.txt`, `.pdf`, `.docx`) into `data/literature/`
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
│   ├── run_demo.py
│   └── test_piper.py
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

### 4) Add Tamil source documents

Place one or more Tamil source files in:

```text
data/literature/
```

Supported file types:

- `.txt` (UTF-8 plain text)
- `.pdf` (text-searchable PDFs)
- `.docx` (Microsoft Word documents)

Note: scanned/image-only PDFs are not supported yet.

A sample text file is already included.

### 5) Build the vector DB

```bash
python rag/ingest.py --config config.yaml
```

Optional: clean rebuild of the vector DB directory before ingestion.

```bash
python rag/ingest.py --config config.yaml --rebuild-store
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

### 8) Run live microphone voice assistant

This loop records from your default microphone at **16 kHz**, transcribes with Faster-Whisper, prints the recognized Tamil text, and then runs the same RAG + Ollama pipeline used by text QA.

```bash
python scripts/run_voice_assistant.py --config config.yaml
```

Optional flags:

```bash
python scripts/run_voice_assistant.py --duration 8 --top-k 5 --speak
```

Press `Ctrl+C` at any time to stop the assistant loop.

### Tamil transcription quality tuning

For better Tamil recognition accuracy, tune the `speech` section in `config.yaml`:

```yaml
speech:
  whisper_model_size: medium   # options: base, small, medium
  language: auto               # auto for Tamil+English mixed speech, ta for Tamil-only
  beam_size: 8
  best_of: 5
  temperature: 0.0
```

`small` is CPU-friendly, while `medium` usually improves accuracy on clear speech.
Use `language: auto` to preserve English words in Latin script for mixed Tamil-English questions.

## CLI commands

```bash
python rag/ingest.py --config config.yaml
python rag/ingest.py --config config.yaml --rebuild-store
python scripts/run_demo.py --question "திருக்குறளில் தலைமை பற்றி என்ன சொல்லப்பட்டுள்ளது?"
python scripts/run_demo.py --config config.yaml --question "திருக்குறளில் தலைமை பற்றி என்ன சொல்லப்படுகிறது?" --speak
python scripts/run_demo.py --config config.yaml --audio path/to/question.wav --speak
python scripts/test_piper.py
python scripts/run_voice_assistant.py --config config.yaml
```

## Piper diagnostic script

Use this standalone check to validate Piper model loading and Tamil synthesis runtime:

```bash
python scripts/test_piper.py
```

Optional: override model and test phrase without editing `config.yaml`:

```bash
python scripts/test_piper.py --model-path data/models/piper/en_US-lessac-medium.onnx --text "Hello world"
```

By default, the diagnostic uses `"Hello world"` for `en_*` models and `"வணக்கம்"` for `ta_*` models. It prints model/runtime diagnostics (sample rate, speaker info, synthesize entry, WAV header/chunk activity) and saves audio to `data/output/piper_test.wav`.


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
  model_path: data/models/piper/ta_IN-kani-medium.onnx
  config_path: data/models/piper/ta_IN-kani-medium.onnx.json
  output_file: data/output/output_answer.wav
  auto_play: true
```

Troubleshooting (Windows Unicode): to avoid stdin/console encoding issues like `surrogates not allowed`, synthesis now uses Piper's Python API directly (no subprocess CLI), while still sanitizing reply text before synthesis.

Troubleshooting (Piper runtime/WAV errors): if `wave.Error: # channels not specified` appears, run `python scripts/test_piper.py` to capture the underlying synthesis exception before WAV close masks it. The diagnostic now uses the installed Piper API (`voice.synthesize(text)` generator), initializes WAV header fields from emitted chunks, and reports precise failure context (model path, config path, test text, sample rate, chunk count, byte count). If synthesis emits zero chunks/bytes, it raises a dedicated zero-audio runtime error.

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
