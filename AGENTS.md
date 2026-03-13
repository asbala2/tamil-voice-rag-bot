# \# AGENTS.md

# 

# \## Project purpose

# Build a local Tamil voice RAG bot:

# audio -> Whisper -> RAG -> Ollama -> XTTS

# 

# \## Environment

# \- Python 3.11

# \- Windows-first

# \- CPU-friendly defaults

# \- UTF-8 safe for Tamil text

# 

# \## Rules

# \- Keep modules separate

# \- Prefer simple scripts over heavy frameworks

# \- Add docstrings

# \- Do not break existing CLI commands

# \- Keep config in config.yaml where practical

# 

# \## Validation

# Before finishing a task:

# \- run the changed script at least once

# \- update README if commands changed

# \- keep imports minimal

# 

# \## Main commands

# \- python rag/ingest.py --config config.yaml

# \- python scripts/run\_demo.py --question "திருக்குறளில் தலைமை பற்றி என்ன சொல்லப்பட்டுள்ளது?"

