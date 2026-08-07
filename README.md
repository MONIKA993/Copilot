# Agentic FIFA Assistant

This project runs a small agentic assistant that answers questions about a FIFA dataset.

## Features
- Web UI served from `index.html`.
- Agentic backend in `promt.py` with a small tool catalog for dataset queries.
- Optional LLM integration via the `OPENAI_API_KEY` environment variable for planning and natural-language final answers.

## Run locally
1. Open a terminal and change to the project folder:

```powershell
cd C:\Users\RSMDH-LPT-002\Downloads\copilot
```

2. (Optional) Export your OpenAI API key:

```powershell
$env:OPENAI_API_KEY = 'sk-...'
```

3. Start the agent server:

```powershell
python promt.py
```

4. Open http://localhost:8000 in your browser and ask questions, or POST to `/ask`:

```powershell
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"prompt":"Which country has the most wins?"}'
```

## Files
- `index.html` - Main webpage and simple chat UI
- `promt.py` - Python server and agent logic (tools, planner, LLM integration)
- `tests/` - Unit tests for the tool catalog

## Notes
- If `OPENAI_API_KEY` is not set the agent will use an internal fallback planner and answer directly from the dataset.
- The web UI shows available tools and preserves a short session memory in the browser's `localStorage`.
