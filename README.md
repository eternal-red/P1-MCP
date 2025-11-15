# P1-MCP

A minimal MCP (Model Context Protocol) that has RAG integrations and can connect with the other P1 microservices

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the main program (example):

```bash
python main.py
```

3. Run local tests/examples:

```bash
python -m local_testing.test_client
```

## Repository layout

- `main.py` — project entry point / example runner
- `parser.py` — input parsing utilities
- `logical_functions/` — helpers used for calculations and RAG logic (`cal_functions.py`, `rag_functions.py`)
- `default_KB_data/` — sample knowledge base (embeddings and PDF chunks)
- `local_testing/` — small local test harnesses (`test_client.py`)
- `requirements.txt` — Python dependencies

## Contact / License

This is a small demo project. See repository owner for license and contribution details.
