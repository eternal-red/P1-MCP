'''
This file is used to prepare a singular PDF document for RAG
The document should be less than 10K words
'''

import json
import numpy as np
from pathlib import Path
import pdfplumber
from sentence_transformers import SentenceTransformer

CHUNK_FILE = Path("pdf_chunks.json")
EMB_FILE = Path("pdf_embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_WORDS = 500
OVERLAP_WORDS = 100

def read_pdf_text(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n\n".join([page.extract_text() or "" for page in pdf.pages])

def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + CHUNK_WORDS]))
    return chunks

def build_or_load_index(pdf_path: str):
    # 1) If cache exists, load it
    if CHUNK_FILE.exists() and EMB_FILE.exists():
        chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
        emb = np.load(EMB_FILE)
        model = SentenceTransformer(MODEL_NAME)
        return chunks, emb, model

    # 2) Otherwise, process PDF and save cache
    text = read_pdf_text(pdf_path)
    chunks = chunk_text(text)
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    # Save
    CHUNK_FILE.write_text(json.dumps(chunks), encoding="utf-8")
    np.save(EMB_FILE, emb)

    return chunks, emb, model

# Example usage
chunks, emb, model = build_or_load_index("data.pdf")
print(f"Loaded {len(chunks)} chunks")
