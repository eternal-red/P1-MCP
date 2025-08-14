# mcp_rag_server.py
# pip install fastmcp uvicorn sentence-transformers numpy
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

# -----------------------------------------------------------------------------
# Globals initialized at import time (so they load before uvicorn starts serving)
# -----------------------------------------------------------------------------

CHUNK_FILE = Path("pdf_chunks.json")
EMB_FILE   = Path("pdf_embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"   # must match what you used to embed

# Load chunks
if not CHUNK_FILE.exists():
    raise RuntimeError("Missing pdf_chunks.json (run your chunk step first).")
CHUNKS: List[str] = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))

# Load & normalize embeddings once
if not EMB_FILE.exists():
    raise RuntimeError("Missing pdf_embeddings.npy (run your embedding step first).")
EMB: np.ndarray = np.load(EMB_FILE)
if EMB.dtype != np.float32:
    EMB = EMB.astype(np.float32, copy=False)
EMB = np.ascontiguousarray(EMB)

norms = np.linalg.norm(EMB, axis=1, keepdims=True)
norms[norms == 0.0] = 1.0
EMB /= norms

# Load the encoder ONCE and warm it up
_MODEL: SentenceTransformer = SentenceTransformer(MODEL_NAME)
# Warmup to avoid first-request latency spikes
_MODEL.encode(["warmup"], convert_to_numpy=True, normalize_embeddings=True)

def model() -> SentenceTransformer:
    # Returns the already-loaded global model (never re-loads per request)
    return _MODEL

# -----------------------------------------------------------------------------
# Search helpers
# -----------------------------------------------------------------------------

def _search(query: str, k: int = 4) -> List[Tuple[int, float]]:
    """Return top-k (index, score) using exact cosine with partial selection."""
    q = model().encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    if q.dtype != np.float32:
        q = q.astype(np.float32, copy=False)

    scores = EMB @ q  # cosine similarity (because both are L2-normalized)
    k = max(1, min(k, scores.shape[0]))

    idx_part = np.argpartition(-scores, k - 1)[:k]
    idx = idx_part[np.argsort(-scores[idx_part])]
    return [(int(i), float(scores[i])) for i in idx]

def _preview(text: str, max_chars: int = 240) -> str:
    t = " ".join(text.split())
    return t[:max_chars] + ("…" if len(t) > max_chars else "")

def search_rag(query: str, k: int = 4) -> List[Dict[str, Any]]:
    results = _search(query, k)
    return [{"id": i, "score": s, "preview": _preview(CHUNKS[i])} for i, s in results]

# -----------------------------------------------------------------------------
# MCP server & tools
# -----------------------------------------------------------------------------

mcp = FastMCP(name="rag")

@mcp.tool()
async def search_chunks(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """Vector search over cached chunks. Returns [{id, score, preview}, ...]."""
    return search_rag(query, k)

@mcp.tool()
async def get_chunk(id: int) -> Dict[str, Any]:
    """Return the full text of a chunk by its id."""
    if id < 0 or id >= len(CHUNKS):
        return {"error": f"chunk id {id} out of range (0..{len(CHUNKS)-1})"}
    text = CHUNKS[id]
    return {"id": id, "text": text, "length": len(text)}

@mcp.tool()
async def build_prompt(question: str, chunk_ids: List[int]) -> str:
    """Build a minimal RAG prompt using the selected chunk ids."""
    parts = []
    for cid in chunk_ids:
        if 0 <= cid < len(CHUNKS):
            parts.append(CHUNKS[cid])
    context = "\n\n---\n\n".join(parts)
    prompt = f"""Use ONLY the context to answer. If unknown, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
    return prompt

# -----------------------------------------------------------------------------
# Uvicorn entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = 8081
    # Model + embeddings are already loaded above at import time and remain in RAM.
    uvicorn.run(mcp.streamable_http_app, host="0.0.0.0", port=port)
