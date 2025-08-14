from pathlib import Path
import json
import numpy as np
from typing import Any, List, Dict, Tuple
from sentence_transformers import SentenceTransformer

CHUNK_FILE = Path("default_KB_data/pdf_chunks.json")
EMB_FILE   = Path("default_KB_data/pdf_embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"

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

_MODEL: SentenceTransformer = None

def start_encoder():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
        _MODEL.encode(["warmup"], convert_to_numpy=True, normalize_embeddings=True)

def model() -> SentenceTransformer:
    if _MODEL is None:
        raise RuntimeError("Encoder not started. Call start_encoder() first.")
    return _MODEL

def _search(query: str, k: int = 4) -> List[Tuple[int, float]]:
    q = model().encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    if q.dtype != np.float32:
        q = q.astype(np.float32, copy=False)
    scores = EMB @ q
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

def get_chunk_logic(id: int) -> Dict[str, Any]:
    if id < 0 or id >= len(CHUNKS):
        return {"error": f"chunk id {id} out of range (0..{len(CHUNKS)-1})"}
    text = CHUNKS[id]
    return {"id": id, "text": text, "length": len(text)}

def build_prompt_logic(question: str, chunk_ids: List[int]) -> str:
    parts = []
    for cid in chunk_ids:
        if 0 <= cid < len(CHUNKS):
            parts.append(CHUNKS[cid])
    context = "\n\n---\n\n".join(parts)
    prompt = f"""Use ONLY the context to answer. If unknown, say \"I don't know\".\nContext:\n{context}\nQuestion: {question}\nAnswer:"""
    return prompt
