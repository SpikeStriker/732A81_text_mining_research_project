from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from sentence_transformers import SentenceTransformer
import uvicorn
import numpy as np

app = FastAPI()

# Load model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading embedding model: {MODEL_NAME} on {device}...")
model = SentenceTransformer(MODEL_NAME, device=device)
model.max_seq_length = 512

class EmbeddingRequest(BaseModel):
    texts: List[str]
    normalize: bool = True
    batch_size: int = 32

@app.post("/embed")
async def embed(request: EmbeddingRequest):
    embeddings = model.encode(
        request.texts,
        normalize_embeddings=request.normalize,
        batch_size=request.batch_size,
        show_progress_bar=False
    )
    return {
        "embeddings": embeddings.tolist(),
        "dimension": embeddings.shape[1],
        "count": len(embeddings),
        "model": MODEL_NAME
    }

@app.post("/similarity")
async def similarity(texts: List[str]):
    """Calculate pairwise similarities"""
    embeddings = model.encode(texts, normalize_embeddings=True)
    similarity_matrix = np.dot(embeddings, embeddings.T)
    return {
        "similarities": similarity_matrix.tolist(),
        "texts": texts
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": device,
        "embedding_dim": model.get_sentence_embedding_dimension()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)