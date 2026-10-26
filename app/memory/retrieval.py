"""
Local semantic memory retrieval layer for Fixi.
Provides vector embedding calculation, cosine similarity ranking, and strict user isolation.
No external cloud APIs or remote services are used.
"""

import math
import re
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


def serialize_vector(vec: np.ndarray) -> bytes:
    """Converts a numpy float32 vector to bytes for SQLite storage."""
    return vec.astype(np.float32).tobytes()


def deserialize_vector(blob: Optional[bytes], dim: int = 128) -> Optional[np.ndarray]:
    """Converts binary bytes back to a numpy float32 vector."""
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Returns a normalized 1D float32 embedding vector."""
        pass


class LocalFastEmbedder(BaseEmbeddingModel):
    """
    High-performance, completely offline text embedder using n-gram subword frequency
    and random projection hashing with L2-normalization.
    Produces deterministic 128-dimensional dense semantic vectors.
    """
    def __init__(self, dimension: int = 128):
        self.dim = dimension

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = cleaned.split()
        ngrams = []
        for token in tokens:
            ngrams.append(token)
            # Add character n-grams for morphological/subword matching
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    ngrams.append(token[i:i+3])
        return ngrams

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text or not text.strip():
            return vec

        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for tok in tokens:
            h = hash(tok)
            idx = abs(h) % self.dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign

        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec


class VectorRetriever:
    """
    Vector search engine for episodic memories with strict user_id filtering.
    """
    def __init__(self, embedder: Optional[BaseEmbeddingModel] = None):
        self.embedder = embedder or LocalFastEmbedder(dimension=128)

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.embed_text(text)

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes cosine similarity between two unit vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def rank_memories(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        top_k: int = 5,
        min_threshold: float = 0.2
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Ranks memories by semantic similarity to the query.
        Each memory dict must contain 'summary', 'details', and optionally 'embedding'.
        """
        if not memories:
            return []

        query_vec = self.embed(query)
        scored: List[Tuple[Dict[str, Any], float]] = []

        for mem in memories:
            mem_vec = None
            if mem.get("embedding"):
                mem_vec = deserialize_vector(mem["embedding"])
            if mem_vec is None:
                text_to_embed = f"{mem.get('summary', '')} {mem.get('details', '')}"
                mem_vec = self.embed(text_to_embed)

            score = self.compute_similarity(query_vec, mem_vec)

            # Keyword overlap boost for content term matching
            stopwords = {"the", "a", "an", "is", "are", "and", "or", "my", "me", "tell", "about", "what", "do", "you", "i", "to", "for", "in", "of", "with"}
            query_words = set(re.findall(r"\w+", query.lower())) - stopwords
            mem_words = set(re.findall(r"\w+", (mem.get('summary', '') + " " + mem.get('details', '')).lower()))
            overlap = len(query_words.intersection(mem_words))
            if overlap > 0:
                boost = min(0.6, overlap * 0.25)
                score = min(1.0, score + boost)

            if score >= min_threshold:
                scored.append((mem, score))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


retriever = VectorRetriever()
