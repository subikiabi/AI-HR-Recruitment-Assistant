"""In-memory Vector Store for semantic retrieval of resume and job description chunks."""

import math
from typing import List, Dict, Any, Optional
from src.rag.chunker import DocumentChunk
from src.rag.embeddings import HybridEmbeddingEngine


class InMemoVectorStore:
    """Fast, dependency-free in-memory vector database for RAG."""

    def __init__(self, embedding_engine: Optional[HybridEmbeddingEngine] = None):
        self.embedding_engine = embedding_engine or HybridEmbeddingEngine()
        self.chunks: List[DocumentChunk] = []

    def clear(self):
        """Clears all stored chunks."""
        self.chunks.clear()

    def add_documents(self, chunks: List[DocumentChunk]):
        """Adds and embeds chunks."""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        # Fit corpus for TFIDF fallback
        all_texts = [c.text for c in self.chunks] + texts
        self.embedding_engine.fit_corpus(all_texts)

        # Re-embed all chunks to align vector space dimensions
        all_chunks = self.chunks + chunks
        all_embeddings = self.embedding_engine.embed_texts([c.text for c in all_chunks])

        for c, emb in zip(all_chunks, all_embeddings):
            c.embedding = emb

        self.chunks = all_chunks

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculates cosine similarity between two numeric lists."""
        if not vec_a or not vec_b:
            return 0.0
        
        min_len = min(len(vec_a), len(vec_b))
        if min_len == 0:
            return 0.0

        dot_product = sum(vec_a[i] * vec_b[i] for i in range(min_len))
        norm_a = math.sqrt(sum(x * x for x in vec_a[:min_len]))
        norm_b = math.sqrt(sum(y * y for y in vec_b[:min_len]))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return max(0.0, min(1.0, dot_product / (norm_a * norm_b)))

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Performs semantic search against stored chunks with optional metadata filtering."""
        if not self.chunks or not query.strip():
            return []

        query_vec = self.embedding_engine.embed_query(query)
        scored_results = []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        for chunk in self.chunks:
            # Check metadata filters
            if filter_dict:
                match = True
                for k, v in filter_dict.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Semantic score
            sem_score = 0.0
            if chunk.embedding and query_vec:
                sem_score = self._cosine_similarity(query_vec, chunk.embedding)

            # Keyword lexical overlap boost
            chunk_words = set(chunk.text.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            lex_score = overlap / max(1, len(query_words))

            # Hybrid score (70% semantic, 30% lexical)
            final_score = (0.7 * sem_score) + (0.3 * lex_score)

            scored_results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": round(final_score, 4)
            })

        # Sort descending by score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]
