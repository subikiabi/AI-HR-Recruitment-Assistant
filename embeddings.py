"""Embedding engine supporting both Gemini API and built-in TF-IDF hybrid fallback."""

import math
import re
from typing import List, Dict, Optional
import requests

from src.config import GEMINI_API_KEY, DEFAULT_EMBEDDING_MODEL


class TFIDFEmbeddingEngine:
    """Pure-Python high-efficiency TF-IDF vectorizer and semantic similarity engine.
    Requires no external dependencies, operates offline, and supports fast cosine similarity.
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric tokens and bigrams."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\+\#\.]+\b", text.lower())
        tokens = [t for t in tokens if len(t) > 1]
        
        # Add bigrams for technical phrases like 'fast api', 'rest api', 'machine learning'
        bigrams = []
        for i in range(len(tokens) - 1):
            bigrams.append(f"{tokens[i]}_{tokens[i+1]}")
        return tokens + bigrams

    def fit(self, documents: List[str]):
        """Fit vocabulary and compute inverse document frequency (IDF)."""
        self.doc_count = len(documents)
        if self.doc_count == 0:
            return

        doc_term_freq: Dict[str, int] = {}
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                doc_term_freq[t] = doc_term_freq.get(t, 0) + 1

        self.vocabulary = {term: idx for idx, term in enumerate(sorted(doc_term_freq.keys()))}
        self.idf = {
            term: math.log((1.0 + self.doc_count) / (1.0 + freq)) + 1.0
            for term, freq in doc_term_freq.items()
        }

    def transform(self, text: str) -> List[float]:
        """Transform a document into a normalized TF-IDF vector."""
        if not self.vocabulary:
            return []

        vec = [0.0] * len(self.vocabulary)
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        # Term frequency
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0

        length = len(tokens)
        norm_sq = 0.0
        for term, count in tf.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                val = (count / length) * self.idf.get(term, 1.0)
                vec[idx] = val
                norm_sq += val * val

        # L2 normalization
        norm = math.sqrt(norm_sq)
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec


class GeminiEmbeddingEngine:
    """Uses Google's text-embedding-004 model via Gemini REST API."""

    def __init__(self, api_key: str, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.api_key = api_key
        self.model_name = model_name

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Fetch embeddings for multiple texts using the Gemini API."""
        embeddings = []
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:batchEmbedContents?key={self.api_key}"
        
        requests_payload = [
            {"model": self.model_name, "content": {"parts": [{"text": text[:2048]}]}}
            for text in texts
        ]

        try:
            resp = requests.post(url, json={"requests": requests_payload}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("embeddings", []):
                    embeddings.append(item.get("values", []))
                return embeddings
        except Exception:
            pass

        return []

    def embed_query(self, query: str) -> List[float]:
        """Fetch embedding for a single query."""
        results = self.embed_texts([query])
        return results[0] if results else []


class HybridEmbeddingEngine:
    """Hybrid embedding coordinator that gracefully falls back between Gemini and TF-IDF."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.tfidf_engine = TFIDFEmbeddingEngine()
        self.gemini_engine = GeminiEmbeddingEngine(self.api_key) if self.api_key else None
        self.use_gemini = bool(self.gemini_engine)

    def fit_corpus(self, texts: List[str]):
        """Fits the local TF-IDF engine across all indexed texts."""
        self.tfidf_engine.fit(texts)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.use_gemini and self.gemini_engine:
            emb = self.gemini_engine.embed_texts(texts)
            if emb and len(emb) == len(texts):
                return emb
        
        # Fallback to TF-IDF
        return [self.tfidf_engine.transform(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        if self.use_gemini and self.gemini_engine:
            emb = self.gemini_engine.embed_query(query)
            if emb:
                return emb
        
        return self.tfidf_engine.transform(query)
