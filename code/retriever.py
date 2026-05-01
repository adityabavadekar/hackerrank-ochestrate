"""Stage 2: Hybrid BM25 + vector retriever with company routing and similarity gating."""

import hashlib
import re
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from . import config

from .logger import get_logger
from .models import Company, Document, RetrievedDoc

logger = get_logger(__name__)

_EMBED_MODEL = "all-MiniLM-L6-v2"
_TOP_K = 5
_SIMILARITY_THRESHOLD = 0.30


class Retriever:
    """Hybrid BM25 + dense retriever with per-company namespaces.

    Index is built once at construction time and is read-only thereafter.
    """

    def __init__(self, documents: List[Document]) -> None:
        logger.info("Building retrieval index over %d documents", len(documents))
        self._model = SentenceTransformer(_EMBED_MODEL)
        self._namespaces: Dict[Company, "_Namespace"] = {}
        self._build_index(documents)
        logger.info("Retrieval index ready (%d companies)", len(self._namespaces))

    def _build_index(self, documents: List[Document]) -> None:
        by_company: Dict[Company, List[Document]] = {}
        for doc in documents:
            by_company.setdefault(doc.source, []).append(doc)

        for company, docs in by_company.items():
            logger.debug("Indexing %d docs for company=%s", len(docs), company.value)
            self._namespaces[company] = _Namespace(docs, self._model)

    def retrieve(
        self,
        query: str,
        company: Company,
        top_k: int = _TOP_K,
        threshold: float = _SIMILARITY_THRESHOLD,
    ) -> Tuple[List[RetrievedDoc], bool]:
        """Retrieves top-k docs for a query, optionally restricted to a single company.

        When company is Company.NONE, all namespaces are queried and the highest
        scoring namespace is used — this also infers the company from the corpus.

        Args:
            query: The sanitized ticket text.
            company: Known company or Company.NONE for auto-detection.
            top_k: Number of results to return.
            threshold: Minimum merged score; results below this escalate the ticket.

        Returns:
            Tuple of (retrieved_docs, threshold_met). When threshold_met is False
            the caller should escalate instead of proceeding to Stage 3.
        """
        query = self._clean_query(query)
        
        if company != Company.NONE and company in self._namespaces:
            docs = self._namespaces[company].search(query, top_k)
        else:
            docs = self._search_all(query, top_k)

        if not docs or docs[0].score < threshold:
            logger.warning(
                "Similarity threshold not met (best=%.3f, threshold=%.3f) for query: %r",
                docs[0].score if docs else 0.0,
                threshold,
                query[:80],
            )
            return docs, False

        logger.debug(
            "Retrieved %d docs (best=%.3f) for query: %r",
            len(docs),
            docs[0].score,
            query[:80],
        )
        return docs, True

    def _clean_query(self, query: str) -> str:
        """Normalizes query text by lowercasing, removing punctuation, and collapsing whitespace."""
        q = query.lower()
        q = re.sub(r'[^a-z0-9\s-]', ' ', q)
        return re.sub(r'\s+', ' ', q).strip()

    def _search_all(self, query: str, top_k: int) -> List[RetrievedDoc]:
        """Queries every namespace and returns the best-scoring top_k across all."""
        candidates: List[RetrievedDoc] = []
        for company, ns in self._namespaces.items():
            results = ns.search(query, top_k * 2)
            if results:
                logger.debug(
                    "Namespace %s top score=%.3f", company.value, results[0].score
                )
            candidates.extend(results)

        candidates.sort(key=lambda r: r.score, reverse=True)
        
        seen = set()
        deduped = []
        for c in candidates:
            if c.document.id not in seen:
                seen.add(c.document.id)
                deduped.append(c)
                
        return deduped[:top_k]


class _Namespace:
    """Per-company index combining BM25 and dense vector search."""

    def __init__(self, documents: List[Document], model: SentenceTransformer) -> None:
        self._documents = documents
        self._model = model
        tokenized = [doc.content.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        texts = [doc.content for doc in documents]
        
        # Fast embedding caching
        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode("utf-8"))
        cache_key = h.hexdigest()
        
        config.EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = config.EMBEDDINGS_CACHE_DIR / f"{cache_key}.npy"
        
        if cache_file.exists():
            self._vectors = np.load(cache_file)
        else:
            self._vectors = model.encode(texts, show_progress_bar=False, batch_size=64)
            np.save(cache_file, self._vectors)

    def search(self, query: str, top_k: int) -> List[RetrievedDoc]:
        """Returns top_k results with scores normalized and merged from BM25 + vector."""
        bm25_scores = self._bm25_scores(query)
        vector_scores = self._vector_scores(query)
        merged = bm25_scores + vector_scores

        indices = np.argsort(merged)[::-1][:top_k * 2]
        
        seen = set()
        results = []
        for idx in indices:
            doc = self._documents[idx]
            if doc.id not in seen:
                seen.add(doc.id)
                results.append(RetrievedDoc(document=doc, score=float(merged[idx])))
                if len(results) == top_k:
                    break
        return results

    def _bm25_scores(self, query: str) -> np.ndarray:
        raw = np.array(self._bm25.get_scores(query.lower().split()), dtype=float)
        return _normalize(raw)

    def _vector_scores(self, query: str) -> np.ndarray:
        q_vec = self._model.encode([query], show_progress_bar=False)
        sims = (self._vectors @ q_vec.T).flatten()
        sims = np.clip(sims, 0, 1)
        return _normalize(sims)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalizes to [0, 1]; returns zeros if range is zero."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
