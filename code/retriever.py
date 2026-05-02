"""Stage 2: Hybrid BM25 + vector retriever with company routing and similarity gating."""

import hashlib
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config
from .cache import lru
from .logger import get_logger
from .models import Company, Document, RetrievedDoc

logger = get_logger(__name__)

_EMBED_MODEL = "all-MiniLM-L6-v2"
_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_TOP_K = 5
_SIMILARITY_THRESHOLD = 0.30
_RERANK_THRESHOLD = 0.45   # separate threshold for sigmoid-normalized reranker scores
_BM25_WEIGHT = 0.35
_VECTOR_WEIGHT = 0.65
_PRODUCT_AREA_BOOST = 0.10
_STOPWORDS = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "i", "my", "me", "we", "our", "you", "your",
    "it", "its", "this", "that", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "from", "by", "about", "into",
])

_QUERY_NORMALIZATIONS = [
    (r"\bcan not able to\b", "cannot"),
    (r"\bnot able to\b", "cannot"),
    (r"\bam not able to\b", "cannot"),
    (r"\bi can not\b", "i cannot"),
    (r"\bwhy so\b", "why is this"),
    (r"\bplz\b", "please"),
    (r"\bu\b", "you"),
    (r"\bpls\b", "please"),
    (r"\bdont\b", "do not"),
    (r"\bcant\b", "cannot"),
    (r"\bwont\b", "will not"),
]

_QUERY_EXPANSIONS = {
    "login": "login sign in authentication account access password",
    "signup": "signup register account creation onboarding",
    "payment": "payment billing charge subscription invoice transaction",
    "assessment": "assessment test coding challenge interview evaluation",
    "bug": "bug error issue glitch not working broken",
    "slow": "slow performance latency timeout loading lag",
    "download": "download export file attachment pdf",
    "dashboard": "dashboard analytics report metrics overview",
    "cash": "cash advance ATM withdraw emergency funds card",
    "urgent": "urgent emergency assistance immediate help",
    "emergency": "emergency urgent cash assistance card stolen lost replacement",
    "money": "money funds cash transfer payment card",
    "atm": "ATM cash withdrawal machine card visa",
    "stolen": "stolen lost card block replacement emergency",
    "lost": "lost stolen card block replacement emergency contact",
    "card": "card visa credit debit lost stolen block replacement",
    "traveller": "traveller cheque emergency lost stolen refund",
    "cheque": "traveller cheque emergency refund replacement stolen",
}


def _expand_query(query: str) -> str:
    """Expands queries with domain-specific synonyms to improve retrieval recall."""
    words = query.lower().split()
    extra = []
    for word in words:
        if word in _QUERY_EXPANSIONS:
            # Clean expansion terms the same way as queries
            extra.append(re.sub(r'[^a-z0-9\s-]', ' ', _QUERY_EXPANSIONS[word]))
    if extra:
        expanded = query + " " + " ".join(extra)
        expanded = re.sub(r'\s+', ' ', expanded).strip()
        logger.debug("Query expanded: %r -> %r", query, expanded[:120])
        return expanded
    return query


def _tokenize(text: str) -> List[str]:
    """Lowercases, strips punctuation, removes stopwords for BM25."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class Retriever:
    """Hybrid BM25 + dense retriever with per-company namespaces.

    Index is built once at construction time and is read-only thereafter.
    """

    def __init__(self, documents: List[Document]) -> None:
        logger.info("Building retrieval index over %d documents", len(documents))
        self._model = SentenceTransformer(_EMBED_MODEL)
        self._reranker = CrossEncoder(_RERANK_MODEL)
        self._namespaces: Dict[Company, "_Namespace"] = {}
        self._build_index(documents)
        logger.info("Retrieval index ready (%d companies)", len(self._namespaces))

    def init_semantic_gate(self) -> None:
        """Initializes gate semantic embeddings using the retriever's own model."""
        from . import gate as _gate
        _gate.init_semantic_gate(self._model)

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
        product_area: Optional[str] = None,
    ) -> Tuple[List[RetrievedDoc], bool]:
        """Retrieves top-k docs for a query, optionally restricted to a single company.

        When company is Company.NONE, all namespaces are queried and the highest
        scoring namespace is used — this also infers the company from the corpus.

        Args:
            query: The sanitized ticket text.
            company: Known company or Company.NONE for auto-detection.
            top_k: Number of results to return.
            threshold: Minimum merged score; results below this escalate the ticket.
            product_area: Optional product area for metadata boosting.

        Returns:
            Tuple of (retrieved_docs, threshold_met). When threshold_met is False
            the caller should escalate instead of proceeding to Stage 3.
        """
        query = self._clean_query(query)
        clean_query = query  # save pre-expansion key for embedding cache
        query = _expand_query(query)

        if company != Company.NONE and company in self._namespaces:
            docs = self._namespaces[company].search(query, clean_query, top_k * 2, product_area)
        else:
            docs = self._search_all(query, clean_query, top_k, product_area)

        # Rerank the candidates with cross-encoder; uses _RERANK_THRESHOLD not _SIMILARITY_THRESHOLD
        if docs:
            docs = self._rerank(query, docs, top_k)

        effective_threshold = _RERANK_THRESHOLD if len(docs) > 0 and docs[0].score > 0 else threshold
        if not docs or docs[0].score < effective_threshold:
            logger.warning(
                "Similarity threshold not met (best=%.3f, threshold=%.3f) for query: %r",
                docs[0].score if docs else 0.0,
                effective_threshold,
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
        for pattern, replacement in _QUERY_NORMALIZATIONS:
            q = re.sub(pattern, replacement, q)
        q = re.sub(r'[^a-z0-9\s-]', ' ', q)
        return re.sub(r'\s+', ' ', q).strip()

    def _rerank(self, query: str, docs: List[RetrievedDoc], top_k: int) -> List[RetrievedDoc]:
        """Reranks candidate docs using a cross-encoder. Lets the model handle truncation."""
        pairs = [[query, rd.document.content] for rd in docs]
        try:
            scores = self._reranker.predict(pairs, show_progress_bar=False)
            reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            results = []
            for raw_score, rd in reranked[:top_k]:
                normalized = float(1 / (1 + np.exp(-raw_score / 5)))
                results.append(RetrievedDoc(document=rd.document, score=normalized))
            return results
        except Exception as e:
            logger.warning("Reranker failed, falling back to original order: %s", e)
            return docs[:top_k]

    def boost_by_product_area(self, docs: List[RetrievedDoc], product_area: str) -> List[RetrievedDoc]:
        """Applies in-memory product_area boost to already-retrieved docs and re-sorts."""
        boosted = []
        for rd in docs:
            score = rd.score
            if rd.document.meta.get("product_area", "") == product_area:
                score = min(1.0, score + _PRODUCT_AREA_BOOST)
            boosted.append(RetrievedDoc(document=rd.document, score=score))
        return sorted(boosted, key=lambda r: r.score, reverse=True)

    def _search_all(self, query: str, clean_query: str, top_k: int, product_area: Optional[str] = None) -> List[RetrievedDoc]:
        """Queries every namespace, returns top_k*2 candidates so the reranker has sufficient input."""
        candidates: List[RetrievedDoc] = []
        for company, ns in self._namespaces.items():
            results = ns.search(query, clean_query, top_k * 2, product_area)
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

        return deduped[:top_k * 2]  # return top_k*2 so reranker sees enough candidates


@lru(maxsize=512)
def _encode_query(model: SentenceTransformer, cache_key: str, query: str) -> "np.ndarray":
    """Returns the query embedding, cached by (model, cache_key).

    cache_key is the pre-expansion clean query so that different expansions of
    the same base query reuse the same embedding.  The model is included in the
    key so the cache is safe across different model instances.
    """
    return model.encode([query], show_progress_bar=False, normalize_embeddings=True)


class _Namespace:
    """Per-company index combining BM25 and dense vector search."""

    def __init__(self, documents: List[Document], model: SentenceTransformer) -> None:
        self._documents = documents
        self._model = model

        tokenized = [_tokenize(doc.content) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

        texts = [doc.content for doc in documents]

        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode("utf-8"))
        cache_key = h.hexdigest()

        config.EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = config.EMBEDDINGS_CACHE_DIR / f"{cache_key}.npy"

        if cache_file.exists():
            self._vectors = np.load(cache_file)
        else:
            self._vectors = model.encode(
                texts, show_progress_bar=False, batch_size=64,
                normalize_embeddings=True
            )
            np.save(cache_file, self._vectors)

    def search(self, query: str, clean_query: str, top_k: int, product_area: Optional[str] = None) -> List[RetrievedDoc]:
        """Returns top_k results using weighted BM25 + vector fusion with optional metadata boost."""
        bm25_scores = self._bm25_scores(query)
        vector_scores = self._vector_scores(query, clean_query)  # cache keyed on pre-expansion clean query
        merged = (_BM25_WEIGHT * bm25_scores) + (_VECTOR_WEIGHT * vector_scores)

        # Metadata boost: bump docs matching the router's product_area classification
        if product_area:
            for i, doc in enumerate(self._documents):
                if doc.meta.get("product_area", "") == product_area:
                    merged[i] = min(1.0, merged[i] + _PRODUCT_AREA_BOOST)

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
        tokens = _tokenize(query)
        raw = np.array(self._bm25.get_scores(tokens), dtype=float)
        return _normalize(raw)

    def _vector_scores(self, query: str, cache_key: Optional[str] = None) -> np.ndarray:
        """Cache keyed on clean pre-expansion query; uses module-level lru_cache."""
        key = cache_key if cache_key is not None else query
        q_vec = _encode_query(self._model, key, query)
        sims = (self._vectors @ q_vec.T).flatten()
        sims = np.clip(sims, 0, 1)
        return _normalize(sims)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalizes to [0, 1]; returns zeros if range is zero."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
