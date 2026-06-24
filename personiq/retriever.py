"""
personiq.retriever
~~~~~~~~~~~~~~~~~~
Multi-query hybrid retrieval with optional query expansion.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel

from personiq.config import PersoniqConfig
from personiq.embeddings import EmbeddingEngine
from personiq.models import HybridSearchResult
from personiq.store import MemoryStore

logger = logging.getLogger(__name__)

_EXPAND_PROMPT = """\
Given this user message, write {n} short search queries (one per line) to find \
relevant stored facts about this user. Cover: background, skills, goals, preferences.
Output ONLY the queries, one per line, no numbering.

Message: {message}"""


class MemoryRetriever:
    def __init__(
        self,
        store:    MemoryStore,
        embedder: EmbeddingEngine,
        config:   PersoniqConfig,
        llm:      Optional[BaseChatModel] = None,
    ) -> None:
        self._store   = store
        self._embedder = embedder
        self._config  = config
        self._llm     = llm

    def retrieve(self, user_id: str, query: str,
                 top_k: Optional[int] = None, threshold: Optional[float] = None) -> list[HybridSearchResult]:
        if not query.strip(): return []
        k = top_k or self._config.top_k
        t = threshold or self._config.similarity_threshold
        queries = self._expand_sync(query)
        return self._search(user_id, queries, k, t)

    async def aretrieve(self, user_id: str, query: str,
                        top_k: Optional[int] = None, threshold: Optional[float] = None) -> list[HybridSearchResult]:
        if not query.strip(): return []
        k = top_k or self._config.top_k
        t = threshold or self._config.similarity_threshold
        queries = await self._expand_async(query)
        return await asyncio.to_thread(self._search, user_id, queries, k, t)

    def _expand_sync(self, message: str) -> list[str]:
        n = self._config.retrieval_queries
        if n <= 1 or not self._llm: return [message]
        try:
            from langchain_core.messages import HumanMessage
            r = self._llm.invoke([HumanMessage(content=_EXPAND_PROMPT.format(n=n, message=message))])
            return self._parse_queries(r.content, message, n)
        except Exception as e:
            logger.debug("[personiq] query expansion failed: %s", e)
            return [message]

    async def _expand_async(self, message: str) -> list[str]:
        n = self._config.retrieval_queries
        if n <= 1 or not self._llm: return [message]
        try:
            from langchain_core.messages import HumanMessage
            r = await self._llm.ainvoke([HumanMessage(content=_EXPAND_PROMPT.format(n=n, message=message))])
            return self._parse_queries(r.content, message, n)
        except Exception as e:
            logger.debug("[personiq] async query expansion failed: %s", e)
            return [message]

    @staticmethod
    def _parse_queries(raw: str, original: str, n: int) -> list[str]:
        lines   = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        queries = lines[:n]
        if original not in queries:
            queries.insert(0, original)
        return queries[:n + 1]

    def _search(self, user_id: str, queries: list[str], k: int, t: float) -> list[HybridSearchResult]:
        embeddings = self._embedder.embed_batch(queries)
        best: dict[str, HybridSearchResult] = {}

        for q, emb in zip(queries, embeddings):
            for r in self._store.hybrid_search(
                user_id=user_id, query_text=q, query_embedding=emb,
                top_k=k * 2, threshold=t,
                semantic_weight=self._config.semantic_weight,
                bm25_weight=self._config.bm25_weight,
                recency_weight=self._config.recency_weight,
            ):
                if r.memory.id not in best or r.final_score > best[r.memory.id].final_score:
                    best[r.memory.id] = r

        ranked = sorted(best.values(), key=lambda r: r.final_score, reverse=True)
        if self._config.debug:
            logger.debug("[personiq] %d queries → %d results for '%s'", len(queries), len(ranked), user_id)
        return ranked[:k]