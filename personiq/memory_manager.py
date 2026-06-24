"""
personiq.memory_manager
~~~~~~~~~~~~~~~~~~~~~~~
Central orchestrator: extraction, deduplication, storage, retrieval.
This class is the internal engine. Developers use personiq.adapter.PersoniqAdapter.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np
from langchain_core.messages import BaseMessage, HumanMessage

from personiq.config import PersoniqConfig
from personiq.embeddings import EmbeddingEngine, create_embedding_engine
from personiq.extractor import MemoryExtractor
from personiq.injector import ContextInjector
from personiq.models import HybridSearchResult, Memory, MemoryCategory
from personiq.retriever import MemoryRetriever
from personiq.store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(
        self,
        config:    Optional[PersoniqConfig]   = None,
        llm                                   = None,
        embedder:  Optional[EmbeddingEngine]  = None,
        extractor                             = None,
    ) -> None:
        self._config   = config or PersoniqConfig()
        self._llm      = llm
        self._embedder = embedder or create_embedding_engine(self._config)
        self._extractor: Optional[MemoryExtractor]  = extractor
        self._store:     Optional[MemoryStore]       = None
        self._retriever: Optional[MemoryRetriever]   = None
        self._injector   = ContextInjector(self._config)
        self._started    = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "MemoryManager":
        if self._started: return self

        dim          = self._embedder.dimension
        self._store  = MemoryStore(self._config, embedding_dim=dim)
        self._store.connect()

        if self._extractor is None:
            if not self._llm:
                raise RuntimeError(
                    "[personiq] an LLM is required. Pass any LangChain BaseChatModel."
                )
            self._extractor = MemoryExtractor(llm=self._llm, config=self._config)

        self._retriever = MemoryRetriever(
            store=self._store, embedder=self._embedder,
            config=self._config, llm=self._llm,
        )
        self._started = True
        logger.info("[personiq] started — db=%s", self._config.db_path)
        return self

    def stop(self) -> None:
        if self._store: self._store.close()
        self._started = False

    def __enter__(self)   -> "MemoryManager": return self.start()
    def __exit__(self, *_) -> None:           self.stop()

    def _ok(self) -> None:
        if not self._started:
            raise RuntimeError("[personiq] call start() before use")

    # ── Context retrieval ──────────────────────────────────────────────────────

    def get_context(self, user_id: str, query: str, mode: str = "context") -> str:
        self._ok()
        results = self._retriever.retrieve(user_id=user_id, query=query)
        if mode == "persona":
            return self._injector.build_persona(results)
        return self._injector.build_context(results)

    async def aget_context(self, user_id: str, query: str, mode: str = "context") -> str:
        self._ok()
        results = await self._retriever.aretrieve(user_id=user_id, query=query)
        if mode == "persona":
            return self._injector.build_persona(results)
        return self._injector.build_context(results)

    def get_results(self, user_id: str, query: str) -> list[HybridSearchResult]:
        self._ok()
        return self._retriever.retrieve(user_id=user_id, query=query)

    # ── Extraction ─────────────────────────────────────────────────────────────

    def learn(self, user_id: str, messages: list[BaseMessage]) -> int:
        self._ok()
        return self._process(user_id, messages)

    async def alearn(self, user_id: str, messages: list[BaseMessage]) -> int:
        self._ok()
        if self._config.async_extraction:
            asyncio.create_task(
                self._aprocess(user_id, messages), name=f"personiq-{user_id[:8]}")
            return 0
        return await self._aprocess(user_id, messages)

    # ── Memory management ──────────────────────────────────────────────────────

    def forget(self, user_id: str) -> int:
        self._ok()
        return self._store.delete_user(user_id)

    def list_all(self, user_id: str) -> list[Memory]:
        self._ok()
        return self._store.get_all(user_id)

    def list_by_category(self, user_id: str, cat: MemoryCategory) -> list[Memory]:
        self._ok()
        return self._store.get_by_category(user_id, cat)

    def count(self, user_id: str) -> int:
        self._ok()
        return self._store.count(user_id)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _process(self, user_id: str, messages: list[BaseMessage]) -> int:
        result = self._extractor.extract(messages)
        if not result.facts: return 0
        embeddings = self._embedder.embed_batch([f.content for f in result.facts])
        stored = 0
        for fact, emb in zip(result.facts, embeddings):
            m = Memory(
                user_id=user_id, content=fact.content, category=fact.category,
                confidence=fact.confidence, keywords=",".join(fact.keywords),
                embedding=emb, source_text=_last_human(messages)[:200],
            )
            if self._dedup_and_store(m): stored += 1
        return stored

    async def _aprocess(self, user_id: str, messages: list[BaseMessage]) -> int:
        try:
            result = await self._extractor.aextract(messages)
            if not result.facts: return 0
            embeddings = await asyncio.to_thread(
                self._embedder.embed_batch, [f.content for f in result.facts])
            stored = 0
            for fact, emb in zip(result.facts, embeddings):
                m = Memory(
                    user_id=user_id, content=fact.content, category=fact.category,
                    confidence=fact.confidence, keywords=",".join(fact.keywords),
                    embedding=emb, source_text=_last_human(messages)[:200],
                )
                if await asyncio.to_thread(self._dedup_and_store, m): stored += 1
            return stored
        except Exception as e:
            logger.error("[personiq] async extraction error for '%s': %s", user_id, e)
            return 0

    def _dedup_and_store(self, memory: Memory) -> bool:
        """
        Two-tier deduplication:
        ≥ dedup_threshold  → reinforce existing (no insert)
        ≥ merge_threshold  → LLM merges both into a richer single fact
        < merge_threshold  → new memory, insert normally
        """
        if not memory.embedding:
            self._store.upsert(memory)
            return True

        qvec     = np.array(memory.embedding, dtype=np.float32)
        existing = self._store.get_vectors(memory.user_id)
        best_sim, best_id = 0.0, None

        for mid, vec in existing:
            sim = float(np.dot(qvec, vec))
            if sim > best_sim:
                best_sim, best_id = sim, mid

        if best_sim >= self._config.dedup_threshold and best_id:
            em = self._store.get(best_id)
            if em:
                em.touch()
                self._store.update_confidence(best_id, em)
                logger.debug("[personiq] reinforced (sim=%.3f): %s", best_sim, em.content)
            return False

        from typing import cast
        from personiq.models import Memory

        if best_sim >= self._config.merge_threshold and best_id and self._llm:
            em = cast(Memory | None, self._store.get(best_id))

            if em:
                merged = self._llm_merge(em.content, memory.content)

                if merged:
                    kw1 = em.keywords or ""
                    kw2 = memory.keywords or ""

                    merged_kw = ",".join(
                        sorted(
                            set(filter(None, kw1.split(","))) |
                            set(filter(None, kw2.split(",")))
                        )
                    )

                    self._store.update_content(best_id, merged, merged_kw)
                    logger.debug("[personiq] merged: '%s'", merged)
                    return True

        self._store.upsert(memory)
        return True

    def _llm_merge(self, a: str, b: str) -> Optional[str]:
        try:
            prompt = (
                f"Combine these two overlapping facts about a user into ONE concise sentence.\n"
                f"Return ONLY the sentence.\nFact 1: {a}\nFact 2: {b}"
            )
            r = self._llm.invoke([HumanMessage(content=prompt)])
            return r.content.strip().strip('"\'') or max(a, b, key=len)
        except Exception as e:
            logger.debug("[personiq] LLM merge failed: %s", e)
            return max(a, b, key=len)


def _last_human(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else ""
    return ""