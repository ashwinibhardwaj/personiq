"""
personiq.adapter
~~~~~~~~~~~~~~~~
PersoniqAdapter — the single object developers drop into any LangChain
or LangGraph application to add personalization memory.

Design principles
─────────────────
1. PROVIDER-AGNOSTIC  — accepts any LangChain BaseChatModel
2. BACKEND-INDEPENDENT — personiq's storage is self-contained; it does NOT
   interfere with other memory systems (Mem0, Zep, LangMem, etc.)
   Developers can run personiq alongside any other memory setup
3. USE-CASE AGNOSTIC  — works for chatbots, recommendation engines,
   ad targeting, content personalization, any application needing user context
4. MINIMAL INTEGRATION — two lines in LangGraph, one method call everywhere else

Integration styles (all from one object)
─────────────────────────────────────────

Style 1 — LangGraph nodes (recommended for agents and graphs):

    from personiq import PersoniqAdapter
    piq = PersoniqAdapter(llm=your_llm)

    builder.add_node("personiq_load",   piq.load_node)   # before chat
    builder.add_node("personiq_save",   piq.save_node)   # after chat

Style 2 — Direct context injection (any framework, maximum control):

    ctx = piq.context("alice", user_message)        # bullet-list format
    ctx = piq.persona("alice", user_message)        # natural-language format
    system = ctx + base_system_prompt

    # After response:
    await piq.learn("alice", messages)

Style 3 — Inject into messages list (OpenAI / LCEL format):

    messages = piq.inject(messages, user_id="alice", query=user_message)

Style 4 — Inspect and manage stored memories:

    piq.memories("alice")                    # all memories
    piq.memories("alice", category="goal")   # filtered by category
    piq.forget("alice")                      # GDPR delete
    piq.count("alice")                       # memory count
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from personiq.config import PersoniqConfig
from personiq.memory_manager import MemoryManager
from personiq.models import Memory, MemoryCategory

logger = logging.getLogger(__name__)


class PersoniqAdapter:
    """
    Personalization memory adapter for LangChain and LangGraph applications.

    Brings long-term user memory to any AI application with minimal integration.
    Works alongside other memory systems (Mem0, Zep, LangMem, custom stores).

    Parameters
    ──────────
    llm         : Any LangChain BaseChatModel — used for memory extraction and
                  optional query expansion. Same model you use for your chatbot.
    config      : Optional PersoniqConfig. Sensible defaults if omitted.
    user_id_key : LangGraph state key holding the user_id (default: "user_id")
    mode        : Default context format — "context" (bullets) or "persona"
                  (natural language). Can be overridden per-call.

    Supported LLM providers
    ───────────────────────
    from langchain_groq      import ChatGroq
    from langchain_openai    import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_ollama    import ChatOllama
    from langchain_mistralai import ChatMistralAI
    ... any LangChain BaseChatModel

    Quick start
    ───────────
        from langchain_groq import ChatGroq
        from personiq import PersoniqAdapter

        piq = PersoniqAdapter(llm=ChatGroq(model="llama-3.1-8b-instant"))

        # LangGraph:
        builder.add_node("personiq_load", piq.load_node)
        builder.add_node("personiq_save", piq.save_node)

        # Direct:
        ctx     = piq.context("alice", user_message)
        system  = ctx + base_system_prompt
        await piq.learn("alice", messages)
    """

    def __init__(
        self,
        llm:          BaseChatModel,
        config:       Optional[PersoniqConfig] = None,
        user_id_key:  str                      = "user_id",
        mode:         str                      = "context",
        _manager:     Optional[MemoryManager]  = None,   # for testing
    ) -> None:
        self._llm         = llm
        self._config      = config or PersoniqConfig()
        self._user_id_key = user_id_key
        self._mode        = mode
        self._manager     = _manager
        self._injector    = None

        if _manager is None:
            self._boot()

    # ── Boot ───────────────────────────────────────────────────────────────────

    def _boot(self) -> None:
        from personiq.embeddings import create_embedding_engine
        from personiq.extractor import MemoryExtractor
        from personiq.injector import ContextInjector

        emb            = create_embedding_engine(self._config)
        ext            = MemoryExtractor(llm=self._llm, config=self._config)
        self._manager  = MemoryManager(
            config=self._config, llm=self._llm, embedder=emb, extractor=ext).start()
        self._injector = ContextInjector(self._config)
        logger.info("[personiq] adapter ready — provider: %s", type(self._llm).__name__)

    # ── Context retrieval ──────────────────────────────────────────────────────

    def context(self, user_id: str, query: str) -> str:
        """
        Retrieve personalization context as a structured bullet list.

        Call BEFORE your LLM. Prepend to system prompt.
        Returns "" when no memories exist — safe to prepend unconditionally.

            ctx    = piq.context("alice", user_message)
            system = f"{ctx}\\n{base_system}" if ctx else base_system
        """
        return self._manager.get_context(user_id=user_id, query=query, mode="context")

    def persona(self, user_id: str, query: str) -> str:
        """
        Retrieve personalization context as a natural-language persona paragraph.

        Better than context() for conversational chatbots — makes the AI feel
        like it genuinely knows the user, not like it's reading from a database.
        Also ideal for personalised recommendations and ad copy generation.

            ctx    = piq.persona("alice", user_message)
            system = f"{ctx}\\n{base_system}" if ctx else base_system
        """
        return self._manager.get_context(user_id=user_id, query=query, mode="persona")

    async def acontext(self, user_id: str, query: str) -> str:
        """Async version of context()."""
        return await self._manager.aget_context(user_id=user_id, query=query, mode="context")

    async def apersona(self, user_id: str, query: str) -> str:
        """Async version of persona()."""
        return await self._manager.aget_context(user_id=user_id, query=query, mode="persona")

    def inject(
        self,
        messages: list[dict],
        user_id:  str,
        query:    str,
        mode:     Optional[str] = None,
    ) -> list[dict]:
        """
        Inject personalization context into an OpenAI-style messages list.

        Finds the first system message and prepends context to it.
        If no system message exists, inserts one at position 0.

            messages = piq.inject(messages, user_id="alice", query=user_message)
            response = await llm.ainvoke(messages)
        """
        results = self._manager.get_results(user_id=user_id, query=query)
        m       = mode or self._mode
        return self._injector.inject_into_messages(messages, results, mode=m)

    # ── Memory extraction ──────────────────────────────────────────────────────

    async def learn(self, user_id: str, messages: list[BaseMessage]) -> int:
        """
        Extract and store new memories from a conversation turn.

        Call AFTER each assistant response. Pass the full conversation history
        including both user and assistant messages for best extraction quality.

        Returns number of new memories stored (0 when fire-and-forget is active).

            await piq.learn("alice", messages)
        """
        return await self._manager.alearn(user_id=user_id, messages=messages)

    def learn_sync(self, user_id: str, messages: list[BaseMessage]) -> int:
        """Synchronous version of learn(). Blocks until extraction completes."""
        return self._manager.learn(user_id=user_id, messages=messages)

    # ── LangGraph nodes ────────────────────────────────────────────────────────

    @property
    def load_node(self):
        """
        LangGraph node — attach BEFORE your chat node.

        Reads state[user_id_key] and state["messages"].
        Retrieves relevant memories and writes them to state["memory_context"].

        Your chat node reads state["memory_context"] and prepends it to
        the system prompt before calling the LLM.

            builder.add_node("personiq_load", piq.load_node)
            builder.add_edge("personiq_load", "chat")
        """
        manager     = self._manager
        uid_key     = self._user_id_key
        mode        = self._mode

        async def _load(state: dict[str, Any]) -> dict[str, Any]:
            user_id  = state.get(uid_key) or "default_user"
            messages = state.get("messages", [])
            query    = _last_human(messages)
            if not query:
                return {"memory_context": ""}
            ctx = await manager.aget_context(user_id=user_id, query=query, mode=mode)
            if manager._config.debug:
                logger.debug("[personiq] load_node: %d chars for '%s'", len(ctx), user_id)
            return {"memory_context": ctx}

        _load.__name__ = "personiq_load_node"
        return _load

    @property
    def save_node(self):
        """
        LangGraph node — attach AFTER your chat node.

        Reads state[user_id_key] and state["messages"].
        Extracts and saves new memories from the conversation (fire-and-forget).
        Never blocks the response path.

            builder.add_node("personiq_save", piq.save_node)
            builder.add_edge("chat", "personiq_save")
        """
        manager = self._manager
        uid_key = self._user_id_key

        async def _save(state: dict[str, Any]) -> dict[str, Any]:
            user_id  = state.get(uid_key) or "default_user"
            messages = state.get("messages", [])
            if messages:
                await manager.alearn(user_id=user_id, messages=messages)
                if manager._config.debug:
                    logger.debug("[personiq] save_node: queued for '%s'", user_id)
            return {}

        _save.__name__ = "personiq_save_node"
        return _save

    # ── Memory management ──────────────────────────────────────────────────────

    def memories(
        self,
        user_id:  str,
        category: Optional[str] = None,
    ) -> list[Memory]:
        """
        List stored memories for a user, sorted by importance.

        Args:
            user_id  : The user identifier.
            category : Optional filter — "preference", "goal", "technical",
                       "context", "style", "personal"

        Returns list of Memory objects sorted by importance_score descending.

            for m in piq.memories("alice"):
                print(f"[{m.category}] {m.content}")
        """
        if category:
            try:
                cat = MemoryCategory(category)
                return self._manager.list_by_category(user_id, cat)
            except ValueError:
                pass
        return self._manager.list_all(user_id)

    def forget(self, user_id: str) -> int:
        """
        Delete all memories for a user.

        Use for GDPR compliance, account deletion, or resetting personalization.
        Returns number of deleted records.

            deleted = piq.forget("alice")
        """
        count = self._manager.forget(user_id)
        logger.info("[personiq] forgot %d memories for '%s'", count, user_id)
        return count

    def count(self, user_id: str) -> int:
        """Return total number of stored memories for a user."""
        return self._manager.count(user_id)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def config(self) -> PersoniqConfig:
        """Access the active configuration."""
        return self._config

    @property
    def manager(self) -> MemoryManager:
        """Direct access to the underlying MemoryManager (for power users)."""
        return self._manager

    def __repr__(self) -> str:
        return (
            f"PersoniqAdapter(provider={type(self._llm).__name__!r}, "
            f"mode={self._mode!r}, db={self._config.db_path!r})"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _last_human(messages: list) -> str:
    from langchain_core.messages import HumanMessage
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else ""
    return ""