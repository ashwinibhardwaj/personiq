"""
personiq — personalization memory connector for LangChain and LangGraph.

One adapter. Any LLM. Works alongside any other memory system.
Built for chatbots, recommendation engines, ad personalization, and more.

Quick start
───────────
    from langchain_groq import ChatGroq          # or OpenAI, Anthropic, Ollama …
    from personiq import PersoniqAdapter, PersoniqState

    piq = PersoniqAdapter(llm=ChatGroq(model="llama-3.1-8b-instant"))

    # LangGraph integration:
    builder.add_node("personiq_load", piq.load_node)   # before chat
    builder.add_node("personiq_save", piq.save_node)   # after chat

    builder.set_entry_point("personiq_load")
    builder.add_edge("personiq_load", "chat")
    builder.add_edge("chat",          "personiq_save")
    builder.add_edge("personiq_save", END)
    graph = builder.compile()

    result = await graph.ainvoke({
        "user_id":  "alice",
        "messages": [HumanMessage(content="hello")],
    })

    # Direct usage (any framework):
    ctx     = piq.context("alice", user_message)   # before LLM
    persona = piq.persona("alice", user_message)   # natural-language variant
    await piq.learn("alice", messages)             # after LLM

    # Inspect / manage:
    piq.memories("alice")
    piq.forget("alice")

Supported providers
───────────────────
    langchain_groq        ChatGroq
    langchain_openai      ChatOpenAI
    langchain_anthropic   ChatAnthropic
    langchain_ollama      ChatOllama
    langchain_mistralai   ChatMistralAI
    langchain_huggingface ChatHuggingFace
    ... any LangChain BaseChatModel

Works alongside other memory systems
─────────────────────────────────────
personiq is fully independent — it stores memories in its own SQLite database
and does not interfere with Mem0, Zep, LangMem, or any other memory setup.
Run multiple systems side by side:

    piq    = PersoniqAdapter(llm=llm)           # personiq
    mem0   = MemoryClient(...)                   # Mem0
    # Both work independently on the same conversation
"""

from personiq.adapter         import PersoniqAdapter
from personiq.langgraph_state import PersoniqState
from personiq.config          import PersoniqConfig
from personiq.memory_manager  import MemoryManager
from personiq.models          import Memory, MemoryCategory, HybridSearchResult

__version__ = "0.2.0"
__author__  = "personiq"

__all__ = [
    # Primary API — what 99% of developers need
    "PersoniqAdapter",
    "PersoniqState",
    "PersoniqConfig",
    # Data models — for type hints and advanced usage
    "Memory",
    "MemoryCategory",
    "HybridSearchResult",
    # Internal engine — for custom integrations
    "MemoryManager",
]