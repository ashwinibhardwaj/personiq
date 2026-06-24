<div align="center">

<br />

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ashwinibhardwaj/personiq/assets/personiq-logo.jpg">
  <img alt="personiq" src="https://raw.githubusercontent.com/ashwinibhardwaj/personiq/main/assets/logo-light.svg" height="48">
</picture>

<h3>Personalization memory for LangChain & LangGraph</h3>

<p>Extract → Store → Retrieve → Inject.<br/>One adapter. Any LLM. Two lines of code.</p>

<br/>

[![PyPI version](https://img.shields.io/pypi/v/personiq?style=flat-square&color=3B82F6&labelColor=1e293b&label=pypi)](https://pypi.org/project/personiq/)
[![Python](https://img.shields.io/pypi/pyversions/personiq?style=flat-square&color=3B82F6&labelColor=1e293b)](https://pypi.org/project/personiq/)
[![License](https://img.shields.io/pypi/l/personiq?style=flat-square&color=34d399&labelColor=1e293b)](https://github.com/ashwinibhardwaj/personiq/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/personiq?style=flat-square&color=a78bfa&labelColor=1e293b)](https://pypi.org/project/personiq/)
[![Docs](https://img.shields.io/badge/docs-live-3B82F6?style=flat-square&labelColor=1e293b)](https://ashwinibhardwaj.github.io/personiq/)

<br/>

[**Documentation**](https://ashwinibhardwaj.github.io/personiq/) · [**PyPI**](https://pypi.org/project/personiq/) · [**Quickstart**](#quickstart) · [**Examples**](#integration-styles) · [**Discord**](https://discord.gg/your-server)

<br/>

</div>

---

personiq gives your AI application a **persistent, searchable memory of each user** — extracted automatically from conversation, stored locally, and injected back into every future prompt. No database to manage. No cloud dependency. No friction.

```python
from langchain_groq import ChatGroq
from personiq import PersoniqAdapter

piq = PersoniqAdapter(llm=ChatGroq(model="llama-3.1-8b-instant"))

# before your LLM call
ctx = piq.persona("alice", user_message)
system = f"{ctx}\n{base_system}" if ctx else base_system

# after your LLM call
await piq.learn("alice", messages)
```

After the first conversation, Alice's skills, goals, and preferences are woven into every response — without her repeating herself.

<br/>

## How it works

```
Conversation  ──▶  Extract  ──▶  SQLite + Embeddings + BM25
                                          │
Future prompt ◀──  Inject  ◀──  Hybrid Search (RRF)
```

| Stage | What happens |
|---|---|
| **Extract** | Your LLM reads the conversation and identifies durable facts — skills, goals, preferences, style, personal context |
| **Store** | Facts persist in a local SQLite file with vector embeddings (`all-MiniLM-L6-v2`) and BM25 keyword indexes |
| **Retrieve** | Hybrid search fuses cosine similarity + BM25 + recency decay via Reciprocal Rank Fusion |
| **Inject** | Top-ranked memories are prepended to your system prompt as bullets or a natural-language persona |

<br/>

## Installation

```bash
# install with your LLM provider
pip install personiq[groq]        # Groq — llama, gemma, mixtral
pip install personiq[openai]      # OpenAI
pip install personiq[anthropic]   # Anthropic Claude
pip install personiq[ollama]      # local, no API key
pip install personiq[mistral]     # Mistral
pip install personiq[all]         # Groq + OpenAI + Anthropic
```

> **Requirements:** Python 3.10+ · No external database · No cloud dependency beyond your LLM provider

<br/>

## Quickstart

### 1 — Install

```bash
pip install personiq[groq]
```

### 2 — Create the adapter

```python
from langchain_groq import ChatGroq
from personiq import PersoniqAdapter

piq = PersoniqAdapter(llm=ChatGroq(model="llama-3.1-8b-instant"))
```

### 3 — Add to your loop

```python
from langchain_core.messages import HumanMessage, SystemMessage

async def chat(user_id: str, message: str):
    # retrieve what we know about this user
    ctx = piq.persona(user_id, message)
    system = f"{ctx}\nYou are a helpful assistant." if ctx else "You are a helpful assistant."

    msgs = [SystemMessage(content=system), HumanMessage(content=message)]
    response = await piq._llm.ainvoke(msgs)

    # extract and store new memories (fire-and-forget)
    await piq.learn(user_id, msgs + [response])
    return response.content
```

That's it. personiq handles the rest silently.

<br/>

## Integration styles

### LangGraph nodes *(recommended)*

The cleanest integration — two nodes, zero boilerplate.

```python
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from personiq import PersoniqAdapter, PersoniqState

llm = ChatGroq(model="llama-3.1-8b-instant")
piq = PersoniqAdapter(llm=llm, mode="persona")

async def chat(state: PersoniqState) -> dict:
    ctx = state.get("memory_context", "")
    system = f"{ctx}\nYou are a helpful assistant." if ctx else "You are a helpful assistant."
    msgs = [SystemMessage(content=system)] + list(state["messages"])
    resp = await llm.ainvoke(msgs)
    return {"messages": [resp]}

builder = StateGraph(PersoniqState)
builder.add_node("personiq_load", piq.load_node)  # retrieves memories → state["memory_context"]
builder.add_node("chat", chat)
builder.add_node("personiq_save", piq.save_node)  # saves memories fire-and-forget
builder.set_entry_point("personiq_load")
builder.add_edge("personiq_load", "chat")
builder.add_edge("chat", "personiq_save")
builder.add_edge("personiq_save", END)
graph = builder.compile()

result = await graph.ainvoke({
    "user_id": "alice",
    "messages": [HumanMessage(content="I'm a backend engineer using Go and Postgres.")],
})
```

### Direct context *(any framework)*

```python
# before your LLM call
ctx = piq.context("alice", message)   # structured bullets
ctx = piq.persona("alice", message)   # natural-language paragraph

# async variants
ctx = await piq.acontext("alice", message)
ctx = await piq.apersona("alice", message)

system = f"{ctx}\n{base_system}" if ctx else base_system

# after your LLM call
await piq.learn("alice", messages)
```

### Message injection *(OpenAI-style lists)*

```python
# finds the system message and prepends context
# inserts a new system message at [0] if none exists
messages = piq.inject(messages, user_id="alice", query=message)
response = await llm.ainvoke(messages)
```

<br/>

## Context modes

**`mode="context"`** — structured bullets, best for technical assistants and dashboards:

```
[personiq: what I know about this user]

Technical background:
  • User is an experienced Go developer
  • User works with PostgreSQL and Redis

Goals:
  • User is building a distributed payment service

[end of personiq context]
```

**`mode="persona"`** — natural language, best for conversational chatbots:

```
You already know this user. Here is what you know about them:
They work with Go and PostgreSQL. They're currently focused on building
a distributed payment service. They prefer concise, direct answers.
Use this naturally — don't recite it back, just let it shape your response.
```

```python
# set globally
piq = PersoniqAdapter(llm=llm, mode="persona")

# or override per call
ctx = piq.context(user_id, query)   # always bullets
ctx = piq.persona(user_id, query)   # always natural language
```

<br/>

## Memory categories

personiq classifies every extracted fact into one of six categories, each with its own retrieval boost:

| Category | Boost | What gets stored |
|---|---|---|
| `technical` | ×1.20 | Languages, frameworks, tools, platforms |
| `goal` | ×1.15 | Objectives, projects, problems to solve |
| `preference` | ×1.10 | Likes, dislikes, favourites |
| `context` | ×1.05 | Occupation, location, life stage |
| `personal` | ×1.00 | Name, relationships, values |
| `style` | ×0.90 | Communication style, tone preference |

<br/>

## Works alongside other memory systems

personiq stores data in its own SQLite database. It does not interfere with Mem0, Zep, LangMem, or any other memory setup — run them side by side.

```python
piq  = PersoniqAdapter(llm=llm)   # personiq — personalization facts
mem0 = MemoryClient(...)           # Mem0 — conversation history

# both work independently on the same conversation
ctx  = piq.persona("alice", message)
hist = mem0.get_history("alice")
```

<br/>

## Configuration

```python
from personiq import PersoniqAdapter, PersoniqConfig

config = PersoniqConfig(
    db_path="./myapp.db",
    embedding_backend="local",        # "local" | "openai"
    embedding_model="all-MiniLM-L6-v2",
    top_k=5,                          # memories returned per query
    similarity_threshold=0.20,        # minimum score to include
    semantic_weight=0.60,             # hybrid search weights
    bm25_weight=0.30,
    recency_weight=0.10,
    dedup_threshold=0.85,             # deduplicate above this similarity
    async_extraction=True,            # fire-and-forget learn()
    context_window_turns=6,           # conversation turns analysed
    max_context_chars=1500,           # max chars injected into prompt
    debug=False,
)

piq = PersoniqAdapter(llm=llm, config=config)
```

All settings are also configurable via `PERSONIQ_*` environment variables.

<br/>

## Supported providers

| Provider | Install | Import |
|---|---|---|
| **Groq** | `pip install personiq[groq]` | `from langchain_groq import ChatGroq` |
| **OpenAI** | `pip install personiq[openai]` | `from langchain_openai import ChatOpenAI` |
| **Anthropic** | `pip install personiq[anthropic]` | `from langchain_anthropic import ChatAnthropic` |
| **Ollama** | `pip install personiq[ollama]` | `from langchain_ollama import ChatOllama` |
| **Mistral** | `pip install personiq[mistral]` | `from langchain_mistralai import ChatMistralAI` |
| **HuggingFace** | `pip install personiq[huggingface]` | `from langchain_huggingface import ChatHuggingFace` |

Any `BaseChatModel` from LangChain works.

<br/>

## Memory management

```python
# inspect all memories
for m in piq.memories("alice"):
    print(f"[{m.category:12}] {m.content}  (importance: {m.importance_score:.2f})")

# filter by category
goals = piq.memories("alice", category="goal")
tech  = piq.memories("alice", category="technical")

# count
print(piq.count("alice"))

# GDPR / account deletion
deleted = piq.forget("alice")
print(f"Deleted {deleted} memories")
```

<br/>

## Use cases

personiq is designed for any application where knowing the user improves outcomes:

- **Personalised chatbots** — the assistant remembers name, job, and preferences across sessions
- **Recommendation engines** — surface products and content matching stored preferences and goals
- **AI sales agents** — know the prospect's industry, pain points, and purchase intent across touches
- **Content personalisation** — adapt tone, complexity, and topics to each user automatically
- **Customer support** — remember past issues, preferences, and technical environment per user
- **Ad targeting** — build rich user profiles from natural conversation signals

<br/>

## Contributing

```bash
git clone https://github.com/ashwinibhardwaj/personiq
cd personiq
pip install -e ".[dev]"

# run the full test suite — no API key needed, all LLM calls are mocked
pytest tests/ -v

# lint
ruff check .
```

Pull requests are welcome. Please open an issue first to discuss significant changes.

<br/>

---

<div align="center">

**[Documentation](https://ashwinibhardwaj.github.io/personiq/)** · **[PyPI](https://pypi.org/project/personiq/)** · **[Changelog](https://github.com/ashwinibhardwaj/personiq/blob/main/CHANGELOG.md)** · **[License (MIT)](https://github.com/ashwinibhardwaj/personiq/blob/main/LICENSE)**

<br/>

<sub>Built with ❤️ for the LangChain ecosystem</sub>

</div>
