"""
examples/langgraph_app.py
~~~~~~~~~~~~~~~~~~~~~~~~~
A complete LangGraph chatbot with personiq memory in ~50 lines.

Shows the recommended integration pattern:
  personiq_load → chat → personiq_save

Run:
    export GROQ_API_KEY=gsk_...
    python examples/langgraph_app.py
"""
import asyncio
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

# ── Swap this line for any provider ───────────────────────────────────────────
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant")
# from langchain_openai    import ChatOpenAI;    llm = ChatOpenAI(model="gpt-4o-mini")
# from langchain_anthropic import ChatAnthropic; llm = ChatAnthropic(model="claude-haiku-4-5")
# from langchain_ollama    import ChatOllama;    llm = ChatOllama(model="llama3.2")
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from personiq import PersoniqAdapter, PersoniqState

BASE_SYSTEM = "You are a helpful personal assistant. Be concise and direct."


async def main():
    piq = PersoniqAdapter(llm=llm, mode="persona")   # natural-language memory style

    # ── Your chat node ─────────────────────────────────────────────────────────
    async def chat(state: PersoniqState) -> dict:
        messages    = list(state.get("messages", []))
        mem_context = state.get("memory_context", "")
        system      = f"{mem_context}\n{BASE_SYSTEM}" if mem_context else BASE_SYSTEM

        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=system)
        else:
            messages.insert(0, SystemMessage(content=system))

        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    # ── Build graph ────────────────────────────────────────────────────────────
    builder = StateGraph(PersoniqState)
    builder.add_node("personiq_load", piq.load_node)   # ← personiq
    builder.add_node("chat",          chat)
    builder.add_node("personiq_save", piq.save_node)   # ← personiq

    builder.set_entry_point("personiq_load")
    builder.add_edge("personiq_load", "chat")
    builder.add_edge("chat",          "personiq_save")
    builder.add_edge("personiq_save", END)
    graph = builder.compile()

    # ── Chat loop ──────────────────────────────────────────────────────────────
    user_id = input("Your name: ").strip() or "user"
    history = []
    print(f"\nChatting as '{user_id}'. Commands: 'memories', 'forget', 'quit'\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text or text == "quit":
            break
        if text == "memories":
            for m in piq.memories(user_id):
                print(f"  [{m.category:12}] {m.content}")
            print()
            continue
        if text == "forget":
            piq.forget(user_id)
            print("  All memories cleared.\n")
            continue

        result  = await graph.ainvoke({
            "user_id":  user_id,
            "messages": history + [HumanMessage(content=text)],
        })
        reply   = result["messages"][-1].content
        print(f"\nAssistant: {reply}\n")
        history = result["messages"][-10:]   # keep last 5 turns


if __name__ == "__main__":
    asyncio.run(main())