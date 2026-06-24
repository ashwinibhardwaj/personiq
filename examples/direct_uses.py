"""
examples/direct_usage.py
~~~~~~~~~~~~~~~~~~~~~~~~
Shows piq.context() and piq.learn() without LangGraph.
Works with any framework — FastAPI, Flask, raw asyncio, etc.
"""
import asyncio, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from personiq import PersoniqAdapter

llm = ChatGroq(model="llama-3.1-8b-instant")
piq = PersoniqAdapter(llm=llm)

BASE_SYSTEM = "You are a helpful personal assistant."


async def chat(user_id: str, message: str, history: list) -> str:
    # 1. Get personalization context BEFORE the LLM call
    ctx    = piq.persona(user_id, message)
    system = f"{ctx}\n{BASE_SYSTEM}" if ctx else BASE_SYSTEM

    messages = [SystemMessage(content=system)] + history + [HumanMessage(content=message)]

    # 2. Call your LLM
    response = await llm.ainvoke(messages)

    # 3. Save memories AFTER the LLM call (fire-and-forget)
    all_messages = history + [HumanMessage(content=message), response]
    await piq.learn(user_id, all_messages)

    return response.content


async def main():
    user_id = "alice"
    history = []

    print("Direct usage demo. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text == "quit":
            break

        reply = await chat(user_id, text, history)
        print(f"\nAssistant: {reply}\n")
        history.append(HumanMessage(content=text))


if __name__ == "__main__":
    asyncio.run(main())