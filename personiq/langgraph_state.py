"""
personiq.langgraph_state
~~~~~~~~~~~~~~~~~~~~~~~~
PersoniqState — TypedDict mixin for LangGraph graphs.

Merge into your own state to add personiq fields:

    from personiq import PersoniqState
    from langgraph.graph import MessagesState

    class MyState(MessagesState, PersoniqState):
        my_field: str

    # Then invoke with:
    await graph.ainvoke({"user_id": "alice", "messages": [...]})
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class PersoniqState(TypedDict, total=False):
    """
    LangGraph state mixin for personiq.

    Fields
    ──────
    user_id        : str  — identifies which user's memories to load/save
    memory_context : str  — populated by load_node; read in your chat node
    messages       : list — standard LangGraph messages accumulator
    """
    user_id:        str
    memory_context: str
    messages:       Annotated[list[BaseMessage], add_messages]