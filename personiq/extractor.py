"""
personiq.extractor
~~~~~~~~~~~~~~~~~~
Extracts structured memory facts from conversation windows.
Accepts any LangChain BaseChatModel — provider-agnostic.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from tenacity import retry, stop_after_attempt, wait_exponential

from personiq.config import PersoniqConfig
from personiq.models import ExtractionResult, ExtractedFact, MemoryCategory

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a memory extraction specialist for a personalised AI system.

TASK
Extract durable, meaningful facts about the USER from the conversation.
These facts personalize future responses, recommendations, and experiences.

WHAT TO EXTRACT
- Stable facts: skills, preferences, goals, background, habits, values
- Implicit facts: if someone debugs Rust fluently, extract "User is experienced with Rust"
- Updated facts: if a new statement contradicts an old one, extract the updated version
- Use cases: personalized chatbots, product recommendations, targeted content, user profiling

WHAT TO SKIP
- Temporary states ("I'm tired today"), one-off events, questions the user asks
- Facts about topics, not the user ("Python uses indentation")
- Vague inferences with no evidence ("User might like coffee")

CATEGORIES
  preference  — likes, dislikes, favourites (products, food, tools, brands, content)
  goal        — objectives, projects, needs, problems to solve, purchase intent
  context     — occupation, location, education, life stage, demographics
  style       — communication style, verbosity, tone, formality preference
  technical   — languages, tools, frameworks, platforms, devices
  personal    — name, relationships, values, life events, personality traits

CONFIDENCE
  1.0  explicitly stated  ("I am a backend engineer")
  0.85 strongly implied   ("I've been shipping Go services for years")
  0.70 reasonably inferred (uses technical vocabulary fluently)
  0.55 weak signal        (question style suggests experience level)

KEYWORDS — 3-6 short nouns/terms per fact for keyword search.
  "User prefers dark roast coffee" → ["coffee", "dark roast", "preference"]

OUTPUT — valid JSON only, no markdown, no explanation:
{"facts": [{"content": str, "category": str, "confidence": float, "keywords": [str]}]}
If nothing to extract: {"facts": []}
"""

_HUMAN = "CONVERSATION (oldest to newest):\n{conversation}\n\nExtract durable user facts."

_CATEGORY_MAP: dict[str, MemoryCategory] = {
    "preference": MemoryCategory.PREFERENCE, "preferences": MemoryCategory.PREFERENCE,
    "goal":       MemoryCategory.GOAL,       "goals":       MemoryCategory.GOAL,
    "context":    MemoryCategory.CONTEXT,    "background":  MemoryCategory.CONTEXT,
    "style":      MemoryCategory.STYLE,      "communication": MemoryCategory.STYLE,
    "technical":  MemoryCategory.TECHNICAL,  "tech":        MemoryCategory.TECHNICAL,
    "personal":   MemoryCategory.PERSONAL,
}


def _format_convo(messages: list[BaseMessage], window: int = 6) -> str:
    recent = messages[-(window * 2):] if len(messages) > window * 2 else messages
    lines  = []
    for m in recent:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines) if lines else "(no conversation)"


def _build_chain(llm: BaseChatModel):
    def make_msgs(inputs: dict) -> list:
        from langchain_core.messages import SystemMessage, HumanMessage as HM
        return [SystemMessage(content=_SYSTEM), HM(content=_HUMAN.format(**inputs))]
    return RunnableLambda(make_msgs) | llm | StrOutputParser()


class MemoryExtractor:
    def __init__(self, llm: BaseChatModel, config: PersoniqConfig) -> None:
        self._config = config
        self._chain  = _build_chain(llm)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=False)
    def extract(self, messages: list[BaseMessage]) -> ExtractionResult:
        convo = _format_convo(messages, self._config.context_window_turns)
        if convo == "(no conversation)":
            return ExtractionResult(facts=[])
        try:
            raw = self._chain.invoke({"conversation": convo})
            return self._parse(raw)
        except Exception as e:
            logger.warning("[personiq] extraction failed: %s", e)
            return ExtractionResult(facts=[], raw_response=str(e))

    async def aextract(self, messages: list[BaseMessage]) -> ExtractionResult:
        convo = _format_convo(messages, self._config.context_window_turns)
        if convo == "(no conversation)":
            return ExtractionResult(facts=[])
        try:
            raw = await self._chain.ainvoke({"conversation": convo})
            return self._parse(raw)
        except Exception as e:
            logger.warning("[personiq] async extraction failed: %s", e)
            return ExtractionResult(facts=[], raw_response=str(e))

    def _parse(self, raw: str) -> ExtractionResult:
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        text = re.sub(r"\s*```$", "", text).strip()
        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[personiq] JSON parse failed: %.200s", raw)
            return ExtractionResult(facts=[], raw_response=raw)

        facts = []
        for item in data.get("facts", []):
            try:
                raw_cat  = str(item.get("category", "context")).lower().strip()
                category = _CATEGORY_MAP.get(raw_cat, MemoryCategory.CONTEXT)
                raw_kw   = item.get("keywords", [])
                if isinstance(raw_kw, str):
                    kws = [k.strip().lower() for k in raw_kw.split(",") if k.strip()]
                else:
                    kws = [str(k).strip().lower() for k in raw_kw if k]
                fact = ExtractedFact(
                    content    = str(item["content"]),
                    category   = category,
                    confidence = float(item.get("confidence", 0.9)),
                    keywords   = kws[:8],
                )
                if fact.content:
                    facts.append(fact)
            except Exception as e:
                logger.debug("[personiq] skipping fact: %s | %s", item, e)

        if self._config.debug:
            logger.debug("[personiq] extracted %d fact(s)", len(facts))
        return ExtractionResult(facts=facts, raw_response=raw)