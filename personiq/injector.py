"""
personiq.injector
~~~~~~~~~~~~~~~~~
Builds formatted memory context from HybridSearchResult objects.

Two output modes:
  build_context()  — structured bullet-list block for system prompt injection
  build_persona()  — natural-language paragraph that makes the AI feel like
                     it genuinely knows the user (better for conversational UX)
"""
from __future__ import annotations

from personiq.config import PersoniqConfig
from personiq.models import HybridSearchResult, MemoryCategory

_LABELS = {
    MemoryCategory.TECHNICAL:  "Technical background",
    MemoryCategory.GOAL:       "Goals & current work",
    MemoryCategory.PREFERENCE: "Preferences",
    MemoryCategory.CONTEXT:    "Background",
    MemoryCategory.PERSONAL:   "Personal",
    MemoryCategory.STYLE:      "Communication style",
}

_ORDER = [
    MemoryCategory.TECHNICAL,
    MemoryCategory.GOAL,
    MemoryCategory.PREFERENCE,
    MemoryCategory.CONTEXT,
    MemoryCategory.PERSONAL,
    MemoryCategory.STYLE,
]


class ContextInjector:
    def __init__(self, config: PersoniqConfig) -> None:
        self._config = config

    def build_context(self, results: list[HybridSearchResult]) -> str:
        """
        Structured bullet-list context block.
        Suitable for any LLM — clear, readable, grouped by category.
        Returns "" when no results so callers can prepend unconditionally.
        """
        if not results:
            return ""

        grouped = _group(results)
        lines   = ["[personiq: what I know about this user]"]

        for cat in _ORDER:
            cv    = cat.value if hasattr(cat, "value") else cat
            items = grouped.get(cv, [])
            if not items:
                continue
            label = _LABELS.get(cv, cv.capitalize())
            lines.append(f"\n{label}:")
            for r in items:
                hint = ""
                if r.memory.confidence < 0.70:
                    hint = f"  (confidence {r.memory.confidence:.0%})"
                elif r.memory.decay_factor < 0.55:
                    hint = "  (may be outdated)"
                lines.append(f"  • {r.memory.content}{hint}")

        lines.append("\n[end of personiq context]\n")
        block = "\n".join(lines)

        if len(block) > self._config.max_context_chars:
            block = block[:self._config.max_context_chars - 45] + \
                    "\n  … (truncated)\n[end of personiq context]\n"
        return block

    def build_persona(self, results: list[HybridSearchResult]) -> str:
        """
        Natural-language persona paragraph.
        Makes the AI feel like it genuinely knows the user rather than reading from a list.
        Best for conversational chatbots, sales agents, and personalised assistants.
        """
        if not results:
            return ""

        grouped    = _group(results)
        parts: list[str] = []

        # Technical
        tech = [r.memory.content for r in grouped.get(MemoryCategory.TECHNICAL, [])]
        if tech:
            skills = _shorten_list(tech, 4)
            parts.append(f"They work with {skills}.")

        # Goals
        goals = [r.memory.content for r in grouped.get(MemoryCategory.GOAL, [])]
        if goals:
            parts.append(f"They're currently focused on: {goals[0].lower().rstrip('.')}.")

        # Preferences
        prefs = [r.memory.content for r in grouped.get(MemoryCategory.PREFERENCE, [])]
        if prefs:
            parts.append(prefs[0].rstrip(".") + ".")

        # Context (occupation, location, etc.)
        ctx = [r.memory.content for r in grouped.get(MemoryCategory.CONTEXT, [])]
        for c in ctx[:2]:
            parts.append(c.rstrip(".") + ".")

        # Personal
        personal = [r.memory.content for r in grouped.get(MemoryCategory.PERSONAL, [])]
        for p in personal[:1]:
            parts.append(p.rstrip(".") + ".")

        # Style
        style = [r.memory.content for r in grouped.get(MemoryCategory.STYLE, [])]
        if style:
            parts.append(style[0].rstrip(".") + ".")

        if not parts:
            return ""

        body = " ".join(parts)
        return (
            "You already know this user. Here is what you know about them:\n"
            f"{body}\n"
            "Use this naturally — don't recite it back, just let it shape how you respond."
        )

    def inject_into_system_prompt(self, system: str, results: list[HybridSearchResult],
                                   mode: str = "context") -> str:
        """
        Prepend memory context to a system prompt string.
        mode: "context" (bullet list) | "persona" (natural paragraph)
        """
        ctx = self.build_persona(results) if mode == "persona" else self.build_context(results)
        return f"{ctx}\n{system}" if ctx else system

    def inject_into_messages(self, messages: list[dict], results: list[HybridSearchResult],
                              mode: str = "context") -> list[dict]:
        """Inject into OpenAI-style [{"role": ..., "content": ...}] messages list."""
        ctx = self.build_persona(results) if mode == "persona" else self.build_context(results)
        if not ctx:
            return messages
        messages = list(messages)
        for i, m in enumerate(messages):
            if m.get("role") == "system":
                messages[i] = {**m, "content": f"{ctx}\n{m['content']}"}
                return messages
        messages.insert(0, {"role": "system", "content": ctx})
        return messages


def _group(results: list[HybridSearchResult]) -> dict[str, list[HybridSearchResult]]:
    g: dict[str, list[HybridSearchResult]] = {}
    for r in results:
        cv = r.memory.category
        cv = cv.value if hasattr(cv, "value") else cv
        g.setdefault(cv, []).append(r)
    return g


def _shorten_list(items: list[str], max_n: int) -> str:
    """'User uses Python' + 'User uses Go' → 'Python and Go'."""
    extracted = []
    for item in items[:max_n]:
        # Try to pull the tool/skill name from common patterns
        import re
        m = re.search(r"(?:uses?|works? with|knows?|expert in|experienced? with)\s+([A-Za-z0-9#+.\-/]+)", item, re.I)
        extracted.append(m.group(1) if m else item.rstrip("."))
    if len(extracted) == 1:
        return extracted[0]
    return ", ".join(extracted[:-1]) + f" and {extracted[-1]}"