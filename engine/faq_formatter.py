"""FAQ Formatter — reformule dans la langue de la question."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import settings

SYSTEM = (
    "Tu reçois une QUESTION et une RÉPONSE de référence. "
    "Reformule la RÉPONSE dans la même langue que la QUESTION, en 1-3 lignes. "
    "Sois direct. Pas d'intro, pas de signature."
)


class FAQFormatter:
    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
            num_predict=150,
        )

    @lru_cache(maxsize=512)
    def _cached(self, question: str, answer: str) -> str:
        try:
            r = self._llm.invoke([
                SystemMessage(content=SYSTEM),
                HumanMessage(content=f"QUESTION: {question}\n\nRÉPONSE: {answer}"),
            ])
            return (r.content or answer).strip()
        except Exception:
            return answer  # fallback : réponse brute si LLM échoue

    def format(self, user_question: str, reference_answer: str) -> str:
        return self._cached(user_question, reference_answer)


_formatter: FAQFormatter | None = None

def get_faq_formatter() -> FAQFormatter:
    global _formatter
    if _formatter is None:
        _formatter = FAQFormatter()
    return _formatter