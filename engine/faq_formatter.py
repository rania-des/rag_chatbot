"""
FAQ Formatter — retourne la réponse dans la langue de la question.

Fix v2 :
  - Détection de langue Python (sans LLM) → langue explicite au LLM
  - Si la réponse de référence est déjà dans la bonne langue → retour direct (0 LLM)
  - num_ctx=1024 pour éviter les troncatures
  - Prompt plus directif avec la langue cible en majuscules
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import settings


# ──────────────────────────────────────────────────────────────────────
# Détection de langue (Python pur, 0 LLM)
# ──────────────────────────────────────────────────────────────────────
def detect_lang(text: str) -> str:
    """Retourne 'ar', 'en' ou 'fr'."""
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if re.search(
        r"\b(what|when|where|who|how|why|is|are|can|do|does|my|i\s+have|"
        r"the\s+school|forgot|password|login|account)\b",
        text, re.I,
    ):
        return "en"
    return "fr"


def _answer_lang(text: str) -> str:
    """Détecte la langue d'une réponse."""
    return detect_lang(text)


LANG_NAMES = {
    "ar": "arabe",
    "en": "anglais",
    "fr": "français",
}


# ──────────────────────────────────────────────────────────────────────
# Prompt système — très directif sur la langue
# ──────────────────────────────────────────────────────────────────────
def _build_system(target_lang: str) -> str:
    lang_name = LANG_NAMES.get(target_lang, "français")
    return (
        f"Tu es un assistant scolaire. "
        f"Tu dois reformuler la RÉPONSE ci-dessous EN {lang_name.upper()} UNIQUEMENT. "
        f"Réponds TOUJOURS en {lang_name}. "
        f"1 à 3 lignes, direct, sans introduction ni signature."
    )


# ──────────────────────────────────────────────────────────────────────
# Formateur
# ──────────────────────────────────────────────────────────────────────
class FAQFormatter:
    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,          # déterministe
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
            num_predict=200,
            num_ctx=1024,           # évite les troncatures
        )

    @lru_cache(maxsize=512)
    def _cached(self, question: str, answer: str, target_lang: str) -> str:
        try:
            r = self._llm.invoke([
                SystemMessage(content=_build_system(target_lang)),
                HumanMessage(
                    content=(
                        f"QUESTION : {question}\n\n"
                        f"RÉPONSE À REFORMULER : {answer}\n\n"
                        f"Reformule en {LANG_NAMES[target_lang]}."
                    )
                ),
            ])
            return (r.content or answer).strip()
        except Exception:
            return answer  # fallback : réponse brute

    def format(self, user_question: str, reference_answer: str) -> str:
        question_lang = detect_lang(user_question)
        answer_lang   = _answer_lang(reference_answer)

        # ── Optimisation : si la réponse est déjà dans la bonne langue → direct ──
        if answer_lang == question_lang:
            return reference_answer.strip()

        # ── LLM pour traduire / reformuler dans la bonne langue ──────────────
        return self._cached(user_question, reference_answer, question_lang)


_formatter: Optional[FAQFormatter] = None


def get_faq_formatter() -> FAQFormatter:
    global _formatter
    if _formatter is None:
        _formatter = FAQFormatter()
    return _formatter
