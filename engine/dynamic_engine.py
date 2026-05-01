"""
Moteur DYNAMIC — tool calling Supabase avec Ollama.

Optimisations v5 :
  - Prompt système réécrit pour qwen2.5:1.5b :
      · Court (< 120 tokens) — le petit modèle se perd dans les longs prompts
      · Directif avec exemples inline de tool calling obligatoire
      · Anti-hallucination renforcé pour le menu et l'emploi du temps
  - Shortcut étendu : plus de patterns = moins d'appels LLM inutiles
  - Détection langue de la question → réponse dans la même langue
  - max 4 messages d'historique (économie tokens)
  - 3 tours max de tool calling
"""
from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_ollama import ChatOllama

from config import settings
from tools import ALL_TOOLS


# ──────────────────────────────────────────────────────────────────────
# Détection de langue (simple, basée sur les caractères)
# ──────────────────────────────────────────────────────────────────────
def _detect_lang(text: str) -> str:
    """Détecte la langue dominante : 'ar', 'en', ou 'fr'."""
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if re.search(
        r"\b(what|when|where|who|how|my|i\s+have|do\s+i|is\s+there|show\s+me)\b",
        text, re.I
    ):
        return "en"
    return "fr"


# ──────────────────────────────────────────────────────────────────────
# Shortcut : résultat tool déjà lisible → on l'envoie sans 2e LLM
# ──────────────────────────────────────────────────────────────────────
_CLEAN_RESULT = re.compile(
    r"^("
    # Français — réponses types des tools
    r"Aucun|Menu du|Emploi du temps|Cours du|"
    r"\d+\s+(note|devoir|réunion|paiement|absence)|"
    r"Depuis le|Dernières annonces|Il n[' ]y a pas|"
    # Anglais
    r"No\s+(grades|schedule|homework|meeting|payment|absence)|"
    r"Schedule for|"
    # Arabe
    r"لا\s+|لم\s+|جدول\s+|قائمة\s+|المطعم\s+"
    r")",
    re.IGNORECASE | re.UNICODE,
)


# ──────────────────────────────────────────────────────────────────────
# Prompt système — optimisé pour les petits modèles (1.5b / 3b)
#
# Principes :
#   1. COURT  — max ~100 tokens de system prompt
#   2. ORDRE  — tool d'abord, réponse ensuite (jamais l'inverse)
#   3. DIRECT — pas de politesse inutile qui mange des tokens
#   4. INTERDIT — liste explicite des hallucinations communes
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
Tu es l'assistant d'une école. Tu réponds UNIQUEMENT à partir des outils disponibles.

RÈGLES ABSOLUES :
1. Pour menu, emploi du temps, notes, absences, devoirs, réunions, paiements, annonces → appelle TOUJOURS l'outil AVANT de répondre. JAMAIS de réponse inventée.
2. Réponds dans la même langue que la question (français, arabe ou anglais).
3. Sois bref : 1 à 4 lignes maximum après avoir reçu le résultat de l'outil.
4. Si l'outil répond "Aucun" ou "No data" → dis-le simplement, sans inventer.
5. Ne dis JAMAIS "je ne peux pas accéder" ou "je n'ai pas accès" — tu AS accès via les outils.\
"""

# Rappel injecté si le LLM oublie d'appeler un outil (détection heuristique)
_TOOL_REMINDER = """\
STOP. Tu n'as pas appelé l'outil. La question porte sur des données réelles.
Appelle l'outil approprié maintenant. Ne réponds PAS sans avoir appelé l'outil.\
"""

# Mots-clés qui doivent déclencher un outil — si le LLM répond sans en appeler un,
# on ajoute le rappel pour forcer un 2e tour
_MUST_USE_TOOL = re.compile(
    r"\b(menu|cantine|manger|repas|emploi|horaire|planning|"
    r"cours|note|résultat|moyenne|absence|devoir|réunion|paiement|annonce|"
    r"schedule|grade|homework|attendance|payment|meeting|food|lunch|"
    r"مطعم|وجبة|جدول|درجة|غياب|واجب|اجتماع|مدفوعات|إعلان)\b",
    re.IGNORECASE | re.UNICODE,
)


class DynamicEngine:
    def __init__(self) -> None:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.OLLAMA_TEMPERATURE,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
            num_predict=settings.OLLAMA_NUM_PREDICT,
        )
        self._llm     = llm.bind_tools(ALL_TOOLS)
        self._by_name = {t.name: t for t in ALL_TOOLS}

    def answer(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
    ) -> str:
        msgs: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        # Historique limité aux 4 derniers échanges
        if history:
            msgs.extend(history[-4:])

        msgs.append(HumanMessage(content=query))

        lang = _detect_lang(query)
        last_tool_result: Optional[str] = None
        forced_tool_reminder = False

        for turn in range(3):
            response: AIMessage = self._llm.invoke(msgs)
            msgs.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            content    = (response.content or "").strip()

            # ── Cas 1 : Le LLM a répondu SANS appeler d'outil ──────────────
            if not tool_calls:

                # Shortcut : résultat du dernier tool déjà propre → direct
                if last_tool_result and _CLEAN_RESULT.match(last_tool_result):
                    print("[DynamicEngine] ✂️ Shortcut — résultat tool direct")
                    return _format_result(last_tool_result, lang)

                # On a une réponse texte valide ET pas de tool obligatoire
                if content and not _MUST_USE_TOOL.search(query):
                    return content

                # Le LLM a répondu en texte alors qu'il aurait dû appeler un outil
                # → on injecte un rappel UNE seule fois
                if content and _MUST_USE_TOOL.search(query) and not forced_tool_reminder:
                    print("[DynamicEngine] ⚠️ LLM a répondu sans tool — rappel injecté")
                    msgs.append(HumanMessage(content=_TOOL_REMINDER))
                    forced_tool_reminder = True
                    continue

                # Dernier tour sans outil ni rappel → on retourne ce qu'on a
                if content:
                    return content

                # Réponse vide
                return _no_data_msg(lang)

            # ── Cas 2 : Le LLM a appelé des outils ─────────────────────────
            for call in tool_calls:
                name = call["name"]
                args = call.get("args", {}) or {}
                tid  = call.get("id", "")

                print(f"[DynamicEngine] 🔧 {name}({args})")
                tool = self._by_name.get(name)

                if tool:
                    try:
                        result = str(tool.invoke(args))
                    except Exception as e:
                        result = f"Erreur lors de l'appel à {name} : {e}"
                        print(f"[DynamicEngine] ❌ {e}")
                else:
                    result = f"Outil inconnu : {name}"

                last_tool_result = result
                print(f"[DynamicEngine] 📤 {result[:200]}")
                msgs.append(
                    ToolMessage(content=result, tool_call_id=tid, name=name)
                )

        # Dernier tour : on laisse le LLM synthétiser
        final: AIMessage = self._llm.invoke(msgs)
        text = (final.content or "").strip()
        return text or _no_data_msg(lang)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _no_data_msg(lang: str) -> str:
    msgs = {
        "ar": "لم أتمكن من الحصول على المعلومات المطلوبة. يرجى المحاولة مجدداً.",
        "en": "I couldn't retrieve the requested information. Please try again.",
        "fr": "Je n'ai pas pu obtenir les informations demandées. Réessayez.",
    }
    return msgs.get(lang, msgs["fr"])


def _format_result(result: str, lang: str) -> str:
    """
    Nettoyage minimal du résultat brut d'un tool avant envoi direct.
    Le résultat est déjà lisible — on évite un 2e appel LLM coûteux.
    """
    return result.strip()


# ──────────────────────────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────────────────────────
_engine: Optional[DynamicEngine] = None


def get_dynamic_engine() -> DynamicEngine:
    global _engine
    if _engine is None:
        _engine = DynamicEngine()
    return _engine