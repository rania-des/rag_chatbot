"""
Moteur DYNAMIC — tool calling Supabase avec Ollama.

Optimisations v4 :
  - Prompt court et direct (moins de tokens = plus rapide)
  - 0 retry (le retry doublait le temps)
  - Shortcut : résultat tool déjà propre → renvoi direct sans LLM
  - num_predict limité à 300 tokens
  - keep_alive = -1 (modèle permanent en RAM)
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

# Résultats de tool déjà bien formatés → renvoi direct sans 2e LLM
_CLEAN_RESULT = re.compile(
    r"^(Aucun|Menu du|Emploi du temps|Emploi du temps du|\d+ note|\d+ devoir|"
    r"\d+ réunion|\d+ paiement|Depuis le|Dernières|Il n'y a|No |"
    r"\u0644\u0627 |\u0645\u062a\u0627\u062d)",   # "لا " et "متاح" en arabe
    re.IGNORECASE,
)

SYSTEM_PROMPT = """Tu es l'assistant scolaire d'une école.
RÈGLE ABSOLUE : pour toute question sur le menu, emploi du temps, notes, absences, devoirs, réunions ou paiements, tu DOIS appeler l'outil correspondant. Ne réponds JAMAIS sans avoir appelé un outil pour ces sujets.
Après avoir reçu le résultat de l'outil :
- Réponds dans la langue de la question (fr/ar/en).
- Sois concis : 1-4 lignes.
- Si le résultat dit "Aucun", répète-le sans inventer.
"""


class DynamicEngine:
    def __init__(self) -> None:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.OLLAMA_TEMPERATURE,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
            num_predict=settings.OLLAMA_NUM_PREDICT,
        )
        self._llm    = llm.bind_tools(ALL_TOOLS)
        self._by_name = {t.name: t for t in ALL_TOOLS}

    def answer(self, query: str, history: Optional[List[BaseMessage]] = None) -> str:
        msgs: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        if history:
            msgs.extend(history[-4:])   # max 4 messages d'historique
        msgs.append(HumanMessage(content=query))

        last_tool_result: Optional[str] = None

        for turn in range(3):
            response: AIMessage = self._llm.invoke(msgs)
            msgs.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # Shortcut : résultat du dernier tool déjà propre
                if last_tool_result and _CLEAN_RESULT.match(last_tool_result):
                    print(f"[DynamicEngine] ✂️ Shortcut — résultat direct")
                    return last_tool_result
                content = (response.content or "").strip()
                if content:
                    return content
                return "Je n'ai pas pu répondre à cette question."

            # Exécuter les outils
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
                        result = f"Erreur {name} : {e}"
                        print(f"[DynamicEngine] ❌ {e}")
                else:
                    result = f"Outil inconnu : {name}"

                last_tool_result = result
                print(f"[DynamicEngine] 📤 {result[:150]}")
                msgs.append(ToolMessage(content=result, tool_call_id=tid, name=name))

        # Dernier tour
        final = self._llm.invoke(msgs)
        return (final.content or "").strip() or "Je n'ai pas pu répondre."


_engine: Optional[DynamicEngine] = None

def get_dynamic_engine() -> DynamicEngine:
    global _engine
    if _engine is None:
        _engine = DynamicEngine()
    return _engine