"""
Moteur DYNAMIC — tool calling Supabase avec Ollama.

Optimisations v5 :
  - Prompt système réécrit pour qwen2.5:3b :
      · TRÈS COURT — le petit modèle se perd dans les longs prompts
      · Directif avec formatage explicite pour la mémoire
      · Anti-hallucination renforcé pour les souvenirs
  - num_predict=2048 pour réponses complètes
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
# Prompt TRÈS COURT pour qwen2.5:3b (le petit modèle se perd dans les longs prompts)
# ──────────────────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """Tu es un assistant scolaire.

RÈGLES :
1. menu/emploi/notes/absences/devoirs → appelle outil
2. Réponds en français, 1-4 lignes
3. Ne dis pas "je n'ai pas accès"
"""

# Prompt TRÈS COURT pour les questions générales et la mémoire
# Format: historique + règles strictes pour éviter les hallucinations
GENERAL_SYSTEM_PROMPT_TEMPLATE = """Tu es un assistant scolaire.

HISTORIQUE DE L'ÉLÈVE (ce dont vous avez réellement parlé) :
{memory_profile}

RÈGLES STRICTES :
1. Si l'élève demande ce dont on a parlé → liste EXACTEMENT les topics ci-dessus, n'invente rien.
2. Si l'élève pose une question de maths ou sciences → réponds directement avec les calculs.
3. Si le profil est vide → dis "Je n'ai pas d'historique de notre conversation."
4. Si la question est une salutation → réponds simplement "Bonjour !"
5. Réponds en français, max 5 lignes.
6. Ne mentionne PAS le mot "profil" ou "historique" dans ta réponse.
"""

# Rappel injecté si le LLM oublie d'appeler un outil
_TOOL_REMINDER = """STOP. Appelle l'outil maintenant."""

# Mots-clés qui doivent déclencher un outil
_MUST_USE_TOOL = re.compile(
    r"\b(menu|cantine|manger|repas|emploi|horaire|planning|"
    r"cours|note|résultat|moyenne|absence|devoir|réunion|paiement|annonce|"
    r"schedule|grade|homework|attendance|payment|meeting|food|lunch)\b",
    re.IGNORECASE | re.UNICODE,
)

# Mots-clés pour questions générales (inclut la mémoire)
_GENERAL_QUESTION = re.compile(
    r"\b("
    # Maths/Sciences
    r"math|maths|équation|calcul|fonction|dérivée|intégrale|théorème|"
    r"exercice|correction|problème|formule|"
    # Histoire/Géo
    r"guerre|histoire|géographie|politique|économie|"
    # MÉMOIRE - mots-clés étendus
    r"souviens|rappelle|conversation|précédemment|as-tu|t[' ]es|"
    r"dernière\s+discussion|dernier\s+topic|notre\s+discussion|"
    r"on\s+a\s+parlé|on\s+a\s+discuté|juste\s+avant|tout\s+à\s+l[' ]heure|"
    r"remember|recall|previous|conversation|last\s+time|we\s+talked|"
    r"last\s+discussion|what\s+did\s+we\s+talk|as-tu\s+souvenir|"
    # Arabe - mémoire
    r"تذكر|هل\s+تذكر|محادثة|آخر\s+مرة|تكلمنا|ناقشنا|سبق\s+وتحدثنا"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

# Mots qui indiquent une boucle bloquée
_BLOCKED_PHRASES = re.compile(
    r"(je ne peux pas|désol[ée]|ne peux pas fournir|sans avoir|utiliser l'outil|"
    r"sorry|cannot provide|without having|use the tool)",
    re.IGNORECASE | re.UNICODE,
)


class DynamicEngine:
    def __init__(self) -> None:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.OLLAMA_TEMPERATURE,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
            num_predict=2048,
        )
        self._llm     = llm.bind_tools(ALL_TOOLS)
        self._by_name = {t.name: t for t in ALL_TOOLS}

    def answer(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
        memory_profile: str = "",
    ) -> str:
        # Log mémoire pour débogage
        if memory_profile:
            print(f"[DynamicEngine] 📝 Profil mémoire reçu ({len(memory_profile)} chars)")
            # Afficher les 200 premiers caractères pour vérifier
            print(f"[DynamicEngine] 📝 Début du profil: {memory_profile[:200]}...")
        else:
            print("[DynamicEngine] ⚠️ Profil mémoire vide")

        # ──────────────────────────────────────────────────────────────
        # 1. Questions générales (maths, histoire, MÉMOIRE)
        #    → réponse directe avec le prompt court et le profil mémoire
        # ──────────────────────────────────────────────────────────────
        if _GENERAL_QUESTION.search(query):
            print(f"[DynamicEngine] 📚 Question générale (mémoire incluse): {query[:80]}")
            
            # Construire le prompt avec le profil mémoire
            if memory_profile:
                general_system = f"""Tu es un assistant scolaire. Réponds UNIQUEMENT depuis l'historique ci-dessous.

HISTORIQUE DE NOTRE CONVERSATION :
{memory_profile}

RÈGLES STRICTES :
1. Si on te demande le sujet de la dernière discussion, cite EXACTEMENT le dernier topic.
2. Si on te demande "de quoi on a parlé", liste tous les topics de l'historique.
3. Si la question est un calcul mathématique, réponds directement avec le résultat.
4. Si l'historique est vide ou ne contient pas l'info, dis "Je n'ai pas d'information sur ce sujet dans notre conversation."
5. Réponse courte, max 3 lignes.
6. Ne mentionne PAS le mot "historique" ou "profil" dans ta réponse."""
            else:
                general_system = """Tu es un assistant scolaire.

RÈGLES STRICTES :
1. Si l'élève demande ce dont on a parlé → dis "Je n'ai pas d'historique de notre conversation."
2. Si l'élève pose une question de maths ou sciences → réponds directement avec les calculs.
3. Si la question est une salutation → réponds simplement "Bonjour !"
4. Réponds en français, max 5 lignes."""
            
            msgs: List[BaseMessage] = [SystemMessage(content=general_system)]
            if history:
                msgs.extend(history[-4:])
            msgs.append(HumanMessage(content=query))
            
            try:
                response = self._llm.invoke(msgs)
                result = response.content or _no_data_msg(_detect_lang(query))
                print(f"[DynamicEngine] 📤 Réponse générale: {result[:100]}...")
                return result
            except Exception as e:
                print(f"[DynamicEngine] ❌ Erreur: {e}")
                return _no_data_msg(_detect_lang(query))

        # ──────────────────────────────────────────────────────────────
        # 2. Questions scolaires (menu, notes, etc.)
        #    → tool calling
        # ──────────────────────────────────────────────────────────────
        system = BASE_SYSTEM_PROMPT
        if memory_profile:
            # Pour le tool calling, on ajoute aussi le profil pour contexte
            system += f"\n\nHISTORIQUE ÉLÈVE: {memory_profile[:500]}"
        
        msgs: List[BaseMessage] = [SystemMessage(content=system)]

        if history:
            msgs.extend(history[-4:])

        msgs.append(HumanMessage(content=query))

        lang = _detect_lang(query)
        last_tool_result: Optional[str] = None
        forced_tool_reminder = False
        last_response_content = ""

        for turn in range(3):
            response: AIMessage = self._llm.invoke(msgs)
            msgs.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            content    = (response.content or "").strip()
            last_response_content = content

            if not tool_calls:
                # Shortcut résultat tool
                if last_tool_result and _CLEAN_RESULT.match(last_tool_result):
                    print("[DynamicEngine] ✂️ Shortcut")
                    return _format_result(last_tool_result, lang)

                # Détection boucle bloquée
                if _BLOCKED_PHRASES.search(content) and turn >= 1:
                    print("[DynamicEngine] 🚫 Boucle bloquée")
                    break

                # Réponse valide sans tool obligatoire
                if content and not _MUST_USE_TOOL.search(query):
                    return content

                # Rappel si tool manquant
                if content and _MUST_USE_TOOL.search(query) and not forced_tool_reminder:
                    print("[DynamicEngine] ⚠️ Rappel tool injecté")
                    msgs.append(HumanMessage(content=_TOOL_REMINDER))
                    forced_tool_reminder = True
                    continue

                if content:
                    return content

            else:
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
                            result = f"Erreur: {e}"
                            print(f"[DynamicEngine] ❌ {e}")
                    else:
                        result = f"Outil inconnu: {name}"

                    last_tool_result = result
                    print(f"[DynamicEngine] 📤 {result[:200]}")
                    msgs.append(ToolMessage(content=result, tool_call_id=tid, name=name))

        # ──────────────────────────────────────────────────────────────────
        # 3. Fallback final
        # ──────────────────────────────────────────────────────────────────
        print("[DynamicEngine] 💬 Fallback")
        
        if last_response_content and len(last_response_content) > 20:
            if not _BLOCKED_PHRASES.search(last_response_content):
                return last_response_content
        
        # Fallback simple
        fallback_prompt = "Réponds à cette question simplement et directement: " + query
        try:
            llm_fallback = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.3,
                keep_alive=settings.OLLAMA_KEEP_ALIVE,
                num_predict=1024,
            )
            final_response = llm_fallback.invoke([HumanMessage(content=fallback_prompt)])
            return final_response.content or _no_data_msg(lang)
        except Exception as e:
            print(f"[DynamicEngine] ❌ Erreur fallback: {e}")
            return _no_data_msg(lang)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _no_data_msg(lang: str) -> str:
    msgs = {
        "ar": "لم أتمكن من الحصول على المعلومات المطلوبة.",
        "en": "I couldn't retrieve the requested information.",
        "fr": "Je n'ai pas pu obtenir les informations demandées.",
    }
    return msgs.get(lang, msgs["fr"])


def _format_result(result: str, lang: str) -> str:
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