"""
Gestion de l'historique des conversations (persistance Supabase).

Tables utilisées :
  - chatbot_conversations(id, student_id, title, created_at, updated_at)
  - chatbot_messages(id, conversation_id, role, content, route, citations, created_at)

Titre de conversation généré automatiquement par le LLM après le 1er échange.
"""
from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import settings
from tools.supabase_tools import get_supabase

# ============================================
# LLM pour la génération de titre
# ============================================
_title_llm: Optional[ChatOllama] = None


def _get_title_llm() -> ChatOllama:
    global _title_llm
    if _title_llm is None:
        _title_llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.3,
            num_predict=30,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
        )
    return _title_llm


TITLE_SYSTEM_PROMPT = """Tu reçois une QUESTION d'élève et la RÉPONSE du chatbot.
Génère un TITRE COURT (3-6 mots max) qui résume le sujet de l'échange.

Règles :
- Dans la langue de la question (français, anglais, arabe, dialecte tunisien).
- Pas de guillemets, pas de point final.
- Sois descriptif mais concis.

Exemples :
- Q: "quel est le menu de demain?" → "Menu de demain"
- Q: "mes notes en maths?" → "Notes de maths"
- Q: "comment contacter l'école?" → "Contact de l'école"
- Q: "explique moi la photosynthèse" → "Explication photosynthèse"

Réponds UNIQUEMENT par le titre, rien d'autre."""


def generate_title(question: str, answer: str) -> str:
    """Génère un titre de conversation à partir du 1er échange."""
    try:
        llm = _get_title_llm()
        response = llm.invoke([
            SystemMessage(content=TITLE_SYSTEM_PROMPT),
            HumanMessage(content=f"QUESTION: {question}\n\nRÉPONSE: {answer[:500]}"),
        ])
        title = (response.content or "").strip()
        # Nettoyage : enlève les guillemets, point final, limite la longueur
        title = title.strip('"\'«».').strip()
        if len(title) > 80:
            title = title[:80]
        return title or "Conversation sans titre"
    except Exception as e:
        print(f"[generate_title] Erreur : {e}")
        # Fallback : les 50 premiers caractères de la question
        fallback = question[:50].strip()
        if len(question) > 50:
            fallback += "..."
        return fallback


# ============================================
# CRUD conversations
# ============================================
def create_conversation(student_id: str, first_question: Optional[str] = None) -> dict:
    """Crée une nouvelle conversation vide."""
    sb = get_supabase()
    initial_title = (first_question[:50] + "..." if first_question and len(first_question) > 50
                     else first_question) if first_question else "Nouvelle conversation"

    res = (
        sb.table("chatbot_conversations")
        .insert({
            "student_id": student_id,
            "title": initial_title,
        })
        .execute()
    )
    return res.data[0] if res.data else None


def list_conversations(student_id: str, limit: int = 50) -> List[dict]:
    """Liste les conversations d'un élève, plus récentes d'abord."""
    sb = get_supabase()
    res = (
        sb.table("chatbot_conversations")
        .select("id, title, created_at, updated_at")
        .eq("student_id", student_id)
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def get_conversation_messages(conversation_id: str, student_id: str) -> List[dict]:
    """Récupère les messages d'une conversation, en vérifiant qu'elle appartient à l'élève."""
    sb = get_supabase()

    # Vérifier que la conversation appartient à l'élève (sécurité)
    conv = (
        sb.table("chatbot_conversations")
        .select("id")
        .eq("id", conversation_id)
        .eq("student_id", student_id)
        .limit(1)
        .execute()
    )
    if not conv.data:
        return []

    res = (
        sb.table("chatbot_messages")
        .select("id, role, content, route, citations, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .execute()
    )
    return res.data or []


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    route: Optional[str] = None,
    citations: Optional[list] = None,
) -> dict:
    """Sauvegarde un message dans une conversation."""
    sb = get_supabase()
    data = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
    }
    if route:
        data["route"] = route
    if citations:
        data["citations"] = json.dumps(citations)

    res = sb.table("chatbot_messages").insert(data).execute()
    return res.data[0] if res.data else None


def update_title(conversation_id: str, student_id: str, new_title: str) -> bool:
    """Met à jour le titre d'une conversation (vérifie l'ownership)."""
    sb = get_supabase()
    res = (
        sb.table("chatbot_conversations")
        .update({"title": new_title})
        .eq("id", conversation_id)
        .eq("student_id", student_id)
        .execute()
    )
    return bool(res.data)


def delete_conversation(conversation_id: str, student_id: str) -> bool:
    """Supprime une conversation (et ses messages via CASCADE)."""
    sb = get_supabase()
    res = (
        sb.table("chatbot_conversations")
        .delete()
        .eq("id", conversation_id)
        .eq("student_id", student_id)
        .execute()
    )
    return bool(res.data)


def get_conversation_message_count(conversation_id: str) -> int:
    """Retourne le nombre de messages dans une conversation."""
    sb = get_supabase()
    res = (
        sb.table("chatbot_messages")
        .select("id", count="exact")
        .eq("conversation_id", conversation_id)
        .execute()
    )
    return res.count or 0


def rebuild_history_from_db(conversation_id: str, student_id: str, max_messages: int = 6) -> list:
    """
    Reconstruit l'historique LangChain depuis la BD pour le passer au LLM.
    Ne garde que les N derniers messages pour ne pas surcharger le contexte.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    messages = get_conversation_messages(conversation_id, student_id)
    # Derniers messages seulement
    messages = messages[-max_messages:] if len(messages) > max_messages else messages

    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history
