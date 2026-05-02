# rag_chatbot/memory.py
import asyncio
from datetime import datetime
from config import settings
from supabase import create_client

supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

MAX_MEMORY_ENTRIES = 20

async def get_student_memory(student_id: str) -> str:
    """Retourne le profil mémoire formaté pour le system prompt."""
    def _sync():
        try:
            return supabase.table("student_memory") \
                .select("topic, difficulty, tone, note, last_seen") \
                .eq("student_id", student_id) \
                .order("last_seen", desc=True) \
                .limit(MAX_MEMORY_ENTRIES) \
                .execute()
        except Exception as e:
            print(f"[Memory] Erreur get _sync: {e}")
            return None

    try:
        res = await asyncio.to_thread(_sync)
        
        if not res or not res.data:
            return ""

        # MODIFICATION 2 & 3: Plus de filtre sur difficulty + format amélioré
        lines = []
        for m in res.data:
            # On garde TOUTES les entrées, plus de filtre difficulty
            topic = m.get("topic", "")
            note = m.get("note", "")
            
            if note:
                # Si la note existe, on l'inclut
                lines.append(f"- {topic} : {note}")
            else:
                # Sinon juste le topic
                lines.append(f"- {topic}")

        if not lines:
            return ""

        # MODIFICATION 3: Format plus clair et utile pour le LLM
        profile = "Dernières discussions de l'élève (du plus récent au plus ancien) :\n"
        profile += "\n".join(lines)
        return profile

    except Exception as e:
        print(f"[Memory] Erreur get: {e}")
        return ""


async def update_student_memory(
    student_id: str,
    topic: str,
    difficulty: str = "medium",
    tone: str = "neutral",
    note: str = ""
):
    """Met à jour ou crée une entrée mémoire. Appelé de façon asynchrone."""
    def _sync():
        try:
            supabase.table("student_memory").upsert({
                "student_id": student_id,
                "topic": topic[:200],  # Limiter la longueur
                "difficulty": difficulty,
                "tone": tone,
                "note": note[:500],  # Limiter la longueur
                "last_seen": datetime.utcnow().isoformat(),
            }, on_conflict="student_id,topic").execute()

            # Garder max 20 entrées — supprimer les plus anciennes
            all_entries = supabase.table("student_memory") \
                .select("id, last_seen") \
                .eq("student_id", student_id) \
                .order("last_seen", desc=True) \
                .execute()

            if len(all_entries.data) > MAX_MEMORY_ENTRIES:
                ids_to_delete = [e["id"] for e in all_entries.data[MAX_MEMORY_ENTRIES:]]
                supabase.table("student_memory") \
                    .delete().in_("id", ids_to_delete).execute()
        except Exception as e:
            print(f"[Memory] Erreur update _sync: {e}")

    try:
        await asyncio.to_thread(_sync)
    except Exception as e:
        print(f"[Memory] Erreur update: {e}")