# rag_chatbot/engine/agenda_engine.py
import asyncio
import json
from datetime import datetime, timedelta
from config import settings
from supabase import create_client
import httpx

supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

async def generate_weekly_agenda(student_id: str, memory_profile: str = "") -> dict:
    """Agrège les données élève et génère un planning via LLM."""

    today = datetime.now().date()
    next_week = today + timedelta(days=7)

    # ── Récupérer les données existantes (async to_thread pour Supabase) ──
    def _fetch_data():
        try:
            # FIX: Utiliser subject_id au lieu de subject
            assignments = supabase.table("assignments") \
                .select("title, due_date, subject_id") \
                .eq("student_id", student_id) \
                .gte("due_date", today.isoformat()) \
                .lte("due_date", next_week.isoformat()) \
                .execute().data or []

            grades = supabase.table("grades") \
                .select("subject, grade, max_grade") \
                .eq("student_id", student_id) \
                .order("created_at", desc=True) \
                .limit(10) \
                .execute().data or []

            absences = supabase.table("attendance") \
                .select("date, subject, status") \
                .eq("student_id", student_id) \
                .gte("date", (today - timedelta(days=14)).isoformat()) \
                .execute().data or []
            
            return assignments, grades, absences
        except Exception as e:
            print(f"[Agenda] Erreur fetch data: {e}")
            return [], [], []

    assignments, grades, absences = await asyncio.to_thread(_fetch_data)

    # ── Prompt structuré ──────────────────────────────────────────────
    prompt = f"""Tu es un assistant scolaire. Génère un planning de révision pour la semaine.

Données de l'élève :
- Devoirs à rendre : {json.dumps(assignments, ensure_ascii=False)}
- Notes récentes : {json.dumps(grades, ensure_ascii=False)}
- Absences récentes : {json.dumps(absences, ensure_ascii=False)}
{f'- Profil : {memory_profile}' if memory_profile else ''}

Réponds UNIQUEMENT en JSON valide, format exact :
{{
  "planning": [
    {{
      "jour": "Lundi",
      "date": "2024-01-15",
      "taches": [
        {{"heure": "18h", "duree": "45min", "matiere": "Maths", "description": "Réviser les équations"}}
      ]
    }}
  ],
  "conseils": ["conseil 1", "conseil 2"]
}}"""

    # ── Appel LLM ─────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                }
            )
            raw = res.json()["message"]["content"]
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[Agenda] JSON invalide: {e}")
                print(f"[Agenda] Raw response: {raw[:500]}")
                return {
                    "planning": [],
                    "conseils": ["Impossible de générer le planning pour l'instant."]
                }
    except Exception as e:
        print(f"[Agenda] Erreur appel LLM: {e}")
        return {
            "planning": [],
            "conseils": ["Service de planning temporairement indisponible."]
        }