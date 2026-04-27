"""
Outils Supabase exposés au LLM via tool calling.

SÉCURITÉ CRITIQUE :
---------------------
Le `student_id` n'est JAMAIS passé en paramètre par le LLM. Il est injecté
depuis le contexte de la requête (JWT / session). Le LLM ne peut donc pas
demander les données d'un autre élève.

Chaque fonction est annotée avec un docstring détaillé : c'est ce que le
LLM lit pour décider quand l'appeler.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from langchain_core.tools import tool
from supabase import Client, create_client

from config import settings

# ============================================
# Client Supabase singleton
# ============================================
_client: Optional[Client] = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        settings.validate()
        _client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY
        )
    return _client


# ============================================
# Contexte d'appel (student_id injecté)
# ============================================
class StudentContext:
    """
    Contexte thread-local-style stockant l'identité de l'élève courant.
    Utilisé par les tools pour filtrer les données sans que le LLM
    ait besoin (ou le droit) de manipuler le student_id.
    """

    _current_student_id: Optional[str] = None
    _current_student_id: Optional[str] = None
    _current_class_id: Optional[str] = None  # cache : rempli à la demande via Supabase

    @classmethod
    def set(cls, student_id: str, class_id: Optional[str] = None) -> None:
        """
        Initialise le contexte pour une requête.
        - student_id est obligatoire
        - class_id est optionnel : si non fourni, il sera récupéré automatiquement
          depuis Supabase au moment où un tool en a besoin (lazy loading).
        """
        cls._current_student_id = student_id
        cls._current_class_id = class_id  # peut être None, sera rempli à la demande

    @classmethod
    def get_student_id(cls) -> str:
        if not cls._current_student_id:
            raise RuntimeError("StudentContext non initialisé.")
        return cls._current_student_id

    @classmethod
    def get_class_id(cls) -> Optional[str]:
        """
        Retourne le class_id. S'il n'a pas été fourni explicitement,
        va le chercher dans Supabase (une seule fois, puis met en cache).
        """
        if cls._current_class_id is not None:
            return cls._current_class_id

        if cls._current_student_id is None:
            return None

        # Lazy fetch depuis Supabase
        try:
            sb = get_supabase()
            res = (
                sb.table("students")
                .select("class_id")
                .eq("id", cls._current_student_id)
                .single()
                .execute()
            )
            class_id = (res.data or {}).get("class_id")
            cls._current_class_id = class_id  # cache pour les appels suivants
            return class_id
        except Exception as e:
            print(f"[StudentContext] Impossible de récupérer class_id : {e}")
            return None

    @classmethod
    def clear(cls) -> None:
        cls._current_student_id = None
        cls._current_class_id = None


# ============================================
# Helpers
# ============================================
DAYS_FR = {
    0: "lundi", 1: "mardi", 2: "mercredi", 3: "jeudi",
    4: "vendredi", 5: "samedi", 6: "dimanche",
}

DAYS_ENUM = {
    0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday",
    4: "friday", 5: "saturday", 6: "sunday",
}

# Mapping nom de jour → index (0 = lundi, 6 = dimanche)
# On accepte français, anglais, arabe standard, dialecte tunisien
DAY_NAME_TO_INDEX = {
    # Lundi
    "lundi": 0, "monday": 0, "mon": 0,
    "الاثنين": 0, "الإثنين": 0, "الاتنين": 0,
    "tnin": 0, "lethnine": 0, "lethnin": 0,
    # Mardi
    "mardi": 1, "tuesday": 1, "tue": 1, "tues": 1,
    "الثلاثاء": 1, "الثلاثا": 1,
    "thlatha": 1, "tlata": 1,
    # Mercredi
    "mercredi": 2, "wednesday": 2, "wed": 2,
    "الأربعاء": 2, "الاربعاء": 2,
    "lerba3": 2, "larbaa": 2,
    # Jeudi
    "jeudi": 3, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "الخميس": 3,
    "lekhmis": 3, "khmiss": 3,
    # Vendredi
    "vendredi": 4, "friday": 4, "fri": 4,
    "الجمعة": 4, "الجمعه": 4,
    "jem3a": 4, "jomaa": 4,
    # Samedi
    "samedi": 5, "saturday": 5, "sat": 5,
    "السبت": 5,
    "sebt": 5,
    # Dimanche
    "dimanche": 6, "sunday": 6, "sun": 6,
    "الأحد": 6, "الاحد": 6,
    "lahad": 6, "el a7ad": 6,
}


def _parse_date(date_str: Optional[str]) -> date:
    """
    Parse une date depuis plusieurs formats :
    - Mots-clés : 'today', 'tomorrow', 'yesterday', équivalents fr/ar
    - Format ISO : 'YYYY-MM-DD'
    - Nom d'un jour : 'lundi', 'monday', 'الاثنين'... (= prochain jour correspondant)
    - Nom d'un jour + modificateur : 'lundi prochain', 'ce lundi', 'next monday'
    """
    if not date_str:
        return date.today()

    s = date_str.strip().lower()

    # Aujourd'hui
    if s in ("today", "aujourd'hui", "aujourdhui", "aujourd’hui", "اليوم", "lyoum", "el yom"):
        return date.today()

    # Demain
    if s in ("tomorrow", "demain", "غدا", "غداً", "غدًا", "ghodwa", "ghodowa"):
        return date.today() + timedelta(days=1)

    # Hier
    if s in ("yesterday", "hier", "أمس", "امس", "elbareh", "lbereh"):
        return date.today() - timedelta(days=1)

    # Format ISO
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        pass

    # Nom de jour (avec modificateurs optionnels : "lundi prochain", "ce lundi"...)
    # On retire les modificateurs pour ne garder que le nom du jour
    MODIFIERS = {
        "prochain", "prochaine", "next",
        "ce", "cette", "this",
        "dernier", "dernière", "last", "passé", "passée",
        "القادم", "الماضي", "هذا",
    }
    words = s.replace("-", " ").split()
    day_word = None
    has_last_modifier = any(m in words for m in ("dernier", "dernière", "last", "passé", "passée", "الماضي"))

    for word in words:
        if word in DAY_NAME_TO_INDEX:
            day_word = word
            break

    if day_word is not None:
        target_idx = DAY_NAME_TO_INDEX[day_word]
        today = date.today()
        today_idx = today.weekday()

        if has_last_modifier:
            # Jour passé de la semaine en cours ou précédente
            days_ago = (today_idx - target_idx) % 7
            if days_ago == 0:
                days_ago = 7  # même jour = celui de la semaine passée
            return today - timedelta(days=days_ago)
        else:
            # Par défaut : prochaine occurrence (y compris aujourd'hui si c'est le bon jour)
            days_ahead = (target_idx - today_idx) % 7
            if days_ahead == 0:
                days_ahead = 7  # même jour = celui de la semaine prochaine
            return today + timedelta(days=days_ahead)

    raise ValueError(
        f"Format de date non reconnu : '{date_str}'. "
        "Utilise 'today', 'tomorrow', 'YYYY-MM-DD', ou un nom de jour "
        "('lundi', 'monday', 'lundi prochain'...)."
    )


# ============================================
# TOOLS — chaque fonction est exposée au LLM
# ============================================

@tool
def get_student_schedule(date_str: Optional[str] = None) -> str:
    """
    Retourne l'emploi du temps de l'élève pour un jour donné.

    Args:
        date_str: La date, acceptant de NOMBREUX formats :
          - Format ISO : '2026-04-27'
          - Relatif : 'today', 'aujourd''hui', 'tomorrow', 'demain', 'yesterday', 'hier'
          - Nom de jour : 'lundi', 'monday', 'الاثنين' → prochain jour correspondant
          - Nom + modificateur : 'lundi prochain', 'ce lundi', 'next monday',
            'lundi dernier', 'last monday'
          - Dialecte tunisien : 'ghodwa' (demain), 'lyoum' (aujourd'hui)
          Par défaut = aujourd'hui.

    À utiliser pour répondre aux questions :
    - "Est-ce que j'ai cours demain ?"
    - "Quel est mon emploi du temps aujourd'hui ?"
    - "À quelle heure commence mon premier cours ?"
    """
    target_date = _parse_date(date_str)
    day_of_week = DAYS_ENUM[target_date.weekday()]

    # Récupère le class_id (cache ou lazy fetch depuis Supabase)
    class_id = StudentContext.get_class_id()
    if not class_id:
        return "Aucune classe n'est associée à cet élève."

    sb = get_supabase()

    # Emploi du temps
    slots = (
        sb.table("schedule_slots")
        .select("start_time, end_time, room, subjects(name), teachers(profile_id, profiles(first_name, last_name))")
        .eq("class_id", class_id)
        .eq("day_of_week", day_of_week)
        .eq("is_active", True)
        .order("start_time")
        .execute()
    )

    if not slots.data:
        return f"Aucun cours prévu pour {DAYS_FR[target_date.weekday()]} {target_date.isoformat()}."

    lines = [f"Emploi du temps du {DAYS_FR[target_date.weekday()]} {target_date.isoformat()} :"]
    for s in slots.data:
        subject = (s.get("subjects") or {}).get("name", "?")
        teacher = s.get("teachers") or {}
        prof = (teacher.get("profiles") or {})
        teacher_name = f"{prof.get('first_name', '')} {prof.get('last_name', '')}".strip() or "prof inconnu"
        room = s.get("room") or "?"
        lines.append(
            f"- {s['start_time'][:5]}–{s['end_time'][:5]} : {subject} "
            f"(salle {room}, {teacher_name})"
        )
    return "\n".join(lines)


@tool
def get_canteen_menu(date_str: Optional[str] = None) -> str:
    """
    Retourne le menu de la cantine pour un jour donné.

    Args:
        date_str: La date, acceptant de NOMBREUX formats :
          - Format ISO : '2026-04-27'
          - Relatif : 'today', 'aujourd''hui', 'tomorrow', 'demain', 'yesterday', 'hier'
          - Nom de jour : 'lundi', 'monday', 'الاثنين' → prochain jour correspondant
          - Nom + modificateur : 'lundi prochain', 'ce lundi', 'lundi dernier'
          - Dialecte tunisien : 'ghodwa', 'lyoum'
          Par défaut = aujourd'hui.

    À utiliser pour répondre aux questions :
    - "Quel est le menu d'aujourd'hui ?"
    - "Qu'est-ce qu'on mange demain à la cantine ?"
    - "Quel est le menu du lundi ?" → date_str='lundi'
    - "Menu de lundi prochain ?" → date_str='lundi prochain'
    """
    target_date = _parse_date(date_str)
    sb = get_supabase()

    res = (
        sb.table("canteen_menus")
        .select("*")
        .eq("date", target_date.isoformat())
        .limit(1)
        .execute()
    )

    if not res.data:
        return f"Aucun menu n'est disponible pour le {target_date.isoformat()}."

    m = res.data[0]
    parts = [f"Menu du {target_date.isoformat()} :"]

    starters = m.get("starters") or ([m["starter"]] if m.get("starter") else [])
    if starters:
        parts.append(f"- Entrée : {', '.join(starters)}")

    mains = m.get("main_courses") or ([m["main_course"]] if m.get("main_course") else [])
    if mains:
        parts.append(f"- Plat : {', '.join(mains)}")

    if m.get("side_dish"):
        parts.append(f"- Accompagnement : {m['side_dish']}")

    desserts = m.get("desserts_list") or ([m["dessert"]] if m.get("dessert") else [])
    if desserts:
        parts.append(f"- Dessert : {', '.join(desserts)}")

    if m.get("allergens"):
        parts.append(f"- Allergènes : {', '.join(m['allergens'])}")

    return "\n".join(parts)


@tool
def get_student_grades(subject_name: Optional[str] = None, period: Optional[str] = None) -> str:
    """
    Retourne les notes de l'élève.

    Args:
        subject_name: (optionnel) filtrer par matière, ex: 'Mathématiques', 'Français'.
        period: (optionnel) filtrer par période, ex: 'trimester_1', 'trimester_2'.

    À utiliser pour répondre aux questions :
    - "Quelles sont mes notes ?"
    - "Quelle est ma moyenne en maths ?"
    - "Mes notes du trimestre 2 ?"
    """
    sb = get_supabase()
    student_id = StudentContext.get_student_id()

    query = (
        sb.table("grades")
        .select("score, max_score, coefficient, title, period, grade_date, subjects(name)")
        .eq("student_id", student_id)
        .order("grade_date", desc=True)
        .limit(30)
    )

    if period:
        query = query.eq("period", period)

    res = query.execute()
    rows = res.data or []

    # Filtrage matière côté Python (plus tolérant qu'un LIKE SQL)
    if subject_name:
        needle = subject_name.lower()
        rows = [
            r for r in rows
            if needle in ((r.get("subjects") or {}).get("name", "").lower())
        ]

    if not rows:
        return "Aucune note trouvée pour ces critères."

    # Calcul de la moyenne pondérée
    total_weighted = sum(float(r["score"]) * float(r.get("coefficient") or 1) for r in rows)
    total_coef = sum(float(r.get("coefficient") or 1) for r in rows)
    avg = total_weighted / total_coef if total_coef else 0

    lines = [f"{len(rows)} note(s) trouvée(s). Moyenne pondérée : {avg:.2f}/20"]
    for r in rows[:10]:
        subject = (r.get("subjects") or {}).get("name", "?")
        title = r.get("title") or "évaluation"
        lines.append(
            f"- {r['grade_date']} · {subject} · {title} : "
            f"{r['score']}/{r.get('max_score', 20)} (coef {r.get('coefficient', 1)})"
        )
    if len(rows) > 10:
        lines.append(f"... et {len(rows) - 10} autres.")
    return "\n".join(lines)


@tool
def get_student_assignments(upcoming_only: bool = True) -> str:
    """
    Retourne la liste des devoirs/travaux de l'élève.

    Args:
        upcoming_only: si True, ne retourne que les devoirs non encore rendus.

    À utiliser pour répondre aux questions :
    - "Quels devoirs dois-je rendre ?"
    - "Y a-t-il des devoirs pour cette semaine ?"
    """
    sb = get_supabase()

    # Récupère le class_id (cache ou lazy fetch)
    class_id = StudentContext.get_class_id()
    if not class_id:
        return "Aucune classe n'est associée à cet élève."

    query = (
        sb.table("assignments")
        .select("id, title, description, type, due_date, subjects(name)")
        .eq("class_id", class_id)
        .order("due_date")
        .limit(20)
    )
    if upcoming_only:
        query = query.gte("due_date", datetime.now().isoformat())

    res = query.execute()
    if not res.data:
        return "Aucun devoir à rendre." if upcoming_only else "Aucun devoir trouvé."

    lines = [f"{len(res.data)} devoir(s) :"]
    for a in res.data:
        subj = (a.get("subjects") or {}).get("name", "?")
        due = a.get("due_date", "").split("T")[0]
        lines.append(f"- {due} · {subj} · {a.get('type', 'homework')} : {a['title']}")
    return "\n".join(lines)


@tool
def get_student_attendance(from_date: Optional[str] = None) -> str:
    """
    Retourne les absences/retards récents de l'élève.

    Args:
        from_date: date de départ au format 'YYYY-MM-DD'. Par défaut = 30 derniers jours.

    À utiliser pour :
    - "Ai-je des absences ?"
    - "Combien de retards ce mois-ci ?"
    """
    sb = get_supabase()
    student_id = StudentContext.get_student_id()

    start = (
        _parse_date(from_date) if from_date else (date.today() - timedelta(days=30))
    )

    res = (
        sb.table("attendance")
        .select("date, status, reason")
        .eq("student_id", student_id)
        .gte("date", start.isoformat())
        .order("date", desc=True)
        .execute()
    )

    rows = res.data or []
    if not rows:
        return f"Aucune absence/retard depuis le {start.isoformat()}."

    # Compteurs
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    summary = ", ".join(f"{v} {k}" for k, v in counts.items())
    lines = [f"Depuis le {start.isoformat()} : {summary}."]
    for r in rows[:10]:
        reason = f" (raison : {r['reason']})" if r.get("reason") else ""
        lines.append(f"- {r['date']} · {r['status']}{reason}")
    return "\n".join(lines)


@tool
def get_announcements(limit: int = 5) -> str:
    """
    Retourne les dernières annonces pour la classe de l'élève.

    Args:
        limit: nombre maximum d'annonces à retourner (défaut 5).

    À utiliser pour :
    - "Y a-t-il des annonces ?"
    - "Quelles sont les actualités ?"
    """
    sb = get_supabase()
    class_id = StudentContext.get_class_id()

    query = (
        sb.table("announcements")
        .select("title, content, is_pinned, published_at")
        .order("is_pinned", desc=True)
        .order("published_at", desc=True)
        .limit(limit)
    )

    # On filtre sur les annonces de la classe OU les annonces générales (class_id null)
    if class_id:
        query = query.or_(f"class_id.eq.{class_id},class_id.is.null")
    else:
        query = query.is_("class_id", "null")

    res = query.execute()
    if not res.data:
        return "Aucune annonce récente."

    lines = ["Dernières annonces :"]
    for a in res.data:
        pin = "📌 " if a.get("is_pinned") else ""
        date_str = (a.get("published_at") or "").split("T")[0]
        lines.append(f"{pin}[{date_str}] {a['title']} — {a['content'][:200]}")
    return "\n".join(lines)


@tool
def get_student_meetings(status: Optional[str] = None, upcoming_only: bool = True) -> str:
    """
    Retourne les réunions parents-professeurs concernant l'élève.

    Args:
        status: (optionnel) filtrer par statut : 'requested', 'confirmed',
                'cancelled', 'completed'.
        upcoming_only: si True (défaut), ne retourne que les réunions à venir.

    À utiliser pour répondre aux questions :
    - "Mes parents ont-ils des réunions prévues avec mes profs ?"
    - "Quand est la prochaine rencontre parents-profs ?"
    - "Y a-t-il une réunion de classe bientôt ?"
    - "شكون من الأستاذة طلب إجتماع مع والديا؟"
    """
    sb = get_supabase()
    student_id = StudentContext.get_student_id()
    class_id = StudentContext.get_class_id()

    # Deux cas : réunions individuelles (student_id) OU réunions de classe (class_id)
    query_individual = (
        sb.table("meetings")
        .select(
            "id, status, scheduled_at, duration_minutes, location, notes, "
            "is_class_meeting, teachers(profile_id, profiles(first_name, last_name))"
        )
        .eq("student_id", student_id)
        .order("scheduled_at", desc=False)
        .limit(20)
    )
    if status:
        query_individual = query_individual.eq("status", status)
    if upcoming_only:
        query_individual = query_individual.gte("scheduled_at", datetime.now().isoformat())

    rows = (query_individual.execute().data) or []

    # Réunions de classe (is_class_meeting=True sur la classe de l'élève)
    if class_id:
        query_class = (
            sb.table("meetings")
            .select(
                "id, status, scheduled_at, duration_minutes, location, notes, "
                "is_class_meeting, teachers(profile_id, profiles(first_name, last_name))"
            )
            .eq("class_id", class_id)
            .eq("is_class_meeting", True)
            .order("scheduled_at", desc=False)
            .limit(20)
        )
        if status:
            query_class = query_class.eq("status", status)
        if upcoming_only:
            query_class = query_class.gte("scheduled_at", datetime.now().isoformat())

        class_rows = (query_class.execute().data) or []
        existing_ids = {r["id"] for r in rows}
        rows.extend(r for r in class_rows if r["id"] not in existing_ids)

    if not rows:
        return "Aucune réunion parents-professeurs prévue." if upcoming_only else "Aucune réunion trouvée."

    rows.sort(key=lambda r: r.get("scheduled_at") or "")

    lines = [f"{len(rows)} réunion(s) :"]
    for m in rows:
        when = (m.get("scheduled_at") or "?").replace("T", " ")[:16]
        status_str = m.get("status", "?")
        duration = m.get("duration_minutes", 30)
        location = m.get("location") or "lieu non précisé"
        teacher = m.get("teachers") or {}
        prof = teacher.get("profiles") or {}
        teacher_name = f"{prof.get('first_name', '')} {prof.get('last_name', '')}".strip() or "prof inconnu"
        meeting_type = "réunion de classe" if m.get("is_class_meeting") else f"avec {teacher_name}"

        lines.append(
            f"- {when} · {meeting_type} · {duration} min · {location} · [{status_str}]"
        )
        if m.get("notes"):
            lines.append(f"  note : {m['notes'][:100]}")

    return "\n".join(lines)


@tool
def get_student_payments(status: Optional[str] = None) -> str:
    """
    Retourne les paiements de l'élève.

    Args:
        status: (optionnel) filtrer par statut : 'pending', 'paid', 'overdue'.

    À utiliser pour :
    - "Ai-je des paiements en attente ?"
    - "Quels sont mes frais de scolarité ?"
    """
    sb = get_supabase()
    student_id = StudentContext.get_student_id()

    query = (
        sb.table("payments")
        .select("type, amount, status, description, due_date, paid_at")
        .eq("student_id", student_id)
        .order("due_date", desc=True)
    )
    if status:
        query = query.eq("status", status)

    res = query.execute()
    if not res.data:
        return "Aucun paiement trouvé."

    lines = [f"{len(res.data)} paiement(s) :"]
    for p in res.data:
        lines.append(
            f"- {p.get('due_date', '?')} · {p['type']} · {p['amount']} TND "
            f"· [{p['status']}] {p.get('description', '')}"
        )
    return "\n".join(lines)


# Liste de tous les tools exportés vers le LLM
ALL_TOOLS = [
    get_student_schedule,
    get_canteen_menu,
    get_student_grades,
    get_student_assignments,
    get_student_attendance,
    get_announcements,
    get_student_meetings,
    get_student_payments,
]