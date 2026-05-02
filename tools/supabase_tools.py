"""
Outils Supabase exposés au LLM via tool calling.

SÉCURITÉ CRITIQUE :
---------------------
Le `student_id` n'est JAMAIS passé en paramètre par le LLM. Il est injecté
depuis le contexte de la requête (JWT / session). Le LLM ne peut donc pas
demander les données d'un autre élève.

RÈGLE ANTI-HALLUCINATION :
---------------------------
Chaque tool retourne exactement les données de Supabase.
Le LLM NE DOIT PAS inventer ou compléter — il reformule uniquement ce qui est retourné.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional, Tuple

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
        _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    return _client


# ============================================
# Contexte d'appel (student_id injecté)
# ============================================
class StudentContext:
    _current_student_id: Optional[str] = None
    _current_class_id: Optional[str] = None

    @classmethod
    def set(cls, student_id: str, class_id: Optional[str] = None) -> None:
        cls._current_student_id = student_id
        cls._current_class_id = class_id

    @classmethod
    def get_student_id(cls) -> str:
        if not cls._current_student_id:
            raise RuntimeError("StudentContext non initialisé.")
        return cls._current_student_id

    @classmethod
    def get_class_id(cls) -> Optional[str]:
        if cls._current_class_id is not None:
            return cls._current_class_id
        if cls._current_student_id is None:
            return None
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
            cls._current_class_id = class_id
            return class_id
        except Exception as e:
            print(f"[StudentContext] Impossible de récupérer class_id : {e}")
            return None

    @classmethod
    def clear(cls) -> None:
        cls._current_student_id = None
        cls._current_class_id = None


# ============================================
# Helpers — jours
# ============================================
DAYS_FR = {
    0: "lundi", 1: "mardi", 2: "mercredi", 3: "jeudi",
    4: "vendredi", 5: "samedi", 6: "dimanche",
}

# Enum exact attendu par la colonne day_of_week de Supabase
# ⚠️  "sunday" n'existe PAS dans l'enum — les écoles n'ont pas cours le dimanche
DAYS_ENUM = {
    0: "monday",
    1: "tuesday",
    2: "wednesday",
    3: "thursday",
    4: "friday",
    5: "saturday",
    # 6 (dimanche) intentionnellement absent → intercepté en amont
}

# Jours sans cours (pas dans l'enum DB)
NO_SCHOOL_DAYS: set[int] = {6}   # dimanche = 6 en weekday()

# Message renvoyé quand on demande l'emploi du temps un jour férié/repos
def _no_school_msg(target_date: "date", lang: str = "fr") -> str:  # type: ignore[name-defined]
    day = DAYS_FR[target_date.weekday()]
    msgs = {
        "fr": f"Pas de cours le {day} {target_date.isoformat()} (jour de repos).",
        "en": f"No classes on {day} {target_date.isoformat()} (day off).",
        "ar": f"لا دروس يوم {day} {target_date.isoformat()} (يوم عطلة).",
    }
    return msgs.get(lang, msgs["fr"])

DAY_NAME_TO_INDEX = {
    # Lundi
    "lundi": 0, "monday": 0, "mon": 0, "lun": 0,
    "الاثنين": 0, "الإثنين": 0, "الاتنين": 0,
    "tnin": 0, "lethnine": 0, "lethnin": 0, "ithnayn": 0,
    # Mardi
    "mardi": 1, "tuesday": 1, "tue": 1, "tues": 1, "mar": 1,
    "الثلاثاء": 1, "الثلاثا": 1,
    "thlatha": 1, "tlata": 1,
    # Mercredi
    "mercredi": 2, "wednesday": 2, "wed": 2, "mer": 2,
    "الأربعاء": 2, "الاربعاء": 2,
    "lerba3": 2, "larbaa": 2, "arba3a": 2,
    # Jeudi
    "jeudi": 3, "thursday": 3, "thu": 3, "thur": 3, "jeu": 3,
    "الخميس": 3,
    "lekhmis": 3, "khmiss": 3, "khamis": 3,
    # Vendredi
    "vendredi": 4, "friday": 4, "fri": 4, "ven": 4,
    "الجمعة": 4, "الجمعه": 4,
    "jem3a": 4, "jomaa": 4, "jum3a": 4,
    # Samedi
    "samedi": 5, "saturday": 5, "sat": 5, "sam": 5,
    "السبت": 5,
    "sebt": 5, "sbet": 5,
    # Dimanche
    "dimanche": 6, "sunday": 6, "sun": 6, "dim": 6,
    "الأحد": 6, "الاحد": 6,
    "lahad": 6, "el a7ad": 6, "ahad": 6,
}


def _parse_date(date_str: Optional[str]) -> date:
    """
    Parse une date depuis plusieurs formats.

    BUG FIX : "même jour" = aujourd'hui, pas la semaine prochaine.
    On ne saute à la semaine suivante QUE si le modificateur est 'prochain'/'next'.
    """
    if not date_str:
        return date.today()

    s = date_str.strip().lower()

    # ── Mots-clés : aujourd'hui ──────────────────────────────────────────────
    TODAY_KEYWORDS = {
        "today", "aujourd'hui", "aujourdhui", "ajourd'hui",
        "اليوم", "lyoum", "el yom", "yawm", "nhar",
        # dialecte tunisien
        "lyouma", "elyoum",
    }
    if s in TODAY_KEYWORDS:
        return date.today()

    # ── Mots-clés : demain ───────────────────────────────────────────────────
    TOMORROW_KEYWORDS = {
        "tomorrow", "demain",
        "غدا", "غداً", "غدًا", "ghodwa", "ghodowa", "ghodua",
        "bokra", "boukra",
    }
    if s in TOMORROW_KEYWORDS:
        return date.today() + timedelta(days=1)

    # ── Mots-clés : hier ─────────────────────────────────────────────────────
    YESTERDAY_KEYWORDS = {
        "yesterday", "hier",
        "أمس", "امس", "elbareh", "lbereh", "bareh",
    }
    if s in YESTERDAY_KEYWORDS:
        return date.today() - timedelta(days=1)

    # ── Format ISO ───────────────────────────────────────────────────────────
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        pass

    # ── Nom de jour (avec modificateurs optionnels) ──────────────────────────
    NEXT_MODIFIERS = {"prochain", "prochaine", "next", "القادم"}
    LAST_MODIFIERS = {"dernier", "dernière", "last", "passé", "passée", "الماضي"}
    THIS_MODIFIERS = {"ce", "cette", "this", "هذا"}

    words = s.replace("-", " ").split()
    has_next = any(m in words for m in NEXT_MODIFIERS)
    has_last = any(m in words for m in LAST_MODIFIERS)
    # "ce lundi" = lundi de CETTE semaine (pas forcément le prochain)
    has_this = any(m in words for m in THIS_MODIFIERS)

    day_word = next((w for w in words if w in DAY_NAME_TO_INDEX), None)

    if day_word is not None:
        target_idx = DAY_NAME_TO_INDEX[day_word]
        today = date.today()
        today_idx = today.weekday()  # 0=lundi

        if has_last:
            days_ago = (today_idx - target_idx) % 7
            if days_ago == 0:
                days_ago = 7
            return today - timedelta(days=days_ago)

        if has_this:
            # "ce lundi" → lundi de la semaine courante (peut être passé)
            days_diff = (target_idx - today_idx) % 7
            if days_diff > 3:
                days_diff -= 7  # si trop loin dans le futur, c'est la semaine passée
            return today + timedelta(days=days_diff)

        if has_next:
            # "lundi prochain" → strictement la semaine suivante
            days_ahead = (target_idx - today_idx) % 7
            if days_ahead == 0:
                days_ahead = 7
            return today + timedelta(days=days_ahead)

        # ── FIX CRITIQUE : sans modificateur ────────────────────────────────
        # "lundi" = le prochain lundi Y COMPRIS aujourd'hui si c'est lundi.
        # L'ancienne version faisait days_ahead=7 si days_ahead==0,
        # ce qui renvoyait la semaine suivante quand l'élève voulait aujourd'hui.
        days_ahead = (target_idx - today_idx) % 7
        return today + timedelta(days=days_ahead)  # 0 = aujourd'hui ✓

    raise ValueError(
        f"Format de date non reconnu : '{date_str}'. "
        "Utilise 'today', 'tomorrow', 'YYYY-MM-DD', ou un nom de jour."
    )


def _week_bounds(offset: int = 0) -> Tuple[date, date]:
    """
    Retourne (lundi, dimanche) d'une semaine.
    offset=0 → semaine courante
    offset=-1 → semaine précédente
    offset=1 → semaine prochaine
    """
    today = date.today()
    monday = today - timedelta(days=today.weekday()) + timedelta(weeks=offset)
    sunday = monday + timedelta(days=6)
    return monday, sunday


def _parse_period_range(period_str: Optional[str]) -> Tuple[Optional[date], Optional[date]]:
    """
    Convertit un mot-clé de période en (date_debut, date_fin).
    Retourne (None, None) si non reconnu (= pas de filtre).

    Accepte (FR / EN / dialecte tunisien) :
    - "cette semaine", "this week", "hal jem3a"
    - "la semaine dernière", "last week", "ljom3a lli fattet"
    - "ce mois", "this month", "hal chhar"
    - "le mois dernier", "last month"
    - "aujourd'hui", "today"
    - "trimestre 1/2/3", "trimester_1/2/3"
    """
    if not period_str:
        return None, None

    s = period_str.strip().lower()

    # Aujourd'hui
    if any(k in s for k in ("today", "aujourd", "lyoum", "elyoum", "yawm", "nhar")):
        d = date.today()
        return d, d

    # Cette semaine
    if any(k in s for k in (
        "cette semaine", "this week", "semaine en cours", "semaine actuelle",
        "hal jem3a", "hal jom3a", "hath ljom3a", "الأسبوع الحالي", "هذا الأسبوع",
        "current week", "la semaine",
    )):
        return _week_bounds(0)

    # Semaine dernière
    if any(k in s for k in (
        "semaine dernière", "semaine passée", "last week", "previous week",
        "ljom3a lli fattet", "الأسبوع الماضي",
    )):
        return _week_bounds(-1)

    # Ce mois
    if any(k in s for k in (
        "ce mois", "ce mois-ci", "this month", "hal chhar", "هذا الشهر",
        "le mois", "month",
    )):
        today = date.today()
        first = today.replace(day=1)
        return first, today

    # Mois dernier
    if any(k in s for k in (
        "mois dernier", "mois passé", "last month", "الشهر الماضي",
    )):
        today = date.today()
        first_this = today.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start, last_month_end

    return None, None


# ============================================
# TOOLS
# ============================================

@tool
def get_student_schedule(date_str: Optional[str] = None) -> str:
    """
    Retourne l'emploi du temps de l'élève pour un jour donné.

    IMPORTANT : Appelle ce tool DÈS QUE la question porte sur l'emploi du temps,
    les cours, les séances ou les horaires — peu importe comment c'est formulé.

    Args:
        date_str: La date cible. Valeurs acceptées :
          ┌─ Relatif ──────────────────────────────────────────────────────┐
          │  "today" / "aujourd'hui" / "lyoum"   → aujourd'hui             │
          │  "tomorrow" / "demain" / "ghodwa"    → demain                  │
          │  "yesterday" / "hier" / "lbereh"     → hier                    │
          ├─ Nom de jour ───────────────────────────────────────────────────┤
          │  "lundi" / "monday" / "الاثنين"      → ce lundi (si futur),    │
          │                                         ou aujourd'hui si c'est│
          │                                         lundi                   │
          │  "lundi prochain" / "next monday"    → strictement semaine+1   │
          │  "lundi dernier" / "last monday"     → lundi de la sem passée  │
          ├─ ISO ───────────────────────────────────────────────────────────┤
          │  "2026-05-12"                         → date exacte            │
          └─────────────────────────────────────────────────────────────────┘
          Par défaut (date_str=None ou vide) = AUJOURD'HUI.

    Exemples de questions → date_str à passer :
      "j'ai cours aujourd'hui ?"      → date_str="today"  (ou None)
      "j'ai cours demain ?"           → date_str="tomorrow"
      "mon emploi du temps de lundi"  → date_str="lundi"
      "cours de mercredi prochain"    → date_str="mercredi prochain"
      "est-ce que j'ai cours ?"       → date_str=None  (= aujourd'hui)
      "عندي دروس اليوم؟"              → date_str="today"
      "شنوا عندي غدوة؟"              → date_str="tomorrow"
    """
    try:
        target_date = _parse_date(date_str)
    except ValueError as e:
        return f"Je n'ai pas pu interpréter la date '{date_str}'. {e}"

    # ── Jour sans cours (dimanche ou autre jour hors enum) ───────────────────
    if target_date.weekday() in NO_SCHOOL_DAYS:
        return _no_school_msg(target_date)

    day_of_week = DAYS_ENUM.get(target_date.weekday())
    if day_of_week is None:
        return f"Jour non géré dans l'emploi du temps : {DAYS_FR.get(target_date.weekday(), '?')}."

    day_label = DAYS_FR[target_date.weekday()]

    class_id = StudentContext.get_class_id()
    if not class_id:
        return "Aucune classe n'est associée à cet élève."

    sb = get_supabase()
    slots = (
        sb.table("schedule_slots")
        .select(
            "start_time, end_time, room, "
            "subjects(name), "
            "teachers(profile_id, profiles(first_name, last_name))"
        )
        .eq("class_id", class_id)
        .eq("day_of_week", day_of_week)
        .eq("is_active", True)
        .order("start_time")
        .execute()
    )

    if not slots.data:
        # Message précis : on indique le VRAI jour demandé, pas "demain"
        return (
            f"Aucun cours prévu pour le {day_label} {target_date.isoformat()}. "
            f"(Vérifie que l'emploi du temps est bien saisi pour ce jour.)"
        )

    lines = [f"Emploi du temps du {day_label} {target_date.isoformat()} :"]
    for s in slots.data:
        subject = (s.get("subjects") or {}).get("name", "Matière inconnue")
        teacher = s.get("teachers") or {}
        prof = teacher.get("profiles") or {}
        teacher_name = (
            f"{prof.get('first_name', '')} {prof.get('last_name', '')}".strip()
            or "prof non renseigné"
        )
        room = s.get("room") or "salle non précisée"
        lines.append(
            f"- {s['start_time'][:5]}–{s['end_time'][:5]} : {subject} "
            f"(salle {room}, {teacher_name})"
        )
    return "\n".join(lines)


@tool
def get_canteen_menu(date_str: Optional[str] = None) -> str:
    """
    Retourne le menu de la cantine pour un jour donné.

    IMPORTANT : Utilise ce tool pour TOUTE question sur la cantine, le repas,
    ce qu'on mange, le déjeuner — même si c'est formulé de façon familière.

    Args:
        date_str: La date cible (mêmes formats que get_student_schedule).
          Par défaut = aujourd'hui.

    Exemples de questions → date_str à passer :
      "quel est le menu aujourd'hui ?"   → date_str="today"
      "qu'est-ce qu'on mange demain ?"   → date_str="tomorrow"
      "menu du lundi prochain"           → date_str="lundi prochain"
      "menu de lundi ?"                  → date_str="lundi"
      "شنوا في الكانتين اليوم؟"          → date_str="today"
    """
    try:
        target_date = _parse_date(date_str)
    except ValueError as e:
        return f"Je n'ai pas pu interpréter la date '{date_str}'. {e}"

    sb = get_supabase()
    res = (
        sb.table("canteen_menus")
        .select("*")
        .eq("date", target_date.isoformat())
        .limit(1)
        .execute()
    )

    if not res.data:
        return (
            f"Aucun menu n'est disponible pour le {target_date.isoformat()}. "
            f"Le menu n'a peut-être pas encore été publié."
        )

    m = res.data[0]
    day_label = DAYS_FR[target_date.weekday()]
    parts = [f"Menu du {day_label} {target_date.isoformat()} :"]

    starters = m.get("starters") or ([m["starter"]] if m.get("starter") else [])
    if starters:
        parts.append(f"- Entrée : {', '.join(str(x) for x in starters)}")

    mains = m.get("main_courses") or ([m["main_course"]] if m.get("main_course") else [])
    if mains:
        parts.append(f"- Plat principal : {', '.join(str(x) for x in mains)}")

    if m.get("side_dish"):
        parts.append(f"- Accompagnement : {m['side_dish']}")

    desserts = m.get("desserts_list") or ([m["dessert"]] if m.get("dessert") else [])
    if desserts:
        parts.append(f"- Dessert : {', '.join(str(x) for x in desserts)}")

    if m.get("allergens"):
        parts.append(f"- ⚠️ Allergènes : {', '.join(m['allergens'])}")

    return "\n".join(parts)


@tool
def get_student_grades(
    subject_name: Optional[str] = None,
    period: Optional[str] = None,
) -> str:
    """
    Retourne les notes et la moyenne de l'élève.

    IMPORTANT : Utilise ce tool pour TOUTE question sur les notes, les moyennes,
    les résultats, les évaluations — même formulée simplement.

    Args:
        subject_name: (optionnel) nom de la matière à filtrer.
          Accepte des noms partiels et différentes langues :
          'maths', 'math', 'mathématiques', 'رياضيات',
          'arabe', 'arabic', 'عربية', 'français', 'french',
          'physique', 'phys', 'svt', 'histoire', 'anglais', ...
        period: (optionnel) 'trimester_1', 'trimester_2', 'trimester_3'.
          Accepte aussi : '1er trimestre', 'trimestre 1', 'T1',
          'premier trimestre', 'first trimester'.

    Exemples de questions → arguments à passer :
      "mes notes ?"                       → subject_name=None, period=None
      "ma moyenne en maths ?"             → subject_name="maths"
      "mes notes d'arabe ?"               → subject_name="arabe"
      "mes résultats du trimestre 2 ?"    → period="trimester_2"
      "quelle est ma note en physique ?"  → subject_name="physique"
      "معدلي في العربية؟"                 → subject_name="arabe"
      "شنوا درجاتي؟"                      → subject_name=None
    """
    sb = get_supabase()
    student_id = StudentContext.get_student_id()

    # ── Normaliser period ────────────────────────────────────────────────────
    period_normalized = period
    if period:
        p = period.strip().lower()
        if any(k in p for k in ("1", "premier", "first", "t1", "uno")):
            period_normalized = "trimester_1"
        elif any(k in p for k in ("2", "deuxième", "second", "t2", "dos")):
            period_normalized = "trimester_2"
        elif any(k in p for k in ("3", "troisième", "third", "t3", "tres")):
            period_normalized = "trimester_3"

    query = (
        sb.table("grades")
        .select("score, max_score, coefficient, title, period, grade_date, subjects(name)")
        .eq("student_id", student_id)
        .order("grade_date", desc=True)
        .limit(50)
    )
    if period_normalized:
        query = query.eq("period", period_normalized)

    res = query.execute()
    rows = res.data or []

    # ── Filtrage matière côté Python (tolérant aux fautes / abréviations) ────
    if subject_name:
        needle = subject_name.strip().lower()
        # Mapping abréviations → mots-clés
        SUBJECT_ALIASES = {
            "maths": "math", "mathématiques": "math", "رياضيات": "math",
            "arabe": "arab", "arabic": "arab", "عربية": "arab", "عربي": "arab",
            "français": "fran", "french": "fran", "فرنسية": "fran",
            "anglais": "angl", "english": "angl", "إنجليزية": "angl",
            "physique": "phys", "physics": "phys", "فيزياء": "phys",
            "svt": "svt", "bio": "bio", "biologie": "bio",
            "histoire": "hist", "history": "hist", "تاريخ": "hist",
            "géo": "geo", "géographie": "geo", "geography": "geo",
            "info": "info", "informatique": "info",
            "eco": "eco", "économie": "eco",
            "philo": "philo", "philosophie": "philo",
            "sport": "sport", "eps": "sport",
        }
        search_key = next(
            (v for k, v in SUBJECT_ALIASES.items() if k in needle or needle in k),
            needle
        )
        rows = [
            r for r in rows
            if search_key in ((r.get("subjects") or {}).get("name", "").lower())
            or needle in ((r.get("subjects") or {}).get("name", "").lower())
        ]

    if not rows:
        subject_info = f" pour '{subject_name}'" if subject_name else ""
        period_info = f" (période : {period_normalized})" if period_normalized else ""
        return f"Aucune note trouvée{subject_info}{period_info}."

    # ── Calcul moyenne pondérée ───────────────────────────────────────────────
    valid = [r for r in rows if r.get("score") is not None]
    if valid:
        total_w = sum(float(r["score"]) * float(r.get("coefficient") or 1) for r in valid)
        total_c = sum(float(r.get("coefficient") or 1) for r in valid)
        avg = total_w / total_c if total_c else 0
        avg_str = f"{avg:.2f}/20"
    else:
        avg_str = "N/A"

    subject_label = f" ({subject_name})" if subject_name else ""
    lines = [f"{len(rows)} note(s){subject_label} — Moyenne : {avg_str}"]
    for r in rows[:15]:
        subject = (r.get("subjects") or {}).get("name", "?")
        title = r.get("title") or "évaluation"
        score = r.get("score", "?")
        max_s = r.get("max_score", 20)
        coef = r.get("coefficient", 1)
        gdate = (r.get("grade_date") or "?")[:10]
        lines.append(f"- {gdate} · {subject} · {title} : {score}/{max_s} (coef {coef})")
    if len(rows) > 15:
        lines.append(f"... et {len(rows) - 15} autres notes.")
    return "\n".join(lines)


@tool
def get_student_assignments(upcoming_only: bool = True) -> str:
    """
    Retourne la liste des devoirs/travaux de l'élève.

    IMPORTANT : Utilise ce tool pour TOUTE question sur les devoirs,
    le travail à faire, les DM, les rendus.

    Args:
        upcoming_only: si True (défaut), ne retourne que les devoirs à venir.
          Passe False pour avoir tous les devoirs (y compris passés).

    Exemples de questions :
      "j'ai des devoirs ?"                → upcoming_only=True
      "qu'est-ce que je dois rendre ?"    → upcoming_only=True
      "mes devoirs de la semaine"         → upcoming_only=True
      "شنوا عندي نحضّر؟"                  → upcoming_only=True
    """
    sb = get_supabase()
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

    lines = [f"{len(res.data)} devoir(s) à rendre :"]
    for a in res.data:
        subj = (a.get("subjects") or {}).get("name", "?")
        due = (a.get("due_date") or "?")[:10]
        desc = a.get("description", "")
        desc_short = f" — {desc[:80]}" if desc else ""
        lines.append(
            f"- {due} · {subj} · {a.get('type', 'devoir')} : {a['title']}{desc_short}"
        )
    return "\n".join(lines)


@tool
def get_student_attendance(
    period: Optional[str] = None,
    from_date: Optional[str] = None,
) -> str:
    """
    Retourne les absences et retards de l'élève pour une période.

    IMPORTANT : Utilise ce tool pour TOUTE question sur les absences,
    les retards, la présence — quelle que soit la formulation.

    Args:
        period: Période en langage naturel. Exemples acceptés :
          ┌────────────────────────────────────────────────────────────────┐
          │  "cette semaine" / "this week" / "hal jem3a"                   │
          │  "semaine dernière" / "last week"                              │
          │  "ce mois" / "ce mois-ci" / "this month" / "hal chhar"        │
          │  "mois dernier" / "last month"                                 │
          │  "aujourd'hui" / "today"                                       │
          └────────────────────────────────────────────────────────────────┘
          Si period est fourni, from_date est ignoré.
          Par défaut (period=None, from_date=None) = 30 derniers jours.

        from_date: date de début au format 'YYYY-MM-DD' (si period non fourni).

    Exemples de questions → arguments à passer :
      "j'ai des absences cette semaine ?"       → period="cette semaine"
      "combien de retards ce mois-ci ?"         → period="ce mois"
      "absences la semaine dernière ?"          → period="semaine dernière"
      "j'ai été absent aujourd'hui ?"           → period="aujourd'hui"
      "mes absences depuis le 1er avril ?"      → from_date="2026-04-01"
      "عندي غيابات هذا الأسبوع؟"               → period="cette semaine"
      "كم مرة تأخرت هذا الشهر؟"                → period="ce mois"
    """
    sb = get_supabase()
    student_id = StudentContext.get_student_id()

    # ── Déterminer la plage de dates ─────────────────────────────────────────
    start: Optional[date] = None
    end: Optional[date] = None
    period_label = "les 30 derniers jours"

    if period:
        start, end = _parse_period_range(period)
        if start:
            period_label = period

    if start is None:
        if from_date:
            try:
                start = _parse_date(from_date)
                period_label = f"depuis le {start.isoformat()}"
            except ValueError:
                pass

    if start is None:
        start = date.today() - timedelta(days=30)

    # ── Requête Supabase ─────────────────────────────────────────────────────
    query = (
        sb.table("attendance")
        .select("date, status, reason, schedule_slots(start_time, end_time, subjects(name))")
        .eq("student_id", student_id)
        .gte("date", start.isoformat())
        .order("date", desc=True)
    )
    if end:
        query = query.lte("date", end.isoformat())

    res = query.execute()
    rows = res.data or []

    # ── Filtrer : ne montrer que les absences et retards (pas les présences) ─
    absences = [r for r in rows if r["status"] in ("absent", "late")]

    if not absences:
        return f"Aucune absence ni retard pour {period_label}. ✅"

    counts: dict[str, int] = {}
    for r in absences:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    summary_parts = []
    if counts.get("absent"):
        summary_parts.append(f"{counts['absent']} absence(s)")
    if counts.get("late"):
        summary_parts.append(f"{counts['late']} retard(s)")

    lines = [f"Pour {period_label} : {', '.join(summary_parts)}."]
    for r in absences[:15]:
        slot = r.get("schedule_slots") or {}
        subj = (slot.get("subjects") or {}).get("name", "")
        time_range = ""
        if slot.get("start_time") and slot.get("end_time"):
            time_range = f" · {slot['start_time'][:5]}–{slot['end_time'][:5]}"
        subject_str = f" · {subj}" if subj else ""
        reason = f" (raison : {r['reason']})" if r.get("reason") else ""
        status_label = "Absent" if r["status"] == "absent" else "Retard"
        lines.append(
            f"- {r['date']}{subject_str}{time_range} → {status_label}{reason}"
        )
    if len(absences) > 15:
        lines.append(f"... et {len(absences) - 15} autres.")
    return "\n".join(lines)


@tool
def get_announcements(limit: int = 5) -> str:
    """
    Retourne les dernières annonces pour la classe de l'élève.

    IMPORTANT : Utilise ce tool pour TOUTE question sur les actualités,
    les nouvelles, les annonces, les communications de l'école.

    Args:
        limit: nombre maximum d'annonces (défaut 5, max 10).

    Exemples de questions :
      "y a-t-il des annonces ?"
      "quoi de neuf à l'école ?"
      "annonces importantes ?"
      "شنوا في الأخبار؟"
    """
    sb = get_supabase()
    class_id = StudentContext.get_class_id()

    query = (
        sb.table("announcements")
        .select("title, content, is_pinned, published_at")
        .order("is_pinned", desc=True)
        .order("published_at", desc=True)
        .limit(min(limit, 10))
    )
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
        date_str = (a.get("published_at") or "")[:10]
        content_short = (a.get("content") or "")[:200]
        lines.append(f"{pin}[{date_str}] {a['title']} — {content_short}")
    return "\n".join(lines)


@tool
def get_student_meetings(
    status: Optional[str] = None,
    upcoming_only: bool = True,
) -> str:
    """
    Retourne les réunions parents-professeurs concernant l'élève.

    Args:
        status: (optionnel) 'requested', 'confirmed', 'cancelled', 'completed'.
        upcoming_only: si True (défaut), réunions à venir uniquement.

    Exemples de questions :
      "mes parents ont une réunion avec un prof ?"
      "prochaine réunion parents-profs ?"
      "شكون طلب إجتماع مع والديّ؟"
    """
    sb = get_supabase()
    student_id = StudentContext.get_student_id()
    class_id = StudentContext.get_class_id()

    def _build_meeting_query(table_query):
        if status:
            table_query = table_query.eq("status", status)
        if upcoming_only:
            table_query = table_query.gte("scheduled_at", datetime.now().isoformat())
        return table_query

    base_select = (
        "id, status, scheduled_at, duration_minutes, location, notes, "
        "is_class_meeting, teachers(profile_id, profiles(first_name, last_name))"
    )

    q_ind = _build_meeting_query(
        sb.table("meetings").select(base_select)
        .eq("student_id", student_id)
        .order("scheduled_at").limit(20)
    )
    rows = q_ind.execute().data or []

    if class_id:
        q_cls = _build_meeting_query(
            sb.table("meetings").select(base_select)
            .eq("class_id", class_id)
            .eq("is_class_meeting", True)
            .order("scheduled_at").limit(20)
        )
        existing_ids = {r["id"] for r in rows}
        rows.extend(r for r in (q_cls.execute().data or []) if r["id"] not in existing_ids)

    if not rows:
        return "Aucune réunion parents-professeurs prévue."

    rows.sort(key=lambda r: r.get("scheduled_at") or "")
    lines = [f"{len(rows)} réunion(s) :"]
    for m in rows:
        when = (m.get("scheduled_at") or "?").replace("T", " ")[:16]
        dur = m.get("duration_minutes", 30)
        loc = m.get("location") or "lieu non précisé"
        teacher = (m.get("teachers") or {}).get("profiles") or {}
        tname = f"{teacher.get('first_name', '')} {teacher.get('last_name', '')}".strip() or "prof inconnu"
        mtype = "réunion de classe" if m.get("is_class_meeting") else f"avec {tname}"
        lines.append(f"- {when} · {mtype} · {dur} min · {loc} · [{m.get('status', '?')}]")
        if m.get("notes"):
            lines.append(f"  note : {m['notes'][:100]}")
    return "\n".join(lines)


@tool
def get_student_payments(status: Optional[str] = None) -> str:
    """
    Retourne les paiements et frais de scolarité de l'élève.

    Args:
        status: (optionnel) 'pending', 'paid', 'overdue'.

    Exemples de questions :
      "j'ai des paiements en attente ?"
      "mes frais de scolarité ?"
      "est-ce que j'ai des dettes ?"
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
        paid_info = f" (payé le {p['paid_at'][:10]})" if p.get("paid_at") else ""
        lines.append(
            f"- {p.get('due_date', '?')[:10]} · {p['type']} · {p.get('amount', '?')} TND "
            f"· [{p['status']}]{paid_info} {p.get('description', '')}"
        )
    return "\n".join(lines)


# ============================================
# Export
# ============================================
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
