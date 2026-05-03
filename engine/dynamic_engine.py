"""
Moteur DYNAMIC v8 — Fixes critiques :

1. PRÉ-EXTRACTION DE DATE (côté Python, avant tout LLM)
   "j'ai cours demain?" → on détecte "demain", on calcule la date,
   on l'injecte explicitement dans le message → le LLM ne peut plus se tromper.

2. BOUCLE PROPRE : break dès qu'on a un résultat tool → plus de double appel.

3. SYNTHÈSE via llm_synth (sans tools) → plus de re-appel tool accidentel.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_ollama import ChatOllama

from config import settings
from tools import ALL_TOOLS
from engine.router import _normalize   # normalisation abréviations (binôme)


# ──────────────────────────────────────────────────────────────────────
# Détection de langue
# ──────────────────────────────────────────────────────────────────────
def _detect_lang(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if re.search(r"\b(what|when|where|who|how|my|i\s+have|do\s+i|show)\b", text, re.I):
        return "en"
    return "fr"


# ──────────────────────────────────────────────────────────────────────
# PRÉ-EXTRACTION DE DATE (Python pur, 0 LLM)
#
# Détecte les mots-clés temporels dans la question et retourne :
#  - la date ISO calculée (ex: "2026-05-03")
#  - le label humain (ex: "demain (lundi 2026-05-03)")
#  - date_str à passer au tool (ex: "tomorrow")
#
# POURQUOI : qwen2.5:3b ignore souvent le docstring du tool et appelle
# get_student_schedule({}) sans date. En injectant la date directement
# dans le message utilisateur, le modèle la "voit" et la passe correctement.
# ──────────────────────────────────────────────────────────────────────
_DAYS_FR = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]
_DAYS_EN = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

_DAY_MAP: Dict[str, int] = {}
for _i, _d in enumerate(_DAYS_FR): _DAY_MAP[_d] = _i
for _i, _d in enumerate(_DAYS_EN): _DAY_MAP[_d] = _i
# dialecte tunisien + arabe
_DAY_MAP.update({
    "الاثنين":0, "الإثنين":0, "tnin":0, "lethnin":0,
    "الثلاثاء":1, "tlata":1, "thlatha":1,
    "الأربعاء":2, "lerba3":2, "larbaa":2,
    "الخميس":3,   "lekhmis":3,"khmiss":3,
    "الجمعة":4,   "jem3a":4,  "jomaa":4,
    "السبت":5,    "sebt":5,   "sbet":5,
    "الأحد":6,    "lahad":6,
})

def _extract_date_hint(query: str) -> Tuple[Optional[date], Optional[str]]:
    """
    Retourne (date_cible, date_str_pour_tool) si un mot-clé temporel est détecté.
    Retourne (None, None) sinon.
    """
    s = query.lower().strip()
    today = date.today()

    # Aujourd'hui
    if any(k in s for k in [
        "aujourd'hui", "aujourd hui", "ajourd'hui", "today",
        "اليوم", "lyoum", "elyoum", "lyouma",
    ]):
        return today, "today"

    # Demain
    if any(k in s for k in [
        "demain", "tomorrow",
        "غدا", "غداً", "ghodwa", "ghodoa", "bokra", "boukra",
    ]):
        return today + timedelta(days=1), "tomorrow"

    # Hier
    if any(k in s for k in [
        "hier", "yesterday",
        "أمس", "elbareh", "lbereh", "bareh",
    ]):
        return today - timedelta(days=1), "yesterday"

    # Nom de jour ("lundi", "mardi prochain", etc.)
    words = re.sub(r"[?!،؟]", " ", s).split()
    is_next = any(w in words for w in ["prochain","prochaine","next","القادم"])
    is_last = any(w in words for w in ["dernier","dernière","last","الماضي"])

    for w in words:
        if w in _DAY_MAP:
            target_idx = _DAY_MAP[w]
            today_idx = today.weekday()

            if is_last:
                ago = (today_idx - target_idx) % 7 or 7
                d = today - timedelta(days=ago)
            elif is_next:
                ahead = (target_idx - today_idx) % 7 or 7
                d = today + timedelta(days=ahead)
            else:
                ahead = (target_idx - today_idx) % 7
                d = today + timedelta(days=ahead)

            return d, d.isoformat()

    return None, None


def _build_enriched_query(query: str) -> str:
    """
    Injecte la date calculée directement dans la question pour que
    le LLM la passe correctement au tool, même s'il est petit.

    Exemple :
      "j'ai cours demain?" 
      → "j'ai cours demain?
         [CONTEXTE DATE: demain = mardi 2026-05-05, date_str="tomorrow"]"
    """
    target_date, date_str = _extract_date_hint(query)
    if target_date is None:
        return query

    day_name = _DAYS_FR[target_date.weekday()]
    hint = (
        f"\n[CONTEXTE DATE: la date mentionnée = {day_name} {target_date.isoformat()}"
        f', utilise date_str="{date_str}" dans le tool]'
    )
    return query + hint


# ──────────────────────────────────────────────────────────────────────
# Détection "doit utiliser un tool"
# ──────────────────────────────────────────────────────────────────────
_MUST_USE_TOOL = re.compile(
    r"\b(menu|cantine|manger|repas|déjeuner|emploi|horaire|planning|cours|séance|"
    r"note|résultat|moyenne|bulletin|absence|absent|retard|présence|devoir|dm|"
    r"réunion|paiement|facture|frais|annonce|actualité|"
    r"schedule|grade|homework|attendance|payment|meeting|food|lunch|"
    r"مطعم|وجبة|أكل|جدول|درجة|غياب|واجب|اجتماع|مدفوعات|إعلان|كانتين|نوت|دروس)\b",
    re.IGNORECASE | re.UNICODE,
)

_HALLUCINATION = re.compile(
    r"(je n[' ]ai pas accès|je ne peux pas accéd|cannot access|"
    r"don[' ]t have access|لا أستطيع الوصول)",
    re.IGNORECASE,
)

# Résultats lisibles → retour direct sans 2ème LLM (économie 20-60s)
_DIRECT_RESULT = re.compile(
    r"^(Emploi du temps du|Menu du|Aucun cours|Pas de cours|"
    r"Il n[' ]y a pas cours|Aucun menu|\d+ note\(s\)|Aucune note|"
    r"Pour (cette semaine|ce mois|les \d+|aujourd|hier)|"
    r"Aucune absence ni retard|\d+ absence|\d+ devoir|Aucun devoir|"
    r"Dernières annonces|Aucune annonce|No classes on|لا دروس|لا يوجد)",
    re.IGNORECASE | re.UNICODE,
)


# ──────────────────────────────────────────────────────────────────────
# PRÉ-ROUTAGE PYTHON — outil forcé sans passer par le LLM
#
# POURQUOI : qwen2.5:1.5b confond régulièrement les outils sur les
# questions courtes ("Mes devoirs ?", "Mes absences ?", etc.)
# Solution : on détecte le mot-clé en Python et on retourne directement
# le nom de l'outil à appeler — 0 LLM pour la sélection.
#
# Priorité : l'ordre des règles compte (la première qui matche gagne).
# ──────────────────────────────────────────────────────────────────────
_TOOL_RULES: list = [
    # (pattern, tool_name)
    # ── Devoirs — DOIT être avant "cours" car "devoir de cours" peut matcher ──
    (re.compile(
        r"\b(devoir|devoirs|dm|homework|assignment|assignments|"
        r"travaux?\s+[àa]\s+rendre|rendre|à\s+remettre|due|deadline|"
        r"واجب|واجباتي|مهمة|مهام)\b",
        re.IGNORECASE | re.UNICODE,
    ), "get_student_assignments"),

    # ── Menu / cantine ──
    (re.compile(
        r"\b(menu|cantine|canteen|cafétéria|manger|repas|déjeuner|dîner|"
        r"nourriture|food|lunch|dinner|plat|qu[' ]est[- ]ce\s+qu[' ]on\s+mange|"
        r"مطعم|وجبة|أكل|طعام|غداء|ماذا\s+(نأكل|يوجد\s+للأكل)|ما\s+هو\s+طعام)\b",
        re.IGNORECASE | re.UNICODE,
    ), "get_canteen_menu"),

    # ── Emploi du temps / cours du jour ──
    (re.compile(
        r"\b(emploi\s+du\s+temps|edt|planning|cours\s+(de\s+)?(aujourd|demain|lundi|mardi|"
        r"mercredi|jeudi|vendredi|cette\s+semaine)|j[' ]ai\s+cours|j[' ]ai\s+quoi|"
        r"mes\s+cours|schedule|timetable|class|جدول|جدولي|حصص|حصصي|حصة|مواعيد\s+الدروس)\b",
        re.IGNORECASE | re.UNICODE,
    ), "get_student_schedule"),

    # ── Notes / résultats / moyenne ──
    (re.compile(
        r"\b(note|notes|résultat|résultats|moyenne|bulletin|relevé|"
        r"combien\s+j[' ]ai\s+eu|ma\s+note|mes\s+notes|grade|grades|"
        r"average|gpa|score|درجة|درجاتي|نقطة|نقاطي|معدل|نتيجة|نتائجي)\b",
        re.IGNORECASE | re.UNICODE,
    ), "get_student_grades"),

    # ── Absences / retards / présence — APRÈS devoirs pour éviter confusion ──
    (re.compile(
        r"\b(absence|absences|absent|retard|retards|présence|"
        r"combien\s+de\s+fois\s+(j[' ]ai\s+)?(été\s+)?absent|"
        r"attendance|tardiness|غياب|غياباتي|تأخر|حضور)\b",
        re.IGNORECASE | re.UNICODE,
    ), "get_student_attendance"),

    # ── Réunions parents-profs ──
    (re.compile(
        r"\b(réunion|réunions|rendez[- ]vous|parents[- ]profs|"
        r"meeting|meetings|appointment|اجتماع|اجتماعاتي|موعد)\b",
        re.IGNORECASE | re.UNICODE,
    ), "get_student_meetings"),

    # ── Paiements / frais ──
    (re.compile(
        r"\b(paiement|paiements|facture|frais|solde|dois[- ]je\s+payer|"
        r"payment|payments|fee|fees|invoice|tuition|مدفوعات|رسوم|فاتورة)\b",
        re.IGNORECASE | re.UNICODE,
    ), "get_student_payments"),

    # ── Annonces ──
    (re.compile(
        r"\b(annonce|annonces|actualité|actualités|news|announcement|"
        r"إعلان|إعلانات|أخبار)\b",
        re.IGNORECASE | re.UNICODE,
    ), "get_announcements"),
]


# ──────────────────────────────────────────────────────────────────────
# Résolution contextuelle — questions de suivi sans topic explicite
# Ex: "et pour le mardi ?" après "le menu lundi ?"
# ──────────────────────────────────────────────────────────────────────

# Détecte une question de suivi : juste un jour/moment, sans mot-clé de topic
_FOLLOWUP = re.compile(
    r"^\s*(et\s+)?(pour\s+)?(le\s+|la\s+|les\s+|au\s+)?"
    r"("
    # ── Jours / moments ──────────────────────────────────────────────
    r"lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"الاثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت|الأحد|"
    r"demain|après[- ]demain|hier|aujourd[\' ]?hui|"
    r"ce\s+soir|ce\s+matin|cette\s+semaine|la\s+semaine\s+(prochaine|dernière)|"
    r"tomorrow|yesterday|today|tonight|"
    # ── Changement de langue ─────────────────────────────────────────
    r"(en\s+)?(français|french|fr)|"
    r"(en\s+)?(anglais|english|en)|"
    r"(en\s+)?(arabe|arabic|ar)|"
    r"(بال)?(عربية|عربي)|"
    r"(بال)?(فرنسية|فرنسي)|"
    r"(بال)?(إنجليزية|إنجليزي)|"
    r"in\s+(french|english|arabic)"
    r")"
    r"\s*[?؟!]?\s*$",
    re.IGNORECASE | re.UNICODE,
)

# Signaux de topic dans les messages précédents → quel tool hériter
_TOPIC_SIGNALS: list = [
    (re.compile(r"(menu|cantine|repas|déjeuner|plat|manger|طعام|وجبة|مطعم|aucun menu|no menu)", re.I|re.U), "get_canteen_menu"),
    (re.compile(r"(cours|séance|emploi du temps|edt|planning|schedule|جدول|حصص|aucun cours|no class)", re.I|re.U), "get_student_schedule"),
    (re.compile(r"(note|résultat|moyenne|grade|bulletin|درجة|معدل|aucune note)", re.I|re.U), "get_student_grades"),
    (re.compile(r"(absence|retard|absent|attendance|غياب|aucune absence)", re.I|re.U), "get_student_attendance"),
    (re.compile(r"(devoir|homework|assignment|واجب|aucun devoir)", re.I|re.U), "get_student_assignments"),
    (re.compile(r"(réunion|meeting|اجتماع|aucune réunion)", re.I|re.U), "get_student_meetings"),
    (re.compile(r"(paiement|frais|fee|مدفوعات|aucun paiement)", re.I|re.U), "get_student_payments"),
]



def _build_tool_args(tool_name: str, query: str) -> Dict[str, Any]:
    """
    Construit les arguments à passer au tool selon son nom et la question.
    Utilisé par le pré-routage Python pour éviter que le LLM choisisse les args.
    """
    args: Dict[str, Any] = {}

    # Tools qui acceptent date_str
    if tool_name in ("get_student_schedule", "get_canteen_menu"):
        _, date_str = _extract_date_hint(query)
        if date_str:
            args["date_str"] = date_str

    # Tools qui acceptent upcoming_only
    if tool_name == "get_student_assignments":
        args["upcoming_only"] = True

    # Tools qui acceptent period
    if tool_name == "get_student_attendance":
        # Détecter si on demande une période spécifique
        q = query.lower()
        if any(k in q for k in ["cette semaine", "this week", "semaine"]):
            args["period"] = "cette semaine"
        elif any(k in q for k in ["ce mois", "this month", "mois"]):
            args["period"] = "ce mois"

    return args


def _topic_from_history(history: Optional[List]) -> Optional[str]:
    """
    Parcourt les messages récents (user + assistant) pour détecter
    quel topic était en cours → permet d\'hériter du contexte.
    """
    if not history:
        return None
    for msg in reversed(history[-6:]):
        text = ""
        if hasattr(msg, "content"):
            text = str(msg.content or "")
        elif isinstance(msg, dict):
            text = str(msg.get("content", ""))
        if not text:
            continue
        for pattern, tool_name in _TOPIC_SIGNALS:
            if pattern.search(text):
                return tool_name
    return None



# ──────────────────────────────────────────────────────────────────────
# Détection de langue cible dans un suivi ("et en français ?")
# ──────────────────────────────────────────────────────────────────────
_LANG_REQUEST = re.compile(
    r"(en\s+)?(français|french)|(en\s+)?(anglais|english)|(en\s+)?(arabe|arabic)"
    r"|(بال)?(عربية|عربي)|(بال)?(فرنسية|فرنسي)|(بال)?(إنجليزية|إنجليزي)"
    r"|in\s+(french|english|arabic)",
    re.IGNORECASE | re.UNICODE,
)

_LANG_MAP = {
    "français": "fr", "french": "fr", "fr": "fr", "فرنسية": "fr", "فرنسي": "fr",
    "anglais": "en", "english": "en", "en": "en", "إنجليزية": "en", "إنجليزي": "en",
    "arabe": "ar", "arabic": "ar", "ar": "ar", "عربية": "ar", "عربي": "ar",
}

def _extract_target_lang(query: str) -> Optional[str]:
    """Extrait la langue cible d'un suivi de langue, ex: 'et en français?' → 'fr'"""
    m = _LANG_REQUEST.search(query.lower())
    if not m:
        return None
    matched = next((g for g in m.groups() if g), "")
    return _LANG_MAP.get(matched.strip(), None)


def _preroute_tool(
    query: str,
    history: Optional[List] = None,
) -> Optional[str]:
    """
    Retourne le nom du tool à appeler (Python pur, 0 LLM).

    Étape 1 : matching direct sur la question courante.
    Étape 2 : si la question est un suivi temporel sans topic
              (ex: "et pour le mardi ?"), hérite du topic de l\'historique.
    """
    # Étape 1 — matching direct sur les mots-clés de la question
    for pattern, tool_name in _TOOL_RULES:
        if pattern.search(query):
            return tool_name

    # Étape 2 — suivi contextuel (question courte = juste un jour/moment)
    if _FOLLOWUP.match(query):
        inherited = _topic_from_history(history)
        if inherited:
            print(f"[DynamicEngine] 🔗 Suivi contextuel → {inherited}")
            return inherited

    return None


# ──────────────────────────────────────────────────────────────────────
# Parse tool calls en JSON texte (fallback pour petits modèles)
# ──────────────────────────────────────────────────────────────────────
def _parse_json_tool_calls(content: str) -> List[Dict[str, Any]]:
    results = []
    seen: set = set()
    for pat in [r"```(?:json)?\s*(\{.*?\})\s*```", r'\{\s*"name"\s*:\s*"(\w+)"[^{}]*\}']:
        for m in re.finditer(pat, content, re.DOTALL):
            raw = m.group(0) if '"name"' in m.group(0) else m.group(1)
            if raw in seen:
                continue
            seen.add(raw)
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    obj = json.loads(raw.replace("'", '"'))
                except Exception:
                    continue
            name = obj.get("name") or obj.get("function") or obj.get("tool")
            if not name:
                continue
            args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            results.append({"name": name, "args": args, "id": f"jtc_{len(results)}"})
    return results


# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
Tu es l'assistant d'une école. Tu as des outils pour lire les vraies données.

RÈGLES :
1. Menu, cours, notes, absences, devoirs → appelle l'outil AVANT de répondre.
2. Si le message contient [CONTEXTE DATE: ... date_str="X"], passe exactement X comme date_str.
3. Réponds TOUJOURS dans la langue demandée explicitement. Si la question dit "en français" → réponds en français. Si elle dit "en arabe" → réponds en arabe. Sinon, utilise la langue de la question.
4. CONTEXTE : si la question est courte et imprécise ("et en français ?", "et pour les maths ?"), lis les messages précédents pour comprendre de quoi on parle, puis rappelle le même outil avec les mêmes paramètres.
5. Maximum 4 lignes après le résultat de l'outil.
6. Si l'outil dit "Aucun" → dis-le simplement, n'invente rien.
7. Ne dis jamais "je n'ai pas accès".\
"""

_REMINDER = """\
Tu n'as pas appelé l'outil. Appelle le BON outil maintenant selon le sujet :
  devoirs / homework / travaux → get_student_assignments()
  absences / retards / présence → get_student_attendance()
  notes / résultats / moyenne → get_student_grades()
  menu / cantine / repas → get_canteen_menu(date_str="today")
  cours / emploi du temps → get_student_schedule(date_str="today")
  réunions / meetings → get_student_meetings()
  paiements / frais → get_student_payments()
  annonces / actualités → get_announcements()
Si [CONTEXTE DATE] est présent, utilise exactement le date_str indiqué.\
"""


def _synth_prompt(result: str, lang: str) -> str:
    lang_word = {"fr": "français", "en": "anglais", "ar": "arabe"}.get(lang, "français")
    return (
        f"Résultat de la base de données :\n{result}\n\n"
        f"Reformule en {lang_word} en 2-4 lignes. "
        f"Utilise UNIQUEMENT ces données. N'ajoute rien."
    )


# ──────────────────────────────────────────────────────────────────────
# Moteur principal
# ──────────────────────────────────────────────────────────────────────
class DynamicEngine:
    def __init__(self) -> None:
        _base = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
            num_predict=200,
            num_ctx=2048,
        )
        self._llm = _base.bind_tools(ALL_TOOLS)
        self._llm_synth = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
            num_predict=150,
            num_ctx=1024,
        )
        self._by_name = {t.name: t for t in ALL_TOOLS}

    def answer(self, query: str, history: Optional[List[BaseMessage]] = None, memory_profile: str = "") -> str:
        lang = _detect_lang(query)
        needs_tool = bool(_MUST_USE_TOOL.search(query))

        # ── Normalisation apostrophes/tirets (mobile, clavier FR) ─────────
        query = _normalize(query)

        # ── Pré-routage Python : outil forcé sans LLM ────────────────────
        # Gère aussi les suivis de langue : "et en français ?" après des notes en AR
        forced_tool = _preroute_tool(query, history)

        # Détecter si c'est un suivi de langue (ex: "et en français ?")
        # → on récupère les données du même tool mais on force la langue de réponse
        target_lang = _extract_target_lang(query)
        if target_lang and not forced_tool:
            # Pas de mot-clé de topic → suivi de langue pur
            inherited_tool = _topic_from_history(history)
            if inherited_tool:
                forced_tool = inherited_tool
                print(f"[DynamicEngine] 🌐 Suivi langue ({target_lang}) → {forced_tool}")

        # Utiliser la langue cible si détectée, sinon garder la langue de la question
        effective_lang = target_lang or lang

        if forced_tool:
            print(f"[DynamicEngine] 🎯 Pré-routage → {forced_tool}")
            fn = self._by_name.get(forced_tool)
            if fn:
                args: Dict[str, Any] = _build_tool_args(forced_tool, query)
                try:
                    result = str(fn.invoke(args))
                    print(f"[DynamicEngine] 📤 {result[:300]}")
                    return self._finalize(result, effective_lang, query)
                except Exception as e:
                    print(f"[DynamicEngine] ❌ Pré-routage échoué ({forced_tool}): {e}")
                    # On continue vers le LLM en fallback

        # ── Enrichissement de la requête avec date pré-calculée ──────────
        enriched_query = _build_enriched_query(query)
        if enriched_query != query:
            print(f"[DynamicEngine] 📅 Date injectée : {enriched_query.split('[CONTEXTE')[1][:60]}…")

        tool_results: List[str] = []
        reminder_sent = False

        msgs: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        if history:
            msgs.extend(history[-4:])  # 4 messages = 2 échanges = contexte suffisant
        msgs.append(HumanMessage(content=enriched_query))

        for turn in range(3):
            response: AIMessage = self._llm.invoke(msgs)

            content = (response.content or "").strip()
            native_tcs = list(getattr(response, "tool_calls", None) or [])
            json_tcs = _parse_json_tool_calls(content) if not native_tcs else []
            tool_calls = native_tcs or json_tcs

            # ── Aucun tool call ────────────────────────────────────────
            if not tool_calls:

                # Hallucination → forcer tool
                if content and _HALLUCINATION.search(content) and not reminder_sent:
                    print("[DynamicEngine] ⚠️  Hallucination → rappel")
                    msgs.append(response)
                    msgs.append(HumanMessage(content=_REMINDER))
                    reminder_sent = True
                    continue

                # Tool obligatoire mais pas appelé (1 rappel max)
                if needs_tool and not tool_results and not reminder_sent:
                    print(f"[DynamicEngine] ⚠️  Tour {turn}: tool manquant → rappel")
                    msgs.append(response)
                    msgs.append(HumanMessage(content=_REMINDER))
                    reminder_sent = True
                    continue

                # On a des résultats → finaliser
                if tool_results:
                    return self._finalize("\n\n".join(tool_results), lang, query)

                # Réponse textuelle valide
                if content:
                    return content

                return _no_data(lang)

            # ── Exécution des tool calls ───────────────────────────────
            msgs.append(response)  # on ajoute l'AIMessage AVANT les ToolMessages

            executed_any = False
            for call in tool_calls:
                name = call.get("name", "")
                args = call.get("args", {}) or {}
                tid = call.get("id", f"tc_{turn}")

                # Nettoyer les paramètres inconnus injectés par le LLM
                # (qwen2.5:3b injecte parfois "student_id", "date" au lieu de "date_str")
                args = _sanitize_args(name, args)

                # Correction de date si le LLM a oublié date_str malgré l'enrichissement
                if name == "get_student_schedule" and "date_str" not in args:
                    _, date_str = _extract_date_hint(query)
                    if date_str:
                        args["date_str"] = date_str
                        print(f"[DynamicEngine] 🔧 date_str auto-injecté: {date_str}")

                if name == "get_canteen_menu" and "date_str" not in args:
                    _, date_str = _extract_date_hint(query)
                    if date_str:
                        args["date_str"] = date_str

                print(f"[DynamicEngine] 🔧 {name}({args})")
                fn = self._by_name.get(name)
                if fn:
                    try:
                        result = str(fn.invoke(args))
                    except Exception as e:
                        result = f"Erreur outil {name} : {e}"
                        print(f"[DynamicEngine] ❌ {e}")
                else:
                    result = f"Outil inconnu : {name}"

                print(f"[DynamicEngine] 📤 {result[:300]}")
                tool_results.append(result)
                msgs.append(ToolMessage(content=result, tool_call_id=tid, name=name))
                executed_any = True

            # ── BREAK immédiat après exécution ─────────────────────────
            # On ne re-boucle PAS pour éviter le double appel tool observé dans les logs.
            # On finalise directement avec le résultat obtenu.
            if executed_any:
                break

        # Finalisation
        if tool_results:
            return self._finalize("\n\n".join(tool_results), lang, query)
        return _no_data(lang)

    def _finalize(self, tool_result: str, lang: str, query: str) -> str:
        """
        Retour direct si résultat lisible → économie 20-60s.
        Synthèse LLM uniquement si le résultat est complexe/brut.
        """
        stripped = tool_result.strip()

        if _DIRECT_RESULT.match(stripped):
            print("[DynamicEngine] ✂️  Retour direct (0 LLM synthèse)")
            return stripped

        lang_word = {"fr": "français", "en": "anglais", "ar": "arabe"}.get(lang, "français")
        prompt = (
            f"Question : {query}\n\nDonnées :\n{tool_result}\n\n"
            f"Réponds en {lang_word} en 2 à 4 lignes. Utilise uniquement ces données."
        )
        try:
            resp = self._llm_synth.invoke([
                SystemMessage(content="Reformule des données scolaires brièvement. N'invente rien."),
                HumanMessage(content=prompt),
            ])
            text = (resp.content or "").strip()
            return text if text else stripped
        except Exception as e:
            print(f"[DynamicEngine] ❌ Synthèse échouée : {e}")
            return stripped


def _sanitize_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retire les paramètres que le LLM invente et qui n'existent pas dans le tool.
    Ex: qwen2.5:3b envoie parfois {"student_id": None, "date": "2026-05-02"}
        au lieu de {"date_str": "tomorrow"}.
    """
    VALID_PARAMS: Dict[str, set] = {
        "get_student_schedule"  : {"date_str"},
        "get_canteen_menu"      : {"date_str"},
        "get_student_grades"    : {"subject_name", "period"},
        "get_student_attendance": {"period", "from_date"},
        "get_student_assignments": {"upcoming_only"},
        "get_announcements"     : {"limit"},
        "get_student_meetings"  : {"status", "upcoming_only"},
        "get_student_payments"  : {"status"},
    }
    valid = VALID_PARAMS.get(tool_name)
    if valid is None:
        return args  # tool inconnu → on laisse passer

    cleaned = {k: v for k, v in args.items() if k in valid}

    # Tenter de récupérer date_str si le LLM a utilisé un alias
    if tool_name in ("get_student_schedule", "get_canteen_menu"):
        for alias in ("date", "day", "jour", "target_date"):
            if alias in args and "date_str" not in cleaned:
                cleaned["date_str"] = str(args[alias])
                break

    return cleaned


# ──────────────────────────────────────────────────────────────────────
def _no_data(lang: str) -> str:
    return {
        "ar": "لم أتمكن من الحصول على المعلومات. يرجى المحاولة مجدداً.",
        "en": "I couldn't retrieve the information. Please try again.",
        "fr": "Je n'ai pas pu obtenir les informations. Réessayez.",
    }.get(lang, "Je n'ai pas pu obtenir les informations. Réessayez.")


_engine: Optional[DynamicEngine] = None


def get_dynamic_engine() -> DynamicEngine:
    global _engine
    if _engine is None:
        _engine = DynamicEngine()
    return _engine