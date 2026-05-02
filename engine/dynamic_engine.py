"""
Moteur DYNAMIC v9 — Fusion complète :
- Pré-extraction de date et pré-routage Python (côté distant)
- Support mémoire et prompt court optimisé (côté local)
- Synthèse via llm_synth + fallback amélioré
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
# ──────────────────────────────────────────────────────────────────────
_DAYS_FR = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]
_DAYS_EN = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

_DAY_MAP: Dict[str, int] = {}
for _i, _d in enumerate(_DAYS_FR): _DAY_MAP[_d] = _i
for _i, _d in enumerate(_DAYS_EN): _DAY_MAP[_d] = _i
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
    s = query.lower().strip()
    today = date.today()

    if any(k in s for k in ["aujourd'hui", "aujourd hui", "ajourd'hui", "today", "اليوم", "lyoum", "elyoum", "lyouma"]):
        return today, "today"

    if any(k in s for k in ["demain", "tomorrow", "غدا", "غداً", "ghodwa", "ghodoa", "bokra", "boukra"]):
        return today + timedelta(days=1), "tomorrow"

    if any(k in s for k in ["hier", "yesterday", "أمس", "elbareh", "lbereh", "bareh"]):
        return today - timedelta(days=1), "yesterday"

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
# Détection et patterns
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

_DIRECT_RESULT = re.compile(
    r"^(Emploi du temps du|Menu du|Aucun cours|Pas de cours|"
    r"Il n[' ]y a pas cours|Aucun menu|\d+ note\(s\)|Aucune note|"
    r"Pour (cette semaine|ce mois|les \d+|aujourd|hier)|"
    r"Aucune absence ni retard|\d+ absence|\d+ devoir|Aucun devoir|"
    r"Dernières annonces|Aucune annonce|No classes on|لا دروس|لا يوجد)",
    re.IGNORECASE | re.UNICODE,
)

_GENERAL_QUESTION = re.compile(
    r"\b("
    r"math|maths|équation|calcul|fonction|dérivée|intégrale|théorème|"
    r"exercice|correction|problème|formule|"
    r"guerre|histoire|géographie|politique|économie|"
    r"souviens|rappelle|conversation|précédemment|as-tu|t[' ]es|"
    r"dernière\s+discussion|dernier\s+topic|notre\s+discussion|"
    r"on\s+a\s+parlé|on\s+a\s+discuté|juste\s+avant|tout\s+à\s+l[' ]heure|"
    r"remember|recall|previous|conversation|last\s+time|we\s+talked|"
    r"last\s+discussion|what\s+did\s+we\s+talk|as-tu\s+souvenir|"
    r"تذكر|هل\s+تذكر|محادثة|آخر\s+مرة|تكلمنا|ناقشنا|سبق\s+وتحدثنا"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

_BLOCKED_PHRASES = re.compile(
    r"(je ne peux pas|désol[ée]|ne peux pas fournir|sans avoir|utiliser l'outil|"
    r"sorry|cannot provide|without having|use the tool)",
    re.IGNORECASE | re.UNICODE,
)


# ──────────────────────────────────────────────────────────────────────
# PRÉ-ROUTAGE PYTHON
# ──────────────────────────────────────────────────────────────────────
_TOOL_RULES: list = [
    (re.compile(r"\b(devoir|devoirs|dm|homework|assignment|assignments|"
                r"travaux?\s+[àa]\s+rendre|rendre|à\s+remettre|due|deadline|"
                r"واجب|واجباتي|مهمة|مهام)\b", re.IGNORECASE | re.UNICODE), "get_student_assignments"),
    (re.compile(r"\b(menu|cantine|canteen|cafétéria|manger|repas|déjeuner|dîner|"
                r"nourriture|food|lunch|dinner|plat|qu[' ]est[- ]ce\s+qu[' ]on\s+mange|"
                r"مطعم|وجبة|أكل|طعام|غداء|ماذا\s+(نأكل|يوجد\s+للأكل)|ما\s+هو\s+طعام)\b", 
                re.IGNORECASE | re.UNICODE), "get_canteen_menu"),
    (re.compile(r"\b(emploi\s+du\s+temps|edt|planning|cours\s+(de\s+)?(aujourd|demain|lundi|mardi|"
                r"mercredi|jeudi|vendredi|cette\s+semaine)|j[' ]ai\s+cours|j[' ]ai\s+quoi|"
                r"mes\s+cours|schedule|timetable|class|جدول|جدولي|حصص|حصصي|حصة|مواعيد\s+الدروس)\b",
                re.IGNORECASE | re.UNICODE), "get_student_schedule"),
    (re.compile(r"\b(note|notes|résultat|résultats|moyenne|bulletin|relevé|"
                r"combien\s+j[' ]ai\s+eu|ma\s+note|mes\s+notes|grade|grades|"
                r"average|gpa|score|درجة|درجاتي|نقطة|نقاطي|معدل|نتيجة|نتائجي)\b",
                re.IGNORECASE | re.UNICODE), "get_student_grades"),
    (re.compile(r"\b(absence|absences|absent|retard|retards|présence|"
                r"combien\s+de\s+fois\s+(j[' ]ai\s+)?(été\s+)?absent|"
                r"attendance|tardiness|غياب|غياباتي|تأخر|حضور)\b",
                re.IGNORECASE | re.UNICODE), "get_student_attendance"),
    (re.compile(r"\b(réunion|réunions|rendez[- ]vous|parents[- ]profs|"
                r"meeting|meetings|appointment|اجتماع|اجتماعاتي|موعد)\b",
                re.IGNORECASE | re.UNICODE), "get_student_meetings"),
    (re.compile(r"\b(paiement|paiements|facture|frais|solde|dois[- ]je\s+payer|"
                r"payment|payments|fee|fees|invoice|tuition|مدفوعات|رسوم|فاتورة)\b",
                re.IGNORECASE | re.UNICODE), "get_student_payments"),
    (re.compile(r"\b(annonce|annonces|actualité|actualités|news|announcement|"
                r"إعلان|إعلانات|أخبار)\b", re.IGNORECASE | re.UNICODE), "get_announcements"),
]

def _preroute_tool(query: str) -> Optional[str]:
    for pattern, tool_name in _TOOL_RULES:
        if pattern.search(query):
            return tool_name
    return None


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
BASE_SYSTEM_PROMPT = """Tu es un assistant scolaire.

RÈGLES :
1. menu/emploi/notes/absences/devoirs → appelle outil
2. Réponds en français, 1-4 lignes
3. Ne dis pas "je n'ai pas accès"
"""

SYSTEM_PROMPT = """Tu es l'assistant d'une école. Tu as des outils pour lire les vraies données.

RÈGLES :
1. Menu, cours, notes, absences, devoirs → appelle l'outil AVANT de répondre.
2. Si le message contient [CONTEXTE DATE: ... date_str="X"], passe exactement X comme date_str.
3. Réponds dans la même langue que la question.
4. Maximum 4 lignes après le résultat de l'outil.
5. Si l'outil dit "Aucun" → dis-le simplement, n'invente rien.
6. Ne dis jamais "je n'ai pas accès".
"""

_REMINDER = """Tu n'as pas appelé l'outil. Appelle le BON outil maintenant selon le sujet :
  devoirs / homework / travaux → get_student_assignments()
  absences / retards / présence → get_student_attendance()
  notes / résultats / moyenne → get_student_grades()
  menu / cantine / repas → get_canteen_menu(date_str="today")
  cours / emploi du temps → get_student_schedule(date_str="today")
  réunions / meetings → get_student_meetings()
  paiements / frais → get_student_payments()
  annonces / actualités → get_announcements()
"""


# ──────────────────────────────────────────────────────────────────────
# Moteur principal
# ──────────────────────────────────────────────────────────────────────
class DynamicEngine:
    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
            num_predict=2048,
            num_ctx=2048,
        ).bind_tools(ALL_TOOLS)
        
        self._llm_synth = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
            num_predict=150,
            num_ctx=1024,
        )
        self._by_name = {t.name: t for t in ALL_TOOLS}

    def answer(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
        memory_profile: str = "",
    ) -> str:
        # Log mémoire
        if memory_profile:
            print(f"[DynamicEngine] 📝 Profil mémoire reçu ({len(memory_profile)} chars)")
        else:
            print("[DynamicEngine] ⚠️ Profil mémoire vide")

        lang = _detect_lang(query)

        # ── Questions générales (maths, mémoire) ──────────────────────────
        if _GENERAL_QUESTION.search(query):
            print(f"[DynamicEngine] 📚 Question générale: {query[:80]}")
            return self._handle_general_query(query, history, memory_profile, lang)

        # ── Pré-routage Python ───────────────────────────────────────────
        forced_tool = _preroute_tool(query)
        if forced_tool:
            print(f"[DynamicEngine] 🎯 Pré-routage → {forced_tool}")
            fn = self._by_name.get(forced_tool)
            if fn:
                args: Dict[str, Any] = {}
                if forced_tool in ("get_student_schedule", "get_canteen_menu"):
                    _, date_str = _extract_date_hint(query)
                    if date_str:
                        args["date_str"] = date_str
                try:
                    result = str(fn.invoke(args))
                    return self._finalize(result, lang, query)
                except Exception as e:
                    print(f"[DynamicEngine] ❌ Pré-routage échoué: {e}")

        # ── Enrichissement de la requête avec date ────────────────────────
        enriched_query = _build_enriched_query(query)
        if enriched_query != query:
            print("[DynamicEngine] 📅 Date injectée")

        needs_tool = bool(_MUST_USE_TOOL.search(query))
        tool_results: List[str] = []
        reminder_sent = False

        system = SYSTEM_PROMPT if not memory_profile else SYSTEM_PROMPT + f"\n\nHISTORIQUE ÉLÈVE: {memory_profile[:500]}"
        msgs: List[BaseMessage] = [SystemMessage(content=system)]
        if history:
            msgs.extend(history[-4:])
        msgs.append(HumanMessage(content=enriched_query))

        for turn in range(3):
            response: AIMessage = self._llm.invoke(msgs)

            content = (response.content or "").strip()
            native_tcs = list(getattr(response, "tool_calls", None) or [])
            json_tcs = _parse_json_tool_calls(content) if not native_tcs else []
            tool_calls = native_tcs or json_tcs

            if not tool_calls:
                if content and _HALLUCINATION.search(content) and not reminder_sent:
                    print("[DynamicEngine] ⚠️ Hallucination → rappel")
                    msgs.append(response)
                    msgs.append(HumanMessage(content=_REMINDER))
                    reminder_sent = True
                    continue

                if needs_tool and not tool_results and not reminder_sent:
                    print(f"[DynamicEngine] ⚠️ Tour {turn}: tool manquant → rappel")
                    msgs.append(response)
                    msgs.append(HumanMessage(content=_REMINDER))
                    reminder_sent = True
                    continue

                if tool_results:
                    return self._finalize("\n\n".join(tool_results), lang, query)

                if content and not needs_tool:
                    return content

                if content:
                    return content

                return _no_data(lang)

            # ── Exécution des tool calls ───────────────────────────────────
            msgs.append(response)
            executed_any = False

            for call in tool_calls:
                name = call.get("name", "")
                args = call.get("args", {}) or {}
                tid = call.get("id", f"tc_{turn}")

                args = self._sanitize_args(name, args)

                # Correction de date
                if name in ("get_student_schedule", "get_canteen_menu") and "date_str" not in args:
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
                else:
                    result = f"Outil inconnu : {name}"

                print(f"[DynamicEngine] 📤 {result[:300]}")
                tool_results.append(result)
                msgs.append(ToolMessage(content=result, tool_call_id=tid, name=name))
                executed_any = True

            if executed_any:
                break

        if tool_results:
            return self._finalize("\n\n".join(tool_results), lang, query)
        return _no_data(lang)

    def _handle_general_query(
        self, 
        query: str, 
        history: Optional[List[BaseMessage]], 
        memory_profile: str, 
        lang: str
    ) -> str:
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
            return response.content or _no_data(lang)
        except Exception as e:
            print(f"[DynamicEngine] ❌ Erreur: {e}")
            return _no_data(lang)

    def _finalize(self, tool_result: str, lang: str, query: str) -> str:
        stripped = tool_result.strip()

        if _DIRECT_RESULT.match(stripped):
            print("[DynamicEngine] ✂️ Retour direct")
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

    def _sanitize_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        VALID_PARAMS: Dict[str, set] = {
            "get_student_schedule": {"date_str"},
            "get_canteen_menu": {"date_str"},
            "get_student_grades": {"subject_name", "period"},
            "get_student_attendance": {"period", "from_date"},
            "get_student_assignments": {"upcoming_only"},
            "get_announcements": {"limit"},
            "get_student_meetings": {"status", "upcoming_only"},
            "get_student_payments": {"status"},
        }
        valid = VALID_PARAMS.get(tool_name)
        if valid is None:
            return args

        cleaned = {k: v for k, v in args.items() if k in valid}

        if tool_name in ("get_student_schedule", "get_canteen_menu"):
            for alias in ("date", "day", "jour", "target_date"):
                if alias in args and "date_str" not in cleaned:
                    cleaned["date_str"] = str(args[alias])
                    break

        return cleaned


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