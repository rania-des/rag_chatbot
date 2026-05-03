"""
Routeur hybride — 4 branches : GREET, FAQ, DYNAMIC, COURSE.

Logique ENTIÈREMENT basée sur règles (0 appel LLM pour le routage).

Ordre de priorité :
  1. GREET   — salutations simples (regex ou mots clés) → réponse instantanée
  2. COURSE  — course_id fourni + question pédagogique → RAG sur cours
  3. DYNAMIC — mots-clés données BD / personnels → Supabase
  4. FAQ     — similarité vectorielle ≥ seuil (adaptatif selon langue) → FAISS
  5. DYNAMIC — par défaut si rien ne matche assez

Seuils FAQ (v2 améliorée) :
  - Seuil de base : 0.78 (configurable via settings.FAQ_SIMILARITY_THRESHOLD)
  - Boost arabe   : -0.06 (car E5 multilingue sous-score l'arabe)
  - Seuil effectif FR/EN : 0.78
  - Seuil effectif AR    : 0.72
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Tuple

from langchain_core.documents import Document

from config import settings
from engine.faq_engine import get_faq_engine


class Route(str, Enum):
    GREET   = "GREET"
    FAQ     = "FAQ"
    DYNAMIC = "DYNAMIC"
    COURSE  = "COURSE"


# ══════════════════════════════════════════════════════════════════════
# 0. NORMALISATION DES ABRÉVIATIONS (version binôme)
# ══════════════════════════════════════════════════════════════════════
_ABREVIATIONS = {
    # Français SMS
    "pb": "problème",
    "stp": "s'il te plaît",
    "svp": "s'il vous plaît",
    "msg": "message",
    "rdv": "rendez-vous",
    "tjrs": "toujours",
    "bcp": "beaucoup",
    "dc": "donc",
    "pk": "pourquoi",
    "pcq": "parce que",
    "qd": "quand",
    "ac": "avec",
    "ss": "sans",
    "tt": "tout",
    "ts": "tous",
    "nv": "nouveau",
    "mtn": "maintenant",
    "auj": "aujourd'hui",
    "dem": "demain",
    "exerc": "exercice",
    "exo": "exercice",
    "exos": "exercices",
    "correcc": "correction",
    "corec": "correction",
    "explic": "explication",
    "ds": "dans",
    "pr": "pour",
    "vs": "vous",
    "ct": "c'était",
    "jvx": "je veux",
    "jsp": "je ne sais pas",
    "chui": "je suis",
    "g": "j'ai",
    "ya": "il y a",
    "kelke": "quelque",
    # Abréviations scolaires
    "maths": "mathématiques",
    "svt": "sciences de la vie et de la terre",
    "eps": "éducation physique",
    "em": "emploi du temps",
    "edt": "emploi du temps",
    "interro": "interrogation",
    "controle": "contrôle",
    "dm": "devoir maison",
    "tp": "travaux pratiques",
    "td": "travaux dirigés",
    "cours": "cours",
    "notes": "notes",
    "abs": "absences",
    "absences": "absences",
    "devoirs": "devoirs",
    "menu": "menu",
    "cantine": "cantine",
    "heure": "heure",
    "planning": "planning",
    "schedule": "emploi du temps",
    "grade": "note",
    "grades": "notes",
}


def _normalize(text: str) -> str:
    """Remplace les abréviations par leur forme complète pour le matching."""
    words = re.findall(r"[\w'\-]+|[^\w\s]", text, re.UNICODE)
    result = []
    for w in words:
        if re.match(r"[\w'\-]+$", w):
            w_lower = w.lower()
            if w_lower in _ABREVIATIONS:
                result.append(_ABREVIATIONS[w_lower])
            else:
                result.append(w)
        else:
            result.append(w)
    normalized = ""
    for i, w in enumerate(result):
        if i > 0 and not re.match(r"^[.,!?;:)]$", w):
            normalized += " "
        normalized += w
    return normalized


# ══════════════════════════════════════════════════════════════════════
# 1. GREET — réponse prédéfinie, 0 LLM, 0 Supabase
# ══════════════════════════════════════════════════════════════════════
_GREET_REGEX = re.compile(
    r"^\s*("
    r"bonjour|bonsoir|salut|coucou|bonne\s+journée|bonne\s+soirée|bonne\s+nuit|"
    r"bonne\s+matinée|bonne\s+année|bienvenue|au\s+revoir|à\s+bientôt|"
    r"hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|good\s+night|"
    r"howdy|greetings|welcome|goodbye|bye|see\s+you|"
    r"مرحبا|أهلا|أهلاً|السلام\s+عليكم|وعليكم\s+السلام|صباح\s+الخير|"
    r"مساء\s+الخير|تصبح\s+على\s+خير|مرحباً|وداعاً|إلى\s+اللقاء|"
    r"oui|non|ok|d'accord|merci|شكرا|شكراً|عفواً|thanks|thank\s+you"
    r")\s*[!،,\.؟?]*\s*$",
    re.IGNORECASE | re.UNICODE,
)

# Ensemble des mots pour reconnaissance des phrases courtes (rajout utilisateur)
_GREET_WORDS = {
    "bonjour", "bonsoir", "salut", "coucou", "merci", "d'accord", "ok",
    "oui", "non", "thanks", "thank", "you", "bye", "hello", "hi", "hey",
    "مرحبا", "شكرا", "عفواً", "تمام", "حسناً", "نعم", "لا",
}

def _is_greet_phrase(text: str) -> bool:
    """Détecte les phrases courtes (1 à 5 mots) dont tous les mots sont dans _GREET_WORDS."""
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    words = cleaned.split()
    return 1 <= len(words) <= 5 and all(w in _GREET_WORDS for w in words)


GREET_RESPONSES = {
    "ar": (
        "مرحباً! 👋 كيف يمكنني مساعدتك اليوم؟\n"
        "يمكنني عرض جدولك الدراسي، نقاطك، قائمة المطعم، غياباتك، واجباتك، "
        "أو مساعدتك في فهم أي درس إذا رفعت الملف. فقط اسألني!"
    ),
    "en": (
        "Hello! 👋 How can I help you today?\n"
        "I can show your schedule, grades, cafeteria menu, absences, homework, "
        "or help you understand any course if you upload the file. Just ask!"
    ),
    "fr": (
        "Bonjour ! 👋 Comment puis-je vous aider ?\n"
        "Je peux afficher votre emploi du temps, vos notes, le menu de la cantine, "
        "vos absences, vos devoirs, ou vous expliquer un cours si vous uploadez le fichier. "
        "Posez votre question !"
    ),
}


def _greet_lang(query: str) -> str:
    if re.search(r"[\u0600-\u06FF]", query):
        return "ar"
    if re.search(r"\b(hello|hi|hey|good|morning|evening|bye|thanks|thank)\b", query, re.I):
        return "en"
    return "fr"


def get_greet_response(query: str) -> str:
    return GREET_RESPONSES[_greet_lang(query)]


# ══════════════════════════════════════════════════════════════════════
# 2. DYNAMIC — données temps réel / personnelles / mémoire
# ══════════════════════════════════════════════════════════════════════
_DYNAMIC = re.compile(
    r"""
    # ── CANTINE / REPAS ────────────────────────────────────────────────────
    \b(
      menu|menus|cantine|canteen|cafétéria|cafeteria|
      manger|repas|plat|plats|déjeuner|dîner|lunch|dinner|
      nourriture|food|réfectoire|
      qu[' ]est[- ]ce\s+qu[' ]il\s+y\s+a\s+[àa]\s+manger|
      qu[' ]est[- ]ce\s+qu[' ]on\s+mange|on\s+mange\s+quoi|
      il\s+y\s+a\s+quoi\s+[àa]\s+manger|
      what[' ]s\s+for\s+(lunch|dinner|breakfast)|what\s+are\s+we\s+eating|
      مطعم|وجبة|أكل|طعام|غداء|عشاء|ماذا\s+نأكل|ماذا\s+يوجد\s+للأكل
    )\b |

    # ── EMPLOI DU TEMPS / COURS ─────────────────────────────────────────────
    \b(
      emploi\s+du\s+temps|edt|planning|plan\s+de\s+cours|
      mes\s+cours|j[' ]ai\s+cours|j[' ]ai\s+quoi|qu[' ]est[- ]ce\s+que\s+j[' ]ai|
      quelle\s+heure\s+commence|à\s+quelle\s+heure\s+(commence|finit|j[' ]ai)|
      schedule|timetable|my\s+classes|today[' ]s\s+class|
      when\s+(do\s+i\s+have|is\s+my)\s+class|what\s+(class|course)\s+do\s+i\s+have|
      جدولي|حصتي|مواعيد\s+الدراسة|متى\s+(عندي|لدي)\s+(حصة|درس)
    )\b |

    # ── NOTES / RÉSULTATS (données PERSONNELLES) ─────────────────────────────
    \b(
      mes\s+notes?|ma\s+note|ma\s+moyenne|mes\s+moyennes?|
      mes\s+résultats?|mon\s+bulletin|mon\s+relevé|
      j[' ]ai\s+eu|j[' ]ai\s+obtenu|combien\s+j[' ]ai\s+eu|
      quelle\s+est\s+ma\s+moyenne|
      my\s+(grade|grades|result|results|average|gpa|transcript)|
      how\s+did\s+i\s+do|what\s+(grade|score)\s+did\s+i\s+get|
      درجاتي|نقاطي|معدلي|نتائجي|كشف\s+النقاط\s+(الخاص\s+بي)?
    )\b |

    # ── ABSENCES / RETARDS (données PERSONNELLES) ────────────────────────────
    \b(
      mes\s+absences?|mes\s+retards?|j[' ]ai\s+(combien\s+d[' ])?absences?|
      suis[- ]je\s+absent|ai[- ]je\s+(des\s+)?absences?|
      combien\s+de\s+fois\s+(j[' ]ai\s+)?(été\s+)?absent|
      my\s+(attendance|absence|absences)|am\s+i\s+absent|
      do\s+i\s+have\s+any\s+absences?|how\s+many\s+times\s+(was\s+i|did\s+i)\s+absent|
      غيابي|غياباتي|تأخري|هل\s+تغيبت|كم\s+مرة\s+تغيبت
    )\b |

    # ── DEVOIRS (données PERSONNELLES) ──────────────────────────────────────
    \b(
      mes\s+devoirs?|j[' ]ai\s+(un\s+devoir|des\s+devoirs)|
      qu[' ]est[- ]ce\s+que\s+j[' ]ai\s+à\s+(faire|rendre)|
      qu[' ]est[- ]ce\s+qu[' ]il\s+faut\s+rendre|à\s+remettre|
      my\s+(homework|assignment)|what\s+homework\s+do\s+i\s+have|what\s+is\s+due|
      واجبي|واجباتي|ما\s+(الذي\s+)?(علي|يجب\s+علي)\s+تسليمه
    )\b |

    # ── ANNONCES ─────────────────────────────────────────────────────────────
    \b(
      annonces?|actualités?|quoi\s+de\s+neuf|informations?\s+importantes?|
      announcement|news|updates?|what[' ]s\s+new|
      إعلان|إعلانات|أخبار|مستجدات|ما\s+الجديد
    )\b |

    # ── RÉUNIONS ──────────────────────────────────────────────────────────────
    \b(
      mes\s+réunions?|ma\s+prochaine\s+réunion|
      rendez[- ]vous\s+(parents?|avec\s+le\s+prof)|conseil\s+de\s+classe|
      my\s+(meeting|appointment)|
      اجتماعاتي|موعد\s+(مع\s+المعلم|الأولياء)|لقاء\s+(الأولياء|الآباء)
    )\b |

    # ── PAIEMENTS ────────────────────────────────────────────────────────────
    \b(
      mes\s+paiements?|est[- ]ce\s+que\s+j[' ]ai\s+payé|
      dois[- ]je\s+payer|montant\s+dû|reste\s+à\s+payer|
      my\s+(payment|fee|invoice)|do\s+i\s+owe|
      مدفوعاتي|هل\s+دفعت|ما\s+(يجب\s+علي)\s+دفعه
    )\b |

    # ── TEMPORALITÉ (impose données temps réel) ───────────────────────────────
    \b(
      aujourd[' ]hui|ce\s+matin|ce\s+midi|ce\s+soir|maintenant|
      demain|après[- ]demain|hier|avant[- ]hier|
      cette\s+semaine|la\s+semaine\s+prochaine|la\s+semaine\s+dernière|
      ce\s+mois[- ]ci|ce\s+trimestre|
      lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|
      today|tomorrow|yesterday|this\s+week|next\s+week|last\s+week|
      this\s+month|monday|tuesday|wednesday|thursday|friday|saturday|sunday|
      اليوم|الآن|غداً?|أمس|هذا\s+الأسبوع|الأسبوع\s+القادم|الأسبوع\s+الماضي|
      الاثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت|الأحد|
      lyoum|elyoum|ghodwa|bokra|lbereh|hal\s+jem3a|hal\s+chhar
    )\b |

    # ── MÉMOIRE LONG-TERME / SOUVENIRS ──────────────────────────────────────
    \b(
      souvien|souviens|rappelle|rappelles|rappel|dernière\s+fois|on\s+a\s+parlé|
      précédemment|avant|hier\s+(on|tu|j[' ]ai)|tu\s+te\s+souviens|est-ce\s+que\s+tu\s+te\s+souviens|
      dernière\s+discussion|dernier\s+topic|notre\s+conversation|
      remember|last\s+time|we\s+talked|last\s+discussion|do\s+you\s+remember|
      تذكر|هل\s+تذكر|آخر\s+مرة|سبق\s+وتحدثنا|محادثتنا
    )\b |

    # ── POSSESSIFS 1ÈRE PERSONNE ────────────────────────────────────────────
    \bmes\s+(notes?|devoirs?|absences?|cours|retards?|paiements?|
             réunions?|moyennes?|résultats?|professeurs?)\b |
    \bmon\s+(emploi|bulletin|résultat|prof|professeur)\b |
    \bma\s+(note|moyenne|classe)\b |
    \bj[' ]ai\s+(combien|quoi|cours|un\s+devoir|des\s+absences?|payé)\b |
    \bje\s+(dois|veux\s+voir|voudrais\s+voir|souhaite\s+voir|vais\s+avoir)\b |
    \best[- ]ce\s+que\s+j[' ]ai\b |
    \bai[- ]je\b |
    \bmy\s+(grade|grades|schedule|class|homework|assignment|
             attendance|absence|result|average|payment)\b |
    \b(do|did)\s+i\s+(have|get|pass)\b |
    \bعندي\b | \bلدي\b | \bدرجاتي\b | \bجدولي\b | \bغيابي\b | \bواجبي\b |
    \bنقاطي\b | \bمعدلي\b | \bمدفوعاتي\b
    """,
    re.IGNORECASE | re.UNICODE | re.VERBOSE,
)

# Questions admin qui forcent DYNAMIC même si un cours est actif
_ADMIN = re.compile(
    r"\b("
    r"menu|cantine|canteen|manger|repas|nourriture|food|"
    r"mes\s+notes?|mes\s+résultats?|ma\s+moyenne|mon\s+bulletin|"
    r"emploi\s+du\s+temps|horaire|schedule|timetable|planning|"
    r"absence|absences|retard|attendance|"
    r"devoir|devoirs|homework|assignment|"
    r"paiement|paiements|payment|frais|fee|"
    r"réunion|meeting|annonce|announcement|"
    r"souvien|souviens|rappelle|remember|last\s+time|"
    r"مطعم|وجبة|درجة|جدول|غياب|واجب|مدفوعات|إعلان|تذكر"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

# ── Détection de langue (pour le boost arabe) ────────────────────────────────
def _is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))

# ── Seuil FAQ (version adaptative) ────────────────────────────────────────────
_FAQ_THRESHOLD_BASE = getattr(settings, "FAQ_SIMILARITY_THRESHOLD", 0.78)
_FAQ_ARABIC_BOOST = 0.06  # seuil effectif arabe = base - 0.06


# ══════════════════════════════════════════════════════════════════════
# Routeur
# ══════════════════════════════════════════════════════════════════════
class Router:
    def __init__(self) -> None:
        self._faq = get_faq_engine()

    def route(
        self,
        query: str,
        course_id: Optional[str] = None,
        memory_profile: str = "",  # non utilisé actuellement, réservé
    ) -> Tuple[Route, Optional[Document], float]:
        """
        Retourne (route, faq_doc_ou_None, score).
        Aucun appel LLM — 100% règles.
        """
        # Normalisation des abréviations
        q_original = query.strip()
        q_normalized = _normalize(q_original)
        q = q_normalized  # on utilise la version normalisée pour le matching

        if q_normalized != q_original:
            print(f"[Router] 🔧 Normalisation: {q_original[:50]}... → {q_normalized[:50]}...")

        # 1. GREET (regex ou phrase courte)
        if _GREET_REGEX.match(q) or _is_greet_phrase(q_original):
            print(f"[Router] 👋 GREET: {q_original!r}")
            return Route.GREET, None, 0.0

        # 2. COURSE — cours uploadé actif, question pédagogique
        if course_id and not _ADMIN.search(q):
            print(f"[Router] 📚 COURSE: course_id={course_id}")
            return Route.COURSE, None, 0.0

        # 3. DYNAMIC — mot-clé données BD / personnel / temporel / mémoire
        if _DYNAMIC.search(q):
            print(f"[Router] ⚡ DYNAMIC (keyword): {q_original[:80]!r}")
            return Route.DYNAMIC, None, 0.0

        # 4. FAQ — recherche vectorielle avec seuil adaptatif (selon la langue)
        is_ar = _is_arabic(q_original)
        threshold = (
            _FAQ_THRESHOLD_BASE - _FAQ_ARABIC_BOOST
            if is_ar
            else _FAQ_THRESHOLD_BASE
        )

        match = self._faq.best_match(q_original)  # on utilise l'original pour la FAQ
        if match:
            doc, score = match
            if score >= threshold:
                lang_tag = "AR" if is_ar else "FR/EN"
                print(
                    f"[Router] 📖 FAQ ({lang_tag}, score={score:.3f}≥{threshold:.3f}): "
                    f"{q_original[:80]!r}"
                )
                return Route.FAQ, doc, score
            else:
                print(
                    f"[Router] ℹ️  FAQ score trop bas ({score:.3f}<{threshold:.3f}) "
                    f"→ DYNAMIC: {q_original[:60]!r}"
                )

        # 5. DYNAMIC par défaut
        print(
            f"[Router] ⚡ DYNAMIC (default, score={match[1] if match else 0:.3f}): "
            f"{q_original[:80]!r}"
        )
        return Route.DYNAMIC, None, match[1] if match else 0.0


# ══════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════
_router: Optional[Router] = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router