"""
Routeur hybride — 4 branches : GREET, FAQ, DYNAMIC, COURSE.

Logique ENTIÈREMENT basée sur règles (0 appel LLM pour le routage).

Ordre de priorité :
  1. GREET   — salutations simples (regex)         → réponse instantanée
  2. COURSE  — course_id fourni + question pédago  → RAG sur cours
  3. DYNAMIC — mots-clés données BD / personnels   → Supabase
  4. FAQ     — similarité vectorielle ≥ seuil      → FAISS
  5. DYNAMIC — par défaut si rien ne matche assez

Philosophie DYNAMIC :
  Toute question dont la réponse DÉPEND de l'élève ou du MOMENT
  doit passer par Supabase, même si la formulation est ambiguë.
  Vaut mieux appeler un tool inutilement que rater une donnée réelle.
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
# 0. NORMALISATION DES ABRÉVIATIONS
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
    "ds": "devoir surveillé",
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
    # Séparer la ponctuation pour mieux traiter les mots
    words = re.findall(r"[\w'\-]+|[^\w\s]", text, re.UNICODE)
    result = []
    for w in words:
        # Si c'est un mot (pas de la ponctuation)
        if re.match(r"[\w'\-]+$", w):
            w_lower = w.lower()
            # Remplacer si c'est une abréviation
            if w_lower in _ABREVIATIONS:
                result.append(_ABREVIATIONS[w_lower])
            else:
                result.append(w)
        else:
            # Ponctuation inchangée
            result.append(w)
    
    # Reconstruire la phrase en gardant les espaces après les mots
    normalized = ""
    for i, w in enumerate(result):
        if i > 0 and not re.match(r"^[.,!?;:)]$", w):
            normalized += " "
        normalized += w
    
    return normalized


# ══════════════════════════════════════════════════════════════════════
# 1. GREET — réponse prédéfinie, 0 LLM, 0 Supabase
# ══════════════════════════════════════════════════════════════════════
_GREET = re.compile(
    r"^\s*("
    # Français
    r"bonjour|bonsoir|salut|coucou|bonne\s+journée|bonne\s+soirée|bonne\s+nuit|"
    r"bonne\s+matinée|bonne\s+année|bienvenue|au\s+revoir|à\s+bientôt|"
    # Anglais
    r"hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|good\s+night|"
    r"howdy|greetings|welcome|goodbye|bye|see\s+you|"
    # Arabe standard
    r"مرحبا|أهلا|أهلاً|السلام\s+عليكم|وعليكم\s+السلام|صباح\s+الخير|"
    r"مساء\s+الخير|تصبح\s+على\s+خير|مرحباً|وداعاً|إلى\s+اللقاء|"
    # Réponses courtes sans contexte
    r"oui|non|ok|d'accord|merci|شكرا|شكراً|عفواً|thanks|thank\s+you"
    r")\s*[!،,\.؟?]*\s*$",
    re.IGNORECASE | re.UNICODE,
)

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
    if re.search(r"\b(hello|hi|hey|good|morning|evening|bye|thanks)\b", query, re.I):
        return "en"
    return "fr"


def get_greet_response(query: str) -> str:
    return GREET_RESPONSES[_greet_lang(query)]


# ══════════════════════════════════════════════════════════════════════
# 2. DYNAMIC — tout ce qui nécessite Supabase
#
# Règle de conception :
#   Si la réponse CORRECTE dépend de l'identité de l'élève OU du moment
#   présent, c'est DYNAMIC. Mieux vaut sur-déclencher que rater.
#
# On couvre :
#   - Cantine / repas / nourriture (menu = TOUJOURS dynamique)
#   - Emploi du temps / planning / cours du jour
#   - Notes / résultats / moyennes / bulletins
#   - Absences / retards / présence
#   - Devoirs / exercices à rendre
#   - Annonces / actualités de classe
#   - Réunions parents-profs
#   - Paiements / frais en attente
#   - Tout ce qui contient "aujourd'hui", "demain", "cette semaine", etc.
#   - Pronoms personnels 1ère personne (mes, mon, ma, j'ai, je dois...)
#   - MÉMOIRE LONG-TERME : souvenirs, rappels, conversations précédentes
# ══════════════════════════════════════════════════════════════════════
_DYNAMIC = re.compile(
    r"""
    # ── CANTINE / REPAS (TOUJOURS dynamique — menu change chaque jour) ─────
    \b(
      menu|menus|cantine|canteen|cafétéria|cafeteria|
      manger|repas|plat|plats|déjeuner|dîner|lunch|dinner|
      nourriture|food|cuisine|chef|self|réfectoire|
      qu[' ]est[- ]ce\s+qu[' ]il\s+y\s+a\s+[àa]\s+manger|
      qu[' ]est[- ]ce\s+qu[' ]on\s+mange|
      on\s+mange\s+quoi|il\s+y\s+a\s+quoi\s+[àa]\s+manger|
      what[' ]s\s+for\s+(lunch|dinner|breakfast|eat)|
      what\s+are\s+we\s+eating|
      مطعم|وجبة|أكل|طعام|غداء|عشاء|ماذا\s+نأكل|ماذا\s+يوجد\s+للأكل
    )\b |

    # ── EMPLOI DU TEMPS / COURS DU JOUR ────────────────────────────────────
    \b(
      emploi\s+du\s+temps|edt|horaires?\s+(de\s+)?(cours|classe|mes\s+cours)|planning|plan\s+de\s+cours|
      cours\s+(de\s+la\s+)?journée|programme\s+(de\s+la\s+)?journée|
      mes\s+cours|j[' ]ai\s+cours|j[' ]ai\s+quoi|qu[' ]est[- ]ce\s+que\s+j[' ]ai|
      quelle\s+heure\s+commence|à\s+quelle\s+heure|jusqu[' ]à\s+quelle\s+heure|
      schedule|timetable|class\s+schedule|my\s+classes|today[' ]s\s+class|
      when\s+(do\s+i\s+have|is\s+my)\s+class|what\s+(class|course)\s+do\s+i\s+have|
      جدول|جدولي|حصص|حصتي|مواعيد\s+الدراسة|متى\s+(عندي|لدي)\s+(حصة|درس)
    )\b |

    # ── NOTES / RÉSULTATS / MOYENNE ─────────────────────────────────────────
    \b(
      note|notes|résultat|résultats|moyenne|moyennes|bulletin|bulletins|
      bilan\s+scolaire|relevé\s+de\s+notes|score|scores|
      j[' ]ai\s+eu|j[' ]ai\s+obtenu|combien\s+j[' ]ai\s+eu|ma\s+note|
      mes\s+résultats|mon\s+bulletin|quelle\s+est\s+ma\s+moyenne|
      grade|grades|result|results|average|gpa|transcript|report\s+card|
      how\s+did\s+i\s+do|what\s+(grade|score)\s+did\s+i\s+get|
      درجة|درجاتي|نقطة|نقاطي|معدل|نتيجة|نتائجي|كشف\s+النقاط
    )\b |

    # ── ABSENCES / PRÉSENCE / RETARDS ───────────────────────────────────────
    \b(
      absence|absences|absent|présence|présences|retard|retards|
      j[' ]ai\s+(combien\s+d[' ])?absences?|mes\s+absences?|
      suis[- ]je\s+absent|ai[- ]je\s+(des\s+)?absences?|
      combien\s+de\s+fois\s+(j[' ]ai\s+)?(été\s+)?absent|
      attendance|absent|tardiness|late|
      am\s+i\s+absent|do\s+i\s+have\s+any\s+absences?|
      غياب|غياباتي|تأخر|تأخراتي|حضور|هل\s+تغيبت
    )\b |

    # ── DEVOIRS / EXERCICES À RENDRE ────────────────────────────────────────
    \b(
      devoir|devoirs|exercice|exercices|travail\s+à\s+rendre|tâche|tâches|
      rendre|à\s+remettre|délai|date\s+limite|
      qu[' ]est[- ]ce\s+que\s+j[' ]ai\s+à\s+faire|qu[' ]est[- ]ce\s+qu[' ]il\s+faut\s+rendre|
      homework|assignment|assignments|due\s+date|deadline|
      what\s+(homework|assignment)\s+do\s+i\s+have|what\s+is\s+due|
      واجب|واجباتي|تمرين|مهام|ما\s+الذي\s+(علي|يجب\s+علي)\s+تسليمه
    )\b |

    # ── ANNONCES / ACTUALITÉS ───────────────────────────────────────────────
    \b(
      annonce|annonces|actualité|actualités|notification|notifications|
      nouveauté|nouveautés|quoi\s+de\s+neuf|informations?\s+importantes?|
      news|announcement|announcements|updates?|what[' ]s\s+new|
      إعلان|إعلانات|أخبار|مستجدات|ما\s+الجديد
    )\b |

    # ── RÉUNIONS PARENTS-PROFS ──────────────────────────────────────────────
    \b(
      réunion|réunions|rencontre|rendez[- ]vous|rdv|conseil\s+de\s+classe|
      réunion\s+parents|parents[- ]profs|prochaine\s+réunion|
      meeting|meetings|appointment|parent[- ]teacher|
      اجتماع|اجتماعاتي|موعد|مواعيد|لقاء\s+(الأولياء|الآباء)
    )\b |

    # ── PAIEMENTS / FRAIS ───────────────────────────────────────────────────
    \b(
      paiement|paiements|facture|factures|frais|solde|
      dois[- ]je\s+payer|est[- ]ce\s+que\s+j[' ]ai\s+payé|
      montant\s+dû|reste\s+à\s+payer|en\s+attente\s+de\s+paiement|
      payment|payments|fee|fees|invoice|tuition|balance\s+due|
      مدفوعات|رسوم|فاتورة|ما\s+(يجب|عليّ)\s+دفعه|رصيد
    )\b |

    # ── TEMPORALITÉ — impose une donnée en temps réel ───────────────────────
    \b(
      aujourd[' ]hui|ce\s+matin|ce\s+midi|ce\s+soir|maintenant|
      demain|après[- ]demain|
      hier|avant[- ]hier|
      cette\s+semaine|la\s+semaine\s+prochaine|la\s+semaine\s+dernière|
      ce\s+mois[- ]ci|ce\s+trimestre|ce\s+semestre|
      lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|
      today|tomorrow|yesterday|this\s+week|next\s+week|last\s+week|
      this\s+month|monday|tuesday|wednesday|thursday|friday|saturday|sunday|
      اليوم|الآن|غداً?|أمس|هذا\s+الأسبوع|الأسبوع\s+القادم|الأسبوع\s+الماضي|
      الاثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت|الأحد
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

# Mots qui signalent une question admin même quand un cours est actif
# → force le passage en DYNAMIC plutôt que COURSE
_ADMIN = re.compile(
    r"\b("
    r"menu|cantine|canteen|cafétéria|manger|repas|nourriture|food|"
    r"note|notes|résultat|résultats|moyenne|bulletin|grade|grades|"
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

_FAQ_THRESHOLD = max(0.88, settings.FAQ_SIMILARITY_THRESHOLD)


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
        memory_profile: str = "",
    ) -> Tuple[Route, Optional[Document], float]:
        """
        Retourne (route, faq_doc_ou_None, score).
        Aucun appel LLM — 100% règles.
        
        Args:
            query: Question de l'élève
            course_id: ID du cours uploadé (optionnel)
            memory_profile: Profil mémoire de l'élève (optionnel, utilisé dans DYNAMIC)
        """
        # ═══════════════════════════════════════════════════════════════
        # NORMALISATION : remplacer les abréviations pour le matching
        # ═══════════════════════════════════════════════════════════════
        q_original = query.strip()
        q_normalized = _normalize(q_original)
        
        # Utiliser la version normalisée pour le matching
        q = q_normalized
        
        # Log de la normalisation (utile pour déboguer)
        if q_normalized != q_original:
            print(f"[Router] 🔧 Normalisation: {q_original[:50]}... → {q_normalized[:50]}...")

        # 1. GREET — salutation / message très court sans contenu
        if _GREET.match(q):
            print(f"[Router] 👋 GREET: {q_original!r}")
            return Route.GREET, None, 0.0

        # 2. COURSE — cours uploadé actif et question pédagogique
        #    (si la question parle aussi de données admin → DYNAMIC quand même)
        if course_id and not _ADMIN.search(q):
            print(f"[Router] 📚 COURSE: course_id={course_id}")
            return Route.COURSE, None, 0.0

        # 3. DYNAMIC — mot-clé données BD / personnel / temporel / mémoire
        if _DYNAMIC.search(q):
            print(f"[Router] ⚡ DYNAMIC (keyword): {q_original[:80]!r}")
            return Route.DYNAMIC, None, 0.0

        # 4. FAQ — recherche vectorielle avec seuil haut
        match = self._faq.best_match(q_original)  # Utiliser l'original pour la FAQ
        if match:
            doc, score = match
            if score >= _FAQ_THRESHOLD:
                print(f"[Router] 📖 FAQ (score={score:.3f}): {q_original[:80]!r}")
                return Route.FAQ, doc, score

        # 5. DYNAMIC par défaut — mieux que risquer une FAQ incorrecte
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