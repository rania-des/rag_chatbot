"""
Routeur hybride — 4 branches : GREET, FAQ, DYNAMIC, COURSE.

Logique ENTIÈREMENT basée sur règles (0 appel LLM pour le routage).
C'est le changement clé : l'ancien code appelait le LLM pour les
cas ambigus, ce qui ajoutait 15-30s à chaque réponse. Ici, on ne
l'appelle jamais pour router — on se trompe rarement et c'est
infiniment plus rapide.

Ordre de priorité :
  1. GREET   — salutations simples (regex)   → réponse instantanée
  2. COURSE  — course_id fourni et question pédagogique
  3. DYNAMIC — mots-clés personnels/temporels/données BD (regex large)
  4. FAQ     — similarité vectorielle >= seuil haut
  5. DYNAMIC — par défaut si aucune FAQ ne matche assez
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
# SALUTATIONS — réponse prédéfinie, 0 appel LLM (< 1 ms)
# ══════════════════════════════════════════════════════════════════════
_GREET = re.compile(
    r"^\s*("
    # Français
    r"bonjour|bonsoir|salut|coucou|bonne\s+journée|bonne\s+soirée|bonne\s+nuit|"
    r"bonne\s+matinée|bonne\s+année|bienvenue|"
    # Anglais
    r"hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|good\s+night|"
    r"howdy|greetings|welcome|"
    # Arabe standard
    r"مرحبا|أهلا|أهلاً|السلام\s+عليكم|وعليكم\s+السلام|صباح\s+الخير|"
    r"مساء\s+الخير|تصبح\s+على\s+خير|مرحباً|"
    # Abréviations courantes
    r"bjr|bsr|slt|cc|oui|non|ok|merci|شكرا|شكراً"
    r")\s*[!،,\.؟?]*\s*$",
    re.IGNORECASE | re.UNICODE,
)

GREET_RESPONSES = {
    "ar": "مرحباً! 👋 كيف يمكنني مساعدتك اليوم؟\nاسألني عن درجاتك أو جدولك الدراسي أو قائمة الطعام أو أي شيء آخر.",
    "en": "Hello! 👋 How can I help you today?\nAsk me about your grades, schedule, canteen menu, or upload a course.",
    "fr": "Bonjour ! 👋 Comment puis-je vous aider ?\nPosez-moi une question sur vos notes, emploi du temps, cantine, absences, ou uploadez un cours.",
}

def _greet_lang(query: str) -> str:
    if re.search(r'[\u0600-\u06FF]', query): return "ar"
    if re.search(r'\b(hello|hi|hey|good|morning|evening)\b', query, re.I): return "en"
    return "fr"

def get_greet_response(query: str) -> str:
    return GREET_RESPONSES[_greet_lang(query)]


# ══════════════════════════════════════════════════════════════════════
# DYNAMIC — mots-clés qui IMPOSENT un accès Supabase
# Règle : si la réponse DÉPEND de l'élève ou du MOMENT → DYNAMIC
# ══════════════════════════════════════════════════════════════════════
_DYNAMIC = re.compile(
    r"""
    # ── Données cantine (toujours dynamique — menu change chaque jour) ──
    \b(menu|cantine|canteen|manger|repas|plat|déjeuner|lunch|
       dîner|dinner|nourriture|food|مطعم|وجبة|أكل|كنتين)\b |

    # ── Possessifs personnels ──
    \bmes\s+(notes?|devoirs?|absences?|cours|retards?|paiements?|
             réunions?|moyennes?|résultats?)\b |
    \bmon\s+(emploi|cours|devoir|prof|professeur|dernier|prochain|
             bulletin|résultat)\b |
    \bma\s+(note|moyenne|classe|dernière|prochaine|liste)\b |
    \bmy\s+(grade|grades|schedule|class|homework|assignment|attendance|
             absence|score|result)\b |

    # ── Verbes à la 1ère personne ──
    \bj[' ']?ai\b | \bai[- ]?je\b |
    \bje\s+(dois|ai|vais|voudrais|souhaite|veux)\b |
    \bdo\s+i\s+have\b | \bam\s+i\b | \bi\s+have\b |

    # ── Temporalité (=données en temps réel) ──
    \b(aujourd[' ']?hui|demain|hier|ce\s+matin|ce\s+soir)\b |
    \b(cette\s+semaine|ce\s+mois|ce\s+trimestre|ce\s+semestre)\b |
    \b(la\s+semaine\s+(prochaine|passée|dernière))\b |
    \b(le\s+mois\s+(prochain|dernier|passé))\b |
    \b(today|tomorrow|yesterday|this\s+week|this\s+month|next\s+week)\b |

    # ── Emploi du temps / cours ──
    \b(emploi\s+du\s+temps|edt|horaire|cours\s+(de|d')?\s*\w+|séance)\b |
    \b(schedule|timetable|class\s+today|class\s+tomorrow)\b |

    # ── Notes / résultats ──
    \b(note|notes|résultat|résultats|moyenne|bulletin|score|bilan)\b |
    \b(grades?|results?|average|gpa|transcript)\b |

    # ── Absences / présence ──
    \b(absence|absences|retard|retards|présence|présences)\b |
    \b(absent|attendance|tardiness)\b |

    # ── Devoirs ──
    \b(devoir|devoirs|exercice|exercices|travail\s+à\s+rendre)\b |
    \b(homework|assignment|assignments|due\s+date)\b |

    # ── Paiements ──
    \b(paiement|paiements|facture|frais|scolarité|cotisation)\b |
    \b(payment|payments|invoice|fee|tuition)\b |

    # ── Annonces / actualités ──
    \b(annonce|annonces|actualité|actualités|news|événement)\b |

    # ── Réunions ──
    \b(réunion|réunions|rencontre|parents|rdv)\b |
    \b(meeting|meetings|appointment)\b |

    # ── Arabe standard ──
    عندي | لدي | اليوم | غدا | غداً | درجاتي | حصصي | مالي |
    جدولي | غيابي | واجبي |

    # ── Arabizi / dialecte tunisien ──
    \b(3andi|andi|lyoum|ghodwa|ghodowa|nhar)\b
    """,
    re.IGNORECASE | re.UNICODE | re.VERBOSE,
)

# Mots qui signalent une question admin même quand un cours est actif
_ADMIN = re.compile(
    r'\b(menu|cantine|canteen|notes?|grades?|emploi|schedule|absence|'
    r'devoir|homework|paiement|payment|réunion|meeting|annonce)\b',
    re.IGNORECASE | re.UNICODE,
)

# Seuil FAQ : très haut pour éviter les faux positifs
_FAQ_THRESHOLD = max(0.88, settings.FAQ_SIMILARITY_THRESHOLD)


class Router:
    def __init__(self) -> None:
        self._faq = get_faq_engine()

    def route(
        self,
        query: str,
        course_id: Optional[str] = None,
    ) -> Tuple[Route, Optional[Document], float]:
        """
        Retourne (route, faq_doc_ou_None, score).
        Aucun appel LLM — 100% règles.
        """
        q = query.strip()

        # 1. GREET — salutations / messages très courts
        if _GREET.match(q):
            print(f"[Router] 👋 GREET: {q!r}")
            return Route.GREET, None, 0.0

        # 2. COURSE — cours uploadé actif et question pédagogique
        if course_id and not _ADMIN.search(q):
            print(f"[Router] 📚 COURSE: course_id={course_id}")
            return Route.COURSE, None, 0.0

        # 3. DYNAMIC — mot-clé personnel/temporel/données BD
        if _DYNAMIC.search(q):
            print(f"[Router] ⚡ DYNAMIC (keyword): {q[:60]!r}")
            return Route.DYNAMIC, None, 0.0

        # 4. FAQ — recherche vectorielle avec seuil très haut
        match = self._faq.best_match(q)
        if match:
            doc, score = match
            if score >= _FAQ_THRESHOLD:
                print(f"[Router] 📖 FAQ (score={score:.3f}): {q[:60]!r}")
                return Route.FAQ, doc, score

        # 5. DYNAMIC par défaut — mieux que de risquer une FAQ incorrecte
        print(f"[Router] ⚡ DYNAMIC (default, score={match[1] if match else 0:.3f}): {q[:60]!r}")
        return Route.DYNAMIC, None, match[1] if match else 0.0


_router: Optional[Router] = None

def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router