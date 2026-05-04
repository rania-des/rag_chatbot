"""
Moteur COURSE — RAG sur cours uploadé avec citations.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import settings
from ingestion import get_course_store

# Import de la fonction de détection de langue depuis dynamic_engine (ou copie locale)
try:
    from engine.dynamic_engine import _detect_lang
except ImportError:
    # Fallback simple si non disponible
    def _detect_lang(text: str) -> str:
        import re
        if re.search(r"[\u0600-\u06FF]", text):
            return "ar"
        if re.search(r"\b(what|when|where|who|how|my|i\s+have|do\s+i|show)\b", text, re.I):
            return "en"
        return "fr"


TOP_K = 4

@dataclass
class Citation:
    filename: str
    page: int
    excerpt: str

@dataclass
class CourseAnswer:
    answer: str
    citations: List[Citation]

QA_PROMPT = (
    "Tu es un tuteur pédagogique. Réponds UNIQUEMENT à partir des EXTRAITS fournis. "
    "Si la réponse n'est pas dans les extraits, dis-le. "
    "IMPORTANT : tu DOIS répondre dans la même langue que la QUESTION de l'utilisateur. "
    "Si la question est en français, réponds en français. Si elle est en anglais, en anglais. "
    "Ne change pas de langue. 3-8 lignes max."
)

EXPLAIN_PROMPT = (
    "Tu es un tuteur pédagogique. Explique le PASSAGE fourni de façon pédagogique : "
    "décompose les concepts, donne des exemples. "
    "Réponds dans la langue du passage. 6-12 lignes max."
)


class CourseEngine:
    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.2,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
            num_predict=500,
        )

    def _retrieve(self, course_id: str, student_id: str, query: str, k: int = TOP_K):
        session = get_course_store().get(course_id, student_id)
        if session is None:
            raise ValueError(f"Cours introuvable : {course_id}. Réuploadez le fichier.")
        docs = session.vector_store.similarity_search_with_score(f"query: {query}", k=k)
        return session, docs

    def _context(self, docs) -> tuple[str, List[Citation]]:
        parts, cits, seen = [], [], set()
        for i, (doc, _) in enumerate(docs, 1):
            page = doc.metadata.get("page", 0)
            raw  = doc.metadata.get("raw_text", "")
            fn   = doc.metadata.get("filename", "cours")
            parts.append(f"[Extrait {i} — page {page}]\n{raw}")
            key = (fn, page)
            if key not in seen:
                cits.append(Citation(fn, page if isinstance(page, int) else 0, raw[:150]))
                seen.add(key)
        return "\n\n".join(parts), cits

    def answer_question(self, course_id: str, student_id: str, question: str) -> CourseAnswer:
        # 1. Détection des demandes d'extraction brute de texte (sans LLM)
        extraction_keywords = re.compile(
            r"(extrait|extrais|donne le texte|affiche le contenu|texte brut|récupère le texte|"
            r"extraire le texte|restituer le texte|show the text|give me the text|"
            r"اعطيني النص|استخرج النص)",
            re.IGNORECASE
        )
        if extraction_keywords.search(question):
            # Récupérer le meilleur chunk
            try:
                _, docs = self._retrieve(course_id, student_id, question, k=1)
            except ValueError:
                return CourseAnswer("Cours introuvable ou non accessible.", [])
            if docs:
                doc = docs[0][0]
                raw_text = doc.metadata.get("raw_text", "")
                # Nettoyer et limiter la longueur (optionnel)
                if len(raw_text) > 2000:
                    raw_text = raw_text[:2000] + "..."
                return CourseAnswer(
                    answer=f"Voici le texte extrait du cours :\n{raw_text}",
                    citations=[
                        Citation(
                            filename=doc.metadata.get("filename", "cours"),
                            page=doc.metadata.get("page", 0),
                            excerpt=raw_text[:150]
                        )
                    ]
                )
            else:
                return CourseAnswer("Aucun passage trouvé dans ce cours.", [])

        # 2. Comportement normal : recherche + génération par LLM
        try:
            _, docs = self._retrieve(course_id, student_id, question)
        except ValueError as e:
            return CourseAnswer(str(e), [])
        if not docs:
            return CourseAnswer("Aucun passage pertinent trouvé dans ton cours.", [])

        ctx, cits = self._context(docs)

        # 3. Appel au LLM avec le prompt de base
        response = self._llm.invoke([
            SystemMessage(content=QA_PROMPT),
            HumanMessage(content=f"QUESTION : {question}\n\nEXTRAITS :\n{ctx}"),
        ])
        answer = (response.content or "").strip()

        # 4. Correction de la langue : forcer la même langue que la question
        lang_q = _detect_lang(question)
        lang_a = _detect_lang(answer)
        if lang_q != lang_a and lang_q in ('fr', 'en', 'ar'):
            prompt_lang = {"fr": "français", "en": "anglais", "ar": "arabe"}[lang_q]
            retry_prompt = (
                f"La réponse précédente n'était pas dans la bonne langue. "
                f"Réponds uniquement en {prompt_lang} à la question suivante, "
                f"en utilisant uniquement les extraits fournis.\n"
                f"QUESTION : {question}\n\nEXTRAITS :\n{ctx}"
            )
            second_response = self._llm.invoke([
                SystemMessage(content="Tu es un tuteur. Réponds dans la langue de la question."),
                HumanMessage(content=retry_prompt)
            ])
            answer = (second_response.content or "").strip()

        return CourseAnswer(answer, cits)

    def explain_passage(self, course_id: str, student_id: str, passage: str) -> CourseAnswer:
        try:
            _, docs = self._retrieve(course_id, student_id, passage, k=TOP_K + 2)
        except ValueError as e:
            return CourseAnswer(str(e), [])
        if not docs:
            return CourseAnswer("Passage non trouvé dans le cours.", [])
        ctx, cits = self._context(docs)
        r = self._llm.invoke([
            SystemMessage(content=EXPLAIN_PROMPT),
            HumanMessage(content=f"PASSAGE : \"{passage}\"\n\nCONTEXTE :\n{ctx}"),
        ])
        return CourseAnswer((r.content or "").strip(), cits)


_engine: CourseEngine | None = None

def get_course_engine() -> CourseEngine:
    global _engine
    if _engine is None:
        _engine = CourseEngine()
    return _engine