"""
Moteur COURSE — RAG sur cours uploadé avec citations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import settings
from ingestion import get_course_store

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
    "Réponds dans la langue de la question. 3-8 lignes max."
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
        _, docs = self._retrieve(course_id, student_id, question)
        if not docs:
            return CourseAnswer("Aucun passage pertinent trouvé dans ton cours.", [])
        ctx, cits = self._context(docs)
        r = self._llm.invoke([
            SystemMessage(content=QA_PROMPT),
            HumanMessage(content=f"QUESTION : {question}\n\nEXTRAITS :\n{ctx}"),
        ])
        return CourseAnswer((r.content or "").strip(), cits)

    def explain_passage(self, course_id: str, student_id: str, passage: str) -> CourseAnswer:
        _, docs = self._retrieve(course_id, student_id, passage, k=TOP_K + 2)
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