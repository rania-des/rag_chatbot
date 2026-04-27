"""
Pipeline d'ingestion pour la branche COURSE.

Flux :
  1. parse_document() → extrait (page_num, text) depuis n'importe quel format
  2. chunk_documents() → découpe en chunks de ~500 tokens avec overlap 100
  3. On crée un index FAISS EN MÉMOIRE pour ce cours uniquement
  4. On garde le mapping course_id → FAISS en mémoire (session-scope)

IMPORTANT : rien n'est sauvegardé sur disque. Le cours disparaît au
redémarrage du serveur ou à la suppression explicite.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from ingestion.parsers import parse_document

# Chunking : 500 tokens approx = 2000 chars, overlap 100 tokens = 400 chars
# (règle de base : 1 token ≈ 4 chars pour du texte latin, moins pour l'arabe)
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

# Session TTL : combien de temps un cours reste en mémoire (en secondes)
# 2 heures = une session de chat raisonnable
SESSION_TTL_SECONDS = 2 * 60 * 60


# ============================================
# Modèle interne
# ============================================
@dataclass
class CourseSession:
    """Un cours uploadé, stocké en mémoire pour une session."""
    course_id: str
    student_id: str
    filename: str
    vector_store: FAISS
    total_chunks: int
    total_pages: int
    uploaded_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return (time.time() - self.uploaded_at) > SESSION_TTL_SECONDS


# ============================================
# Store en mémoire (singleton)
# ============================================
class CourseSessionStore:
    """Conserve les cours uploadés en mémoire, scopés par élève."""

    def __init__(self) -> None:
        # {course_id: CourseSession}
        self._sessions: Dict[str, CourseSession] = {}
        # {student_id: [course_id, ...]} pour l'isolation
        self._by_student: Dict[str, List[str]] = {}

    def add(self, session: CourseSession) -> None:
        self._cleanup_expired()
        self._sessions[session.course_id] = session
        self._by_student.setdefault(session.student_id, []).append(session.course_id)

    def get(self, course_id: str, student_id: str) -> CourseSession | None:
        """
        Retourne le cours SEULEMENT s'il appartient à l'élève demandeur.
        C'est la protection critique contre l'accès aux cours d'autres élèves.
        """
        self._cleanup_expired()
        session = self._sessions.get(course_id)
        if session is None:
            return None
        if session.student_id != student_id:
            # Tentative d'accès à un cours qui ne t'appartient pas
            return None
        if session.is_expired():
            self.delete(course_id)
            return None
        return session

    def list_for_student(self, student_id: str) -> List[CourseSession]:
        self._cleanup_expired()
        course_ids = self._by_student.get(student_id, [])
        return [
            self._sessions[cid] for cid in course_ids if cid in self._sessions
        ]

    def delete(self, course_id: str) -> bool:
        session = self._sessions.pop(course_id, None)
        if session is None:
            return False
        student_courses = self._by_student.get(session.student_id, [])
        if course_id in student_courses:
            student_courses.remove(course_id)
        return True

    def _cleanup_expired(self) -> None:
        expired = [cid for cid, s in self._sessions.items() if s.is_expired()]
        for cid in expired:
            self.delete(cid)


_store: CourseSessionStore | None = None


def get_course_store() -> CourseSessionStore:
    global _store
    if _store is None:
        _store = CourseSessionStore()
    return _store


# ============================================
# Embeddings (partagés avec la branche FAQ)
# ============================================
_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ============================================
# Ingestion principale
# ============================================
def ingest_course(
    file_path: str,
    filename: str,
    student_id: str,
) -> CourseSession:
    """
    Pipeline complet : parse → chunk → embed → index.

    Args:
        file_path: chemin du fichier uploadé (temp).
        filename: nom original.
        student_id: UUID de l'élève propriétaire.

    Returns:
        Un CourseSession prêt à être interrogé.

    Raises:
        ValueError: si le fichier n'est pas parseable ou vide.
    """
    # 1. Parse
    pages = parse_document(file_path, filename)
    if not pages:
        raise ValueError(
            f"Aucun texte extrait de {filename}. "
            "Fichier vide, corrompu, ou format non supporté."
        )

    total_pages = len(pages)

    # 2. Chunking avec préservation du numéro de page
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "، ", " ", ""],  # arabe-friendly
        length_function=len,
    )

    documents: List[Document] = []
    for page_num, page_text in pages:
        chunks = splitter.split_text(page_text)
        for chunk_idx, chunk in enumerate(chunks):
            # Préfixe "passage:" pour E5 (comme dans la FAQ)
            documents.append(
                Document(
                    page_content=f"passage: {chunk}",
                    metadata={
                        "page": page_num,
                        "chunk_idx": chunk_idx,
                        "filename": filename,
                        "raw_text": chunk,  # texte sans préfixe E5, pour les citations
                    },
                )
            )

    if not documents:
        raise ValueError(f"Aucun chunk généré depuis {filename}.")

    # 3. Embedding + indexation
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    # 4. Création et enregistrement de la session
    course_id = str(uuid.uuid4())
    session = CourseSession(
        course_id=course_id,
        student_id=student_id,
        filename=filename,
        vector_store=vector_store,
        total_chunks=len(documents),
        total_pages=total_pages,
    )

    get_course_store().add(session)
    print(
        f"[ingestion] Cours indexé : {filename} "
        f"({total_pages} pages, {len(documents)} chunks) → course_id={course_id}"
    )
    return session
