"""
Moteur FAQ : RAG sur qa_dataset.json + questions.json.

Corrections par rapport au main.py original :
- UNE Q/R = UN Document (au lieu d'un gros blob)
- Embeddings MULTILINGUE (E5 multilingual au lieu de all-mpnet anglais)
- Persistance de l'index FAISS sur disque (évite de réindexer à chaque démarrage)
- Le retriever retourne le score de similarité pour le seuil de routage
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


class FAQEngine:
    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            # E5 recommande le prefix "passage:" / "query:"
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector_store: FAISS | None = None

    # --------------------------------------
    # Construction / chargement de l'index
    # --------------------------------------
    def build_or_load(self, force_rebuild: bool = False) -> None:
        """Charge l'index FAISS depuis le disque, ou le reconstruit."""
        index_path = Path(settings.FAISS_INDEX_PATH)

        if index_path.exists() and not force_rebuild:
            print(f"[FAQ] Chargement de l'index depuis {index_path}")
            self._vector_store = FAISS.load_local(
                str(index_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            return

        print("[FAQ] Construction de l'index FAISS...")
        docs = self._load_documents()
        if not docs:
            raise RuntimeError(
                f"Aucun document FAQ trouvé dans {settings.DATA_DIR}"
            )

        self._vector_store = FAISS.from_documents(docs, self._embeddings)
        self._vector_store.save_local(str(index_path))
        print(f"[FAQ] Index construit : {len(docs)} documents, sauvegardé dans {index_path}")

    def _load_documents(self) -> List[Document]:
        """Charge les fichiers JSON et crée UN Document par Q/R."""
        data_dir = Path(settings.DATA_DIR)
        docs: List[Document] = []

        # On ne charge QUE les vraies FAQ statiques.
        # Les données dynamiques (cantine, notes, emploi du temps, annonces...)
        # sont lues en temps réel depuis Supabase via la branche dynamique.
        for filename in ["qa_dataset.json"]:
            filepath = data_dir / filename
            if not filepath.exists():
                print(f"[FAQ] Fichier absent : {filepath}")
                continue

            with filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                question = item.get("question", "").strip()
                answer = item.get("answer", "").strip()
                category = item.get("category", "général")
                if not question or not answer:
                    continue

                # IMPORTANT : le contenu indexé = la QUESTION
                # (pour que la similarité question-question fonctionne bien)
                # La réponse est stockée en métadonnée.
                docs.append(
                    Document(
                        page_content=f"passage: {question}",
                        metadata={
                            "answer": answer,
                            "category": category,
                            "source": filename,
                            "original_question": question,
                            "doc_id": f"{filename}:{i}",
                        },
                    )
                )

        return docs

    # --------------------------------------
    # Recherche
    # --------------------------------------
    def search(self, query: str, k: int | None = None) -> List[Tuple[Document, float]]:
        """
        Recherche les Q/R les plus similaires avec leur score.

        Returns:
            Liste de tuples (Document, score) où score est la similarité
            cosinus (entre 0 et 1, 1 = identique).
        """
        if self._vector_store is None:
            raise RuntimeError("FAQEngine non initialisé. Appelle build_or_load().")

        k = k or settings.FAQ_TOP_K
        # E5 attend "query:" comme prefix pour les requêtes
        prefixed_query = f"query: {query}"

        # FAISS retourne un score de DISTANCE L2 → on le convertit en similarité
        results = self._vector_store.similarity_search_with_score(prefixed_query, k=k)

        # Avec normalize_embeddings=True, la distance L2 et la similarité cosinus
        # sont liées : cos_sim = 1 - (L2² / 2). On simplifie en : sim = 1 - L2/2
        # (approximation suffisante pour du thresholding)
        out: List[Tuple[Document, float]] = []
        for doc, dist in results:
            similarity = max(0.0, 1.0 - float(dist) / 2.0)
            out.append((doc, similarity))
        return out

    def best_match(self, query: str) -> Tuple[Document, float] | None:
        """Retourne le meilleur match ou None si aucun résultat."""
        results = self.search(query, k=1)
        return results[0] if results else None


# Singleton partagé
_engine: FAQEngine | None = None


def get_faq_engine() -> FAQEngine:
    global _engine
    if _engine is None:
        _engine = FAQEngine()
        _engine.build_or_load()
    return _engine
