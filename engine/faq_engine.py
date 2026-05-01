"""
Moteur FAQ : RAG sur qa_dataset.json.

Améliorations v5 :
  - Indexation de la QUESTION + toutes les VARIATIONS de chaque entrée
    → FAISS couvre ~600 formulations au lieu de ~90 (6x plus de surface de matching)
  - Chaque document garde la MÊME réponse en métadonnée (group_id = question principale)
  - Toujours UN Document par texte indexé, pour que la similarité soit précise
  - Support du champ `lang` pour filtrage éventuel
  - Persistance FAISS inchangée (rebuild si le dataset change)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


class FAQEngine:
    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector_store: Optional[FAISS] = None

    # ──────────────────────────────────────────────
    # Construction / chargement de l'index
    # ──────────────────────────────────────────────
    def build_or_load(self, force_rebuild: bool = False) -> None:
        """Charge l'index FAISS depuis le disque, ou le reconstruit."""
        index_path = Path(settings.FAISS_INDEX_PATH)

        if index_path.exists() and not force_rebuild:
            print(f"[FAQ] Chargement index depuis {index_path}")
            self._vector_store = FAISS.load_local(
                str(index_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            return

        print("[FAQ] Construction de l'index FAISS (question + variations)...")
        docs = self._load_documents()
        if not docs:
            raise RuntimeError(
                f"Aucun document FAQ trouvé dans {settings.DATA_DIR}"
            )

        self._vector_store = FAISS.from_documents(docs, self._embeddings)
        self._vector_store.save_local(str(index_path))
        print(f"[FAQ] Index construit : {len(docs)} vecteurs → {index_path}")

    def _load_documents(self) -> List[Document]:
        """
        Charge qa_dataset.json et crée UN Document par texte indexé.

        Stratégie d'indexation :
          - Pour chaque entrée : on indexe la `question` PRINCIPALE
            + chaque élément du tableau `variations`.
          - Tous ces documents pointent vers la MÊME réponse (group_id).
          - Résultat : ~600 vecteurs pour ~93 entrées, ce qui multiplie
            les chances de matcher une question mal formulée.
        """
        data_dir = Path(settings.DATA_DIR)
        docs: List[Document] = []

        filepath = data_dir / "qa_dataset.json"
        if not filepath.exists():
            print(f"[FAQ] Fichier absent : {filepath}")
            return docs

        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            question  = item.get("question", "").strip()
            answer    = item.get("answer", "").strip()
            category  = item.get("category", "général")
            lang      = item.get("lang", "fr")
            entry_id  = item.get("id", f"entry_{i}")
            variations = item.get("variations", [])

            if not question or not answer:
                continue

            # Métadonnées communes à tous les vecteurs de cette entrée
            base_meta = {
                "answer":            answer,
                "category":          category,
                "lang":              lang,
                "source":            "qa_dataset.json",
                "original_question": question,
                "group_id":          entry_id,
            }

            # 1. Question principale
            docs.append(Document(
                page_content=f"passage: {question}",
                metadata={**base_meta, "doc_id": f"{entry_id}:q"},
            ))

            # 2. Chaque variation → vecteur séparé, même réponse
            for j, var in enumerate(variations):
                var = var.strip()
                if not var:
                    continue
                docs.append(Document(
                    page_content=f"passage: {var}",
                    metadata={**base_meta, "doc_id": f"{entry_id}:v{j}"},
                ))

        print(f"[FAQ] {len(data)} entrées → {len(docs)} vecteurs indexés")
        return docs

    # ──────────────────────────────────────────────
    # Recherche
    # ──────────────────────────────────────────────
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        lang: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Recherche les Q/R les plus similaires.

        Args:
            query : question de l'élève
            k     : nombre de résultats (défaut : settings.FAQ_TOP_K)
            lang  : si fourni, filtre sur la langue (optionnel — désactivé
                    par défaut car E5 multilingual gère bien le cross-lingual)

        Returns:
            Liste de (Document, score_cosinus ∈ [0, 1]).
        """
        if self._vector_store is None:
            raise RuntimeError("FAQEngine non initialisé. Appelle build_or_load().")

        k = k or settings.FAQ_TOP_K
        prefixed = f"query: {query}"

        # On demande k*3 si on filtre sur la langue (pour compenser le filtre)
        fetch_k = k * 3 if lang else k
        results = self._vector_store.similarity_search_with_score(prefixed, k=fetch_k)

        out: List[Tuple[Document, float]] = []
        seen_groups: set = set()

        for doc, dist in results:
            # Dédoublonnage : si deux variations du même groupe matchent,
            # on ne garde que la meilleure (la première, vu qu'on est trié)
            group = doc.metadata.get("group_id", doc.metadata.get("doc_id"))
            if group in seen_groups:
                continue
            seen_groups.add(group)

            # Filtre langue optionnel
            if lang and doc.metadata.get("lang") != lang:
                continue

            # Distance L2 → similarité cosinus (approximation suffisante)
            similarity = max(0.0, 1.0 - float(dist) / 2.0)
            out.append((doc, similarity))

            if len(out) >= k:
                break

        return out

    def best_match(
        self,
        query: str,
        lang: Optional[str] = None,
    ) -> Optional[Tuple[Document, float]]:
        """Retourne le meilleur match (dédoublonné) ou None."""
        results = self.search(query, k=1, lang=lang)
        return results[0] if results else None


# ──────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────
_engine: Optional[FAQEngine] = None


def get_faq_engine() -> FAQEngine:
    global _engine
    if _engine is None:
        _engine = FAQEngine()
        _engine.build_or_load()
    return _engine