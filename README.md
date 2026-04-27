# Chatbot RAG scolaire (3 branches)

Chatbot RAG à **trois branches** pour plateforme éducative :

1. **FAQ** — questions répétitives (politique, contact, comment accéder à ses notes...) via retrieval vectoriel sur un corpus Q/R préparé.
2. **Dynamique** — questions personnelles (mes notes, mon emploi du temps, le menu d'aujourd'hui...) via **tool calling** sur 8 fonctions Supabase typées.
3. **Cours** — l'élève uploade un cours (PDF, Word, PowerPoint, texte, image scannée), pose des questions dessus, obtient des réponses **avec citations**.

Le tout est 100 % **gratuit et local** : Ollama + embeddings HuggingFace multilingues.

## Architecture

```
                      Question + student_id (+ course_id ?)
                                    ↓
                            [Routeur 3 branches]
                                    ↓
          ┌─────────────────┬───────────────────┬───────────────────┐
          ↓                 ↓                   ↓
        FAQ              DYNAMIC              COURSE
          ↓                 ↓                   ↓
     FAISS + E5       LLM + 8 tools     FAISS par cours
     qa_dataset       Supabase live    (session 2h)
          ↓                 ↓                   ↓
          └─────────────────┴───────────────────┘
                            ↓
                   LLM de formulation (Qwen2.5)
                            ↓
                   Réponse + sources/citations
```

## Pré-requis

### 1. Ollama + modèle

```bash
ollama pull qwen2.5:7b-instruct
```

> Si machine plus modeste : `qwen2.5:3b-instruct`. Plus puissante : `qwen2.5:14b-instruct`.

### 2. Dépendances système pour les parseurs

```bash
# Linux (Ubuntu/Debian)
sudo apt install tesseract-ocr tesseract-ocr-ara tesseract-ocr-fra poppler-utils

# macOS
brew install tesseract tesseract-lang poppler

# Windows
# - Tesseract : https://github.com/UB-Mannheim/tesseract/wiki
# - Poppler   : https://github.com/oschwartz10612/poppler-windows
```

Sans ces paquets, l'OCR et le rendu PDF → image ne fonctionneront pas. Le reste (PDF texte, docx, pptx, txt) marche sans rien.

### 3. Python 3.10+

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Variables d'environnement

```bash
cp .env.example .env
# Renseigner SUPABASE_URL et SUPABASE_SERVICE_KEY
```

## Démarrage

```bash
uvicorn main:app --reload --port 8000
```

Au 1er lancement, l'index FAISS de la FAQ est construit (~10 s).
Les lancements suivants chargent l'index depuis le disque (instantané).

## Workflow utilisateur type

### Cas 1 — Question FAQ
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "uuid-eleve",
    "message": "How can I contact the school?"
  }'
```
→ `route: "FAQ"`, réponse tirée de `qa_dataset.json`.

### Cas 2 — Question dynamique
```bash
curl -X POST http://localhost:8000/chat \
  -d '{
    "student_id": "uuid-eleve",
    "class_id": "uuid-classe",
    "message": "Est-ce que j'\''ai cours demain ?"
  }'
```
→ `route: "DYNAMIC"`, le LLM appelle `get_student_schedule("tomorrow")`.

### Cas 3 — Upload + Q/R sur un cours

**Étape 1 : uploader le cours**
```bash
curl -X POST http://localhost:8000/courses/upload \
  -F "student_id=uuid-eleve" \
  -F "file=@/chemin/vers/chapitre_5_photosynthese.pdf"
```
Réponse :
```json
{
  "course_id": "abc-123-def-456",
  "filename": "chapitre_5_photosynthese.pdf",
  "total_pages": 12,
  "total_chunks": 34,
  "message": "Cours indexé avec succès..."
}
```

**Étape 2 : poser des questions**
```bash
curl -X POST http://localhost:8000/chat \
  -d '{
    "student_id": "uuid-eleve",
    "message": "Explique-moi le cycle de Calvin",
    "course_id": "abc-123-def-456"
  }'
```
Réponse :
```json
{
  "answer": "Le cycle de Calvin est une série de réactions...",
  "route": "COURSE",
  "citations": [
    {"filename": "chapitre_5_photosynthese.pdf", "page": 7, "excerpt": "Le cycle de Calvin, aussi appelé..."},
    {"filename": "chapitre_5_photosynthese.pdf", "page": 8, "excerpt": "..."}
  ]
}
```

**Étape 3 : demander l'explication d'un passage précis**
```bash
curl -X POST http://localhost:8000/courses/explain \
  -d '{
    "student_id": "uuid-eleve",
    "course_id": "abc-123-def-456",
    "passage": "La régénération du RuBP nécessite 3 ATP et consomme du NADPH produit par la phase claire."
  }'
```

### Cas 4 — Poser une question admin pendant un cours ouvert
```bash
curl -X POST http://localhost:8000/chat \
  -d '{
    "student_id": "uuid-eleve",
    "message": "C'\''est quoi le menu de demain ?",
    "course_id": "abc-123-def-456"
  }'
```
→ Le routeur détecte le mot "menu" → bypass du COURSE, passe en DYNAMIC. L'élève n'a pas besoin de fermer son cours.

## Endpoints

| Méthode | URL | Description |
|---------|-----|-------------|
| POST | `/chat` | Conversation (routage automatique) |
| POST | `/courses/upload` | Uploade et indexe un cours |
| POST | `/courses/explain` | Explication d'un passage précis |
| GET  | `/courses/mine?student_id=X` | Liste les cours en session |
| DELETE | `/courses/{id}?student_id=X` | Supprime un cours |
| POST | `/reindex` | Reconstruit l'index FAQ |
| GET  | `/health` | Healthcheck |

## Sécurité

### Isolation des cours entre élèves
Chaque upload crée un `CourseSession` étiqueté avec le `student_id`. Le store refuse l'accès si un autre élève tente de consulter un cours qui ne lui appartient pas — même avec le bon `course_id` en main.

```python
# Dans ingestion/course_ingestion.py :
def get(self, course_id: str, student_id: str) -> CourseSession | None:
    session = self._sessions.get(course_id)
    if session.student_id != student_id:
        return None  # Refus d'accès
```

### Isolation des données dynamiques
Le `student_id` **n'est jamais** un paramètre que le LLM peut manipuler. Il est injecté dans `StudentContext` avant l'appel du LLM et lu par les tools depuis ce contexte. Résultat : impossible pour le LLM d'halluciner un accès à un autre élève.

### Expiration automatique
Les cours uploadés expirent au bout de **2 heures** (configurable via `SESSION_TTL_SECONDS` dans `ingestion/course_ingestion.py`). Aucune donnée ne reste sur disque après expiration.

## Limites et restrictions

- **Stockage temporaire uniquement** : les cours disparaissent au redémarrage du serveur. Pour la persistance, voir la section "Évolutions possibles".
- **Max 50 MB par fichier** (ajustable dans `main.py`).
- **OCR** : nécessite Tesseract installé au niveau OS. Support de l'arabe + français + anglais si les packs de langue sont installés.
- **Pas de streaming SSE** pour le moment — le client attend la réponse complète.

## Structure des fichiers

```
rag_chatbot/
├── main.py                         # FastAPI + orchestration
├── config.py                       # Settings
├── engine/
│   ├── router.py                   # Routeur 3 branches
│   ├── faq_engine.py               # Retrieval FAQ vectoriel
│   ├── faq_formatter.py            # Reformule les FAQ dans la bonne langue
│   ├── dynamic_engine.py           # Tool calling Supabase
│   └── course_engine.py            # RAG sur cours uploadé
├── ingestion/
│   ├── parsers.py                  # PDF, docx, pptx, txt, images (OCR)
│   └── course_ingestion.py         # Pipeline parse + chunk + embed + index
├── tools/
│   └── supabase_tools.py           # 8 fonctions typées
├── data/
│   └── qa_dataset.json             # FAQ statiques
├── faiss_index/                    # Créé automatiquement
├── .env.example
├── requirements.txt
└── README.md
```

## Évolutions possibles

- **Persistance des cours** : stockage dans Supabase Storage + table `student_courses` avec les embeddings en pgvector.
- **Streaming SSE** : `StreamingResponse` FastAPI pour afficher la réponse au fur et à mesure.
- **Génération de quiz** : outil supplémentaire qui génère des QCM sur un cours.
- **Historique persistant** : remplacer le `deque` en mémoire par Redis ou une table Supabase.
