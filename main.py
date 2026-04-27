"""
FastAPI application — point d'entrée du chatbot RAG.

v4 — Gestion des conversations persistées dans Supabase.

Endpoints :
  POST   /chat                           — conversation (routage 3 branches)
  POST   /courses/upload                 — uploade un cours
  POST   /courses/explain                — explique un passage
  GET    /courses/mine                   — liste les cours
  DELETE /courses/{course_id}            — supprime un cours
  GET    /conversations                  — liste les conversations d'un élève
  GET    /conversations/{id}/messages    — messages d'une conversation
  PATCH  /conversations/{id}             — renomme une conversation
  DELETE /conversations/{id}             — supprime une conversation
  POST   /reindex                        — reconstruit l'index FAQ
  GET    /health                         — healthcheck

Exécution :
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from conversations import (
    create_conversation,
    delete_conversation,
    generate_title,
    get_conversation_message_count,
    get_conversation_messages,
    list_conversations,
    rebuild_history_from_db,
    save_message,
    update_title,
)
from engine import (
    Route,
    get_course_engine,
    get_dynamic_engine,
    get_faq_engine,
    get_faq_formatter,
    get_router,
)
from engine.router import get_greet_response
from ingestion import get_course_store, ingest_course
from tools import StudentContext
from tools.supabase_tools import get_supabase


# ================================================
# Résolution profile_id → student_id
# ================================================
_student_id_cache: Dict[str, str] = {}   # cache léger profile_id → student_id


def resolve_student_id(raw_id: str) -> str:
    """
    Accepte indifféremment :
      - un students.id  (UUID direct dans la table students)
      - un profile_id   (auth.users.id = profiles.id = students.profile_id)

    Retourne toujours le students.id.
    Met en cache pour éviter des allers-retours Supabase répétés.
    """
    if raw_id in _student_id_cache:
        return _student_id_cache[raw_id]

    try:
        sb = get_supabase()

        # Essai 1 : c'est directement un students.id ?
        res = sb.table("students").select("id").eq("id", raw_id).limit(1).execute()
        if res.data:
            _student_id_cache[raw_id] = raw_id
            return raw_id

        # Essai 2 : c'est un profile_id (auth.users.id) ?
        res2 = sb.table("students").select("id").eq("profile_id", raw_id).limit(1).execute()
        if res2.data:
            student_id = res2.data[0]["id"]
            _student_id_cache[raw_id] = student_id
            return student_id

    except Exception as e:
        print(f"[resolve_student_id] Erreur : {e}")

    # Fallback : renvoyer l'id brut (le back-end renverra une erreur claire si invalide)
    return raw_id


# ================================================
# Schémas API
# ================================================
class ChatRequest(BaseModel):
    student_id: str = Field(..., description="UUID de l'élève")
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = Field(
        None,
        description=(
            "UUID de la conversation. Si omis, une NOUVELLE conversation est "
            "créée automatiquement. Si fourni, le message est ajouté à la "
            "conversation existante (avec l'historique en contexte)."
        ),
    )
    course_id: Optional[str] = Field(
        None,
        description="UUID du cours uploadé (branche COURSE)",
    )
    class_id: Optional[str] = Field(
        None,
        description="Optionnel — auto-récupéré depuis student_id si absent",
    )


class CitationOut(BaseModel):
    filename: str
    page: int
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    conversation_id: str
    conversation_title: str
    faq_score: Optional[float] = None
    sources: Optional[List[Dict]] = None
    citations: Optional[List[CitationOut]] = None


class UploadResponse(BaseModel):
    course_id: str
    filename: str
    total_pages: int
    total_chunks: int
    message: str


class ExplainRequest(BaseModel):
    student_id: str
    course_id: str
    passage: str = Field(..., min_length=10, max_length=5000)


class CourseInfo(BaseModel):
    course_id: str
    filename: str
    total_pages: int
    total_chunks: int
    uploaded_at: float


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    route: Optional[str] = None
    citations: Optional[List] = None
    created_at: str


class RenameRequest(BaseModel):
    student_id: str
    title: str = Field(..., min_length=1, max_length=200)


# ================================================
# FastAPI app
# ================================================
app = FastAPI(
    title="Chatbot RAG scolaire",
    description="FAQ + Supabase dynamique + cours uploadés + historique persistent",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    print("[startup] Initialisation du FAQ engine...")
    get_faq_engine()
    print("[startup] Initialisation du router...")
    get_router()
    print("[startup] Initialisation du dynamic engine...")
    get_dynamic_engine()
    print("[startup] Initialisation du course engine...")
    get_course_engine()
    get_faq_formatter()

    # Warmup du modèle Ollama
    print("[startup] 🔥 Préchauffage du modèle Ollama (30-60s)...")
    try:
        import httpx
        httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": "hi",
                "stream": False,
                "keep_alive": settings.OLLAMA_KEEP_ALIVE,
                "options": {"num_predict": 1},
            },
            timeout=120.0,
        )
        print(f"[startup] ✓ Modèle {settings.OLLAMA_MODEL} en RAM")
    except Exception as e:
        print(f"[startup] ⚠️  Warmup échoué : {e}")

    print("[startup] ✓ Prêt à recevoir des requêtes.")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": settings.OLLAMA_MODEL}


# ================================================
# ENDPOINT PRINCIPAL : /chat
# ================================================
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # ── Résolution automatique profile_id → student_id ──────────────────
    # Le front-end envoie parfois user.id (= auth.users.id = profile_id)
    # au lieu du students.id. On résout automatiquement.
    student_id = resolve_student_id(req.student_id)
    if student_id != req.student_id:
        print(f"[chat] Résolution : {req.student_id[:8]}… → {student_id[:8]}… (profile_id → student_id)")

    StudentContext.set(student_id=student_id, class_id=req.class_id)

    try:
        # 1. Gérer la conversation (créer ou récupérer)
        is_new_conversation = False
        if req.conversation_id:
            conversation_id = req.conversation_id
            history = rebuild_history_from_db(
                conversation_id, student_id, max_messages=settings.MAX_HISTORY_MESSAGES
            )
            conversation_title = None
        else:
            # Nouvelle conversation
            conv = create_conversation(student_id, first_question=req.message)
            if not conv:
                raise HTTPException(status_code=500, detail="Impossible de créer la conversation.")
            conversation_id = conv["id"]
            conversation_title = conv["title"]
            history = []
            is_new_conversation = True

        # 2. Sauvegarder le message utilisateur
        save_message(conversation_id, role="user", content=req.message)

        # 3. Router + dispatcher
        router = get_router()
        route, faq_doc, score = router.route(req.message, course_id=req.course_id)

        citations_data = None

        if route == Route.GREET:
            # Salutation instantanée — 0 appel LLM
            answer = get_greet_response(req.message)
            sources = None
            citations = None

        elif route == Route.FAQ and faq_doc is not None:
            formatter = get_faq_formatter()
            answer = formatter.format(
                user_question=req.message,
                reference_answer=faq_doc.metadata["answer"],
            )
            sources = [{
                "category": faq_doc.metadata.get("category"),
                "matched_question": faq_doc.metadata.get("original_question"),
                "similarity": round(score, 3),
            }]
            citations = None

        elif route == Route.COURSE:
            if not req.course_id:
                raise HTTPException(
                    status_code=400, detail="course_id requis pour la branche COURSE."
                )
            engine = get_course_engine()
            try:
                result = engine.answer_question(
                    course_id=req.course_id,
                    student_id=student_id,
                    question=req.message,
                )
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
            answer = result.answer
            sources = None
            citations = [
                CitationOut(filename=c.filename, page=c.page, excerpt=c.excerpt)
                for c in result.citations
            ]
            citations_data = [
                {"filename": c.filename, "page": c.page, "excerpt": c.excerpt}
                for c in result.citations
            ]

        else:  # DYNAMIC
            engine = get_dynamic_engine()
            answer = engine.answer(req.message, history=history)
            sources = None
            citations = None

        # 4. Sauvegarder la réponse
        save_message(
            conversation_id,
            role="assistant",
            content=answer,
            route=route.value,
            citations=citations_data,
        )

        # 5. Si c'est une nouvelle conversation, générer un titre après le 1er échange
        if is_new_conversation:
            try:
                title = generate_title(req.message, answer)
                update_title(conversation_id, student_id, title)
                conversation_title = title
            except Exception as e:
                print(f"[chat] Échec génération titre : {e}")

        return ChatResponse(
            answer=answer,
            route=route.value,
            conversation_id=conversation_id,
            conversation_title=conversation_title or "Conversation",
            faq_score=round(score, 3) if score else None,
            sources=sources,
            citations=citations,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback as tb
        print(f"[chat] ❌ ERREUR — {type(e).__name__}: {e}")
        tb.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    finally:
        StudentContext.clear()


# ================================================
# ENDPOINTS CONVERSATIONS
# ================================================
@app.get("/conversations", response_model=List[ConversationOut])
def get_conversations(student_id: str, limit: int = 50) -> List[ConversationOut]:
    """Liste les conversations d'un élève (plus récentes d'abord)."""
    convs = list_conversations(student_id, limit=limit)
    return [
        ConversationOut(
            id=c["id"],
            title=c["title"],
            created_at=c["created_at"],
            updated_at=c["updated_at"],
        )
        for c in convs
    ]


@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageOut])
def get_messages(conversation_id: str, student_id: str) -> List[MessageOut]:
    """Récupère tous les messages d'une conversation."""
    msgs = get_conversation_messages(conversation_id, student_id)
    if not msgs:
        # Peut signifier : conversation vide OU conversation qui n'appartient pas à l'élève
        # On distingue par le count
        count = get_conversation_message_count(conversation_id)
        if count > 0:
            # La conversation existe mais n'appartient pas à l'élève
            raise HTTPException(status_code=403, detail="Accès refusé à cette conversation.")
    return [
        MessageOut(
            id=m["id"],
            role=m["role"],
            content=m["content"],
            route=m.get("route"),
            citations=m.get("citations"),
            created_at=m["created_at"],
        )
        for m in msgs
    ]


@app.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, req: RenameRequest) -> Dict[str, str]:
    """Renomme une conversation."""
    ok = update_title(conversation_id, req.student_id, req.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation introuvable.")
    return {"status": "ok", "new_title": req.title}


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str, student_id: str) -> Dict[str, str]:
    """Supprime une conversation et tous ses messages."""
    ok = delete_conversation(conversation_id, student_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation introuvable.")
    return {"status": "deleted", "conversation_id": conversation_id}


# ================================================
# ENDPOINTS DE GESTION DE COURS (inchangés)
# ================================================
@app.post("/courses/upload", response_model=UploadResponse)
async def upload_course(
    student_id: str = Form(...),
    file: UploadFile = File(...),
) -> UploadResponse:
    """Uploade un cours (PDF, docx, txt, md, pptx, ou image scannée)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Fichier sans nom.")

    suffix = os.path.splitext(file.filename)[1]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Fichier vide.")
            if len(content) > 50 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 50 MB).")
            tmp.write(content)
            tmp_path = tmp.name

        try:
            session = ingest_course(
                file_path=tmp_path,
                filename=file.filename,
                student_id=student_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return UploadResponse(
            course_id=session.course_id,
            filename=session.filename,
            total_pages=session.total_pages,
            total_chunks=session.total_chunks,
            message=(
                f"Cours indexé avec succès. Passe course_id='{session.course_id}' dans /chat."
            ),
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/courses/explain", response_model=ChatResponse)
def explain_passage(req: ExplainRequest) -> ChatResponse:
    """Explication détaillée d'un passage précis du cours."""
    engine = get_course_engine()
    try:
        result = engine.explain_passage(
            course_id=req.course_id,
            student_id=req.student_id,
            passage=req.passage,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return ChatResponse(
        answer=result.answer,
        route=Route.COURSE.value,
        conversation_id="",  # explain ne crée pas de conversation
        conversation_title="",
        citations=[
            CitationOut(filename=c.filename, page=c.page, excerpt=c.excerpt)
            for c in result.citations
        ],
    )


@app.get("/courses/mine", response_model=List[CourseInfo])
def list_my_courses(student_id: str) -> List[CourseInfo]:
    sessions = get_course_store().list_for_student(student_id)
    return [
        CourseInfo(
            course_id=s.course_id,
            filename=s.filename,
            total_pages=s.total_pages,
            total_chunks=s.total_chunks,
            uploaded_at=s.uploaded_at,
        )
        for s in sessions
    ]


@app.delete("/courses/{course_id}")
def delete_course(course_id: str, student_id: str) -> Dict[str, str]:
    store = get_course_store()
    session = store.get(course_id, student_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Cours introuvable.")
    store.delete(course_id)
    return {"status": "deleted", "course_id": course_id}


# ================================================
# MAINTENANCE
# ================================================
@app.post("/reindex")
def reindex() -> Dict[str, str]:
    """Force la reconstruction de l'index FAISS FAQ."""
    engine = get_faq_engine()
    engine.build_or_load(force_rebuild=True)
    return {"status": "ok", "message": "Index FAISS reconstruit."}