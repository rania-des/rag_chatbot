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
  GET    /student/agenda                 — planning de révision
  POST   /courses/fiche                  — génère une fiche de révision (JSON)
  POST   /courses/fiche/pdf              — génère une fiche de révision (PDF)

Exécution :
    uvicorn main:app --reload --port 8000 --host 0.0.0.0
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
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
from engine.agenda_engine import generate_weekly_agenda
from engine.fiche_engine import generate_fiche, generate_fiche_pdf
from ingestion import get_course_store, ingest_course
from memory import get_student_memory, update_student_memory
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


class FicheRequest(BaseModel):
    student_id: str
    course_id: str
    difficulty: str = "medium"


# Stockage temporaire des cours uploadés pour la génération de fiches
# Structure: {course_id: {"text": str, "filename": str, "student_id": str}}
_active_courses: Dict[str, dict] = {}


# ================================================
# FastAPI app
# ================================================
app = FastAPI(
    title="Chatbot RAG scolaire",
    description="FAQ + Supabase dynamique + cours uploadés + historique persistent",
    version="4.0.0",
)

# Configuration CORS améliorée pour supporter l'upload de fichiers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
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
async def chat(req: ChatRequest) -> ChatResponse:

    # ── Résolution automatique profile_id → student_id ──────────────────
    student_id = resolve_student_id(req.student_id)
    StudentContext.set(student_id=student_id, class_id=req.class_id)

    try:
        # ── Récupérer le profil mémoire de l'élève ──────────────────────
        memory_profile = await get_student_memory(student_id)

        router = get_router()
        route, faq_doc, score = router.route(
            query=req.message,
            course_id=req.course_id,
            memory_profile=memory_profile,
        )

        # ════════════════════════════════════════════════════════════
        # GREET — réponse instantanée, 0 appel Supabase, 0 appel LLM
        # ════════════════════════════════════════════════════════════
        if route == Route.GREET:
            print("[chat] 👋 GREET")
            return ChatResponse(
                answer=get_greet_response(req.message),
                route=Route.GREET.value,
                conversation_id=req.conversation_id or "",
                conversation_title="",
                faq_score=None, sources=None, citations=None,
            )

        # ════════════════════════════════════════════════════════════
        # Toutes les autres routes → gestion de la conversation
        # ════════════════════════════════════════════════════════════
        is_new = False
        if req.conversation_id:
            conversation_id    = req.conversation_id
            conversation_title = None
            history = rebuild_history_from_db(
                conversation_id, student_id,
                max_messages=settings.MAX_HISTORY_MESSAGES,
            )
        else:
            conv = create_conversation(student_id, first_question=req.message)
            if not conv:
                raise HTTPException(status_code=500,
                                    detail="Impossible de créer la conversation.")
            conversation_id    = conv["id"]
            conversation_title = conv["title"]
            history            = []
            is_new             = True

        # Sauvegarder le message utilisateur
        save_message(conversation_id, role="user", content=req.message)

        # ── Dispatch ────────────────────────────────────────────────
        citations_data = None

        if route == Route.FAQ and faq_doc is not None:
            formatter = get_faq_formatter()
            answer = formatter.format(
                user_question=req.message,
                reference_answer=faq_doc.metadata["answer"],
            )
            sources = [{
                "category":         faq_doc.metadata.get("category"),
                "matched_question": faq_doc.metadata.get("original_question"),
                "similarity":       round(score, 3),
            }]
            citations = None

        elif route == Route.COURSE:
            if not req.course_id:
                raise HTTPException(status_code=400,
                                    detail="course_id requis pour la branche COURSE.")
            engine = get_course_engine()
            try:
                result = engine.answer_question(
                    course_id=req.course_id,
                    student_id=student_id,
                    question=req.message,
                )
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
            answer    = result.answer
            sources   = None
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
            answer = engine.answer(
                query=req.message,
                history=history,
                memory_profile=memory_profile,
            )
            sources   = None
            citations = None

        # ── Sauvegarder la réponse ──────────────────────────────────
        save_message(
            conversation_id,
            role="assistant",
            content=answer,
            route=route.value,
            citations=citations_data,
        )

        # ── Mise à jour mémoire asynchrone (ne bloque pas la réponse) ──
        if route.value in ("DYNAMIC", "COURSE", "FAQ"):
            asyncio.create_task(update_student_memory(
                student_id=student_id,
                topic=req.message[:80],
                difficulty="medium",
                tone="casual",
                note=answer[:300]
            ))

        # ── Générer le titre si nouvelle conversation ────────────────
        if is_new:
            try:
                title = generate_title(req.message, answer)
                update_title(conversation_id, student_id, title)
                conversation_title = title
            except Exception as e:
                print(f"[chat] Titre non généré : {e}")

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
        print(f"[chat] ❌ ERREUR — {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}") from e
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
        count = get_conversation_message_count(conversation_id)
        if count > 0:
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
# ENDPOINTS DE GESTION DE COURS
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
            
            # Extraire le texte complet depuis le vector_store
            full_text_parts = []
            try:
                # Récupérer tous les chunks du vector store
                chunks = session.vector_store.similarity_search("", k=100)
                for doc in chunks:
                    # Essayer d'extraire le texte original
                    text = doc.metadata.get("raw_text", doc.page_content)
                    full_text_parts.append(text)
                full_text = "\n\n".join(full_text_parts)
                print(f"[upload] Texte extrait: {len(full_text)} caractères")
            except Exception as e:
                print(f"[upload] Erreur extraction texte: {e}")
                full_text = ""
            
            _active_courses[session.course_id] = {
                "text": full_text,
                "filename": session.filename,
                "student_id": student_id,
            }
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
        conversation_id="",
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
    _active_courses.pop(course_id, None)
    return {"status": "deleted", "course_id": course_id}


# ================================================
# NOUVEAUX ENDPOINTS
# ================================================
@app.get("/student/agenda")
async def get_agenda(student_id: str):
    """Génère un planning de révision personnalisé pour la semaine."""
    memory = await get_student_memory(student_id)
    agenda = await generate_weekly_agenda(student_id, memory)
    return agenda


@app.post("/courses/fiche")
async def generate_course_fiche(req: FicheRequest):
    """Génère une fiche de révision complète (résumé, QCM, flashcards) depuis un cours uploadé."""
    course = _active_courses.get(req.course_id)
    if not course:
        raise HTTPException(404, "Cours introuvable — uploadez-le d'abord")
    
    if course.get("student_id") != req.student_id:
        raise HTTPException(403, "Ce cours n'appartient pas à cet élève")
    
    if not course.get("text"):
        raise HTTPException(422, "Texte du cours non disponible pour la génération de fiche")
    
    fiche = await generate_fiche(course["text"], req.difficulty)
    return fiche


@app.post("/courses/fiche/pdf")
async def generate_course_fiche_pdf(req: FicheRequest):
    """Génère une fiche de révision au format PDF."""
    course = _active_courses.get(req.course_id)
    if not course:
        raise HTTPException(404, "Cours introuvable — uploadez-le d'abord")
    
    if course.get("student_id") != req.student_id:
        raise HTTPException(403, "Ce cours n'appartient pas à cet élève")
    
    if not course.get("text"):
        raise HTTPException(422, "Texte du cours non disponible pour la génération de fiche")
    
    fiche = await generate_fiche(course["text"], req.difficulty)
    pdf_bytes = generate_fiche_pdf(fiche, course["filename"])
    
    # Nettoyer le nom du fichier pour le téléchargement
    base_name = course["filename"].replace('.pdf', '').replace('.docx', '').replace('.txt', '').replace('.md', '')
    filename = f"fiche-{base_name}.pdf"
    
    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ================================================
# MAINTENANCE
# ================================================
@app.post("/reindex")
def reindex() -> Dict[str, str]:
    """Force la reconstruction de l'index FAISS FAQ."""
    engine = get_faq_engine()
    engine.build_or_load(force_rebuild=True)
    return {"status": "ok", "message": "Index FAISS reconstruit."}


# ================================================
# POINT D'ENTRÉE
# ================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )