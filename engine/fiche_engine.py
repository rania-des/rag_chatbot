# rag_chatbot/engine/fiche_engine.py
import json
import unicodedata
from datetime import datetime
from typing import Optional
import httpx
from config import settings
from fpdf import FPDF


def _clean(text: str) -> str:
    """Convertit les caractères spéciaux en ASCII lisible."""
    if not text:
        return ""
    return unicodedata.normalize('NFKD', str(text)).encode('latin-1', 'replace').decode('latin-1')


async def generate_fiche(course_text: str, difficulty: str = "medium") -> dict:
    """
    Génère une fiche de révision complète à partir du texte du cours.
    
    Args:
        course_text: Texte extrait du cours
        difficulty: Niveau de difficulté (easy, medium, hard)
    
    Returns:
        Dictionnaire contenant résumé, points clés, QCM et flashcards
    """
    
    prompt = f"""Tu es un professeur expert. Génère une fiche de révision à partir du cours ci-dessous.

Cours :
{course_text[:8000]}

Niveau de difficulté demandé : {difficulty}

Réponds UNIQUEMENT en JSON valide, sans aucun texte avant ou après, avec cette structure exacte :
{{
  "resume": "Résumé concis du cours en 5-7 phrases",
  "points_cles": ["Point clé 1", "Point clé 2", "Point clé 3", "Point clé 4", "Point clé 5"],
  "qcm": [
    {{
      "question": "Question du QCM",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "reponse_correcte": "Option A"
    }}
  ],
  "flashcards": [
    {{"question": "Question flashcard", "reponse": "Réponse flashcard"}}
  ]
}}

Génère entre 3 et 5 QCM et entre 5 et 8 flashcards.
Adapte la difficulté : si difficulty=hard, ajoute des questions plus complexes et des pièges dans les options.
"""

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                }
            )
            
            raw = response.json()["message"]["content"]
            
            # Nettoyer la réponse si nécessaire (enlever les backticks)
            if raw.startswith("```json"):
                raw = raw[7:-3]
            elif raw.startswith("```"):
                raw = raw[3:-3]
            
            result = json.loads(raw)
            
            # Valeurs par défaut si certains champs manquent
            result.setdefault("resume", "Résumé non disponible")
            result.setdefault("points_cles", [])
            result.setdefault("qcm", [])
            result.setdefault("flashcards", [])
            
            return result
            
    except Exception as e:
        print(f"[FicheEngine] Erreur génération: {e}")
        return {
            "resume": f"Erreur lors de la génération de la fiche: {str(e)}",
            "points_cles": [],
            "qcm": [],
            "flashcards": []
        }


def generate_fiche_pdf(fiche: dict, filename: str) -> bytes:
    """
    Génère un PDF à partir des données de la fiche de révision.
    
    Args:
        fiche: Dictionnaire contenant resume, points_cles, qcm, flashcards
        filename: Nom du fichier source
    
    Returns:
        Bytes du PDF généré
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=25)
    
    # ============================================================
    # HEADER - Bleu
    # ============================================================
    pdf.set_fill_color(96, 165, 250)  # blue-400
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_xy(0, 8)
    pdf.cell(210, 12, _clean('FICHE DE REVISION'), align='C')
    
    # ============================================================
    # NOM DU FICHIER
    # ============================================================
    pdf.set_y(35)
    pdf.set_fill_color(243, 244, 246)  # gray-100
    pdf.set_text_color(30, 64, 120)    # blue-800
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_x(10)
    pdf.cell(190, 10, _clean(f'Cours : {filename}'), fill=True, border=0)
    pdf.ln(14)
    
    # ============================================================
    # RESUME
    # ============================================================
    pdf.set_fill_color(96, 165, 250)   # blue-400
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_x(10)
    pdf.cell(190, 10, _clean('  RESUME'), fill=True, border=0)
    pdf.ln(12)
    
    pdf.set_text_color(55, 65, 81)     # gray-700
    pdf.set_font('Helvetica', '', 10)
    resume = fiche.get('resume', 'Résumé non disponible')
    pdf.set_x(10)
    pdf.multi_cell(190, 6, _clean(resume))
    pdf.ln(8)
    
    # ============================================================
    # POINTS CLES
    # ============================================================
    pdf.set_fill_color(96, 165, 250)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_x(10)
    pdf.cell(190, 10, _clean('  POINTS CLES'), fill=True, border=0)
    pdf.ln(12)
    
    pdf.set_text_color(55, 65, 81)
    pdf.set_font('Helvetica', '', 10)
    points_cles = fiche.get('points_cles', [])
    
    for i, point in enumerate(points_cles, 1):
        # Puces
        pdf.set_x(10)
        pdf.cell(6, 6, _clean(f'{i}.'))
        pdf.set_fill_color(96, 165, 250)
        pdf.rect(pdf.get_x() - 2, pdf.get_y() + 2, 3, 3, 'F')
        pdf.multi_cell(184, 6, _clean(f' {point}'))
    
    pdf.ln(8)
    
    # ============================================================
    # QCM
    # ============================================================
    pdf.set_fill_color(96, 165, 250)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_x(10)
    pdf.cell(190, 10, _clean('  QCM'), fill=True, border=0)
    pdf.ln(12)
    
    qcm_list = fiche.get('qcm', [])
    
    for i, q in enumerate(qcm_list, 1):
        # Question
        pdf.set_text_color(30, 64, 120)   # blue-800
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_x(10)
        pdf.multi_cell(190, 6, _clean(f'Q{i}. {q.get("question", "")}'))
        
        # Options
        pdf.set_text_color(55, 65, 81)    # gray-700
        pdf.set_font('Helvetica', '', 9)
        for opt in q.get('options', []):
            pdf.set_x(14)
            pdf.multi_cell(186, 5, _clean(f'  • {opt}'))
        
        # Réponse correcte
        pdf.set_text_color(5, 150, 105)   # green-600
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_x(10)
        pdf.multi_cell(190, 5, _clean(f'Reponse : {q.get("reponse_correcte", "")}'))
        pdf.ln(6)
    
    pdf.ln(4)
    
    # ============================================================
    # FLASHCARDS
    # ============================================================
    # Vérifier si on a besoin d'une nouvelle page
    if pdf.get_y() > 230:
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=25)
    
    pdf.set_fill_color(96, 165, 250)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_x(10)
    pdf.cell(190, 10, _clean('  FLASHCARDS'), fill=True, border=0)
    pdf.ln(12)
    
    flashcards = fiche.get('flashcards', [])
    
    for i, f in enumerate(flashcards, 1):
        # Question
        pdf.set_fill_color(249, 250, 251)  # gray-50
        pdf.set_text_color(30, 64, 120)    # blue-800
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_x(10)
        pdf.multi_cell(190, 6, _clean(f'{i}. Q : {f.get("question", "")}'), fill=True, border=1)
        
        # Réponse
        pdf.set_text_color(55, 65, 81)     # gray-700
        pdf.set_font('Helvetica', '', 9)
        pdf.set_x(14)
        pdf.multi_cell(186, 6, _clean(f'R : {f.get("reponse", "")}'))
        pdf.ln(4)
    
    # ============================================================
    # FOOTER
    # ============================================================
    pdf.set_y(-20)
    pdf.set_text_color(107, 114, 128)     # gray-500
    pdf.set_font('Helvetica', '', 7)
    pdf.cell(190, 5, _clean(f'Genere le {datetime.now().strftime("%d/%m/%Y")} — OMNIA Plateforme educative'), align='C')
    
    # Retourner les bytes du PDF
    return pdf.output()