"""Configuration centralisée."""
import os
from typing import Union
from dotenv import load_dotenv

load_dotenv()

def _keep_alive() -> Union[int, str]:
    raw = os.getenv("OLLAMA_KEEP_ALIVE", "-1").strip()
    try:
        return int(raw)          # "-1" → -1 (int), "0" → 0
    except ValueError:
        return raw               # "24h" → "24h" (str avec unité)

class Settings:
    # ── Ollama ────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # Modèles recommandés par vitesse croissante :
    #   qwen2.5:1.5b  (800 Mo — très rapide, ~5-10s,  tool calling OK)
    #   qwen2.5:3b    (1.9 Go — rapide,   ~15-25s, tool calling bon)
    #   qwen2.5:7b    (4.7 Go — lent CPU, ~35-45s, tool calling excellent)
    OLLAMA_MODEL: str      = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    OLLAMA_KEEP_ALIVE: Union[int, str] = _keep_alive()
    OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "300"))

    # ── Embeddings ────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
    )

    # ── Supabase ──────────────────────────────────────────────────────
    SUPABASE_URL: str         = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    # ── RAG ───────────────────────────────────────────────────────────
    FAQ_SIMILARITY_THRESHOLD: float = float(
        os.getenv("FAQ_SIMILARITY_THRESHOLD", "0.88")
    )
    FAQ_TOP_K: int            = int(os.getenv("FAQ_TOP_K", "3"))
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "4"))

    # ── Chemins ───────────────────────────────────────────────────────
    BASE_DIR: str         = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str         = os.path.join(BASE_DIR, "data")
    FAISS_INDEX_PATH: str = os.path.join(BASE_DIR, "faiss_index")

    def validate(self) -> None:
        missing = [k for k in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY")
                   if not getattr(self, k)]
        if missing:
            raise RuntimeError(
                f"Variables manquantes dans .env : {', '.join(missing)}"
            )

settings = Settings()
print(f"[config] model={settings.OLLAMA_MODEL}  keep_alive={settings.OLLAMA_KEEP_ALIVE!r}")