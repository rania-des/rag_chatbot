"""
model_profiles.py
─────────────────────────────────────────────────────────────────────────────
Profils de configuration par modèle Ollama.

Utilisation :
    from model_profiles import apply_profile
    apply_profile("qwen2.5:7b")   # à appeler avant de créer le moteur

Ou dans config.py :
    OLLAMA_MODEL = "qwen2.5:7b"
    OLLAMA_PROFILE = ModelProfiles.get(OLLAMA_MODEL)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelProfile:
    # ── Paramètres Ollama ──────────────────────────────────────────────
    model: str
    temperature: float
    num_predict: int          # tokens max en sortie
    num_ctx: int              # fenêtre de contexte (tokens)

    # ── Comportement tool calling ──────────────────────────────────────
    native_tool_calling: bool   # True = tool_calls fiable, False = parse JSON texte
    max_turns: int              # tours max dans la boucle tool calling
    system_prompt_style: str    # "short" | "detailed"
                                # short   → < 100 tokens (petits modèles)
                                # detailed → prompt plus riche (≥ 7b)

    # ── Langue & format ───────────────────────────────────────────────
    supports_arabic: bool       # False = éviter les réponses en arabe
    synthesis_needed: bool      # True = appel LLM séparé pour reformuler
                                # False = retourner résultat tool directement

    # ── Notes ─────────────────────────────────────────────────────────
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Profils
# ─────────────────────────────────────────────────────────────────────────────
PROFILES: dict[str, ModelProfile] = {

    # ── ✅ RECOMMANDÉ — meilleur compromis vitesse/qualité ────────────────────
    "qwen2.5:7b": ModelProfile(
        model               = "qwen2.5:7b",
        temperature         = 0.1,
        num_predict         = 512,
        num_ctx             = 8192,
        native_tool_calling = True,   # tool_calls natif très fiable
        max_turns           = 4,
        system_prompt_style = "detailed",
        supports_arabic     = True,
        synthesis_needed    = True,
        notes=(
            "Meilleur choix global. Tool calling natif fiable, "
            "excellente compréhension FR+AR+dialecte tunisien. "
            "Requiert ~6 GB RAM. Téléchargement : ollama pull qwen2.5:7b"
        ),
    ),

    # ── ✅ ALTERNATIVE légère (< 4 GB RAM) ────────────────────────────────────
    "qwen2.5:3b": ModelProfile(
        model               = "qwen2.5:3b",
        temperature         = 0.1,
        num_predict         = 384,
        num_ctx             = 4096,
        native_tool_calling = True,   # majoritairement natif, parfois JSON texte
        max_turns           = 4,
        system_prompt_style = "detailed",
        supports_arabic     = True,
        synthesis_needed    = True,
        notes=(
            "Bon compromis mémoire/qualité. Recommandé si < 8 GB RAM. "
            "Téléchargement : ollama pull qwen2.5:3b"
        ),
    ),

    # ── ⚠️ Modèle actuel — déconseillé pour le tool calling ──────────────────
    "qwen2.5:1.5b": ModelProfile(
        model               = "qwen2.5:1.5b",
        temperature         = 0.0,    # 0 pour limiter l'imprévisibilité
        num_predict         = 256,
        num_ctx             = 2048,
        native_tool_calling = False,  # retourne JSON texte → parser nécessaire
        max_turns           = 5,      # plus de tours car moins fiable
        system_prompt_style = "short",
        supports_arabic     = True,
        synthesis_needed    = False,  # trop petit pour bien synthétiser
        notes=(
            "Très rapide mais tool calling instable. "
            "La boucle JSON texte de dynamic_engine compense partiellement. "
            "Recommandé seulement si RAM < 4 GB."
        ),
    ),

    # ── ✅ Alternative FR-focalisée ───────────────────────────────────────────
    "mistral:7b": ModelProfile(
        model               = "mistral:7b",
        temperature         = 0.1,
        num_predict         = 512,
        num_ctx             = 8192,
        native_tool_calling = False,  # tool calling via prompt engineering
        max_turns           = 4,
        system_prompt_style = "detailed",
        supports_arabic     = False,  # AR très faible
        synthesis_needed    = True,
        notes=(
            "Excellent en français, faible en arabe. "
            "Tool calling non natif → le parser JSON texte doit être actif. "
            "Téléchargement : ollama pull mistral:7b"
        ),
    ),

    # ── ✅ Bonne alternative généraliste ──────────────────────────────────────
    "llama3.1:8b": ModelProfile(
        model               = "llama3.1:8b",
        temperature         = 0.1,
        num_predict         = 512,
        num_ctx             = 8192,
        native_tool_calling = True,
        max_turns           = 4,
        system_prompt_style = "detailed",
        supports_arabic     = False,   # AR basique
        synthesis_needed    = True,
        notes=(
            "Bon généraliste, tool calling natif. "
            "Arabe très limité — déconseillé pour les élèves arabophones. "
            "Téléchargement : ollama pull llama3.1:8b"
        ),
    ),
}

# Alias (noms courts acceptés dans config.py)
ALIASES: dict[str, str] = {
    "qwen7b"      : "qwen2.5:7b",
    "qwen3b"      : "qwen2.5:3b",
    "qwen1.5b"    : "qwen2.5:1.5b",
    "mistral"     : "mistral:7b",
    "llama3"      : "llama3.1:8b",
    "llama3.1"    : "llama3.1:8b",
}


def get_profile(model_name: str) -> ModelProfile:
    """
    Retourne le profil d'un modèle.
    Accepte le nom complet ou un alias.
    Si le modèle est inconnu, retourne un profil générique sûr.
    """
    resolved = ALIASES.get(model_name, model_name)
    if resolved in PROFILES:
        return PROFILES[resolved]

    # Profil générique pour les modèles non listés
    print(f"[ModelProfile] ⚠️  Modèle '{model_name}' non listé — profil générique utilisé.")
    return ModelProfile(
        model               = model_name,
        temperature         = 0.1,
        num_predict         = 512,
        num_ctx             = 4096,
        native_tool_calling = False,  # prudent par défaut
        max_turns           = 5,
        system_prompt_style = "detailed",
        supports_arabic     = True,
        synthesis_needed    = True,
        notes               = "Profil générique.",
    )


def print_comparison() -> None:
    """Affiche un tableau de comparaison dans le terminal."""
    print("\n" + "═" * 78)
    print(f"{'MODÈLE':<20} {'RAM':<8} {'TOOL':<8} {'ARABE':<8} {'VITESSE':<10} NOTES")
    print("═" * 78)
    infos = {
        "qwen2.5:7b"  : ("~6 GB",  "✅ Natif", "✅",     "★★★★☆"),
        "qwen2.5:3b"  : ("~3 GB",  "✅ Natif", "✅",     "★★★★★"),
        "qwen2.5:1.5b": ("~2 GB",  "⚠️ JSON",  "✅",     "★★★★★"),
        "mistral:7b"  : ("~5 GB",  "⚠️ JSON",  "❌",     "★★★★☆"),
        "llama3.1:8b" : ("~6 GB",  "✅ Natif", "❌",     "★★★☆☆"),
    }
    for name, (ram, tool, ar, speed) in infos.items():
        marker = " ← ACTUEL" if name == "qwen2.5:1.5b" else (
                 " ← RECOMMANDÉ" if name == "qwen2.5:7b" else "")
        print(f"{name:<20} {ram:<8} {tool:<10} {ar:<8} {speed:<10}{marker}")
    print("═" * 78 + "\n")


if __name__ == "__main__":
    print_comparison()
    for name, p in PROFILES.items():
        print(f"\n[{name}]\n  → {p.notes}")
