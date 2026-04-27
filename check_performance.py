"""
Script de diagnostic de performance — à lancer dans le venv.
Vérifie que Ollama est chaud et que le modèle est en RAM.

Usage : python check_performance.py
"""
import os, time, json
import urllib.request

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL      = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

def check_model_loaded():
    """Vérifie si le modèle est en RAM via /api/ps"""
    try:
        r = urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=5)
        data = json.loads(r.read())
        models = data.get("models", [])
        for m in models:
            if MODEL in m.get("name", ""):
                print(f"✅ Modèle EN RAM : {m['name']} ({m.get('size_vram', 'N/A')} VRAM)")
                return True
        print(f"❌ Modèle NON chargé en RAM. Premier appel sera lent.")
        return False
    except Exception as e:
        print(f"⚠️  Impossible de vérifier : {e}")
        return False

def warmup_model():
    """Envoie un ping pour charger le modèle avec keep_alive=-1"""
    print(f"🔥 Chargement du modèle {MODEL} en RAM (peut prendre 60-90s)...")
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps({
                "model": MODEL,
                "prompt": "ok",
                "stream": False,
                "keep_alive": -1,  # permanent
                "options": {"num_predict": 1},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=180)
        elapsed = time.time() - t0
        print(f"✅ Modèle chargé en {elapsed:.1f}s")
    except Exception as e:
        print(f"❌ Erreur warmup : {e}")

def benchmark():
    """Mesure le temps d'une génération simple"""
    print(f"\n📊 Benchmark : génération d'une réponse courte...")
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps({
                "model": MODEL,
                "prompt": "Réponds juste 'OK'.",
                "stream": False,
                "keep_alive": -1,
                "options": {"num_predict": 5},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        r = urllib.request.urlopen(req, timeout=300)
        data = json.loads(r.read())
        elapsed = time.time() - t0
        print(f"✅ Réponse en {elapsed:.1f}s : {data.get('response', '?')!r}")
        
        if elapsed < 10:
            print("🚀 Excellent ! Réponses rapides attendues.")
        elif elapsed < 30:
            print("✅ Acceptable pour une démo.")
        elif elapsed < 60:
            print("⚠️  Lent. Ferme d'autres applications pour libérer la RAM.")
        else:
            print("❌ Très lent. Considère qwen2.5:3b-instruct ou Groq.")
    except Exception as e:
        print(f"❌ Erreur benchmark : {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("Diagnostic de performance du chatbot RAG")
    print("=" * 50)
    loaded = check_model_loaded()
    if not loaded:
        warmup_model()
    benchmark()
    print("\n💡 Conseil : Lance uvicorn APRÈS ce script pour bénéficier du modèle chaud.")