#!/usr/bin/env bash
set -euo pipefail

log() { echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

# Resolve repo directory
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

log "📦 Mise à jour des paquets et installation des outils de base"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-dev build-essential git

log "🐍 Création de l'environnement virtuel Python (.venv)"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

log "📚 Installation des dépendances depuis requirements.txt"
pip install -r requirements.txt

log "🧰 Détection CUDA et installation de PyTorch (GPU si possible)"
TORCH_INDEX_URL=""
CUDA_VER=""
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VER=$(nvidia-smi | grep -i "CUDA Version" | sed -E 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/' || true)
  log "NVIDIA détecté. Version CUDA: ${CUDA_VER:-inconnue}"
  case "$CUDA_VER" in
    12.4*) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124" ;;
    12.3*) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu123" ;;
    12.1*) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" ;;
    11.8*) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118" ;;
    *) TORCH_INDEX_URL="" ;;
  esac
else
  log "nvidia-smi non trouvé; installation de PyTorch CPU."
fi

if [[ -n "$TORCH_INDEX_URL" ]]; then
  log "Installation de PyTorch avec CUDA depuis $TORCH_INDEX_URL"
  pip install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
else
  log "Installation de PyTorch CPU (fallback)"
  pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
fi

log "⚙️ Configuration de la croissance mémoire TensorFlow (optionnel)"
python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print(f"Avertissement: impossible d'activer memory_growth sur {g}: {e}")
print("TF GPUs:", gpus)
PY

log "✅ Vérification GPU TensorFlow et PyTorch"
python - <<'PY'
import tensorflow as tf, torch
print("TensorFlow GPUs:", tf.config.list_physical_devices('GPU'))
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Torch current device:", torch.cuda.get_device_name(torch.cuda.current_device()))
PY

log "🎉 Installation terminée"
echo "\nProchaines étapes:"
echo "  source .venv/bin/activate"
echo "  python train_minute_model_lstm.py --interactive"