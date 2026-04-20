#!/bin/bash
set -e

echo "============================================================"
echo "🎯 Setup Ambiente per Qualcomm X Elite (ARM64) & ONNX Export"
echo "============================================================"

VENV_DIR=".venv_qualcomm"

# 1. Crea VENV se non esiste
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creazione Virtual Environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
fi

# 2. Attiva e installa dipendenze
echo "[*] Attivazione venv..."
source $VENV_DIR/bin/activate

echo "[*] Installazione pacchetti essenziali per PyTorch e ONNX..."
pip install --upgrade pip
pip install torch transformers onnx onnxruntime gliner gliner2 urllib3 requests huggingface_hub

# 3. Esegui export
echo "[*] Esecuzione script di esportazione ONNX..."
python ../05_export_e2e_fragments.py

echo "============================================================"
echo "✅ Esportazione Completata!"
echo "============================================================"
