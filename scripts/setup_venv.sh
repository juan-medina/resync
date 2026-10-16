#!/usr/bin/env bash
set -e

echo "[1/3] Creating virtual environment (.venv)..."
python3 -m venv .venv

echo "[2/3] Upgrading pip..."
source .venv/bin/activate
pip install --upgrade pip

echo "[3/3] Installing dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

echo "Setup complete. Run 'source .venv/bin/activate' to activate."
