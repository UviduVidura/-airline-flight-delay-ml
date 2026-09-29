#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  echo "Creating Python 3.11 virtual environment at .venv..."
  python3.11 -m venv .venv
fi

if [ -f ".venv/bin/activate" ]; then
  echo "Activating virtual environment..."
  . .venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Setup complete. Activate with: source .venv/bin/activate"
