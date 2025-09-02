#!/usr/bin/env bash
# Create and activate venv (Unix/macOS). For Windows, use scripts/env.ps1
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Activated. To use later: source .venv/bin/activate"
