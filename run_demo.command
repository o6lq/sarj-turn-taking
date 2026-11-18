#!/bin/bash
# Mac: double-click to start the demo «متى يسكت الوكيل». It opens http://127.0.0.1:8790/demo by itself.
cd "$(dirname "$0")" || exit 1
if [ ! -f .venv/bin/activate ]; then
  echo "No .venv found. Run: bash setup.sh"; read -r -p "Press Enter to close"; exit 1
fi
source .venv/bin/activate
python -m tools.live
read -r -p "Server stopped. Press Enter to close"
