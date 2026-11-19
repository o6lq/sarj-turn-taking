"""After the code freeze, the scored code must stay identical to the frozen tag.

    FREEZE_TAG=v2.0-freeze .venv/bin/python -m pytest -q tests/test_freeze_guard.py
"""
import os
import subprocess
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1]
FROZEN = ["turn", "my_system.py", "models", "ttbench"]


@pytest.mark.skipif(not os.environ.get("FREEZE_TAG"), reason="set FREEZE_TAG to check the freeze")
def test_scored_code_matches_the_freeze_tag():
    diff = subprocess.run(["git", "diff", "--stat", os.environ["FREEZE_TAG"], "--", *FROZEN],
                          cwd=KIT, capture_output=True, text=True, check=True).stdout
    assert diff == "", f"scored code changed since {os.environ['FREEZE_TAG']}:\n{diff}"
