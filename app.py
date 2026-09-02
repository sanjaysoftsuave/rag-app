"""Entry point for `streamlit run app.py`.

Streamlit needs a script path, and it executes that script directly rather
than importing the package — so `src/` will not be on `sys.path` unless the
project happens to be pip-installed. This repo's checked-in `.venv` is broken
(see CLAUDE.md), which makes "happens to be installed" an unsafe assumption,
so the path is added explicitly here.

The UI itself lives in `src/rag_app/ui.py` so it stays part of the package and
its helpers remain importable by the test suite.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_app.ui import main  # noqa: E402 - must follow the sys.path fix above

main()
