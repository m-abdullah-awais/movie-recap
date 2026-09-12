#!/usr/bin/env python
"""Entry point for the analysis stages of the Local Movie Recap Generator.

Run ``python analyze.py --help`` for the available commands.

``recap.config`` is imported before anything else on purpose. It redirects every
cache and install location into the project directory, and ``HF_HOME`` in
particular has to be set before any library that reads it is imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Line buffering, so progress appears as it happens even when output is piped or
# captured. Python otherwise switches stdout to a large block buffer whenever it
# is not a terminal, which makes a stage that is working normally look hung for
# minutes at a time.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recap import config  # noqa: E402, F401  (imported first for containment)
from recap.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
