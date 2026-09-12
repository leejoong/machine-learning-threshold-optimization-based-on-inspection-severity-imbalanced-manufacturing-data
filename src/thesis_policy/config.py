"""Portable paths for the public reproduction package.

Raw datasets are intentionally kept outside Git.  By default this checkout
expects them in the parent directory, which matches the local layout::

    D:/졸업논문 데이터/
    ├── 논문 공개 코드/   <- this repository
    └── wm811k/, rsw_gun/, ...

On another machine set ``THESIS_DATA_ROOT`` to the directory containing the
dataset folders.  Generated caches, models, and results go to ``outputs/``
unless ``THESIS_OUTPUT_ROOT`` is set.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("THESIS_DATA_ROOT", str(PROJECT_ROOT.parent))).expanduser()
OUTPUT_ROOT = Path(os.environ.get("THESIS_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))).expanduser()
RESULT_ROOT = OUTPUT_ROOT / "results"
MODEL_ROOT = OUTPUT_ROOT / "models"
CACHE_ROOT = OUTPUT_ROOT / "cache"


def ensure_output_dirs() -> None:
    """Create only generated-output directories, never the raw data root."""

    for path in (OUTPUT_ROOT, RESULT_ROOT, MODEL_ROOT, CACHE_ROOT):
        path.mkdir(parents=True, exist_ok=True)
