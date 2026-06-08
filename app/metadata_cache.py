from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterator
from functools import lru_cache


# Lazy-loaded global cache
_METADATA_FILES: list[Path] | None = None
_LOCK = threading.Lock()

def iter_csv_files(directory: Path) -> Iterator[Path]:
    """
    Yield CSV files from a flat directory (non-recursive).
    """
    if not directory.exists():
        raise FileNotFoundError(directory)

    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(".csv"):
                yield Path(entry.path)

@lru_cache(maxsize=1)
def get_metadata_files() -> tuple[Path, ...]:
    return tuple(iter_csv_files(Path("/data/metadata/image")))