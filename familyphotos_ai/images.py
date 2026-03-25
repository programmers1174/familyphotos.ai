from __future__ import annotations

import os
from pathlib import Path


_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}


def iter_image_files(root: str) -> list[str]:
    p = Path(root)
    if p.is_file():
        if p.suffix.lower() in _IMAGE_EXTS:
            return [str(p.resolve())]
        return []

    results: list[str] = []
    if not p.exists():
        return results

    for dirpath, _, filenames in os.walk(p):
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in _IMAGE_EXTS:
                results.append(str((Path(dirpath) / fn).resolve()))

    results.sort()
    return results

