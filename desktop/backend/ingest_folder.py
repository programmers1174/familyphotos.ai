"""
Scan a folder for images, append entries to desktop/photos_db.json, and pre-build
JPEG thumbnails under desktop/thumbnails/ (same layout as main.py).

Run from repo root:
  uv run python desktop/backend/ingest_folder.py --source "D:\\Varshneys\\Pictures\\Sorted"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

_THUMB_MAX_EDGE = 320
_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _thumb_jpeg_path(repo_root: Path, photo_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", photo_id).strip("._-") or "photo"
    d = repo_root / "desktop" / "thumbnails"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"fp_{safe}.jpg"


def _ensure_thumbnail_jpeg(source: Path, dest: Path) -> None:
    from PIL import Image, ImageOps

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im)
        rgb = im.convert("RGB")
        rgb.thumbnail((_THUMB_MAX_EDGE, _THUMB_MAX_EDGE), Image.Resampling.LANCZOS)
        tmp = dest.with_suffix(".jpg.tmp")
        rgb.save(tmp, format="JPEG", quality=82, optimize=True)
        tmp.replace(dest)


def _thumb_worker(job: tuple[str, str, str]) -> tuple[str, str | None]:
    """(photo_id, source_str, dest_str) -> (photo_id, error or None)."""
    try:
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except ImportError:
            pass
        _ensure_thumbnail_jpeg(Path(job[1]), Path(job[2]))
        return job[0], None
    except Exception as e:  # noqa: BLE001 — want per-file errors
        return job[0], str(e)


def _default_db_path() -> Path:
    return _repo_root() / "desktop" / "photos_db.json"


def _max_photo_numeric_id(entries: list[dict[str, Any]]) -> int:
    best = 0
    pat = re.compile(r"^photo-(\d+)$")
    for row in entries:
        if not isinstance(row, dict):
            continue
        m = pat.match(str(row.get("id", "")))
        if m:
            best = max(best, int(m.group(1)))
    return best


def _existing_paths(entries: list[dict[str, Any]]) -> set[str]:
    seen: set[str] = set()
    for row in entries:
        if not isinstance(row, dict):
            continue
        rel = row.get("relativePath") or row.get("file")
        if not rel:
            continue
        try:
            seen.add(str(Path(str(rel)).resolve()))
        except OSError:
            seen.add(str(rel))
    return seen


def _collect_images(root: Path) -> list[Path]:
    out: list[Path] = []
    root = root.resolve()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part.startswith("._") for part in p.parts):
            continue
        if p.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        try:
            if p.stat().st_size == 0:
                continue
        except OSError:
            continue
        out.append(p)
    out.sort(key=lambda x: str(x).lower())
    return out


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest images into photos_db.json and build thumbnails.")
    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Root folder to scan (recursive).",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=_default_db_path(),
        help="Path to photos_db.json",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes for thumbnails (default: min(8, CPU count)).",
    )
    ap.add_argument(
        "--skip-thumbnails",
        action="store_true",
        help="Only update the JSON; thumbnails will be built on first API request.",
    )
    args = ap.parse_args()
    workers = args.workers
    if workers is None:
        workers = max(1, min(8, os.cpu_count() or 4))
    source: Path = args.source.expanduser()
    db_path: Path = args.db.expanduser().resolve()
    if not source.is_dir():
        print(f"error: source is not a directory: {source}", file=sys.stderr)
        return 1
    if not db_path.is_file():
        print(f"error: database not found: {db_path}", file=sys.stderr)
        return 1

    raw = json.loads(db_path.read_text(encoding="utf-8"))
    entries = raw.get("photos")
    if not isinstance(entries, list):
        print('error: "photos" must be a list', file=sys.stderr)
        return 1

    existing = _existing_paths(entries)
    next_num = _max_photo_numeric_id(entries) + 1
    repo = _repo_root()

    images = _collect_images(source)
    new_rows: list[dict[str, str]] = []
    for img in images:
        key = str(img.resolve())
        if key in existing:
            continue
        pid = f"photo-{next_num}"
        next_num += 1
        new_rows.append({"id": pid, "relativePath": key})
        existing.add(key)

    if not new_rows:
        print("Nothing new to add (all images already in database).")
        return 0

    print(f"Adding {len(new_rows)} photos from {source} ...")

    if not args.skip_thumbnails:
        jobs: list[tuple[str, str, str]] = []
        for row in new_rows:
            pid = row["id"]
            src = row["relativePath"]
            dest = str(_thumb_jpeg_path(repo, pid))
            jobs.append((pid, src, dest))

        failed: list[tuple[str, str]] = []
        n = len(jobs)
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_thumb_worker, j): j[0] for j in jobs}
            for fut in as_completed(futures):
                done += 1
                pid, err = fut.result()
                if err:
                    failed.append((pid, err))
                if done % 500 == 0 or done == n:
                    print(f"  thumbnails {done}/{n}")

        if failed:
            fail_path = db_path.parent / "ingest_thumbnail_errors.txt"
            fail_path.write_text(
                "\n".join(f"{pid}\t{msg}" for pid, msg in failed) + "\n",
                encoding="utf-8",
            )
            print(
                f"warning: {len(failed)} thumbnails failed; see {fail_path}",
                file=sys.stderr,
            )

    entries.extend(new_rows)
    raw["photos"] = entries
    _atomic_write_json(db_path, raw)
    print(f"Updated {db_path} (total photos: {len(entries)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
