"""
Rebuild cached JPEG thumbnails under desktop/thumbnails/ for all entries in
desktop/photos_db.json.

Run from repo root:
  uv run python desktop/backend/rebuild_thumbnails.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

_THUMB_MAX_EDGE = 320


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_db_path() -> Path:
    return _repo_root() / "desktop" / "photos_db.json"


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


def _collect_jobs(db_path: Path) -> list[tuple[str, str, str]]:
    raw = json.loads(db_path.read_text(encoding="utf-8"))
    entries = raw.get("photos", [])
    if not isinstance(entries, list):
        raise ValueError('"photos" must be a list')
    repo = _repo_root()

    jobs: list[tuple[str, str, str]] = []
    for row in entries:
        if not isinstance(row, dict):
            continue
        pid = row.get("id")
        rel = row.get("relativePath") or row.get("file")
        if not pid or not rel:
            continue
        src = Path(str(rel).replace("\\", "/"))
        if not src.is_absolute():
            src = (db_path.parent / src).resolve()
        else:
            src = src.resolve()
        if not src.is_file():
            continue
        dest = _thumb_jpeg_path(repo, str(pid))
        jobs.append((str(pid), str(src), str(dest)))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild desktop/thumbnails for all photos_db.json entries.")
    ap.add_argument("--db", type=Path, default=_default_db_path(), help="Path to photos_db.json")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes for thumbnails (default: min(8, CPU count)).",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing .jpg thumbs before rebuilding (recommended).",
    )
    args = ap.parse_args()

    workers = args.workers
    if workers is None:
        workers = max(1, min(8, os.cpu_count() or 4))

    db_path: Path = args.db.expanduser().resolve()
    if not db_path.is_file():
        print(f"error: database not found: {db_path}", file=sys.stderr)
        return 1

    repo = _repo_root()
    thumbs_dir = repo / "desktop" / "thumbnails"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for p in thumbs_dir.glob("*.jpg"):
            try:
                p.unlink()
            except OSError:
                pass

    jobs = _collect_jobs(db_path)
    if not jobs:
        print("No valid photo entries found to thumbnail.")
        return 0

    failed: list[tuple[str, str]] = []
    n = len(jobs)
    done = 0
    print(f"Rebuilding {n} thumbnails with {workers} workers...", flush=True)

    # Threads work well here because Pillow's heavy lifting is in native code
    # and we avoid Windows process-spawn overhead for large libraries.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_thumb_worker, j): j[0] for j in jobs}
        for fut in as_completed(futures):
            done += 1
            pid, err = fut.result()
            if err:
                failed.append((pid, err))
            if done % 500 == 0 or done == n:
                print(f"  {done}/{n}", flush=True)

    if failed:
        fail_path = db_path.parent / "rebuild_thumbnail_errors.txt"
        fail_path.write_text(
            "\n".join(f"{pid}\t{msg}" for pid, msg in failed) + "\n",
            encoding="utf-8",
        )
        print(f"warning: {len(failed)} thumbnails failed; see {fail_path}", file=sys.stderr)
        return 2

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

