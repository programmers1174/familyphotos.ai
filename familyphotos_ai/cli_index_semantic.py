"""Command-line semantic index build (same FAISS store as the desktop app). CUDA required."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _backend_dir() -> Path:
    return _repo_root() / "desktop" / "backend"


def _ensure_semantic_search() -> object:
    b = str(_backend_dir())
    if b not in sys.path:
        sys.path.insert(0, b)
    import semantic_search as ss  # noqa: E402  # type: ignore[import-untyped]

    return ss


def _default_db_path() -> Path:
    return _repo_root() / "desktop" / "photos_db.json"


def _load_db(db_path: Path) -> tuple[Path, list[dict]]:
    if not db_path.is_file():
        raise FileNotFoundError(f"Photo database not found: {db_path}")
    raw = json.loads(db_path.read_text(encoding="utf-8"))
    db_dir = db_path.parent
    photos_root = Path(raw.get("photosRoot", ".")).expanduser()
    if not photos_root.is_absolute():
        photos_root = (db_dir / photos_root).resolve()
    photos_root = photos_root.resolve()
    entries = raw.get("photos", [])
    if not isinstance(entries, list):
        raise ValueError('"photos" must be a list')
    return photos_root, entries


def main() -> int:
    ss = _ensure_semantic_search()
    p = argparse.ArgumentParser(
        description="Build the semantic search FAISS index from photos_db.json. "
        "Requires an NVIDIA GPU and PyTorch with CUDA.",
    )
    p.add_argument(
        "--model",
        required=True,
        choices=sorted(ss.MODELS.keys()),
        help="Embedding model id (same as in the desktop UI).",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=_default_db_path(),
        help=f"Path to photos_db.json (default: {_default_db_path()})",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=_backend_dir() / "indexes",
        help="Output directory for <model>.faiss and <model>_ids.json.",
    )
    args = p.parse_args()

    ss.init_inference_device()

    try:
        photos_root, entries = _load_db(args.db.resolve())
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    args.index_dir.mkdir(parents=True, exist_ok=True)
    n = ss.run_semantic_index_sync(
        args.model,
        entries,
        photos_root,
        args.index_dir.resolve(),
    )
    store = ss.FaissStore(
        args.index_dir.resolve(),
        args.model,
        ss.MODELS[args.model]["embedding_dim"],
    )
    print(
        f"Pending batch: {n} photo(s); total vectors in index: {store.count}; "
        f"device: {ss.device_label()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
