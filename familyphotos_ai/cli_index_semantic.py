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
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU instead of GPU (for benchmarking). Without this flag the program exits if no GPU is found.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="Images per GPU forward pass (default: 128).",
    )
    p.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 autocast (runs in fp32).",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (skips JIT compilation warm-up).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        metavar="N",
        help="CPU threads for parallel image preprocessing (default: 4).",
    )
    args = p.parse_args()

    if args.cpu:
        ss.set_force_cpu()
    ss.set_inference_options(
        batch_size=args.batch_size,
        fp16=not args.no_fp16,
        compile=not args.no_compile,
    )
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
        num_workers=args.num_workers,
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
