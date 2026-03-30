"""
Stage 3: Semantic search using CLIP / SigLiP models and a FAISS vector store.

Architecture
------------
* MODELS registry  — describes every supported embedding model.
* FaissStore       — wraps a FAISS IndexFlatIP with a photo-id mapping; saved to disk.
* embed_image / embed_text — run a model on CUDA only and return an L2-normalised
  numpy vector.
* IndexingManager  — background-threads indexing; exposes status and search.
"""
from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: dict[str, dict[str, Any]] = {
    "clip-vit-base-patch32": {
        "id": "clip-vit-base-patch32",
        "name": "CLIP ViT-B/32",
        "hf_name": "openai/clip-vit-base-patch32",
        "model_type": "clip",
        "description": "OpenAI CLIP Base — fast, good quality (512-dim). Great starting point.",
        "embedding_dim": 512,
        "size_mb": 338,
    },
    "clip-vit-large-patch14": {
        "id": "clip-vit-large-patch14",
        "name": "CLIP ViT-L/14",
        "hf_name": "openai/clip-vit-large-patch14",
        "model_type": "clip",
        "description": "OpenAI CLIP Large — higher quality (768-dim). Slower to download & index.",
        "embedding_dim": 768,
        "size_mb": 1600,
    },
    "siglip-base-patch16-224": {
        "id": "siglip-base-patch16-224",
        "name": "SigLiP Base",
        "hf_name": "google/siglip-base-patch16-224",
        "model_type": "siglip",
        "description": "Google SigLiP Base — modern architecture, excellent quality (768-dim).",
        "embedding_dim": 768,
        "size_mb": 400,
    },
    "siglip-so400m-patch14-384": {
        "id": "siglip-so400m-patch14-384",
        "name": "SigLiP SO400M",
        "hf_name": "google/siglip-so400m-patch14-384",
        "model_type": "siglip",
        "description": "Google SigLiP SO400M — highest quality (1152-dim). Best results, large model.",
        "embedding_dim": 1152,
        "size_mb": 3500,
    },
}


_force_cpu: bool = False
_batch_size: int = 128
_use_fp16: bool = True
_use_compile: bool = True


def set_force_cpu(enabled: bool = True) -> None:
    global _force_cpu
    _force_cpu = enabled


def set_inference_options(
    batch_size: int | None = None,
    fp16: bool | None = None,
    compile: bool | None = None,
) -> None:
    global _batch_size, _use_fp16, _use_compile
    if batch_size is not None:
        _batch_size = batch_size
    if fp16 is not None:
        _use_fp16 = fp16
    if compile is not None:
        _use_compile = compile


def _device() -> torch.device:
    if _force_cpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for semantic embedding models but torch.cuda.is_available() "
            "is False. Install a PyTorch build with CUDA for your NVIDIA driver (see pytorch.org). "
            "Use --cpu to run on CPU instead."
        )
    return torch.device("cuda:0")


def init_inference_device() -> None:
    """Warm up the inference device. Exits if no GPU and --cpu was not requested."""
    try:
        device = _device()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    if device.type == "cuda":
        torch.zeros(1, device=device)
        torch.cuda.synchronize()


def device_label() -> str:
    if _force_cpu:
        return "CPU"
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


# ---------------------------------------------------------------------------
# FAISS vector store
# ---------------------------------------------------------------------------

class FaissStore:
    """
    FAISS IndexFlatIP (inner-product / cosine for L2-normalised vectors)
    with a parallel list of photo-ids, persisted to two files:
      <model_id>.faiss   — binary FAISS index
      <model_id>_ids.json — JSON array of photo-ids in insertion order
    """

    def __init__(self, index_dir: Path, model_id: str, embedding_dim: int) -> None:
        import faiss as _faiss  # local import so the module loads without faiss installed
        self._faiss = _faiss
        self._dir = index_dir
        self._model_id = model_id
        self._dim = embedding_dim
        self._index_path = index_dir / f"{model_id}.faiss"
        self._ids_path = index_dir / f"{model_id}_ids.json"
        self._lock = threading.Lock()
        self._index: Any = None
        self._photo_ids: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._index_path.exists() and self._ids_path.exists():
            self._index = self._faiss.read_index(str(self._index_path))
            self._photo_ids = json.loads(self._ids_path.read_text(encoding="utf-8"))
        else:
            self._index = self._faiss.IndexFlatIP(self._dim)
            self._photo_ids = []

    def save(self) -> None:
        with self._lock:
            self._faiss.write_index(self._index, str(self._index_path))
            self._ids_path.write_text(
                json.dumps(self._photo_ids), encoding="utf-8"
            )

    # ------------------------------------------------------------------
    def indexed_ids(self) -> set[str]:
        with self._lock:
            return set(self._photo_ids)

    def add(self, photo_id: str, embedding: np.ndarray) -> None:
        vec = embedding.reshape(1, -1).astype("float32")
        self._faiss.normalize_L2(vec)
        with self._lock:
            self._index.add(vec)
            self._photo_ids.append(photo_id)

    def search(self, query_embedding: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        with self._lock:
            total = self._index.ntotal
        if total == 0:
            return []
        vec = query_embedding.reshape(1, -1).astype("float32")
        self._faiss.normalize_L2(vec)
        k = min(k, total)
        with self._lock:
            D, I = self._index.search(vec, k)
            ids_snap = list(self._photo_ids)
        results: list[tuple[str, float]] = []
        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(ids_snap):
                results.append((ids_snap[idx], float(dist)))
        return results

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._photo_ids)


# ---------------------------------------------------------------------------
# Model loading & embedding
# ---------------------------------------------------------------------------

_loaded_models: dict[str, tuple] = {}
_model_lock = threading.Lock()

# Warmed at desktop API startup so the UI opens only after weights are ready.
DEFAULT_PRELOAD_MODEL_ID = "clip-vit-base-patch32"


def preload_embedding_model(model_id: str | None = None) -> str:
    """Load one embedding model into GPU/CPU memory (and torch.compile on CUDA). Safe to call twice."""
    mid = model_id or DEFAULT_PRELOAD_MODEL_ID
    if mid not in MODELS:
        raise ValueError(f"Unknown model_id for preload: {mid!r}")
    info = MODELS[mid]
    print(f"[semantic_search] preloading embedding model {mid!r} …", flush=True)
    _load_model(info)
    print(f"[semantic_search] model {mid!r} ready.", flush=True)
    return mid


def _load_model(model_info: dict[str, Any]) -> tuple:
    """Return (model, processor, device, model_type), loading once and caching."""
    mid = model_info["id"]
    with _model_lock:
        if mid in _loaded_models:
            return _loaded_models[mid]

        hf_name = model_info["hf_name"]
        mtype = model_info["model_type"]
        device = _device()

        if mtype == "clip":
            from transformers import CLIPModel, CLIPProcessor
            processor = CLIPProcessor.from_pretrained(hf_name)
            model = CLIPModel.from_pretrained(hf_name).to(device)
        else:  # siglip
            from transformers import SiglipModel, SiglipProcessor
            processor = SiglipProcessor.from_pretrained(hf_name)
            model = SiglipModel.from_pretrained(hf_name).to(device)

        model.eval()
        if _use_compile and device.type == "cuda":
            print(f"[semantic_search] compiling model with torch.compile …", flush=True)
            model = torch.compile(model)
        entry = (model, processor, device, mtype)
        _loaded_models[mid] = entry
        return entry


def _autocast():
    import contextlib
    if _use_fp16 and not _force_cpu:
        return torch.autocast("cuda")
    return contextlib.nullcontext()


def embed_image(model_info: dict[str, Any], image_path: Path) -> np.ndarray:
    from PIL import Image
    model, processor, device, _ = _load_model(model_info)
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad(), _autocast():
        feats = model.get_image_features(**inputs)
    if not isinstance(feats, torch.Tensor):
        feats = feats.pooler_output
    return feats.cpu().float().numpy()[0]


def embed_text(model_info: dict[str, Any], text: str) -> np.ndarray:
    model, processor, device, mtype = _load_model(model_info)
    if mtype == "siglip":
        # SigLiP requires fixed-length padding for text
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(device)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad(), _autocast():
        feats = model.get_text_features(**inputs)
    if not isinstance(feats, torch.Tensor):
        feats = feats.pooler_output
    return feats.cpu().float().numpy()[0]


def _index_pending_into_store(
    model_info: dict[str, Any],
    store: FaissStore,
    pending: list[dict],
    photos_root: Path,
    *,
    checkpoint_every: int = 20,
    after_each: Callable[[], None] | None = None,
    show_progress: bool = False,
    num_workers: int = 4,
) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from tqdm import tqdm

    model, processor, device, _ = _load_model(model_info)

    def _load_one(photo: dict) -> tuple[dict, torch.Tensor | None, str | None]:
        rel = str(photo.get("relativePath") or photo.get("file") or "")
        rel_p = Path(rel.replace("\\", "/"))
        img_path = rel_p if rel_p.is_absolute() else photos_root / rel_p
        if not img_path.is_file():
            return photo, None, None
        try:
            img = Image.open(img_path).convert("RGB")
            t = processor(images=img, return_tensors="pt")["pixel_values"][0]
            return photo, t, None
        except Exception as exc:
            return photo, None, str(exc)

    bar = tqdm(total=len(pending), unit="img", dynamic_ncols=True) if show_progress else None
    total_preprocess_s = 0.0
    total_gpu_s = 0.0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Keep 2 batches of futures in flight at all times so preprocessing
        # of batch N+1 runs while the GPU processes batch N.
        i = 0
        pipeline: list[list] = []

        def _submit_next() -> None:
            nonlocal i
            if i < len(pending):
                batch = pending[i : i + _batch_size]
                i += _batch_size
                pipeline.append([executor.submit(_load_one, p) for p in batch])

        _submit_next()
        _submit_next()

        while pipeline:
            t0 = time.perf_counter()
            futures = pipeline.pop(0)
            _submit_next()  # keep pipeline full

            good: list[tuple[dict, torch.Tensor]] = []
            failed: list[dict] = []
            for future in futures:
                photo, tensor, err = future.result()
                if tensor is not None:
                    good.append((photo, tensor))
                else:
                    failed.append(photo)
                    if err:
                        print(f"\n[semantic_search] load failed: {err}")
            total_preprocess_s += time.perf_counter() - t0

            for photo in failed:
                if after_each:
                    after_each()
            if bar and failed:
                bar.update(len(failed))

            if good:
                photos_b, tensors_b = zip(*good)
                pixel_values = torch.stack(list(tensors_b)).pin_memory().to(device, non_blocking=True)
                t1 = time.perf_counter()
                try:
                    with torch.no_grad(), _autocast():
                        feats = model.get_image_features(pixel_values=pixel_values)
                    if not isinstance(feats, torch.Tensor):
                        feats = feats.pooler_output
                    for photo, emb in zip(photos_b, feats.cpu().float().numpy()):
                        store.add(str(photo["id"]), emb)
                except Exception as exc:
                    print(f"\n[semantic_search] batch forward failed: {exc}")
                total_gpu_s += time.perf_counter() - t1

                for _ in good:
                    if after_each:
                        after_each()
                if bar:
                    bar.update(len(good))

                store.save()

    if bar:
        bar.close()
    if show_progress:
        print(f"preprocess: {total_preprocess_s:.1f}s  gpu forward: {total_gpu_s:.1f}s")


def run_semantic_index_sync(
    model_id: str,
    photos: list[dict],
    photos_root: Path,
    index_dir: Path,
    *,
    checkpoint_every: int = 20,
    num_workers: int = 4,
) -> int:
    """Index photos not yet in the FAISS store. Requires CUDA. Returns how many were pending."""
    if model_id not in MODELS:
        raise ValueError(f"Unknown model_id: {model_id!r}")
    model_info = MODELS[model_id]
    store = FaissStore(index_dir, model_id, model_info["embedding_dim"])
    already = store.indexed_ids()
    pending = [p for p in photos if str(p.get("id", "")) not in already]
    if not pending:
        return 0
    _index_pending_into_store(
        model_info,
        store,
        pending,
        photos_root,
        checkpoint_every=checkpoint_every,
        show_progress=True,
        num_workers=num_workers,
    )
    return len(pending)


# ---------------------------------------------------------------------------
# Indexing manager
# ---------------------------------------------------------------------------

class _Status:
    __slots__ = ("running", "total", "done", "error")

    def __init__(self) -> None:
        self.running = False
        self.total = 0
        self.done = 0
        self.error = ""

    @property
    def state(self) -> str:
        if self.error:
            return "error"
        if self.running:
            return "indexing"
        return "idle"


class IndexingManager:
    """
    Manages one background indexing thread per model and the FAISS stores.
    All public methods are thread-safe.
    """

    def __init__(self, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._lock = threading.Lock()
        self._stores: dict[str, FaissStore] = {}
        self._statuses: dict[str, _Status] = {mid: _Status() for mid in MODELS}

    # ------------------------------------------------------------------
    def _store(self, model_id: str) -> FaissStore:
        """Return (and lazily create) the FaissStore for model_id."""
        if model_id not in self._stores:
            info = MODELS[model_id]
            self._stores[model_id] = FaissStore(
                self._index_dir, model_id, info["embedding_dim"]
            )
        return self._stores[model_id]

    # ------------------------------------------------------------------
    def model_status(self, model_id: str) -> dict:
        with self._lock:
            st = self._statuses[model_id]
            store = self._store(model_id)
            return {
                "model_id": model_id,
                "state": st.state,
                "running": st.running,
                "total": st.total,
                "done": st.done,
                "indexed_count": store.count,
                "error": st.error,
            }

    def all_model_statuses(self) -> dict[str, dict]:
        return {mid: self.model_status(mid) for mid in MODELS}

    # ------------------------------------------------------------------
    def start_indexing(
        self,
        model_id: str,
        photos: list[dict],
        photos_root: Path,
    ) -> bool:
        """Start a background thread that indexes unindexed photos.

        Returns False if the model is unknown or already indexing.
        """
        if model_id not in MODELS:
            return False

        with self._lock:
            st = self._statuses[model_id]
            if st.running:
                return False
            store = self._store(model_id)
            already = store.indexed_ids()
            pending = [p for p in photos if str(p.get("id", "")) not in already]
            st.running = True
            st.error = ""
            st.total = len(pending)
            st.done = 0

        def _run() -> None:
            model_info = MODELS[model_id]
            _store = self._store(model_id)
            try:

                def _tick() -> None:
                    with self._lock:
                        self._statuses[model_id].done += 1

                _index_pending_into_store(
                    model_info,
                    _store,
                    pending,
                    photos_root,
                    checkpoint_every=20,
                    after_each=_tick,
                )
            except Exception as exc:
                with self._lock:
                    self._statuses[model_id].error = str(exc)
            finally:
                with self._lock:
                    self._statuses[model_id].running = False

        threading.Thread(target=_run, daemon=True).start()
        return True

    # ------------------------------------------------------------------
    def search(
        self,
        model_id: str,
        query: str,
        k: int = 50,
    ) -> list[dict[str, Any]]:
        """Return [{photo_id, score}] sorted by score desc."""
        if model_id not in MODELS:
            return []
        model_info = MODELS[model_id]
        with self._lock:
            store = self._store(model_id)
        if store.count == 0:
            return []
        emb = embed_text(model_info, query)
        hits = store.search(emb, k=k)
        return [{"photo_id": pid, "score": score} for pid, score in hits]
