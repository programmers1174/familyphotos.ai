"""
Stage 3: Semantic search using CLIP / SigLiP models and a FAISS vector store.

Architecture
------------
* MODELS registry  — describes every supported embedding model.
* FaissStore       — wraps a FAISS IndexFlatIP with a photo-id mapping; saved to disk.
* embed_image / embed_text — run a model on GPU (or CPU) and return an L2-normalised
  numpy vector.
* IndexingManager  — background-threads indexing; exposes status and search.
"""
from __future__ import annotations

import json
import threading
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


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_label() -> str:
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
        entry = (model, processor, device, mtype)
        _loaded_models[mid] = entry
        return entry


def embed_image(model_info: dict[str, Any], image_path: Path) -> np.ndarray:
    from PIL import Image
    model, processor, device, _ = _load_model(model_info)
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
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
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return feats.cpu().float().numpy()[0]


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
                for i, photo in enumerate(pending):
                    rel = str(photo.get("relativePath") or photo.get("file") or "")
                    rel_p = Path(rel.replace("\\", "/"))
                    img_path = rel_p if rel_p.is_absolute() else photos_root / rel_p
                    if not img_path.is_file():
                        with self._lock:
                            self._statuses[model_id].done += 1
                        continue
                    try:
                        emb = embed_image(model_info, img_path)
                        _store.add(str(photo["id"]), emb)
                    except Exception as exc:
                        print(f"[semantic_search] embed failed for {img_path}: {exc}")
                    with self._lock:
                        self._statuses[model_id].done += 1
                    # Checkpoint every 20 images
                    if (i + 1) % 20 == 0:
                        _store.save()
                _store.save()
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
