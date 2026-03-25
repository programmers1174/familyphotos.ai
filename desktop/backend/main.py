"""
FastAPI server for the Stage 2 / Stage 3 desktop app.
Stage 2 — reads photo metadata from JSON and serves files.
Stage 3 — semantic search via CLIP/SigLiP + FAISS vector store.
Run from repo root: uv run python desktop/backend/main.py
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import uvicorn

import semantic_search as ss

DEFAULT_PORT = int(os.environ.get("FAMILYPHOTOS_PORT", "8765"))

# FAISS indexes live next to main.py (desktop/backend/indexes/)
_INDEX_DIR = Path(__file__).resolve().parent / "indexes"
_indexing_manager = ss.IndexingManager(_INDEX_DIR)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_db_path() -> Path:
    override = os.environ.get("FAMILYPHOTOS_DB")
    if override:
        return Path(override).expanduser().resolve()
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


def _safe_file(photos_root: Path, relative_path: str) -> Path:
    p = Path(relative_path.replace("\\", "/"))
    if p.is_absolute():
        candidate = p.resolve()
    else:
        root = photos_root.resolve()
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=403, detail="Invalid path") from None
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return candidate


_HEIC_SUFFIXES = {".heic", ".heif"}


def _heic_as_jpeg_response(path: Path) -> Response:
    """Chromium/Electron cannot display HEIC in <img>; serve JPEG bytes instead."""
    from PIL import Image

    buf = BytesIO()
    with Image.open(path) as im:
        im.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="familyphotos.ai desktop API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PhotoItem(BaseModel):
    id: str
    relativePath: str
    url: str


class PhotoListResponse(BaseModel):
    photos: list[PhotoItem]
    semantic: bool = False


class ModelInfo(BaseModel):
    id: str
    name: str
    model_type: str
    description: str
    embedding_dim: int
    size_mb: int
    state: str          # idle | indexing | error
    running: bool
    total: int
    done: int
    indexed_count: int
    error: str


class ModelsResponse(BaseModel):
    device: str
    models: list[ModelInfo]


class IndexStartRequest(BaseModel):
    model_id: str


class IndexStatusResponse(BaseModel):
    model_id: str
    state: str
    running: bool
    total: int
    done: int
    indexed_count: int
    error: str


# ---------------------------------------------------------------------------
# Endpoints — health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Endpoints — photos
# ---------------------------------------------------------------------------

@app.get("/api/photos", response_model=PhotoListResponse)
def list_photos(
    q: str | None = Query(None, description="Search query. Semantic if model is provided."),
    model: str | None = Query(None, description="Model id for semantic search."),
) -> PhotoListResponse:
    db_path = _default_db_path()
    try:
        photos_root, entries = _load_db(db_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Invalid database: {e}") from e

    # Build a lookup by id for all valid photos
    valid: dict[str, PhotoItem] = {}
    for row in entries:
        if not isinstance(row, dict):
            continue
        pid = row.get("id")
        rel = row.get("relativePath") or row.get("file")
        if not pid or not rel:
            continue
        rel = str(rel).replace("\\", "/")
        try:
            _safe_file(photos_root, rel)
        except HTTPException:
            continue
        valid[str(pid)] = PhotoItem(
            id=str(pid),
            relativePath=rel,
            url=f"/api/photos/{pid}/file",
        )

    # Semantic search
    if q and q.strip() and model and model in ss.MODELS:
        hits = _indexing_manager.search(model, q.strip())
        ordered: list[PhotoItem] = []
        for hit in hits:
            item = valid.get(hit["photo_id"])
            if item:
                ordered.append(item)
        return PhotoListResponse(photos=ordered, semantic=True)

    # Default: return all photos
    return PhotoListResponse(photos=list(valid.values()), semantic=False)


@app.get("/api/photos/{photo_id}/file")
def photo_file(photo_id: str) -> FileResponse:
    db_path = _default_db_path()
    try:
        photos_root, entries = _load_db(db_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Invalid database: {e}") from e

    rel: str | None = None
    for row in entries:
        if isinstance(row, dict) and str(row.get("id")) == photo_id:
            rel = row.get("relativePath") or row.get("file")
            break
    if not rel:
        raise HTTPException(status_code=404, detail="Unknown photo id")
    rel = str(rel).replace("\\", "/")
    path = _safe_file(photos_root, rel)
    if path.suffix.lower() in _HEIC_SUFFIXES:
        try:
            return _heic_as_jpeg_response(path)
        except Exception as e:  # pragma: no cover - decode / missing codec
            raise HTTPException(
                status_code=500,
                detail=f"Could not decode HEIC image (install pillow-heif): {e}",
            ) from e
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Endpoints — semantic index
# ---------------------------------------------------------------------------

@app.get("/api/index/models", response_model=ModelsResponse)
def index_models() -> ModelsResponse:
    statuses = _indexing_manager.all_model_statuses()
    items: list[ModelInfo] = []
    for mid, info in ss.MODELS.items():
        st = statuses[mid]
        items.append(
            ModelInfo(
                id=mid,
                name=info["name"],
                model_type=info["model_type"],
                description=info["description"],
                embedding_dim=info["embedding_dim"],
                size_mb=info["size_mb"],
                state=st["state"],
                running=st["running"],
                total=st["total"],
                done=st["done"],
                indexed_count=st["indexed_count"],
                error=st["error"],
            )
        )
    return ModelsResponse(device=ss.device_label(), models=items)


@app.post("/api/index/start", response_model=IndexStatusResponse)
def index_start(body: IndexStartRequest) -> IndexStatusResponse:
    model_id = body.model_id
    if model_id not in ss.MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")

    db_path = _default_db_path()
    try:
        photos_root, entries = _load_db(db_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Invalid database: {e}") from e

    started = _indexing_manager.start_indexing(model_id, entries, photos_root)
    if not started:
        # Already running — return current status
        st = _indexing_manager.model_status(model_id)
        return IndexStatusResponse(**st)

    st = _indexing_manager.model_status(model_id)
    return IndexStatusResponse(**st)


@app.get("/api/index/status/{model_id}", response_model=IndexStatusResponse)
def index_status(model_id: str) -> IndexStatusResponse:
    if model_id not in ss.MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    st = _indexing_manager.model_status(model_id)
    return IndexStatusResponse(**st)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT)
