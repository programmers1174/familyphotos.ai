"""
FastAPI server for the Stage 2 desktop app. Reads photo metadata from JSON and serves files.
Run from repo root: uv run python desktop/backend/main.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

DEFAULT_PORT = int(os.environ.get("FAMILYPHOTOS_PORT", "8765"))


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


app = FastAPI(title="familyphotos.ai desktop API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PhotoItem(BaseModel):
    id: str
    relativePath: str
    url: str


class PhotoListResponse(BaseModel):
    photos: list[PhotoItem]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/photos", response_model=PhotoListResponse)
def list_photos(
    q: str | None = Query(None, description="Ignored for now; returns all photos."),
) -> PhotoListResponse:
    _ = q  # reserved for future search
    db_path = _default_db_path()
    try:
        photos_root, entries = _load_db(db_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Invalid database: {e}") from e

    items: list[PhotoItem] = []
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
        items.append(
            PhotoItem(
                id=str(pid),
                relativePath=rel,
                url=f"/api/photos/{pid}/file",
            )
        )
    return PhotoListResponse(photos=items)


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
    return FileResponse(path)


if __name__ == "__main__":
    # Run app object directly so this file works without PYTHONPATH tricks
    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT)
