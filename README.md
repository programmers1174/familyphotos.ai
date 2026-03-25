## familyphotos.ai

### Stage 1 (people identification)

First install deps (recommended, from repo root):

```bash
uv sync
```

Then run commands using the project environment:

```bash
uv run python -m familyphotos_ai.cli --help
```

1) Create an enrollment YAML (see `enroll.sample.yaml`).

2) Enroll reference embeddings:

```bash
uv run python -m familyphotos_ai.cli enroll --enroll-yaml enroll.yaml --out-refs artifacts/references.json
```

3) Scan a folder (recursive) and produce `{full_image_path: [people...]}`:

```bash
uv run python -m familyphotos_ai.cli scan --refs-json artifacts/references.json --images "C:/path/to/photos" --out artifacts/matches.json
```

Notes:
- Output keys are **full paths**.
- Default matching is **lenient (high recall)**; tune via YAML `matching.threshold` or `--threshold`.

### Stage 2 (desktop app)

Stack: **Electron** UI + **FastAPI** / **uvicorn** backend. Photo index is **`desktop/photos_db.json`** (see `desktop/photos_db.sample.json`). Image files live under the folder set by `photosRoot` in that JSON (paths are relative to the JSON file unless `photosRoot` is absolute).

1) Install Python deps from the repo root:

```bash
uv sync
```

2) Edit `desktop/photos_db.json`: set `photosRoot` if needed and add objects `{ "id": "unique-id", "relativePath": "file.jpg" }` for each image.

3) Run the API alone (optional):

```bash
uv run python desktop/backend/main.py
```

4) Run the desktop shell (from repo root, after installing Node dependencies once):

```bash
cd desktop/electron
npm install
npm start
```

Electron starts the backend automatically. The main window has a search bar (search behavior is still “list everything” for now), a grid of thumbnails, and **Esc** closes the full-size overlay.

### Stage 3 (semantic search — CUDA required)

CLIP / SigLIP embedding and FAISS indexing **run on NVIDIA GPU only**. Install a **CUDA-enabled** PyTorch build for your driver ([PyTorch get started](https://pytorch.org/get-started/locally/)). If `torch.cuda.is_available()` is false, the desktop **backend exits on startup** and the indexer CLI exits with code 1 — there is no CPU fallback.

**Index photos from the command line** (same `photos_db.json` and `desktop/backend/indexes/` store as the app):

```bash
uv run familyphotos-index-semantic --model clip-vit-base-patch32
```

Other models: `clip-vit-large-patch14`, `siglip-base-patch16-224`, `siglip-so400m-patch14-384`.

Optional flags:

- `--db` — path to `photos_db.json` (default: `desktop/photos_db.json`)
- `--index-dir` — FAISS output directory (default: `desktop/backend/indexes`)
