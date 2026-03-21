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
