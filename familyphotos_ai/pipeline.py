from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import EnrollConfig
from .deepface_embed import largest_face_index, represent_faces
from .matching import distance, mean_embedding


@dataclass(frozen=True)
class PersonReference:
    name: str
    prototype: np.ndarray
    samples: list[np.ndarray]


def build_references(cfg: EnrollConfig) -> list[PersonReference]:
    refs: list[PersonReference] = []

    for person, image_paths in cfg.people.items():
        embs: list[np.ndarray] = []
        for img_path in image_paths:
            faces = represent_faces(
                img_path,
                model_name=cfg.model.model_name,
                detector_backend=cfg.model.detector_backend,
                enforce_detection=True,
            )
            if len(faces) != 1:
                # Enrollment images are expected to have one face.
                # If multiple are detected anyway, pick the largest to keep running.
                i = largest_face_index(faces)
                embs.append(faces[i].embedding)
            else:
                embs.append(faces[0].embedding)

        proto = mean_embedding(embs)
        refs.append(PersonReference(name=person, prototype=proto, samples=embs))

    return refs


def scan_images(
    image_paths: list[str],
    *,
    cfg: EnrollConfig,
    references: list[PersonReference],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for img_path in image_paths:
        faces = represent_faces(
            img_path,
            model_name=cfg.model.model_name,
            detector_backend=cfg.model.detector_backend,
            enforce_detection=False,
        )

        matched: set[str] = set()
        for f in faces:
            for ref in references:
                d = distance(f.embedding, ref.prototype, cfg.model.distance_metric)
                if d <= cfg.matching.threshold:
                    matched.add(ref.name)

        out[str(Path(img_path).resolve())] = sorted(matched)
    return out


def save_references_json(path: str, cfg: EnrollConfig, refs: list[PersonReference]) -> None:
    payload = {
        "config": asdict(cfg),
        "people": {
            r.name: {
                "prototype": r.prototype.astype(float).tolist(),
                "samples": [s.astype(float).tolist() for s in r.samples],
            }
            for r in refs
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_references_json(path: str) -> tuple[EnrollConfig, list[PersonReference]]:
    from .config import EnrollConfig, MatchingConfig, ModelConfig

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cfg_raw = payload.get("config") or {}
    model_raw = cfg_raw.get("model") or {}
    matching_raw = cfg_raw.get("matching") or {}
    people_raw = payload.get("people") or {}

    cfg = EnrollConfig(
        model=ModelConfig(
            model_name=model_raw.get("model_name", "Facenet512"),
            detector_backend=model_raw.get("detector_backend", "retinaface"),
            distance_metric=model_raw.get("distance_metric", "cosine"),
        ),
        matching=MatchingConfig(threshold=float(matching_raw.get("threshold", 0.55))),
        people={k: [] for k in people_raw.keys()},
    )

    refs: list[PersonReference] = []
    for name, r in people_raw.items():
        proto = np.asarray(r["prototype"], dtype=np.float32)
        samples = [np.asarray(s, dtype=np.float32) for s in (r.get("samples") or [])]
        refs.append(PersonReference(name=name, prototype=proto, samples=samples))

    return cfg, refs

