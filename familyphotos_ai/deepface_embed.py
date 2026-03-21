from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from deepface import DeepFace


@dataclass(frozen=True)
class FaceEmbedding:
    embedding: np.ndarray  # shape (d,)
    facial_area: dict[str, Any] | None = None


def represent_faces(
    image_path: str,
    *,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool = True,
) -> list[FaceEmbedding]:
    reps = DeepFace.represent(
        img_path=image_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
    )

    # DeepFace returns either dict or list[dict] depending on version/settings.
    if isinstance(reps, dict):
        reps_list = [reps]
    else:
        reps_list = list(reps)

    out: list[FaceEmbedding] = []
    for r in reps_list:
        emb = np.asarray(r["embedding"], dtype=np.float32)
        out.append(FaceEmbedding(embedding=emb, facial_area=r.get("facial_area")))
    return out


def largest_face_index(faces: list[FaceEmbedding]) -> int:
    def area(f: FaceEmbedding) -> int:
        a = f.facial_area or {}
        w = int(a.get("w", 0) or 0)
        h = int(a.get("h", 0) or 0)
        return w * h

    best_i = 0
    best_area = -1
    for i, f in enumerate(faces):
        a = area(f)
        if a > best_area:
            best_area = a
            best_i = i
    return best_i

