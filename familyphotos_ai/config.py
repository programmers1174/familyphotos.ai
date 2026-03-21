from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

try:
    import yaml  # type: ignore
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'pyyaml'. Install project dependencies (recommended: `uv sync`), "
        "or `pip install pyyaml` in your active environment."
    ) from e

DistanceMetric = Literal["cosine", "euclidean", "euclidean_l2"]


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "Facenet512"
    detector_backend: str = "retinaface"
    distance_metric: DistanceMetric = "cosine"


@dataclass(frozen=True)
class MatchingConfig:
    # Lenient by default (high recall); tune per dataset.
    threshold: float = 0.55


@dataclass(frozen=True)
class EnrollConfig:
    model: ModelConfig
    matching: MatchingConfig
    people: dict[str, list[str]]


def _require_mapping(x: Any, path: str) -> dict[str, Any]:
    if not isinstance(x, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(x).__name__}")
    return x


def load_enroll_config(yaml_path: str) -> EnrollConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = _require_mapping(raw, "root")

    model_raw = _require_mapping(raw.get("model", {}) or {}, "model")
    matching_raw = _require_mapping(raw.get("matching", {}) or {}, "matching")
    people_raw = _require_mapping(raw.get("people", {}) or {}, "people")

    model = ModelConfig(
        model_name=str(model_raw.get("model_name", ModelConfig.model_name)),
        detector_backend=str(model_raw.get("detector_backend", ModelConfig.detector_backend)),
        distance_metric=str(model_raw.get("distance_metric", ModelConfig.distance_metric)),  # type: ignore[arg-type]
    )
    if model.distance_metric not in ("cosine", "euclidean", "euclidean_l2"):
        raise ValueError(
            f"model.distance_metric must be one of cosine|euclidean|euclidean_l2, got {model.distance_metric!r}"
        )

    matching = MatchingConfig(threshold=float(matching_raw.get("threshold", MatchingConfig.threshold)))

    people: dict[str, list[str]] = {}
    for person, images in people_raw.items():
        if not isinstance(person, str) or not person.strip():
            raise ValueError("people keys must be non-empty strings")
        if not isinstance(images, list) or not images:
            raise ValueError(f"people.{person} must be a non-empty list of image paths")
        paths: list[str] = []
        for p in images:
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"people.{person} contains a non-string/empty image path")
            paths.append(p)
        people[person] = paths

    if not people:
        raise ValueError("people must not be empty")

    return EnrollConfig(model=model, matching=matching, people=people)

