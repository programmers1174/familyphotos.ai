from __future__ import annotations

from typing import Literal

import numpy as np

DistanceMetric = Literal["cosine", "euclidean", "euclidean_l2"]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n


def distance(a: np.ndarray, b: np.ndarray, metric: DistanceMetric) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    if metric == "cosine":
        a_n = _l2_normalize(a)
        b_n = _l2_normalize(b)
        return float(1.0 - float(np.dot(a_n, b_n)))

    if metric == "euclidean":
        return float(np.linalg.norm(a - b))

    if metric == "euclidean_l2":
        a_n = _l2_normalize(a)
        b_n = _l2_normalize(b)
        return float(np.linalg.norm(a_n - b_n))

    raise ValueError(f"Unknown metric: {metric}")


def mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("No embeddings provided")
    stacked = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    return stacked.mean(axis=0)

