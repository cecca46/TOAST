from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial import distance


def build_cost_matrices(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    source_entropy: np.ndarray,
    target_entropy: np.ndarray,
    source_avg_expression: np.ndarray,
    target_avg_expression: np.ndarray,
) -> Dict[str, np.ndarray]:
    matrices = {
        "M": distance.cdist(source_features, target_features).astype(float),
        "C1": distance.cdist(source_coords, source_coords).astype(float),
        "C2": distance.cdist(target_coords, target_coords).astype(float),
        "C3": np.abs(source_entropy[:, np.newaxis] - target_entropy[np.newaxis, :]).astype(float),
        "C4": distance.cdist(source_avg_expression, target_avg_expression).astype(float),
    }
    return matrices


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    max_val = matrix.max() if matrix.size else 0.0
    if max_val > 0:
        return matrix / max_val
    return matrix


def normalize_cost_matrices(matrices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: normalize_matrix(mat) for name, mat in matrices.items()}
