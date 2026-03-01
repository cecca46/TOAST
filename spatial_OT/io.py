from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class SliceData:
    expression: np.ndarray
    coords: np.ndarray
    labels: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None


def _to_numpy(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def load_csv_slice(
    csv_path: str,
    x_col: str,
    y_col: Optional[str] = None,
    *,
    label_col: str,
    feature_cols: Optional[Iterable[str]] = None,
) -> SliceData:
    df = pd.read_csv(csv_path)

    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in {csv_path}.")
    if y_col is not None and y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in {csv_path}.")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in {csv_path}.")

    if y_col is None:
        coords = np.column_stack([df[x_col].to_numpy(dtype=float), np.zeros(len(df), dtype=float)])
    else:
        coords = df[[x_col, y_col]].to_numpy(dtype=float)

    if feature_cols is None:
        excluded = {x_col}
        if y_col is not None:
            excluded.add(y_col)
        excluded.add(label_col)
        candidate_cols = [c for c in df.columns if c not in excluded]
        feature_cols = [
            c
            for c in candidate_cols
            if pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        feature_cols = list(feature_cols)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in {csv_path}: {missing}")

    if len(feature_cols) == 0:
        raise ValueError("No feature columns selected for CSV input.")

    expression = df[feature_cols].to_numpy(dtype=float)
    labels = df[label_col].to_numpy()
    return SliceData(expression=expression, coords=coords, labels=labels)


def load_anndata_slice(
    adata,
    spatial_key: str = "spatial",
    *,
    label_key: str,
    embedding_key: Optional[str] = None,
) -> SliceData:
    if spatial_key not in adata.obsm:
        raise ValueError(f"AnnData is missing adata.obsm['{spatial_key}'].")

    coords = _to_numpy(adata.obsm[spatial_key])
    if coords.shape[1] > 2:
        coords = coords[:, :2]
    if coords.shape[1] < 1:
        raise ValueError("Spatial coordinates must have at least one dimension.")
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0], dtype=float)])

    if label_key not in adata.obs:
        raise ValueError(f"AnnData is missing adata.obs['{label_key}'].")
    labels = adata.obs[label_key].to_numpy()

    features = None
    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            raise ValueError(f"AnnData is missing adata.obsm['{embedding_key}'].")
        features = _to_numpy(adata.obsm[embedding_key])

    expression = _to_numpy(adata.X)
    return SliceData(expression=expression, coords=coords, labels=labels, features=features)
