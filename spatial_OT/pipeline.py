from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
try:
    import ot
except ImportError:  # pragma: no cover - runtime dependency check
    ot = None

from .OT import compute_transport
from .costs import build_cost_matrices, normalize_cost_matrices
from .io import SliceData, load_anndata_slice, load_csv_slice
from .metrics import compute_accuracy_max_prob, compute_jsd_from_transport, mapping_accuracy
from .preprocessing import (
    build_spatial_graph,
    compute_average_neighbor_expression,
    compute_spatial_entropy_from_labels,
    prepare_feature_matrices,
)


@dataclass
class AlignmentResult:
    transport: np.ndarray
    cost_matrices: Dict[str, np.ndarray]
    metrics: Dict[str, float]


def align_slices(
    source: SliceData,
    target: SliceData,
    alpha: float = 0.5,
    epsilon: float = 0.1,
    k: int = 10,
    n_comps: Optional[int] = 50,
    use_spatial_terms: bool = True,
    tol: float = 1e-9,
    max_iter: int = 1000,
) -> AlignmentResult:
    if ot is None:
        raise ImportError("POT is required. Install dependencies with: pip install -r requirements.txt")

    if source.features is not None and target.features is not None:
        source_features = np.asarray(source.features, dtype=float)
        target_features = np.asarray(target.features, dtype=float)
    else:
        source_features, target_features = prepare_feature_matrices(
            source.expression,
            target.expression,
            n_comps=n_comps,
        )

    source_labels = source.labels
    target_labels = target.labels

    if source_labels is None or target_labels is None:
        raise ValueError("Labels are required for both source and target slices.")

    graph1 = build_spatial_graph(source.coords, k=k, labels=source_labels)
    graph2 = build_spatial_graph(target.coords, k=k, labels=target_labels)

    source_entropy = compute_spatial_entropy_from_labels(graph1)
    target_entropy = compute_spatial_entropy_from_labels(graph2)

    source_avg = compute_average_neighbor_expression(graph1, source_features).values
    target_avg = compute_average_neighbor_expression(graph2, target_features).values

    cost_matrices = build_cost_matrices(
        source_features=source_features,
        target_features=target_features,
        source_coords=source.coords,
        target_coords=target.coords,
        source_entropy=source_entropy,
        target_entropy=target_entropy,
        source_avg_expression=source_avg,
        target_avg_expression=target_avg,
    )

    cost_matrices = normalize_cost_matrices(cost_matrices)

    p = ot.unif(source_features.shape[0])
    q = ot.unif(target_features.shape[0])
    G0 = np.outer(p, q)

    if use_spatial_terms:
        transport = compute_transport(
            G0=G0,
            epsilon=epsilon,
            alpha=alpha,
            C1=cost_matrices["C1"],
            C2=cost_matrices["C2"],
            p=p,
            q=q,
            M=cost_matrices["M"],
            C3=cost_matrices["C3"],
            C4=cost_matrices["C4"],
            tol=tol,
            max_iter=max_iter,
        )
    else:
        transport = compute_transport(
            G0=G0,
            epsilon=epsilon,
            alpha=alpha,
            C1=cost_matrices["C1"],
            C2=cost_matrices["C2"],
            p=p,
            q=q,
            M=cost_matrices["M"],
            tol=tol,
            max_iter=max_iter,
        )

    metrics: Dict[str, float] = {}
    metrics["accuracy_max_prob"] = compute_accuracy_max_prob(transport, source_labels, target_labels)
    metrics["mapping_accuracy"] = mapping_accuracy(source_labels, target_labels, transport)
    metrics["JSD"] = compute_jsd_from_transport(
        transport_matrix=transport,
        source_coords=source.coords,
        target_coords=target.coords,
        source_labels=source_labels,
        k=k,
    )

    return AlignmentResult(transport=transport, cost_matrices=cost_matrices, metrics=metrics)


def align_csv(
    source_csv: str,
    target_csv: str,
    x_col: str,
    y_col: Optional[str] = None,
    feature_cols: Optional[list[str]] = None,
    *,
    label_col: str,
    **kwargs,
) -> AlignmentResult:
    source = load_csv_slice(
        csv_path=source_csv,
        x_col=x_col,
        y_col=y_col,
        label_col=label_col,
        feature_cols=feature_cols,
    )
    target = load_csv_slice(
        csv_path=target_csv,
        x_col=x_col,
        y_col=y_col,
        label_col=label_col,
        feature_cols=feature_cols,
    )
    return align_slices(source, target, **kwargs)


def align_anndata(
    source_adata,
    target_adata,
    spatial_key: str = "spatial",
    *,
    label_key: str,
    embedding_key: Optional[str] = None,
    gene_join: Literal["intersection", "none"] = "intersection",
    **kwargs,
) -> AlignmentResult:
    if embedding_key is None:
        if gene_join == "intersection":
            try:
                shared_genes = source_adata.var_names.intersection(target_adata.var_names, sort=False)
            except TypeError:
                shared_genes = source_adata.var_names.intersection(target_adata.var_names)

            if len(shared_genes) == 0:
                raise ValueError("No shared genes found between source and target AnnData objects.")

            source_adata = source_adata[:, shared_genes].copy()
            target_adata = target_adata[:, shared_genes].copy()
        elif gene_join == "none":
            if source_adata.shape[1] != target_adata.shape[1]:
                raise ValueError(
                    "Source and target AnnData have different numbers of genes. "
                    "Use gene_join='intersection' (recommended) or provide a shared embedding_key."
                )
        else:
            raise ValueError("gene_join must be one of: 'intersection', 'none'.")

    source = load_anndata_slice(
        source_adata,
        spatial_key=spatial_key,
        label_key=label_key,
        embedding_key=embedding_key,
    )
    target = load_anndata_slice(
        target_adata,
        spatial_key=spatial_key,
        label_key=label_key,
        embedding_key=embedding_key,
    )
    return align_slices(source, target, **kwargs)
