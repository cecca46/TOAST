from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors


def compute_accuracy_max_prob(transport_matrix: np.ndarray, source_labels, target_labels) -> float:
    source_labels = np.asarray(source_labels)
    target_labels = np.asarray(target_labels)
    max_prob_indices = np.argmax(transport_matrix, axis=1)
    predicted_labels = target_labels[max_prob_indices]
    return float((source_labels == predicted_labels).mean())


def mapping_accuracy(labels1, labels2, transport_matrix: np.ndarray) -> float:
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)

    unique_labels = np.unique(np.concatenate([labels1, labels2]))
    label_map = {label: i for i, label in enumerate(unique_labels)}

    labels1_mapped = pd.Series(labels1).map(label_map).to_numpy()
    labels2_mapped = pd.Series(labels2).map(label_map).to_numpy()
    matches = labels1_mapped[:, None] == labels2_mapped[None, :]

    weighted_accuracy = np.sum(transport_matrix * matches)
    return float(weighted_accuracy / np.sum(transport_matrix))


def compute_transported_adata_argmax(adata_source, adata_target, transport_matrix: np.ndarray):
    max_indices = np.argmax(transport_matrix, axis=1)
    transported = adata_source.copy()
    transported.obsm["spatial"] = adata_target.obsm["spatial"][max_indices]
    return transported


def compute_local_cell_type_distribution(adata, k: int = 10, cell_type_key: str = "cell_type") -> np.ndarray:
    spatial_coords = adata.obsm["spatial"]
    cell_types = adata.obs[cell_type_key].astype("category").cat.codes.values
    unique_types = np.unique(cell_types)
    num_types = len(unique_types)

    knn = NearestNeighbors(n_neighbors=k).fit(spatial_coords)
    neighbors = knn.kneighbors(spatial_coords, return_distance=False)

    local_distributions = np.zeros((spatial_coords.shape[0], num_types))
    for i in range(spatial_coords.shape[0]):
        neighbor_types = cell_types[neighbors[i]]
        for j, cell_type in enumerate(unique_types):
            local_distributions[i, j] = np.sum(neighbor_types == cell_type) / k

    return local_distributions


def compute_js_divergence_before_after(
    adata_source,
    adata_transported,
    k: int = 5,
    cell_type_key: str = "cell_type",
) -> float:
    source_dist = compute_local_cell_type_distribution(adata_source, k=k, cell_type_key=cell_type_key)
    transported_dist = compute_local_cell_type_distribution(adata_transported, k=k, cell_type_key=cell_type_key)

    js_scores = [
        jensenshannon(source_dist[i], transported_dist[i])
        for i in range(source_dist.shape[0])
    ]
    return float(np.mean(js_scores))


def compute_jsd_from_transport(
    transport_matrix: np.ndarray,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    source_labels,
    k: int = 5,
) -> float:
    source_coords = np.asarray(source_coords, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)
    source_labels = np.asarray(source_labels)

    if source_coords.shape[0] != source_labels.shape[0]:
        raise ValueError("source_coords and source_labels must have the same number of rows.")

    if source_coords.shape[0] < 2:
        return 0.0

    mapped_target_idx = np.argmax(transport_matrix, axis=1)
    transported_coords = target_coords[mapped_target_idx]

    unique_labels = np.unique(source_labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    label_codes = np.array([label_to_index[label] for label in source_labels], dtype=int)
    n_types = len(unique_labels)

    n_neighbors = min(k, source_coords.shape[0])
    if n_neighbors < 1:
        return 0.0

    def _local_distributions(coords: np.ndarray) -> np.ndarray:
        knn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
        neighbors = knn.kneighbors(coords, return_distance=False)
        distributions = np.zeros((coords.shape[0], n_types), dtype=float)

        for i in range(coords.shape[0]):
            neighbor_codes = label_codes[neighbors[i]]
            counts = np.bincount(neighbor_codes, minlength=n_types).astype(float)
            distributions[i] = counts / counts.sum()

        return distributions

    before = _local_distributions(source_coords)
    after = _local_distributions(transported_coords)

    js_scores = [jensenshannon(before[i], after[i]) for i in range(before.shape[0])]
    return float(np.mean(js_scores))
