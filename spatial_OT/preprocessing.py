from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def prepare_feature_matrices(
    source_expression: np.ndarray,
    target_expression: np.ndarray,
    n_comps: Optional[int] = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    source_expression = np.asarray(source_expression, dtype=float)
    target_expression = np.asarray(target_expression, dtype=float)

    if n_comps is None:
        return source_expression, target_expression

    stacked = np.vstack([source_expression, target_expression])
    n_comps = int(min(n_comps, stacked.shape[1], stacked.shape[0]))
    if n_comps <= 0:
        return source_expression, target_expression

    pca = PCA(n_components=n_comps)
    embedded = pca.fit_transform(stacked)
    n_source = source_expression.shape[0]
    return embedded[:n_source], embedded[n_source:]


def build_spatial_graph(coords: np.ndarray, k: int = 10, labels: Optional[np.ndarray] = None) -> nx.Graph:
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 2:
        raise ValueError("Need at least 2 observations to build a kNN graph.")

    n_neighbors = min(k + 1, coords.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    graph = nx.Graph()
    for i in range(coords.shape[0]):
        attrs = {"spatial_position": tuple(coords[i, :2])}
        if labels is not None:
            attrs["cell_type"] = labels[i]
        graph.add_node(i, **attrs)

    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            graph.add_edge(i, int(neighbor))

    return graph


def compute_spatial_entropy_from_labels(graph: nx.Graph, label_attr: str = "cell_type") -> np.ndarray:
    entropies = np.zeros(graph.number_of_nodes(), dtype=float)
    for node in graph.nodes:
        neighbor_labels = [graph.nodes[n].get(label_attr) for n in graph.neighbors(node)]
        neighbor_labels = [label for label in neighbor_labels if label is not None]

        if not neighbor_labels:
            entropies[node] = 0.0
            continue

        counts = Counter(neighbor_labels)
        probabilities = np.array(list(counts.values()), dtype=float)
        probabilities /= probabilities.sum()
        entropies[node] = -np.sum(probabilities * np.log(probabilities + 1e-12))
    return entropies


def compute_spatial_entropy_from_distances(coords: np.ndarray, k: int = 10) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    n_neighbors = min(k + 1, coords.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    distances, _ = nbrs.kneighbors(coords)

    neighbor_distances = distances[:, 1:]
    inv_dist = 1.0 / (neighbor_distances + 1e-12)
    probs = inv_dist / inv_dist.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def compute_average_neighbor_expression(graph: nx.Graph, expression_matrix: np.ndarray) -> pd.DataFrame:
    expression_df = pd.DataFrame(np.asarray(expression_matrix, dtype=float))
    averages = []

    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            avg_expression = pd.Series(0.0, index=expression_df.columns)
        else:
            avg_expression = expression_df.iloc[neighbors].mean(axis=0)
        averages.append(avg_expression.values)

    return pd.DataFrame(averages, columns=expression_df.columns)
