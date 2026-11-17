"""Clique label entropy on a fixed cosine similarity graph.

The paper leaves greedy candidate order and duplicate-clique handling unspecified.
Here each node is a seed, candidates are visited by ascending source index, and
identical node sets are retained only at first discovery. Overlapping distinct
cliques remain separate and receive equal weight. These choices are reported.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


def build_cliques(features, threshold: float = .74, min_clique_size: int = 11,
                  *, chunk_size: int = 256, max_edges: int = 20_000_000) -> dict:
    """Build a threshold graph in bounded dense chunks, then greedy cliques.

    ``max_edges`` bounds undirected edges, not the threshold: an excessively dense
    graph fails explicitly rather than silently altering the paper's graph.
    """
    values = np.asarray(features)
    if (values.ndim != 2 or not len(values) or not values.shape[1]
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()):
        raise ValueError('Features must be a nonempty finite numeric [N,D] array.')
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('Cosine threshold must lie in [-1,1].')
    if (type(min_clique_size) is not int or min_clique_size < 2
            or type(chunk_size) is not int or chunk_size < 1
            or type(max_edges) is not int or max_edges < 1):
        raise ValueError('Use clique size >=2 and positive integer chunk_size/max_edges.')
    # Scaling before normalization avoids overflowing the norm of finite inputs.
    values = values.astype(np.float64, copy=False)
    scale = np.abs(values).max(axis=1)
    if (scale == 0).any():
        raise ValueError('Zero feature vectors do not define cosine similarity.')
    values = values / scale[:, None]
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    n = len(values)
    rows, columns, edge_count = [], [], 0
    for start in tqdm(range(0, n, chunk_size), desc="Build similarity graph", unit="batch", mininterval=5):
        end = min(start + chunk_size, n)
        similarities = values[start:end] @ values.T
        # Compute each undirected edge once; self loops never qualify.
        for local, source in enumerate(range(start, end)):
            neighbors = np.flatnonzero(similarities[local, source + 1:] > threshold) + source + 1
            edge_count += len(neighbors)
            if edge_count > max_edges:
                raise ValueError(f'Cosine graph exceeds max_edges={max_edges}; '
                                 'raise the explicit memory limit or use a documented subset.')
            if len(neighbors):
                rows.append(np.full(len(neighbors), source, dtype=np.int32))
                columns.append(neighbors.astype(np.int32))
    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    column = np.concatenate(columns) if columns else np.empty(0, dtype=np.int32)
    graph = csr_matrix((np.ones(2 * edge_count, dtype=bool),
                        (np.r_[row, column], np.r_[column, row])), shape=(n, n))
    graph.sort_indices()
    cliques, seen, discoveries = [], set(), 0
    for seed in tqdm(range(n), desc="Find cliques", unit="img", mininterval=5):
        candidates = graph.indices[graph.indptr[seed]:graph.indptr[seed + 1]]
        clique = [seed]
        # Maintain intersection of neighbors: every accepted node connects to all
        # members, unlike a walk or a connected component.
        while len(candidates):
            candidate = int(candidates[0])
            clique.append(candidate)
            adjacent = graph.indices[graph.indptr[candidate]:graph.indptr[candidate + 1]]
            candidates = np.intersect1d(candidates[1:], adjacent, assume_unique=True)
        if len(clique) >= min_clique_size:
            discoveries += 1
            canonical = tuple(sorted(clique))
            if canonical not in seen:
                seen.add(canonical)
                cliques.append(list(canonical))
    covered = set(node for clique in cliques for node in clique)
    memberships = sum(map(len, cliques))
    return {'cliques': cliques, 'sample_count': n, 'threshold': float(threshold),
            'edge_comparison': 'cosine_similarity > threshold',
            'min_clique_size': min_clique_size, 'edge_count': edge_count,
            'clique_count': len(cliques), 'qualifying_seed_discoveries': discoveries,
            'duplicate_discoveries_removed': discoveries - len(cliques),
            'covered_sample_count': len(covered), 'coverage': len(covered) / n,
            'total_clique_memberships': memberships,
            'seed_order': 'ascending_source_index', 'candidate_order': 'ascending_source_index',
            'duplicate_policy': 'canonical_node_sets_first_discovery_only',
            'overlap_policy': 'distinct_overlapping_cliques_retained_equal_weight',
            'paper_unspecified_implementation_choices': ['greedy_candidate_order', 'duplicate_clique_policy']}


def score(labels, cliques, num_classes: int) -> dict:
    """Mean empirical hard-label entropy (bits) over selected cliques."""
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not len(labels) or labels.dtype.kind not in 'iu'
            or type(num_classes) is not int or num_classes < 2
            or (labels < 0).any() or (labels >= num_classes).any()):
        raise ValueError('Labels must be nonempty integer [N] values in [0,K), K>=2.')
    metadata = {k: v for k, v in cliques.items() if k != 'cliques'} if isinstance(cliques, Mapping) else {}
    selected = cliques['cliques'] if isinstance(cliques, Mapping) else cliques
    if metadata.get('sample_count', len(labels)) != len(labels):
        raise ValueError('Clique graph and label sample counts differ.')
    entropies, covered, memberships = [], set(), 0
    for clique in selected:
        nodes = np.asarray(clique)
        if (nodes.ndim != 1 or not len(nodes) or nodes.dtype.kind not in 'iu'
                or (nodes < 0).any() or (nodes >= len(labels)).any()
                or len(np.unique(nodes)) != len(nodes)):
            raise ValueError('Each clique must contain distinct valid integer source indices.')
        counts = np.bincount(labels[nodes], minlength=num_classes)
        probabilities = counts[counts > 0] / len(nodes)
        entropies.append(float(-(probabilities * np.log2(probabilities)).sum()))
        covered.update(nodes.tolist())
        memberships += len(nodes)
    return {'metric': 'clique_label_entropy', 'score': float(np.mean(entropies)) if entropies else None,
            'status': 'ok' if entropies else 'no_qualifying_cliques', 'higher_is_better': False,
            'units': 'bits', 'sample_count': len(labels), 'num_classes': num_classes,
            'clique_count': len(entropies), 'covered_sample_count': len(covered),
            'coverage': len(covered) / len(labels), 'total_clique_memberships': memberships,
            'clique_entropies': entropies, 'graph': metadata,
            'formula': 'mean_C(-sum_label p(label|C) * log2(p(label|C)))',
            'ground_truth_verified': False}
