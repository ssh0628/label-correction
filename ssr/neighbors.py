"""SSR relabelling, class-balanced k-NN selection and LSL structural labels."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import Sampler
import torch
import torch.nn.functional as F

# Exact k-NN still compares every query with the full bank. Only the number
# of query rows processed together is reduced for large datasets.
MAX_SIMILARITY_BYTES = 256 * 1024 * 1024


def similarity_chunk_size(
    query_count: int,
    bank_count: int,
    element_size: int,
    chunks: int,
) -> int:
    """Respect the requested chunk count and cap each similarity allocation."""
    requested_rows = max(1, (query_count + chunks - 1) // chunks)
    memory_rows = max(1, MAX_SIMILARITY_BYTES // (bank_count * element_size))
    return min(requested_rows, memory_rows)


@torch.no_grad()
def hard_knn_scores(
    queries: Tensor,
    feature_bank_transposed: Tensor,
    feature_labels: Tensor,
    num_classes: int,
    neighbors: int,
    *,
    chunks: int = 1,
) -> Tensor:
    """Uniformly vote over the cosine-nearest labels, including self matches."""
    if chunks < 1:
        raise ValueError("chunks must be positive.")
    scores = torch.zeros(
        queries.size(0),
        num_classes,
        device=feature_labels.device,
        dtype=queries.dtype,
    )
    chunk_size = similarity_chunk_size(
        len(queries), feature_bank_transposed.size(1), queries.element_size(), chunks
    )
    for start in range(0, len(queries), chunk_size):
        end = min(start + chunk_size, len(queries))
        similarity = torch.mm(queries[start:end], feature_bank_transposed)
        neighbor_indices = similarity.topk(k=neighbors, dim=-1).indices
        del similarity
        neighbor_labels = feature_labels[neighbor_indices]
        scores[start:end].scatter_add_(
            dim=1,
            index=neighbor_labels,
            src=torch.ones_like(neighbor_labels, dtype=scores.dtype),
        )
    return scores / neighbors


@torch.no_grad()
def balanced_knn_scores(
    current_features: Tensor,
    feature_bank: Tensor,
    labels: Tensor,
    *,
    num_classes: int = 10,
    neighbors: int = 200,
    chunks: int = 10,
) -> Tensor:
    """Reproduce official SSR global-prior-corrected k-NN scores."""
    if current_features.ndim != 2 or feature_bank.ndim != 2:
        raise ValueError("current_features and feature_bank must be rank-2 tensors.")
    if current_features.size(1) != feature_bank.size(1):
        raise ValueError("query and feature-bank dimensions must match.")
    if labels.numel() != feature_bank.size(0):
        raise ValueError("labels must contain one entry per feature-bank row.")
    if labels.ndim != 1:
        raise ValueError("labels must be a rank-1 tensor.")
    if not 1 <= neighbors <= feature_bank.size(0):
        raise ValueError("neighbors must be in [1, feature-bank size].")
    if chunks < 1:
        raise ValueError("chunks must be positive.")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError("labels contain an invalid class index.")

    counts = torch.bincount(labels, minlength=num_classes).to(feature_bank.dtype) + 1e-10
    class_prior = counts / counts.sum()
    normalized_bank = F.normalize(feature_bank, dim=1)
    normalized_queries = (
        normalized_bank
        if current_features is feature_bank
        else F.normalize(current_features, dim=1)
    )
    if len(normalized_queries) == 0:
        return torch.empty((0, num_classes), device=feature_bank.device)
    scores = hard_knn_scores(
        normalized_queries,
        normalized_bank.T,
        labels,
        num_classes,
        neighbors,
        chunks=chunks,
    )
    scores = scores / class_prior
    return scores / scores.sum(dim=1, keepdim=True)


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_indices: Tensor
    rejected_indices: Tensor
    modified_labels: Tensor
    # confidence > tau_r인 후보. 원래 label과 같은 prediction도 포함한다.
    relabelled_indices: Tensor
    # 실제로 label 값이 달라진 sample만 포함한다.
    changed_indices: Tensor
    confidences: Tensor
    consistency: Tensor


@torch.no_grad()
def relabel_from_predictions(
    noisy_labels: Tensor,
    prediction_probabilities: Tensor,
    threshold: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reset to noisy labels, then replace predictions strictly above theta_r."""
    confidences, predicted_labels = prediction_probabilities.max(dim=1)
    relabelled_indices = torch.where(confidences > threshold)[0]
    modified_labels = noisy_labels.clone().detach()
    modified_labels[relabelled_indices] = predicted_labels[relabelled_indices]
    return modified_labels, relabelled_indices, confidences


@torch.no_grad()
def select_samples(
    features: Tensor,
    noisy_labels: Tensor,
    prediction_probabilities: Tensor,
    *,
    relabel_threshold: float,
    selection_threshold: float,
    neighbors: int,
    chunks: int,
    num_classes: int = 10,
) -> SelectionResult:
    modified_labels, relabelled_indices, confidences = relabel_from_predictions(
        noisy_labels, prediction_probabilities, relabel_threshold
    )
    changed_indices = torch.where(modified_labels.ne(noisy_labels))[0]
    knn_scores = balanced_knn_scores(
        features,
        features,
        modified_labels,
        num_classes=num_classes,
        neighbors=neighbors,
        chunks=chunks,
    )
    label_scores = torch.gather(knn_scores, 1, modified_labels.view(-1, 1)).squeeze(1)
    maximum_scores = knn_scores.max(dim=1).values
    consistency = label_scores / maximum_scores
    selected_indices = torch.where(consistency >= selection_threshold)[0]
    rejected_indices = torch.where(consistency < selection_threshold)[0]
    return SelectionResult(
        selected_indices=selected_indices,
        rejected_indices=rejected_indices,
        modified_labels=modified_labels,
        relabelled_indices=relabelled_indices,
        changed_indices=changed_indices,
        confidences=confidences,
        consistency=consistency,
    )


@torch.no_grad()
def extract_structural_labels(
    features: Tensor,
    relabelled_labels: Tensor,
    *,
    neighbors: int = 20,
    chunks: int = 10,
    num_classes: int = 10,
) -> Tensor:
    """각 source label을 그 source의 k개 이웃에게 전파해 soft target 생성.

    일반 k-NN은 각 query가 자기 이웃들의 label을 가져온다. 이 함수는 반대로
    source i의 relabelled label을 ``topk(sim(f_i, F))``가 가리키는 sample들에
    누적한다. 마지막 행 정규화가 논문 Algorithm 1의 ``T / T.sum(1)``이다.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape (N, D).")
    if relabelled_labels.ndim != 1 or relabelled_labels.numel() != features.size(0):
        raise ValueError("relabelled_labels must have shape (N,).")
    if not 1 <= neighbors <= features.size(0):
        raise ValueError("neighbors must be in [1, N].")
    if chunks < 1:
        raise ValueError("chunks must be positive.")
    if relabelled_labels.numel() == 0:
        raise ValueError("Structural labels require at least one sample.")
    if int(relabelled_labels.min()) < 0 or int(relabelled_labels.max()) >= num_classes:
        raise ValueError("relabelled_labels contain an invalid class index.")

    normalized_features = F.normalize(features, dim=1)
    feature_bank = normalized_features.T
    edge_counts = torch.zeros(
        features.size(0),
        num_classes,
        dtype=normalized_features.dtype,
        device=features.device,
    )
    flat_edge_counts = edge_counts.view(-1)

    sample_count = features.size(0)
    chunk_size = similarity_chunk_size(
        sample_count, sample_count, normalized_features.element_size(), chunks
    )
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        similarity = torch.mm(normalized_features[start:end], feature_bank)
        local_rows = torch.arange(end - start, device=features.device)
        # Self is a valid nearest neighbor in Algorithm 1. Pinning it to the
        # maximum also guarantees every sample receives at least one label.
        similarity[local_rows, local_rows + start] = torch.inf
        neighbor_indices = similarity.topk(k=neighbors, dim=1).indices
        # Release this chunk before allocating the next query-bank matrix.
        del similarity
        source_labels = relabelled_labels[start:end, None].expand(-1, neighbors)
        flat_targets = (
            neighbor_indices.reshape(-1) * num_classes + source_labels.reshape(-1)
        )
        flat_edge_counts.scatter_add_(
            0,
            flat_targets,
            torch.ones_like(flat_targets, dtype=edge_counts.dtype),
        )

    return edge_counts / edge_counts.sum(dim=1, keepdim=True)


class ClassBalancedSampler(Sampler[int]):
    def __init__(self, labels: torch.Tensor, num_classes: int = 10) -> None:
        labels = torch.as_tensor(labels, dtype=torch.long, device="cpu").flatten()
        if labels.numel() == 0:
            raise ValueError("SSR selected no samples; a balanced loader cannot be constructed.")
        if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
            raise ValueError("labels contain an invalid class index.")

        class_counts = torch.bincount(labels, minlength=num_classes)
        maximum_count = int(class_counts.max().item())
        sampled_ids: list[torch.Tensor] = []
        for class_id, class_count_tensor in enumerate(class_counts):
            class_count = int(class_count_tensor.item())
            if class_count == 0:
                continue
            folds = (maximum_count + class_count - 1) // class_count
            class_ids = torch.where(labels == class_id)[0].repeat(folds)
            random_tail = torch.randperm(maximum_count)
            class_ids[-maximum_count:] = class_ids[-maximum_count:][random_tail]
            sampled_ids.append(class_ids[:maximum_count])
        self.ids = torch.cat(sampled_ids)

    def __iter__(self) -> Iterator[int]:
        permutation = torch.randperm(len(self.ids))
        return iter(self.ids[permutation].tolist())

    def __len__(self) -> int:
        return len(self.ids)


class SelectedSampleSampler(Sampler[int]):
    """Update parent-side selection while persistent workers keep a fixed dataset."""

    def __init__(self, selection: SelectionResult, num_classes: int) -> None:
        self.num_classes = num_classes
        self.update(selection)

    def update(self, selection: SelectionResult) -> None:
        indices = selection.selected_indices.detach().cpu()
        balanced = ClassBalancedSampler(
            selection.modified_labels[selection.selected_indices].detach().cpu(),
            num_classes=self.num_classes,
        )
        self.indices, self.balanced = indices, balanced

    def __iter__(self):
        indices = self.indices
        return (int(indices[position]) for position in self.balanced)

    def __len__(self) -> int:
        return len(self.balanced)
