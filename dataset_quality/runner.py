"""Assess pooled original/SSR/Co labels using one frozen feature space."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.artifacts import atomic_save_npz, read_json
from shared.augmentation import build_image_transforms
from shared.backbone import build_encoder
from shared.config import AugmentationConfig, data_config_from_dict
from shared.data import IndexedImageDataset, SplitSource, _corrected_targets, correction_checkpoint, prepare_dataset
from shared.precision import full_precision
from shared.roi import ROICrop

SPLITS = ("train", "val", "test")
METRICS = ("ch", "cle", "credibility")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _cached_array(path, metadata, key):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(saved["metadata"].item()) != metadata:
            raise ValueError(f"Cache input changed: {path}. Choose a new cache directory.")
        return saved[key].copy()


def _cache_array(path, metadata, key, values):
    atomic_save_npz(path, **{key: values, "metadata": np.asarray(json.dumps(metadata))})


@torch.inference_mode()
def extract_features(data, source, boxes, model_name, device, batch_size, workers):
    transform = build_image_transforms(data, AugmentationConfig()).evaluation
    dataset = IndexedImageDataset(source, transform, allow_truncated=data.allow_truncated_images,
        bboxes=boxes, roi=ROICrop(size=data.image_size, method=data.crop_method, random_view=False,
                                scales=data.roi_scales, shift_ratio=data.roi_shift_ratio))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda",
        **({"multiprocessing_context": "spawn", "prefetch_factor": 1} if workers else {}))
    encoder, dimensions = build_encoder(model_name, pretrained=True, drop_path_rate=0.0)
    encoder = encoder.to(device).eval().requires_grad_(False)
    features = torch.empty((len(source.paths), dimensions), dtype=torch.float32)
    seen = torch.zeros(len(source.paths), dtype=torch.bool)
    with full_precision(device):
        for images, indices in tqdm(loader, desc="Extract features", unit="batch", mininterval=5):
            if indices.unique().numel() != len(indices) or seen[indices].any():
                raise ValueError("Reference loader repeated an image.")
            values = encoder(images.to(device, dtype=torch.float32, non_blocking=True)).float().cpu()
            if values.shape != (len(indices), dimensions) or not torch.isfinite(values).all():
                raise ValueError("Reference encoder returned invalid features.")
            features[indices], seen[indices] = values, True
    if not seen.all():
        raise ValueError("Reference loader omitted images.")
    return features.numpy()


def _corrected_labels(data, sources, candidates):
    labels, provenance = {}, {}
    for name, files in candidates.items():
        if name == "original" or set(files) != set(SPLITS):
            raise ValueError("Each candidate must provide Train, Val and Test; original is reserved.")
        loaded = {split: _corrected_targets(data, sources[split], Path(files[split])) for split in SPLITS}
        identities = []
        for _, report in loaded.values():
            checkpoint = correction_checkpoint(report["metadata"])
            identities.append((checkpoint["sha256"], checkpoint["completed_epochs"]))
        digest, epoch = identities[0]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or type(epoch) is not int or epoch < 1 or any(pair != identities[0] for pair in identities)):
            raise ValueError(f"{name}: Train, Val and Test corrections must come from the same Teacher.")
        labels[name] = np.concatenate([loaded[split][0].numpy() for split in SPLITS])
        provenance[name] = {
            "teacher_identity_verified": True, "teacher_sha256": digest, "teacher_completed_epochs": epoch,
            "artifacts": {split: {k: v for k, v in loaded[split][1].items() if k != "metadata"}
                          for split in SPLITS},
        }
    return labels, provenance


def _scores(metric, features, label_sets, cache, options, device):
    feature_hash = hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
    if metric == "ch":
        from .ch import score
        return {name: score(features, labels) for name, labels in label_sets.items()}
    if metric == "cle":
        from .cle import build_cliques, score
        metadata = {"features_sha256": feature_hash, **{k: options[k] for k in
                    ("threshold", "min_clique_size", "max_edges")}}
        path = cache / "cliques.npz"
        graph = _cached_array(path, metadata, "graph")
        if graph is None:
            graph = build_cliques(features, **{k: v for k, v in options.items() if k != "num_classes"})
            _cache_array(path, metadata, "graph", np.asarray(json.dumps(graph)))
        else:
            graph = json.loads(graph.item())
        return {name: score(labels, graph, options["num_classes"]) for name, labels in label_sets.items()}
    from .credibility import estimate_credibility, nearest_neighbor_triplets
    metadata = {"features_sha256": feature_hash, "neighbors": 2, "similarity": "cosine", "exclude_self": True}
    path = cache / "neighbor_triplets.npz"
    triplets = _cached_array(path, metadata, "triplets")
    if triplets is None:
        triplets = nearest_neighbor_triplets(features, chunk_size=options["chunk_size"], device=device).numpy()
        _cache_array(path, metadata, "triplets", triplets)
    return {name: estimate_credibility(labels, triplets, options["num_classes"],
                    seed=options["seed"], restarts=options["restarts"], steps=options["fit_steps"])
            for name, labels in label_sets.items()}


def run(config_path, candidates, *, cache_dir, metrics=METRICS, device="auto",
        reference_model=None, batch_size=128, num_workers=0, chunk_size=256,
        threshold=.74, min_clique_size=11, max_edges=20_000_000, seed=0, restarts=5, fit_steps=1000):
    """Prepare all three splits once, then compare labels with shared features and neighbors."""
    metrics = tuple(metrics)
    if not metrics or not set(metrics) <= set(METRICS):
        raise ValueError(f"Choose metrics from {METRICS}.")
    if batch_size < 1 or num_workers < 0 or chunk_size < 1 or restarts < 1 or fit_steps < 1 or seed < 0:
        raise ValueError("Use positive sizes/counts and nonnegative worker count/seed.")
    config_path, cache = Path(config_path).resolve(), Path(cache_dir)
    saved = read_json(config_path)
    data = data_config_from_dict(saved["data"], base_dir=config_path.parent)
    sources, bboxes = prepare_dataset(data)
    if set(sources) != set(SPLITS):
        raise ValueError("Quality assessment requires Train, Val and Test datasets.")
    source = SplitSource("all", tuple(p for split in SPLITS for p in sources[split].paths),
                        torch.cat([sources[split].labels for split in SPLITS]),
                        tuple(p for split in SPLITS for p in sources[split].raw_paths))
    boxes = tuple(box for split in SPLITS for box in bboxes[split]) if bboxes else None
    original, classes = source.labels.numpy(), data.num_classes
    corrected, provenance = _corrected_labels(data, sources, candidates)
    label_sets = {"original": original, **corrected}
    boundaries, offset = [], 0
    for split in SPLITS:
        count = len(sources[split].paths)
        boundaries.append({"split": split, "start": offset, "stop": offset + count,
                           "sample_count": count, "paths_sha256": _hash(sources[split].paths)})
        offset += count
    model_name = reference_model or saved.get("model", {}).get("name", "convnextv2_tiny")
    reference = json.loads(json.dumps({
        "input": "dataset_config", "data": asdict(data), "paths_sha256": _hash(source.paths),
        "boxes_sha256": _hash(boxes), "teacher_weights_used": False,
        "pooling": "train_val_test_single_feature_space", "split_order": list(SPLITS),
        "split_boundaries": boundaries, "cross_split_neighbors_allowed": True,
        "original_labels_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
        "reference_model": model_name, "encoder_frozen": True, "view": "fixed_center",
    }, default=str))
    target = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    features = _cached_array(cache / "reference_features.npz", reference, "features")
    if features is None:
        features = extract_features(data, source, boxes, model_name, target, batch_size, num_workers)
        _cache_array(cache / "reference_features.npz", reference, "features", features)
    if (features.ndim != 2 or features.shape[0] != len(original) or features.shape[1] < 1
            or features.dtype.kind not in "fiu" or not np.isfinite(features).all()):
        raise ValueError("Features must be a finite [N,D] matrix aligned with labels.")
    reference.update(candidate_sources=provenance,
                     features_sha256=hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest())
    parameters = {
        "ch": {},
        "cle": {"threshold": threshold, "min_clique_size": min_clique_size, "max_edges": max_edges,
                "chunk_size": chunk_size, "num_classes": classes},
        "credibility": {"chunk_size": chunk_size, "seed": seed, "restarts": restarts,
                        "fit_steps": fit_steps, "num_classes": classes},
    }
    counts = {name: {"changed_count": int(np.count_nonzero(labels != original)),
                     "configured_num_classes": classes,
                     "configured_class_counts": np.bincount(labels, minlength=classes).tolist()}
              for name, labels in label_sets.items()}
    reports = {}
    for metric in metrics:
        scores = _scores(metric, features, label_sets, cache, parameters[metric], target)
        reports[metric] = {
            "metric": metric, "status": "completed", "sample_count": len(original),
            "class_names": list(data.class_names), "reference": reference,
            "parameters": {"split": "all", "reference_model": model_name, "device": str(target),
                           "batch_size": batch_size, "num_workers": num_workers, **parameters[metric]},
            "ground_truth_verified": False, "model_training_performed": False,
            "results": {name: {**values, **counts[name]} for name, values in scores.items()},
        }
        for name, score in scores.items():
            print(f"{metric} {name}={score.get('score')}", flush=True)
    return reports
