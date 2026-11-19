"""Process-local runtime, deterministic epoch seeds and atomic checkpoints."""
from __future__ import annotations

import json
import hashlib
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from shared.runtime import close_loader


def comparable_config(payload: dict) -> dict:
    """Compare experiment settings independently of their output location."""
    result = {key: value for key, value in payload.items() if key != "runtime"}
    if isinstance(result.get("correction"), dict):
        result["correction"] = dict(result["correction"])
        # Older runs without this optional field used the shared RNG policy.
        result["correction"].setdefault("model_seeds", None)
        from .config import RETIRED_CORRECTION_FIELDS
        result["correction"] = {key: value for key, value in result["correction"].items()
                                if key not in RETIRED_CORRECTION_FIELDS}
    if isinstance(result.get("pipeline"), dict):
        from .config import RETIRED_PIPELINE_FIELDS
        # Heldout label optimization does not change model training. Corrected
        # retraining separately binds the exact Validation artifact by hash.
        result["pipeline"] = {key: value for key, value in result["pipeline"].items()
                              if key not in RETIRED_PIPELINE_FIELDS | {"holdout_correction_epochs"}}
    return result


class EpochSeededDataset(torch.utils.data.Dataset):
    """Persistent workers see epoch changes; augmentation also survives resume exactly."""
    def __init__(self, dataset, seed, *, shared=False):
        self.dataset, self.seed = dataset, seed
        self.epoch = torch.zeros((), dtype=torch.int64)
        if shared:
            self.epoch.share_memory_()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        seed = (self.seed + int(self.epoch) * 100003 + int(index) * 9176) % 2**32
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_forward_seed(model_seed: int, epoch: int, step: int) -> int:
    """Stable per-model forward stream, independent of other models' RNG use."""
    if any(type(value) is not int or value < 0 for value in (model_seed, epoch, step)):
        raise ValueError("Model seed, epoch and step must be nonnegative integers.")
    payload = f"teacher-forward-v1:{model_seed}:{epoch}:{step}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


@contextmanager
def isolated_model_rng(seed: int, device):
    """Isolate model randomness from data and peer models, including on resume.

    Only the active CUDA device is seeded and restored; unlike manual_seed_all,
    this cannot change untracked RNG streams on other accelerators. Python and
    NumPy states are included because a model constructor may use either.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("Model seed must be a nonnegative integer.")
    device = torch.device(device)
    cuda_devices = ([torch.cuda.current_device() if device.index is None else device.index]
                    if device.type == "cuda" else [])
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            # Python accepts arbitrary nonnegative integers; torch needs uint64.
            torch_seed = seed % 2**64
            torch.random.default_generator.manual_seed(torch_seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(torch_seed)
            random.seed(seed)
            np.random.seed(seed % 2**32)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def teacher_seed_metadata(model_seeds) -> dict:
    return {"teacher_model_seeds": None if model_seeds is None else list(model_seeds),
            "model_seed_policy": ("legacy_shared_rng" if model_seeds is None
                                  else "independent_initialization_and_forward_v1")}


def validate_teacher_seed_metadata(state, model_seeds) -> dict:
    """Accept missing seed metadata only for the original shared-RNG policy."""
    expected = teacher_seed_metadata(model_seeds)
    saved = {"teacher_model_seeds": state.get("teacher_model_seeds"),
             "model_seed_policy": state.get("model_seed_policy", "legacy_shared_rng")}
    config_seeds = state.get("config", {}).get("correction", {}).get("model_seeds")
    for value in (config_seeds, saved["teacher_model_seeds"]):
        if value is not None and (
                not isinstance(value, list) or len(value) != 2
                or any(type(seed) is not int or seed < 0 for seed in value)
                or value[0] == value[1]):
            raise ValueError("Teacher model seed provenance contains invalid model_seeds.")
    if saved != expected or config_seeds != expected["teacher_model_seeds"]:
        raise ValueError("Teacher model seed provenance mismatch. Use a new runtime.name "
                         "when changing model_seeds or the model RNG policy.")
    return expected


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def make_loader(dataset, config, device, *, training=False, epoch=0, batch_size=None):
    t = config.training
    workers = t.num_workers if training else t.eval_num_workers
    generator = torch.Generator().manual_seed(config.seed + epoch * 100003 + (17 if training else 29))
    kwargs = {}
    if workers:
        kwargs.update(multiprocessing_context="spawn", prefetch_factor=t.prefetch_factor,
                      persistent_workers=True)
    if training:
        dataset = EpochSeededDataset(dataset, config.seed, shared=bool(workers))
        dataset.epoch.fill_(epoch)
    # A new DataLoader iterator consumes its generator for the worker base
    # seed, but a persistent iterator's epoch reset does not. Keep shuffle RNG
    # separate so restarting workers cannot change training sample order.
    sampler = (RandomSampler(dataset, generator=torch.Generator().manual_seed(
        config.seed + epoch * 100003 + 17)) if training else None)
    return DataLoader(dataset, batch_size=batch_size or (t.batch_size if training else t.eval_batch_size),
                      sampler=sampler, shuffle=False, drop_last=False, num_workers=workers,
                      pin_memory=torch.device(device).type == "cuda", generator=generator,
                      worker_init_fn=seed_worker, **kwargs)


def set_loader_epoch(loader, config, epoch):
    loader.generator.manual_seed(config.seed + epoch * 100003 + 17)
    if isinstance(loader.sampler, RandomSampler):
        loader.sampler.generator.manual_seed(config.seed + epoch * 100003 + 17)
    if isinstance(loader.dataset, EpochSeededDataset):
        loader.dataset.epoch.fill_(epoch)


@contextmanager
def managed_loader(*args, **kwargs):
    loader = make_loader(*args, **kwargs)
    try:
        yield loader
    finally:
        close_loader(loader)


def start_run(run_dir, resume):
    """Open only this run's last.pt; fresh runs never overwrite an existing folder."""
    run_dir = Path(run_dir).resolve()
    if resume is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir, None
    if Path(resume).resolve() != run_dir / "last.pt" or not Path(resume).is_file():
        raise ValueError("Resume requires this run's last.pt checkpoint.")
    if not (run_dir / "config.json").is_file():
        raise FileNotFoundError(run_dir / "config.json")
    return run_dir, load_checkpoint(resume)


def load_checkpoint(path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def validate_checkpoint_history(run_dir, checkpoint) -> None:
    """Reject sidecars inconsistent with last.pt before any resume writes.

    Sidecars are saved before last.pt, so a crash can leave a newer best or
    stage checkpoint. Detect that window rather than silently mixing epochs;
    this check deliberately preserves every existing file for recovery.
    """
    run_dir = Path(run_dir)
    completed = checkpoint.get("completed_epochs")

    def inconsistent(path, reason):
        raise ValueError(f"Checkpoint history inconsistent: {path.name} {reason}. "
                         "Refusing to resume; existing files were preserved.")

    if type(completed) is not int or completed < 0:
        inconsistent(run_dir / "last.pt", "has invalid completed_epochs")

    def student_validation_provenance(state, path):
        reference = state.get("selection_label_reference", "original")
        validation_hash = state.get("corrected_validation_labels_sha256")
        if state.get("corrected_labels_sha256") is not None:
            if (reference != "teacher_corrected" or not isinstance(validation_hash, str)
                    or not validation_hash):
                inconsistent(path, "is missing corrected Validation selection provenance")
        elif reference != "original" or validation_hash is not None:
            inconsistent(path, "mixes original training with corrected Validation selection")
        recorded_reference = state.get("metrics", {}).get("validation_label_reference")
        if recorded_reference is not None and recorded_reference != reference:
            inconsistent(path, "has validation metrics from a different label reference")
        return reference, validation_hash

    student_validation = None
    if checkpoint.get("kind") == "student":
        student_validation = student_validation_provenance(checkpoint, run_dir / "last.pt")
    names = ("best_macro_f1.pt",) if student_validation is not None else ("correction_end.pt",)
    paths = [run_dir / name for name in names]
    for path in paths:
        if not path.is_file():
            continue
        sidecar = load_checkpoint(path)
        epoch = sidecar.get("completed_epochs")
        if type(epoch) is not int or not 0 <= epoch <= completed:
            inconsistent(path, f"has completed_epochs={epoch}, beyond last.pt epoch {completed}")
        if any(sidecar.get(key) != checkpoint.get(key)
               for key in ("kind", "dataset_fingerprint", "corrected_labels_sha256",
                           "correction_checkpoint_sha256")):
            inconsistent(path, "does not match last.pt run provenance")
        if student_validation is not None:
            if student_validation_provenance(sidecar, path) != student_validation:
                inconsistent(path, "does not match last.pt Validation labels or selection reference")
        if checkpoint.get("kind") == "teacher" and any(
                sidecar.get(key) != checkpoint.get(key)
                for key in ("model_class_weighting", "class_counts", "class_weights",
                            "class_weight_label_reference", "class_weight_reduction")):
            inconsistent(path, "does not match last.pt teacher class weighting")
        if checkpoint.get("kind") == "teacher":
            for state in (checkpoint, sidecar):
                try:
                    validate_teacher_seed_metadata(
                        state, checkpoint.get("config", {}).get("correction", {}).get("model_seeds"))
                except ValueError:
                    inconsistent(path, "does not match last.pt teacher model seed policy")
        selection = sidecar.get("selection", {})
        if path.name.startswith("best_"):
            criterion = path.stem.removeprefix("best_")
            if student_validation is not None:
                reference = selection.get("label_reference", "original")
                if reference != student_validation[0]:
                    inconsistent(path, "selects its best model using a different Validation label reference")
            expected = checkpoint.get("best_scores", {}).get(criterion)
            if (expected is None or selection.get("criterion") != criterion
                    or selection.get("score") != expected):
                inconsistent(path, f"does not match last.pt best_scores[{criterion!r}]")
            best_epoch = checkpoint.get("best_epochs", {}).get(criterion)
            if best_epoch is not None and epoch != best_epoch:
                inconsistent(path, f"does not match last.pt best_epochs[{criterion!r}]")
        elif path.name == "correction_end.pt":
            if (epoch != checkpoint.get("config", {}).get("correction", {}).get("correction_end_epoch")
                    or selection.get("criterion") != "end_of_label_updates"):
                inconsistent(path, "does not match the saved correction end epoch/selection")
        del sidecar


def reconcile_metrics(path, completed_epochs):
    """Discard rows newer than an explicitly resumed checkpoint; never duplicate epochs."""
    path = Path(path)
    if not path.exists():
        return
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["completed_epochs"] <= completed_epochs:
            rows.append(row)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    temporary.replace(path)


def configure_runtime(config, device):
    seed_everything(config.seed)
    torch.use_deterministic_algorithms(config.runtime.deterministic)
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = not config.runtime.deterministic
