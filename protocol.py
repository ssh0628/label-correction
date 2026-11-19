"""Saved-run state and label provenance shared by the experiment commands."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from shared.artifacts import read_json, file_sha256 as sha256
from shared.data import correction_checkpoint

SPLITS = ("train", "val", "test")
CORRECTION_FILES = {
    "ssr": {split: f"{split}_correction.npz" for split in SPLITS},
    "co": {split: f"{split}_correction.npz" for split in SPLITS},
}
CORRECTION_MODELS = {"ssr": "best_train_total_loss.pt", "co": "correction_end.pt"}
EVALUATION_MODELS = {
    "baseline": "best_macro_f1.pt",
    "ssr/teacher": "best_macro_f1.pt", "ssr/student": "best_corrected_macro_f1.pt",
    "co/teacher": "last.pt", "co/student": "best_macro_f1.pt",
}


def checkpoint_name(method, role):
    return EVALUATION_MODELS["baseline" if method == "baseline" else f"{method}/{role}"]


def archive(directory, output):
    """Preserve a stage before explicitly starting afresh."""
    if directory.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        target = output.parent / ".archive" / "runs" / stamp / directory.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(directory), str(target))


def validate_student_selection(directory):
    summary = read_json(directory / "run_summary.json", {})
    if summary and summary.get("selection_label_reference") != "teacher_corrected":
        raise ValueError(f"{directory}: previous Student used original Val. Use --fresh to preserve it and restart.")


def validate_cached_student(method, directory, teacher, config=None):
    """Newly copied Teacher/label files must agree with a completed Student."""
    summary = read_json(directory / "run_summary.json", {})
    targets = {split: read_json(directory / name, {}) for split, name in
               (("train", "training_targets.json"), ("val", "validation_targets.json"), ("test", "test_targets.json"))}
    metadata = targets["train"].get("metadata", {})
    expected_teacher = (summary.get("correction_checkpoint_sha256") or correction_checkpoint(metadata)["sha256"])
    checkpoint = teacher / CORRECTION_MODELS[method]
    if expected_teacher and checkpoint.is_file() and sha256(checkpoint) != expected_teacher:
        raise ValueError("Student source Teacher changed; use --fresh to retrain this Student.")
    validate_student_selection(directory)
    expected = {split: target.get("artifact_sha256") for split, target in targets.items()}
    expected["train"] = summary.get("corrected_labels_sha256") or expected["train"]
    expected["val"] = (summary.get("corrected_validation_labels_sha256") or expected["val"]
                       or read_json(directory / "evaluation.json", {}).get("corrected_validation_labels_sha256"))
    evaluation = read_json(directory / "evaluation.json", {})
    reference = evaluation.get("corrected_references", {}).get(method, {})
    expected["test"] = reference.get("artifact_sha256") or expected["test"]
    if method == "co":
        from cocorrecting.artifacts import validate_holdout_metadata
        if config is None:
            from cocorrecting.config import load_config
            path = directory / "config.json"
            if not path.is_file():
                raise ValueError("Cached Co Student requires its correction configuration and "
                                 "frozen-teacher Validation provenance; use --fresh.")
            config = load_config(path)
        validation_metadata = targets["val"].get("metadata", {})
        validate_holdout_metadata(config, validation_metadata)
        validation_checkpoint = correction_checkpoint(validation_metadata)
        if (validation_metadata.get("split") != "val" or not expected_teacher
                or validation_checkpoint["sha256"] != expected_teacher
                or validation_checkpoint["completed_epochs"] != config.correction.correction_end_epoch
                or validation_checkpoint["selection"].get("criterion") != "end_of_label_updates"):
            raise ValueError("Cached Co Student Validation labels use a different Teacher; use --fresh.")
        if (not expected["val"] or not expected["test"]
                or not (teacher / "corrections" / CORRECTION_FILES[method]["test"]).is_file()):
            raise ValueError("Cached Co Student requires matching optimized Validation/Test labels; "
                             "legacy prediction references cannot be reused. Use --fresh.")
    names = CORRECTION_FILES[method]
    for split, name in names.items():
        path = teacher / "corrections" / name
        if not path.is_file():
            continue
        if expected[split] and sha256(path) != expected[split]:
            raise ValueError(f"Cached Student {split} labels changed; use --fresh to retrain this Student.")
        if expected_teacher or method == "co":
            import numpy as np
            with np.load(path, allow_pickle=False) as arrays:
                meta = json.loads(str(arrays["metadata"].item()))
            teacher_hash = correction_checkpoint(meta)["sha256"]
            if ((expected_teacher and teacher_hash != expected_teacher) or meta.get("split") != split):
                raise ValueError(f"Cached Student {split} labels use a different Teacher; use --fresh.")
            if method == "co" and split in {"val", "test"}:
                validate_holdout_metadata(config, meta)


def training_state(directory, *, corrected, resume=False):
    """Reject original-Val students before resuming or accepting their results."""
    if not directory.exists():
        return "new"
    status = read_json(directory / "status.json", {})
    if status.get("status") == "pending_corrected_validation" and set(p.name for p in directory.iterdir()) == {"status.json"}:
        return "pending"
    summary = read_json(directory / "run_summary.json", {})
    if corrected:
        validate_student_selection(directory)
    config = read_json(directory / "config.json", {})
    epochs = config.get("training", {}).get("epochs")
    completed = summary.get("training_completed_epochs", summary.get("completed_epochs", 0))
    if (epochs and completed is not None and completed >= epochs
            and summary.get("status") not in {"failed", "running", "interrupted"}):
        return "completed"
    if resume and (directory / "last.pt").is_file():
        return "resume"
    raise FileExistsError(f"{directory}: incomplete run or reports without checkpoints. "
                          "Copy saved models here, use --resume for Co, or --fresh for a new run.")


def shared_baseline_config(ssr_config, co_config):
    """Verify the common conditions before sharing one original-label baseline."""
    from ssr.pipeline import student_config as ssr_student
    from cocorrecting.config import student_config as co_student
    ssr, co = ssr_student(ssr_config, Path("output/baseline")), co_student(co_config)
    for name in ("seed", "data", "augmentation"):
        a, b = getattr(ssr, name), getattr(co, name)
        if a != b:
            raise ValueError(f"Shared baseline mismatch: {name}.")
    for name in ("name", "pretrained", "drop_path_rate"):
        if getattr(ssr.model, name) != getattr(co.model, name):
            raise ValueError(f"Shared baseline mismatch: model.{name}.")
    if co.model.pretrained_weights is not None:
        raise ValueError("Custom Co pretrained weights require a matching baseline configuration.")
    if ssr.runtime.deterministic != co.runtime.deterministic:
        raise ValueError("Shared baseline mismatch: deterministic runtime.")
    for name in ("epochs", "batch_size", "learning_rate", "encoder_learning_rate", "optimizer", "momentum", "weight_decay", "amp", "channels_last"):
        if getattr(ssr.training, name) != getattr(co.training, name):
            raise ValueError(f"Shared baseline mismatch: training.{name}.")
    if ssr.training.scheduler_eta_min_ratio != co.training.student_min_lr_ratio:
        raise ValueError("Shared baseline mismatch: cosine minimum learning rate.")


def correction_paths(method, directory):
    paths = {split: directory / name for split, name in CORRECTION_FILES[method].items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Train, Val and Test corrections are required: " + ", ".join(missing))
    return paths


def validate_export_sources(method, config, paths, checkpoint):
    """Keep copied native exports portable while rejecting mixed Teacher states."""
    import numpy as np
    identities = []
    for split, path in paths.items():
        with np.load(path, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"].item()))
        if metadata.get("split") != split:
            raise ValueError(f"Correction split mismatch: {path}")
        source = correction_checkpoint(metadata)
        identity = (source["sha256"], source["completed_epochs"])
        if method == "ssr":
            selection = source["selection"]
            if (selection.get("method") != "train_loss" or selection.get("split") != "train"
                    or selection.get("criterion") != "total_loss"):
                raise ValueError("SSR labels must come from the minimum Train total-loss teacher.")
            threshold = config.ssr.relabel_threshold
        else:
            selection = source["selection"]
            if (selection.get("criterion") != "end_of_label_updates"
                    or identity[1] != config.correction.correction_end_epoch):
                raise ValueError("Co labels must come from the correction-end teacher.")
            if split != "train":
                from cocorrecting.artifacts import validate_holdout_metadata
                validate_holdout_metadata(config, metadata)
        if (not isinstance(identity[0], str) or len(identity[0]) != 64
                or any(character not in "0123456789abcdef" for character in identity[0])
                or type(identity[1]) is not int or identity[1] < 1):
            raise ValueError(f"Missing Teacher identity in {path}")
        if method == "ssr" and metadata.get("correction_threshold") != threshold:
            raise ValueError(f"Correction threshold differs from the saved configuration: {path}")
        identities.append((identity, selection))
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Train, Validation and Test corrections must use the same Teacher checkpoint.")
    if checkpoint.is_file() and sha256(checkpoint) != identities[0][0][0]:
        raise ValueError("Saved corrections do not match the current Teacher checkpoint.")
