"""Configuration and checkpoint policy for the central experiment runner."""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import json

from .config import CONFIG, ExperimentConfig
from shared.config import SplitConfig

BASELINE_MODEL = "best_macro_f1.pt"
PRIMARY_MODEL = "best_corrected_macro_f1.pt"
BASELINE_SELECTION = {"method": "best", "split": "validation", "criterion": "macro_f1",
                      "label_reference": "original"}
PRIMARY_SELECTION = {**BASELINE_SELECTION, "label_reference": "teacher_corrected"}


def load_config(path: Path, run_dir: Path | None = None) -> ExperimentConfig:
    """Read current or historical configs without rewriting the source file."""
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SSR config must be a JSON object.")
    raw = {name: value for name, value in raw.items()
           if name not in {"label_wave", "pipeline", "quality_metrics"}}
    unknown = set(raw) - {field.name for field in fields(CONFIG)}
    if unknown:
        raise ValueError(f"Unknown SSR config sections: {sorted(unknown)}")
    path_fields = {"data": {"root", "image_root", "annotation_root"},
                   "runtime": {"output_root"},
                   "training": {"corrected_labels_path", "corrected_validation_labels_path",
                                "corrected_test_labels_path"}}
    tuple_fields = {"data": {"class_names", "mean", "std", "roi_scales"},
                    "augmentation": {"color_jitter"}}
    updates = {}
    for name, value in raw.items():
        if name == "seed":
            updates[name] = value
            continue
        if not isinstance(value, dict):
            raise ValueError(f"SSR section {name} must be an object.")
        value = dict(value)
        if name == "training":
            for legacy in ("save_best_ce", "save_best_total"):
                value.pop(legacy, None)
        if name == "runtime":
            value.pop("run_id", None)
            value["name"] = value.get("name") or "teacher"
        for key in path_fields.get(name, ()):
            if value.get(key) is not None:
                target = Path(value[key]).expanduser()
                value[key] = (target if target.is_absolute() else path.parent / target).resolve()
        for key in tuple_fields.get(name, ()):
            if key in value:
                value[key] = tuple(value[key])
        if name == "data":
            for key in ("train", "validation", "test"):
                if value.get(key) is not None:
                    value[key] = SplitConfig(**value[key])
        updates[name] = replace(getattr(CONFIG, name), **value)
    config = replace(CONFIG, **updates)
    if run_dir is not None:
        config = _directory(config, run_dir)
    config.validate()
    return config


def _directory(config: ExperimentConfig, run_dir: Path) -> ExperimentConfig:
    directory = Path(run_dir).expanduser().resolve()
    return replace(config, runtime=replace(config.runtime, output_root=directory.parent,
                                           name=directory.name))


def teacher_config(config: ExperimentConfig, run_dir: Path, *, epochs: int | None = None):
    return _directory(replace(config,
        structural_labels=replace(config.structural_labels, enabled=True),
        training=replace(config.training, method="ssr", ce_loss="ce",
            corrected_labels_path=None, corrected_validation_labels_path=None,
            corrected_test_labels_path=None, epochs=config.training.epochs if epochs is None else epochs)), run_dir)


def student_config(config: ExperimentConfig, run_dir: Path,
                   corrections: dict[str, Path] | None = None, *, epochs: int | None = None):
    if corrections is not None and (set(corrections) != {"train", "val", "test"}
                                     or not all(corrections.values())):
        raise ValueError("Corrected retraining requires Train, Validation and Test correction files.")
    labels = {name: Path(path).expanduser().resolve() for name, path in (corrections or {}).items()}
    return _directory(replace(config,
        data=replace(config.data, multi_roi=True),
        structural_labels=replace(config.structural_labels, enabled=False),
        training=replace(config.training, method="ce", ce_loss="wce",
            corrected_labels_path=labels.get("train"), corrected_validation_labels_path=labels.get("val"),
            corrected_test_labels_path=labels.get("test"), epochs=config.training.epochs if epochs is None else epochs)), run_dir)


def primary_checkpoint(run_dir: Path) -> tuple[str, dict]:
    training = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8")).get("training", {})
    if training.get("corrected_labels_path") is not None:
        if any(training.get(name) is None for name in
               ("corrected_validation_labels_path", "corrected_test_labels_path")):
            raise ValueError(f"Corrected student requires Validation and Test corrections: {run_dir}")
        return PRIMARY_MODEL, PRIMARY_SELECTION
    return BASELINE_MODEL, BASELINE_SELECTION


def validate_selection(selection, run_dir, *, label_reference="original"):
    selection = selection or {}
    if (selection.get("method") != "best" or selection.get("split") != "validation" or selection.get("criterion") != "macro_f1"
            or selection.get("label_reference", "original") != label_reference):
        raise ValueError(f"Expected {label_reference}-validation Macro F1 selection in {run_dir}.")


def selected_model(run_dir: Path, output_dir: Path) -> dict:
    filename, expected = primary_checkpoint(run_dir)
    rows = [json.loads(line) for line in (Path(output_dir) / "checkpoint_results.jsonl").read_text().splitlines()
            if line.strip()]
    row = next((item for item in rows if item.get("filename") == filename), {})
    if row.get("status") != "evaluated":
        raise RuntimeError(f"Missing {expected['label_reference']}-validation Macro F1 evaluation in {output_dir}.")
    validate_selection(row.get("selection"), run_dir, label_reference=expected["label_reference"])
    return {**row, "path": str(Path(run_dir) / filename), "selection": {**expected, **row["selection"]}}
