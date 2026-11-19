"""Run a shared baseline, SSR/Co teachers, corrected-label students and label quality."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from shared.artifacts import atomic_write_json, read_json, file_sha256 as sha256
from protocol import (CORRECTION_MODELS, EVALUATION_MODELS, archive, checkpoint_name,
                      correction_paths, shared_baseline_config, training_state,
                      validate_cached_student, validate_export_sources, validate_student_selection)

ROOT = Path(__file__).resolve().parent


def load_config(method, path, device):
    if method == "co":
        from cocorrecting.config import load_config as load
    else:
        from ssr.pipeline import load_config as load
    config = load(path)
    config = replace(config, runtime=replace(config.runtime, device=device))
    if method == "co":
        config = replace(config, pipeline=replace(config.pipeline, evaluate_27_views=True))
    return config


def stage_config(args, method, saved):
    template = args.co_config if method == "co" else args.ssr_config
    config = load_config(method, saved if saved.is_file() else template, args.device)
    if method == "co" and saved.is_file():
        requested = load_config(method, template, args.device)
        config = replace(config, pipeline=replace(config.pipeline,
            holdout_correction_epochs=requested.pipeline.holdout_correction_epochs))
    return config


def export_labels(method, config, teacher):
    output = teacher / "corrections"
    try:
        paths = correction_paths(method, output)
    except FileNotFoundError:
        paths = None
    checkpoint = teacher / CORRECTION_MODELS[method]
    if paths is not None:
        validate_export_sources(method, config, paths, checkpoint)
        return paths
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing correction files and source weights: {checkpoint}. "
                                "Copy the saved teacher checkpoint here; no training was started.")
    # A report alone is not a complete label export. Preserve incomplete exports
    # before generating all three artifacts at their permanent destination.
    archive(output, teacher.parents[1])
    if method == "ssr":
        from ssr.correction import export_corrections
        export_corrections(config, checkpoint, output, criterion="train_total")
    else:
        from cocorrecting.correction import export_corrections
        export_corrections(config, checkpoint, output)
    paths = correction_paths(method, output)
    validate_export_sources(method, config, paths, checkpoint)
    return paths


def evaluate(method, role, directory, config, corrections, *, force=False):
    if role == "student":
        validate_student_selection(directory)
        if method == "co":
            from cocorrecting.artifacts import validate_holdout_metadata
            validation = read_json(directory / "validation_targets.json", {})
            validate_holdout_metadata(config, validation.get("metadata", {}))
            if (corrections is not None
                    and validation.get("artifact_sha256") != sha256(corrections["val"])):
                raise ValueError("Validation correction changed since model selection; "
                                 "reselect or retrain the model with the current labels.")
    saved = read_json(directory / "evaluation.json", {})
    expected = "teacher_corrected" if role == "student" else "original"
    if not force and saved.get("status") == "evaluated" and saved.get("primary_metrics") is not None:
        if saved.get("primary_label_reference") != expected:
            raise ValueError(f"{directory}: evaluation labels do not match {expected}.")
        if role == "student" and saved.get("selection_label_reference") != "teacher_corrected":
            raise ValueError(f"{directory}: cached Student was selected with original Validation labels.")
        source = directory / saved.get("filename", checkpoint_name(method, role))
        if saved.get("checkpoint_sha256") and source.is_file() and sha256(source) != saved["checkpoint_sha256"]:
            raise ValueError("Cached evaluation checkpoint changed; use --evaluate-only.")
        prediction = Path(saved.get("predictions_path", directory / "evaluation_predictions.npz"))
        if saved.get("predictions_sha256") and prediction.is_file() and sha256(prediction) != saved["predictions_sha256"]:
            raise ValueError("Cached evaluation predictions changed; use --evaluate-only.")
        return saved
    checkpoint = directory / checkpoint_name(method, role)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing evaluation weights: {checkpoint}. "
                                "Copy the saved checkpoint here; no new training was started.")
    references = {method: corrections["test"]} if corrections else {}
    with TemporaryDirectory(prefix="label_evaluation_") as temporary:
        output = Path(temporary) / "evaluation"
        if method in {"ssr", "baseline"}:
            from ssr.evaluation import run
            run(directory, output_dir=output, correction_paths=references, device=config.runtime.device,
                checkpoint_path=checkpoint)
            report = json.loads((output / "checkpoint_results.jsonl").read_text().splitlines()[0])
            report.update(primary_metrics=report["primary_test_metrics"],
                          selection_label_reference=report["primary_label_reference"],
                          aggregation="27_view_logit_avg")
        else:
            from cocorrecting.evaluation import evaluate_checkpoint
            report = evaluate_checkpoint(config, checkpoint, output, correction_paths=references)
        report.update(status="evaluated", role=f"{method}_{role}")
        if report.get("primary_label_reference") != expected:
            raise ValueError(f"{directory}: expected {expected} Test evaluation.")
        for source_name, target_name in (("predictions.npz", "evaluation_predictions.npz"),
                                         ("config.json", "evaluation_config.json")):
            source = output / source_name
            if source.is_file():
                destination = directory / target_name
                shutil.copy2(source, destination)
                if source_name == "predictions.npz":
                    report["predictions_path"] = str(destination.resolve())
        atomic_write_json(directory / "evaluation.json", report, overwrite=True)
    return report


def stage(args, method, role):
    """Follow the same prepare → train/reuse → evaluate sequence for every model."""
    teacher = args.output / method / "teacher"
    if role == "correction":
        config = stage_config(args, method, teacher / "config.json")
        if args.fresh:
            checkpoint = teacher / CORRECTION_MODELS[method]
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing correction weights: {checkpoint}.")
            archive(teacher / "corrections", args.output)
        return export_labels(method, config, teacher)
    directory = args.output / ("baseline" if method == "baseline" else f"{method}/{role}")
    corrected = role == "student"
    if args.fresh:
        archive(directory, args.output)
    if corrected:
        validate_student_selection(directory)
    state = None if args.evaluate_only else training_state(directory, corrected=corrected,
                                                           resume=args.resume and method == "co")
    source = (teacher if corrected else directory) / "config.json"
    base = stage_config(args, method, source)
    saved = read_json(directory / "evaluation.json", {})
    if state == "completed" and saved.get("status") == "evaluated" and saved.get("primary_metrics") is not None:
        if corrected:
            validate_cached_student(method, directory, teacher, base)
        return evaluate(method, role, directory, base, None)
    corrections = export_labels(method, base, teacher) if corrected else None
    if method == "co":
        from cocorrecting.config import student_config
        config = student_config(base) if corrected else base
    else:
        from ssr.pipeline import student_config, teacher_config
        config = teacher_config(base, directory) if role == "teacher" else student_config(base, directory, corrections)
    trained = not args.evaluate_only and state != "completed"
    if trained:
        if state == "pending":
            archive(directory, args.output)
        print(f"{method} {role}: training", flush=True)
        if method != "co":
            from ssr.engine import run
            run(config)
        else:
            resume = directory / "last.pt" if state == "resume" else None
            if role == "teacher":
                from cocorrecting.engine import train_teacher
                train_teacher(config, directory, resume=resume)
            else:
                from cocorrecting.student import train_student
                train_student(config, directory, corrected_labels_path=corrections["train"],
                              corrected_validation_labels_path=corrections["val"], resume=resume)
    result = evaluate(method, role, directory, config, corrections, force=trained or args.evaluate_only)
    (directory / "status.json").unlink(missing_ok=True)
    return result


def quality(args):
    from dataset_quality.runner import run
    candidates = {}
    for method in ("ssr", "co"):
        teacher = args.output / method / "teacher"
        base = stage_config(args, method, teacher / "config.json")
        candidates[method] = export_labels(method, base, teacher)
    reports = run(args.ssr_config, candidates, cache_dir=args.output.parent / ".cache/label_quality",
                  metrics=("ch", "cle", "credibility") if args.metric == "all" else (args.metric,), device=args.device)
    for name, destination in (("original", args.output / "baseline"),
                              ("ssr", args.output / "ssr/teacher"), ("co", args.output / "co/teacher")):
        path = destination / "label_quality.json"
        metrics = read_json(path, {}).get("metrics", {})
        metrics.update({metric: {"result": report["results"][name], "reference": report["reference"],
                                 "parameters": report["parameters"]} for metric, report in reports.items()})
        atomic_write_json(path, {"label_set": name, "scope": "pooled_train_val_test",
                          "ground_truth_verified": False, "metrics": metrics}, overwrite=True)


def inventory(output):
    for name in EVALUATION_MODELS:
        directory = output / name
        report = read_json(directory / "evaluation.json", read_json(directory / "status.json", {}))
        metrics = report.get("primary_metrics") or {}
        weights = sorted(path.name for path in directory.glob("*.pt"))
        required = (CORRECTION_MODELS[name.split("/")[0]],) if name.endswith("teacher") else ()
        missing = [filename for filename in required if not (directory / filename).is_file()]
        if name.endswith("teacher"):
            try:
                correction_paths(name.split("/")[0], directory / "corrections")
                label_status = "complete"
            except FileNotFoundError:
                label_status = "missing_files"
        else:
            label_status = None
        print(json.dumps({"stage": name, "status": report.get("status", "not_run"),
                          "test_macro_f1": metrics.get("macro_f1"),
                          "label_reference": report.get("primary_label_reference"),
                          "checkpoints": weights, "weights_available": bool(weights),
                          "missing_correction_weights": missing, "correction_artifacts": label_status}, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", nargs="?", choices=("all", "baseline", "ssr", "co", "quality"), default="all")
    parser.add_argument("--stage", choices=("all", "teacher", "correction", "student"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--ssr-config", type=Path, default=ROOT / "configs/ssr.json")
    parser.add_argument("--co-config", type=Path, default=ROOT / "configs/co-correcting.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted Co last.pt with unchanged settings")
    parser.add_argument("--fresh", action="store_true", help="Archive selected stages, then start fresh")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate saved models without training")
    parser.add_argument("--check-only", action="store_true", help="List results without loading models")
    parser.add_argument("--metric", choices=("all", "ch", "cle", "credibility"), default="all")
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if args.fresh and (args.resume or args.evaluate_only):
        parser.error("--fresh cannot be combined with --resume or --evaluate-only.")
    if args.stage == "correction" and (args.method not in {"all", "ssr", "co"} or args.evaluate_only):
        parser.error("--stage correction requires SSR/Co and cannot use --evaluate-only.")
    if args.check_only:
        inventory(args.output)
        return
    if args.method == "quality":
        quality(args)
        return
    if args.method in {"all", "baseline"} and args.stage != "correction":
        from ssr.pipeline import load_config as ssr_load
        from cocorrecting.config import load_config as co_load
        co = co_load(args.co_config)
        shared_baseline_config(ssr_load(args.ssr_config), co)
        saved_config = args.output / "baseline/config.json"
        if saved_config.is_file() and not args.fresh:
            saved = ssr_load(saved_config)
            if saved.training.method != "ce" or saved.training.ce_loss != "wce":
                raise ValueError("The shared baseline must use plain WCE training.")
            if any(getattr(saved.training, name) is not None for name in
                   ("corrected_labels_path", "corrected_validation_labels_path", "corrected_test_labels_path")):
                raise ValueError("The shared baseline must use original Train, Val and Test labels.")
            shared_baseline_config(saved, co)
        stage(args, "baseline", "baseline")
    for method in ("ssr", "co"):
        if args.method in {"all", method}:
            for role in ("teacher", "student") if args.stage == "all" else (args.stage,):
                stage(args, method, role)


if __name__ == "__main__":
    main()
