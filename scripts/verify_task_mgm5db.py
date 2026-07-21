"""Verify MGM-5DB schema-v2 transactions against the canonical V3 task."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


MGM_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_ROOT = MGM_ROOT.parent
EVAL_HUB = BENCHMARKS_ROOT / "evaluation-hub"
sys.path.insert(0, str(MGM_ROOT / "scripts"))
sys.path.insert(0, str(EVAL_HUB))

from run_task_mgm5db import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    DEFAULT_PHYLOGENY,
    DEFAULT_TASK_DIR,
    DEFAULT_TOKENIZER,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_MODEL_SHA256,
    EXPECTED_N_GENERA,
    EXPECTED_PHYLOGENY_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    MODEL_NAME,
    VARIANT,
    legacy_inner_split,
)
from utils.data_loader import load_task_contract, validate_task_folds  # noqa: E402
from utils.metrics import compute_unified_metrics  # noqa: E402
from utils.run_manifest import sha256_file, task_payload_fingerprint  # noqa: E402
from utils.unified_io import (  # noqa: E402
    fold_artifact_state,
    make_task_artifact_paths,
)


METRIC_FIELDS = [
    "macro_f1",
    "auroc",
    "aupr",
    "f_max",
    "accuracy",
    "precision_macro",
    "recall_macro",
]

EXPECTED_TASK_MANIFEST_SHA256 = (
    "1ef47a436e77e88ef1128d72c20e6dd7519c5513c49155054d178fbc6010fdc1"
)
EXPECTED_TASK_PAYLOAD_FINGERPRINT = {
    "n_files": 152,
    "tree_sha256": "0980ae729ae2266f9822ad561dd66c027614f9c4edea39cb3f383e4345a15176",
}
EXPECTED_CORPUS_SHA256 = (
    "f10ffe14d2e4384a738e84a9d26694a8e46055520cc30fd5e9fb1d8415d6d78c"
)
EXPECTED_SEEDS = [42, 52, 62]
EXPECTED_N_DISEASES = 11
EXPECTED_N_FOLDS = 69
EXPECTED_N_TRANSACTIONS = EXPECTED_N_FOLDS * len(EXPECTED_SEEDS)
EXPECTED_VALIDATION_FRACTION = 0.15
EXPECTED_INNER_SPLIT_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--summary-out")
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _same_number(observed, expected, tolerance: float = 1e-12) -> bool:
    if expected is None:
        return pd.isna(observed)
    try:
        return math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def artifact_tree_fingerprint(paths: list[Path] | set[Path], output_dir: Path) -> dict:
    """Hash relative artifact paths and file hashes in one stable canonical order."""
    root = output_dir.resolve()
    resolved = [Path(path).resolve() for path in paths]
    records = sorted(
        ((path.relative_to(root).as_posix(), sha256_file(path)) for path in resolved),
        key=lambda item: item[0],
    )
    digest = hashlib.sha256()
    for relative, file_hash in records:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {"n_files": len(records), "tree_sha256": digest.hexdigest()}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    run_manifest = json.loads(manifest_path.read_text())
    contract = run_manifest.get("contract", {})
    task = load_task_contract(args.task_dir)
    if Path(args.task_dir).resolve() != DEFAULT_TASK_DIR.resolve():
        raise ValueError("Formal verification requires the canonical V3 task path")
    if Path(contract.get("task_dir", "")).resolve() != DEFAULT_TASK_DIR.resolve():
        raise ValueError("Run manifest is not bound to the canonical V3 task path")
    if contract.get("protocol_id") != task.contract_id:
        raise ValueError("Run/task protocol mismatch")
    if contract.get("model_name") != MODEL_NAME or contract.get("variant") != VARIANT:
        raise ValueError("Run manifest is not MGM-5DB")
    current_task_payload = task_payload_fingerprint(task.task_dir)
    if current_task_payload != EXPECTED_TASK_PAYLOAD_FINGERPRINT:
        raise ValueError("Canonical task payload differs from the preregistered fingerprint")
    if current_task_payload != contract.get("task_payload_fingerprint"):
        raise ValueError("Current task payload differs from run manifest")
    current_task_manifest_hash = sha256_file(task.task_dir / "label_manifest.json")
    if current_task_manifest_hash != EXPECTED_TASK_MANIFEST_SHA256:
        raise ValueError("Canonical task manifest differs from the preregistered hash")
    if current_task_manifest_hash != contract.get("task_manifest_sha256"):
        raise ValueError("Current task manifest differs from run manifest")
    current_corpus_hash = sha256_file(task.corpus_path)
    if current_corpus_hash != EXPECTED_CORPUS_SHA256:
        raise ValueError("Canonical V3 corpus differs from the preregistered hash")
    if current_corpus_hash != contract.get("corpus_sha256"):
        raise ValueError("Current physical V3 corpus differs from run manifest")
    if sha256_file(contract["runner"]) != contract.get("runner_sha256"):
        raise ValueError("Runner source changed since run manifest")
    for path_field, hash_field in (
        ("model_dir", "model_weights_sha256"),
        ("tokenizer", "tokenizer_sha256"),
        ("phylogeny", "phylogeny_sha256"),
    ):
        path = Path(contract[path_field])
        if path_field == "model_dir":
            path = path / "pytorch_model.bin"
        if sha256_file(path) != contract[hash_field]:
            raise ValueError(f"Current {path_field} differs from run manifest")
    model_config_path = Path(contract["model_dir"]) / "config.json"
    if sha256_file(model_config_path) != contract.get("model_config_sha256"):
        raise ValueError("Current model config differs from run manifest")
    expected_assets = {
        "model_dir": (DEFAULT_MODEL_DIR.resolve(), EXPECTED_MODEL_SHA256),
        "tokenizer": (DEFAULT_TOKENIZER.resolve(), EXPECTED_TOKENIZER_SHA256),
        "phylogeny": (DEFAULT_PHYLOGENY.resolve(), EXPECTED_PHYLOGENY_SHA256),
    }
    for path_field, (expected_path, expected_hash) in expected_assets.items():
        if Path(contract[path_field]).resolve() != expected_path:
            raise ValueError(f"Formal run uses a noncanonical {path_field} path")
        hash_field = {
            "model_dir": "model_weights_sha256",
            "tokenizer": "tokenizer_sha256",
            "phylogeny": "phylogeny_sha256",
        }[path_field]
        if contract.get(hash_field) != expected_hash:
            raise ValueError(f"Formal run uses a noncanonical {path_field} hash")
    if contract.get("model_config_sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("Formal run uses a noncanonical model config hash")
    for source_path, expected_hash in contract.get("support_source_sha256", {}).items():
        if sha256_file(source_path) != expected_hash:
            raise ValueError(f"Source changed since run manifest: {source_path}")

    diseases = list(contract.get("diseases", []))
    seeds = [int(seed) for seed in contract.get("seeds", [])]
    if not diseases or not seeds or len(set(diseases)) != len(diseases) or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid disease/seed contract")
    if diseases != list(task.diseases) or len(diseases) != EXPECTED_N_DISEASES:
        raise ValueError("Formal verification requires all 11 task diseases in canonical order")
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Formal verification requires seeds {EXPECTED_SEEDS}")
    validate_task_folds(task, diseases)
    selected_folds = []
    for disease in diseases:
        selected_folds.extend(task.iter_folds(disease))
    execution_scope = contract.get("execution_scope", {})
    if execution_scope.get("max_folds") is not None or execution_scope.get("smoke_only") is not False:
        raise ValueError("Formal verification refuses smoke or fold-limited execution scope")
    if len(selected_folds) != EXPECTED_N_FOLDS:
        raise ValueError(f"Formal verification requires {EXPECTED_N_FOLDS} outer folds")

    inner = contract.get("inner_validation", {})
    if inner.get("method") != "legacy_random_permutation":
        raise ValueError("Unexpected inner validation method")
    validation_fraction = float(inner["fraction"])
    split_seed = int(inner["seed"])
    if not math.isclose(
        validation_fraction,
        EXPECTED_VALIDATION_FRACTION,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or split_seed != EXPECTED_INNER_SPLIT_SEED:
        raise ValueError("Formal run uses a noncanonical inner validation contract")
    if inner.get("model_seed_independent") is not True:
        raise ValueError("Formal inner split must be fixed across model seeds")
    run_manifest_sha256 = sha256_file(manifest_path)
    expected_metric_paths = set()
    expected_prediction_paths = set()
    expected_completion_paths = set()
    fold_seed_values = defaultdict(list)
    verified = 0

    for spec in selected_folds:
        y_outer = task.labels(spec.disease, spec.train_idx)
        inner_train_idx, _, _, _ = legacy_inner_split(
            spec.train_idx, y_outer, validation_fraction, split_seed
        )
        y_test_expected = task.labels(spec.disease, spec.test_idx)
        for seed in seeds:
            metrics_path, predictions_path, completion_path = make_task_artifact_paths(
                output_dir,
                MODEL_NAME,
                VARIANT,
                seed,
                spec.disease,
                spec.fold,
                spec.test_study,
            )
            expected_metric_paths.add(metrics_path.resolve())
            expected_prediction_paths.add(predictions_path.resolve())
            expected_completion_paths.add(completion_path.resolve())
            identity = {
                "protocol_id": task.contract_id,
                "model_name": MODEL_NAME,
                "variant": VARIANT,
                "seed": seed,
                "disease": spec.disease,
                "fold": spec.fold,
                "test_study": spec.test_study,
            }
            state = fold_artifact_state(
                metrics_path,
                predictions_path,
                completion_path,
                run_manifest_sha256,
                identity,
            )
            if state != "complete":
                raise RuntimeError(f"Incomplete MGM-5DB transaction ({state}): {metrics_path}")
            metrics_frame = pd.read_csv(metrics_path)
            predictions = pd.read_csv(predictions_path)
            if len(metrics_frame) != 1:
                raise ValueError(f"Metrics file must contain one row: {metrics_path}")
            row = metrics_frame.iloc[0]
            expected_identity = {
                "schema_version": 2,
                "protocol_id": task.contract_id,
                "model_name": MODEL_NAME,
                "variant": VARIANT,
                "seed": seed,
                "disease": spec.disease,
                "fold": spec.fold,
                "test_study": spec.test_study,
                "n_train": len(inner_train_idx),
                "n_test": len(spec.test_idx),
                "n_feat": EXPECTED_N_GENERA,
            }
            for field, expected in expected_identity.items():
                observed = row[field]
                if str(observed) != str(expected):
                    raise ValueError(
                        f"{metrics_path} field {field}: observed={observed!r} expected={expected!r}"
                    )
            required_prediction_columns = ["row_idx", "y_true", "y_score"]
            if list(predictions.columns) != required_prediction_columns:
                raise ValueError(f"Unexpected prediction schema: {predictions_path}")
            row_idx = predictions["row_idx"].to_numpy(dtype=np.int64)
            y_true = predictions["y_true"].to_numpy(dtype=np.int64)
            y_score = predictions["y_score"].to_numpy(dtype=np.float64)
            if not np.array_equal(row_idx, spec.test_idx):
                raise ValueError(f"Prediction row order/identity mismatch: {predictions_path}")
            if not np.array_equal(y_true, y_test_expected):
                raise ValueError(f"Prediction labels mismatch task sidecar: {predictions_path}")
            if not np.isfinite(y_score).all() or np.any((y_score < 0) | (y_score > 1)):
                raise ValueError(f"Prediction scores are not finite probabilities: {predictions_path}")
            recomputed = compute_unified_metrics(y_true, y_score)
            for field in METRIC_FIELDS:
                if not _same_number(row[field], recomputed[field]):
                    raise ValueError(
                        f"Metric recomputation mismatch {field}: "
                        f"observed={row[field]} expected={recomputed[field]}"
                    )
            fold_seed_values[(spec.disease, spec.fold)].append(float(row["auroc"]))
            verified += 1

    expected_artifacts = {
        "metrics": expected_metric_paths,
        "predictions": expected_prediction_paths,
        "completion": expected_completion_paths,
    }
    observed_artifacts = {
        "metrics": {path.resolve() for path in output_dir.rglob("*_metrics.csv")},
        "predictions": {path.resolve() for path in output_dir.rglob("*_predictions.csv")},
        "completion": {path.resolve() for path in output_dir.rglob("*_complete.json")},
    }
    for artifact_type, expected_paths in expected_artifacts.items():
        observed_paths = observed_artifacts[artifact_type]
        if observed_paths != expected_paths:
            missing = sorted(str(path) for path in expected_paths - observed_paths)
            extra = sorted(str(path) for path in observed_paths - expected_paths)
            raise ValueError(
                f"{artifact_type} coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
            )
    if verified != EXPECTED_N_TRANSACTIONS or len(expected_metric_paths) != EXPECTED_N_TRANSACTIONS:
        raise ValueError(
            "Formal verification requires exactly "
            f"{EXPECTED_N_TRANSACTIONS} transactions, observed {verified}"
        )

    disease_fold_means = defaultdict(list)
    for (disease, _fold), values in fold_seed_values.items():
        if len(values) != len(seeds):
            raise ValueError(f"Seed coverage mismatch for {disease}")
        disease_fold_means[disease].append(float(np.mean(values)))
    disease_summary = {
        disease: {
            "n_studies": len(disease_fold_means[disease]),
            "auroc_equal_study_mean_after_seed_mean": float(
                np.mean(disease_fold_means[disease])
            ),
        }
        for disease in diseases
    }
    summary = {
        "status": "PASS",
        "verification_scope": "formal_full_preregistered_contract",
        "protocol_id": task.contract_id,
        "model_name": MODEL_NAME,
        "variant": VARIANT,
        "n_diseases": len(diseases),
        "n_outer_folds": len(selected_folds),
        "seeds": seeds,
        "n_verified_transactions": verified,
        "output_dir": str(output_dir),
        "run_manifest_sha256": run_manifest_sha256,
        "metrics_tree_fingerprint": artifact_tree_fingerprint(
            expected_metric_paths, output_dir
        ),
        "predictions_tree_fingerprint": artifact_tree_fingerprint(
            expected_prediction_paths, output_dir
        ),
        "completion_tree_fingerprint": artifact_tree_fingerprint(
            expected_completion_paths, output_dir
        ),
        "disease_summary": disease_summary,
    }
    if args.summary_out:
        _atomic_json(Path(args.summary_out), summary)
    print(json.dumps(summary, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
