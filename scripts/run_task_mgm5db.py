"""Run the converged MGM-5DB checkpoint on the canonical V3 disease task.

This runner is intentionally separate from the legacy V2 ``variant_A/B``
driver.  It reads labels and folds from the registered compact V3 task,
reconstructs MGM tokens with the exact MGM-5DB tokenizer and V3 z-score
statistics, and writes schema-v2 fold transactions with row predictions.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pickle
import shutil
import site
import subprocess
import sys
import time
import uuid
from pathlib import Path

_USER_SITE = site.getusersitepackages()
if site.ENABLE_USER_SITE and _USER_SITE in sys.path:
    raise RuntimeError(
        f"Shared user-site is active ({_USER_SITE}); rerun with "
        "PYTHONNOUSERSITE=1 to prevent environment contamination"
    )

import anndata as ad
import numpy as np
import pandas as pd
import scipy.special
import scipy.sparse
import torch


MGM_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_ROOT = MGM_ROOT.parent
PROJECT_ROOT = BENCHMARKS_ROOT.parent
EVAL_HUB = BENCHMARKS_ROOT / "evaluation-hub"
sys.path.insert(0, str(MGM_ROOT))
sys.path.insert(0, str(EVAL_HUB))

from mgm.src.MicroCorpus import MicroCorpus  # noqa: E402
from utils.assets import refuse_fixed_asset_output  # noqa: E402
from utils.data_loader import (  # noqa: E402
    load_task_contract,
    validate_task_corpus_alignment,
    validate_task_folds,
)
from utils.metrics import compute_unified_metrics  # noqa: E402
from utils.run_manifest import (  # noqa: E402
    assert_source_fingerprint,
    ensure_run_manifest,
    environment_fingerprint,
    sha256_file,
    source_fingerprint,
    task_payload_fingerprint,
)
from utils.unified_io import (  # noqa: E402
    fold_artifact_lock,
    fold_artifact_state,
    make_task_artifact_paths,
    save_fold_completion,
    save_fold_metrics,
    save_fold_predictions,
)


MODEL_NAME = "MGM"
VARIANT = "5DB"
DEFAULT_TASK_DIR = PROJECT_ROOT / "data/MCFCorpusV3/tasks/rm_cc_loso_11"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "ckpt_store/20260617_mgm_5db/pretrained"
DEFAULT_TOKENIZER = (
    PROJECT_ROOT
    / "archive/experiments/20260527_full_benchmark/mgm/assets/our_tokenizer.pkl"
)
DEFAULT_PHYLOGENY = PROJECT_ROOT / "tmp/20260617_mgm_5db/our_phylogeny_v3.csv"

EXPECTED_MODEL_SHA256 = "8e2568e0d023229e2ce56c279862a86eb78f4600ae307c4a57c557f7864e8758"
EXPECTED_CONFIG_SHA256 = "3a24dd7faf69cf37732cc960764a28cf64d54765f9a29c37267cef9a69cb209b"
EXPECTED_TOKENIZER_SHA256 = "0a855e0be57efe91d470c53c8950e726fc096d1fabacbd7b56e84a5d71986d70"
EXPECTED_PHYLOGENY_SHA256 = "321a523345ab48564cb8e8de7ef58b8d269a190a8b33cf9ffba378019f0a79ec"
EXPECTED_N_GENERA = 8114
EXPECTED_VOCAB_SIZE = 8118
MAX_LEN = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--phylogeny", default=str(DEFAULT_PHYLOGENY))
    parser.add_argument("--output-dir", help="New MGM-5DB schema-v2 result root")
    parser.add_argument("--cache-dir", help="Fold corpus cache shared by finetune seeds")
    parser.add_argument("--work-dir", help="Ephemeral per-fit work root; failed fits are retained")
    parser.add_argument("--diseases", nargs="+", default=None)
    parser.add_argument(
        "--allow-noncanonical-task",
        action="store_true",
        help=(
            "Explicitly permit a separately fingerprinted compact task. Default "
            "refuses anything other than the canonical 11-disease/69-fold contract."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--inner-split-seed", type=int, default=0)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Global outer-fold smoke limit; use a dedicated output root",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Repair an incomplete matching transaction; never changes run contract",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Abort after this many consecutive failed fold-seed transactions",
    )
    args = parser.parse_args()
    if not args.dry_run:
        missing = [
            name
            for name in ("output_dir", "cache_dir", "work_dir")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"real runs require: {', '.join('--' + x.replace('_', '-') for x in missing)}")
    return args


def _array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.tobytes())
    return digest.hexdigest()


def _json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def legacy_inner_split(
    outer_train_idx: np.ndarray,
    outer_train_y: np.ndarray,
    validation_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the legacy MGM fallback split exactly, independent of model seed."""
    outer_train_idx = np.asarray(outer_train_idx, dtype=np.int64)
    outer_train_y = np.asarray(outer_train_y, dtype=np.int64)
    if outer_train_idx.ndim != 1 or outer_train_y.shape != outer_train_idx.shape:
        raise ValueError("outer train indices and labels must be equal-length 1D arrays")
    if len(np.unique(outer_train_idx)) != len(outer_train_idx):
        raise ValueError("outer train indices contain duplicates")
    if not np.isin(outer_train_y, [0, 1]).all():
        raise ValueError("outer train labels must be control=0/case=1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")

    permutation = np.random.RandomState(split_seed).permutation(len(outer_train_idx))
    n_val = max(1, int(round(len(outer_train_idx) * validation_fraction)))
    val_pos = permutation[:n_val]
    train_pos = permutation[n_val:]
    train_idx = outer_train_idx[train_pos]
    val_idx = outer_train_idx[val_pos]
    train_y = outer_train_y[train_pos]
    val_y = outer_train_y[val_pos]

    if len(train_idx) == 0 or np.intersect1d(train_idx, val_idx).size:
        raise ValueError("invalid derived inner train/validation split")
    if not np.array_equal(
        np.sort(np.concatenate([train_idx, val_idx])), np.sort(outer_train_idx)
    ):
        raise ValueError("inner train/validation do not cover outer train exactly")
    for label, y in (("train", train_y), ("validation", val_y)):
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(f"derived inner {label} split does not contain both classes")
    return train_idx, val_idx, train_y, val_y


def _load_assets(model_dir: Path, tokenizer_path: Path, phylogeny_path: Path, adata: ad.AnnData):
    model_dir = model_dir.resolve()
    tokenizer_path = tokenizer_path.resolve()
    phylogeny_path = phylogeny_path.resolve()
    weights_path = model_dir / "pytorch_model.bin"
    config_path = model_dir / "config.json"
    for path in (weights_path, config_path, tokenizer_path, phylogeny_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    observed_hashes = {
        "model": sha256_file(weights_path),
        "config": sha256_file(config_path),
        "tokenizer": sha256_file(tokenizer_path),
        "phylogeny": sha256_file(phylogeny_path),
    }
    expected_hashes = {
        "model": EXPECTED_MODEL_SHA256,
        "config": EXPECTED_CONFIG_SHA256,
        "tokenizer": EXPECTED_TOKENIZER_SHA256,
        "phylogeny": EXPECTED_PHYLOGENY_SHA256,
    }
    if observed_hashes != expected_hashes:
        raise RuntimeError(
            f"MGM-5DB asset hash mismatch: expected={expected_hashes} observed={observed_hashes}"
        )

    model_config = json.loads(config_path.read_text())
    with tokenizer_path.open("rb") as handle:
        tokenizer = pickle.load(handle)
    phylogeny = pd.read_csv(phylogeny_path, index_col=0)
    if "Genus" not in adata.var:
        raise ValueError("V3 corpus is missing var['Genus']")
    corpus_genera = adata.var["Genus"].astype(str).tolist()
    phylogeny_genera = phylogeny.index.astype(str).tolist()
    if len(corpus_genera) != EXPECTED_N_GENERA or corpus_genera != phylogeny_genera:
        raise ValueError("V3 var['Genus'] and MGM-5DB phylogeny order do not match exactly")
    if list(tokenizer.toks[:4]) != ["<pad>", "<mask>", "<bos>", "<eos>"]:
        raise ValueError("MGM-5DB tokenizer special-token order is not canonical")
    if list(tokenizer.toks[4:]) != corpus_genera:
        raise ValueError("MGM-5DB tokenizer genus order does not match V3 corpus")
    if int(tokenizer.vocab_size) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Unexpected tokenizer vocab size: {tokenizer.vocab_size}")
    if int(model_config.get("vocab_size", -1)) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Unexpected checkpoint vocab size: {model_config.get('vocab_size')}")
    if [tokenizer.vocab[g] for g in corpus_genera] != list(range(4, EXPECTED_VOCAB_SIZE)):
        raise ValueError("Tokenizer genus token ids are not the frozen contiguous 4..8117 mapping")

    mean = phylogeny["mean"].to_numpy(dtype=np.float64)
    std = phylogeny["std"].to_numpy(dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("MGM-5DB phylogeny mean/std contain invalid values")
    genus_token_ids = np.asarray(
        [tokenizer.vocab[g] for g in corpus_genera], dtype=np.int64
    )
    return {
        "model_dir": model_dir,
        "weights_path": weights_path,
        "config_path": config_path,
        "tokenizer_path": tokenizer_path,
        "phylogeny_path": phylogeny_path,
        "hashes": observed_hashes,
        "model_config": model_config,
        "tokenizer": tokenizer,
        "phylogeny_mean": mean,
        "phylogeny_std": std,
        "genus_token_ids": genus_token_ids,
    }


def _write_split_corpus(
    counts: np.ndarray,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    tokenizer,
    mean: np.ndarray,
    std: np.ndarray,
    genus_token_ids: np.ndarray,
    corpus_path: Path,
    labels_path: Path,
) -> dict:
    counts = np.asarray(counts)
    labels = np.asarray(labels, dtype=np.int64)
    sample_ids = np.asarray(sample_ids, dtype=str)
    if counts.ndim != 2 or counts.shape != (len(labels), EXPECTED_N_GENERA):
        raise ValueError(f"Unexpected split counts shape: {counts.shape}")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Split sample IDs are not unique")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Split labels are outside control=0/case=1")

    tokens = np.full((len(labels), MAX_LEN), tokenizer.pad_token_id, dtype=np.int64)
    lengths = np.empty(len(labels), dtype=np.int64)
    for row_index in range(len(labels)):
        row = np.asarray(counts[row_index], dtype=np.float64)
        total = float(row.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"All-zero or invalid count row for sample {sample_ids[row_index]}")
        present = np.flatnonzero(row > 0)
        scores = (row[present] / total - mean[present]) / std[present]
        order = present[np.argsort(-scores)]
        genus_ids = genus_token_ids[order]
        raw_length = len(genus_ids) + 2
        kept_length = min(raw_length, MAX_LEN)
        tokens[row_index, 0] = tokenizer.bos_token_id
        tokens[row_index, 1 : kept_length - 1] = genus_ids[: kept_length - 2]
        tokens[row_index, kept_length - 1] = tokenizer.eos_token_id
        lengths[row_index] = raw_length

    corpus = MicroCorpus.__new__(MicroCorpus)
    corpus.tokenizer = tokenizer
    corpus.tokens = torch.from_numpy(tokens)
    corpus.max_len = MAX_LEN
    # MGM finetune/predict use corpus.data.index only for label/score alignment.
    corpus.data = pd.DataFrame(index=pd.Index(sample_ids, name="sample_id"))
    with corpus_path.open("wb") as handle:
        pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    label_names = np.where(labels == 1, "case", "control")
    pd.DataFrame({"label": label_names}, index=sample_ids).to_csv(labels_path)
    return {
        "n_samples": int(len(labels)),
        "n_control": int((labels == 0).sum()),
        "n_case": int((labels == 1).sum()),
        "token_length_min": int(lengths.min()),
        "token_length_mean": float(lengths.mean()),
        "token_length_max": int(lengths.max()),
        "n_truncated": int((lengths > MAX_LEN).sum()),
    }


def _validate_cache(cache_dir: Path, expected_identity: dict) -> dict:
    manifest_path = cache_dir / "prepare_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Incomplete MGM fold cache (missing manifest): {cache_dir}")
    payload = json.loads(manifest_path.read_text())
    if payload.get("identity") != expected_identity:
        raise RuntimeError(f"MGM fold cache identity mismatch: {cache_dir}")
    expected_files = payload.get("files", {})
    required = {
        "train_corpus.pkl",
        "val_corpus.pkl",
        "test_corpus.pkl",
        "train_labels.csv",
        "val_labels.csv",
        "test_labels.csv",
    }
    if set(expected_files) != required:
        raise RuntimeError(f"MGM fold cache file manifest mismatch: {cache_dir}")
    observed = {name: sha256_file(cache_dir / name) for name in sorted(required)}
    if observed != expected_files:
        raise RuntimeError(f"MGM fold cache hash mismatch: {cache_dir}")
    return payload


def prepare_fold_cache(
    cache_root: Path,
    task,
    spec,
    adata: ad.AnnData,
    assets: dict,
    validation_fraction: float,
    inner_split_seed: int,
    tokenization_source_sha256: str,
    task_manifest_sha256: str,
    task_payload_fingerprint: dict,
    corpus_sha256: str,
) -> tuple[Path, dict]:
    outer_y = task.labels(spec.disease, spec.train_idx)
    train_idx, val_idx, train_y, val_y = legacy_inner_split(
        spec.train_idx, outer_y, validation_fraction, inner_split_seed
    )
    test_idx = np.asarray(spec.test_idx, dtype=np.int64)
    test_y = task.labels(spec.disease, test_idx)
    if np.intersect1d(np.concatenate([train_idx, val_idx]), test_idx).size:
        raise ValueError(f"{spec.disease} fold {spec.fold}: inner rows overlap outer test")

    identity = {
        "schema_version": 1,
        "protocol_id": task.contract_id,
        "disease": spec.disease,
        "fold": int(spec.fold),
        "test_study": spec.test_study,
        "validation_fraction": validation_fraction,
        "inner_split_seed": inner_split_seed,
        "outer_train_idx_sha256": _array_sha256(spec.train_idx),
        "inner_train_idx_sha256": _array_sha256(train_idx),
        "inner_val_idx_sha256": _array_sha256(val_idx),
        "test_idx_sha256": _array_sha256(test_idx),
        "outer_train_labels_sha256": _array_sha256(outer_y),
        "inner_train_labels_sha256": _array_sha256(train_y),
        "inner_val_labels_sha256": _array_sha256(val_y),
        "test_labels_sha256": _array_sha256(test_y),
        "task_manifest_sha256": task_manifest_sha256,
        "task_payload_fingerprint": task_payload_fingerprint,
        "corpus_sha256": corpus_sha256,
        "tokenizer_sha256": assets["hashes"]["tokenizer"],
        "phylogeny_sha256": assets["hashes"]["phylogeny"],
        "tokenization_source_sha256": tokenization_source_sha256,
        "max_len": MAX_LEN,
    }
    cache_key = _json_sha256(identity)[:20]
    fold_root = cache_root / spec.disease / f"fold_{spec.fold:02d}"
    cache_dir = fold_root / cache_key
    lock_path = fold_root / f".{cache_key}.lock"
    fold_root.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_dir.exists():
            return cache_dir, _validate_cache(cache_dir, identity)

        temporary = fold_root / f".{cache_key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            union_idx = np.unique(np.concatenate([train_idx, val_idx, test_idx]))
            sub = adata[union_idx].to_memory()
            if "counts" not in sub.layers:
                raise ValueError("V3 corpus is missing layers['counts'] required by MGM")
            counts = sub.layers["counts"]
            if scipy.sparse.issparse(counts):
                counts = counts.toarray()
            counts = np.asarray(counts)
            sample_ids_all = sub.obs_names.astype(str).to_numpy()

            split_stats = {}
            for split_name, indices, labels in (
                ("train", train_idx, train_y),
                ("val", val_idx, val_y),
                ("test", test_idx, test_y),
            ):
                positions = np.searchsorted(union_idx, indices)
                if not np.array_equal(union_idx[positions], indices):
                    raise ValueError(f"{split_name} indices cannot be mapped into fold union")
                split_stats[split_name] = _write_split_corpus(
                    counts[positions],
                    sample_ids_all[positions],
                    labels,
                    assets["tokenizer"],
                    assets["phylogeny_mean"],
                    assets["phylogeny_std"],
                    assets["genus_token_ids"],
                    temporary / f"{split_name}_corpus.pkl",
                    temporary / f"{split_name}_labels.csv",
                )

            file_names = [
                "train_corpus.pkl",
                "val_corpus.pkl",
                "test_corpus.pkl",
                "train_labels.csv",
                "val_labels.csv",
                "test_labels.csv",
            ]
            payload = {
                "identity": identity,
                "split_stats": split_stats,
                "files": {name: sha256_file(temporary / name) for name in file_names},
            }
            _atomic_json(temporary / "prepare_manifest.json", payload)
            temporary.rename(cache_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return cache_dir, _validate_cache(cache_dir, identity)


def _run_mgm(args_list: list[str], log_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "from mgm.CLI import main; main()",
        *args_list,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_handle:
        log_handle.write("COMMAND: " + " ".join(command) + "\n")
        log_handle.flush()
        result = subprocess.run(
            command,
            cwd=str(MGM_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"MGM command failed with exit={result.returncode}; see {log_path}"
        )


def align_case_scores(y_score_path: Path, expected_sample_ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(y_score_path, index_col=0)
    frame.index = frame.index.astype(str)
    expected = pd.Index([str(x) for x in expected_sample_ids])
    if frame.index.has_duplicates:
        raise ValueError("MGM y_score contains duplicate sample IDs")
    missing = expected.difference(frame.index)
    extra = frame.index.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            f"MGM y_score sample identity mismatch: missing={missing.tolist()[:5]} "
            f"extra={extra.tolist()[:5]}"
        )
    if not {"control", "case"}.issubset(frame.columns):
        raise ValueError(f"MGM y_score lacks control/case columns: {list(frame.columns)}")
    logits = frame.loc[expected, ["control", "case"]].to_numpy(dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("MGM y_score contains non-finite logits")
    return scipy.special.softmax(logits, axis=1)[:, 1]


def _fit_one(
    cache_dir: Path,
    cache_manifest: dict,
    work_dir: Path,
    model_dir: Path,
    seed: int,
) -> tuple[np.ndarray, dict, dict]:
    if work_dir.exists():
        raise RuntimeError(f"Stale MGM work directory exists: {work_dir}")
    work_dir.mkdir(parents=True)
    model_out = work_dir / "model"
    trainer_log = work_dir / "trainer_log"
    prediction_out = work_dir / "predictions"
    started = time.time()
    try:
        _run_mgm(
            [
                "finetune",
                "--train-corpus",
                str(cache_dir / "train_corpus.pkl"),
                "--val-corpus",
                str(cache_dir / "val_corpus.pkl"),
                "-l",
                str(cache_dir / "train_labels.csv"),
                "--val-labels",
                str(cache_dir / "val_labels.csv"),
                "-m",
                str(model_dir),
                "-o",
                str(model_out),
                "-H",
                str(trainer_log),
                "--seed",
                str(seed),
            ],
            work_dir / "finetune_stdout.log",
        )
        _run_mgm(
            [
                "predict",
                "-i",
                str(cache_dir / "test_corpus.pkl"),
                "-m",
                str(model_out),
                "-l",
                str(cache_dir / "test_labels.csv"),
                "-o",
                str(prediction_out),
                "-E",
                "--seed",
                str(seed),
            ],
            work_dir / "predict_stdout.log",
        )
        labels_frame = pd.read_csv(cache_dir / "test_labels.csv", index_col=0)
        expected_ids = labels_frame.index.astype(str).tolist()
        y_score = align_case_scores(prediction_out / "y_score.csv", expected_ids)
        y_true = labels_frame.iloc[:, 0].map({"control": 0, "case": 1}).to_numpy()
        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("Cached MGM test labels are outside control/case")

        fit_log_path = trainer_log / "finetune_log.csv"
        if not fit_log_path.is_file():
            raise FileNotFoundError(fit_log_path)
        fit_log = pd.read_csv(fit_log_path)
        eval_losses = pd.to_numeric(fit_log.get("eval_loss"), errors="coerce").dropna()
        train_runtimes = pd.to_numeric(fit_log.get("train_runtime"), errors="coerce").dropna()
        summary = {
            "seed": int(seed),
            "elapsed_seconds": time.time() - started,
            "best_eval_loss": float(eval_losses.min()) if len(eval_losses) else None,
            "trainer_runtime_seconds": float(train_runtimes.iloc[-1]) if len(train_runtimes) else None,
            "cache_manifest_sha256": sha256_file(cache_dir / "prepare_manifest.json"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        diagnostics = {
            "fit_log": fit_log_path,
            "finetune_stdout": work_dir / "finetune_stdout.log",
            "predict_stdout": work_dir / "predict_stdout.log",
        }
        return y_score, summary, diagnostics
    except Exception:
        # Failed work is evidence and remains for inspection/retry planning.
        raise


def _copy_diagnostics(diagnostics: dict[str, Path], summary: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for label, source in diagnostics.items():
        shutil.copy2(source, destination / f"{label}{source.suffix}")
    _atomic_json(destination / "fit_summary.json", summary)


def main() -> None:
    args = parse_args()
    if args.max_folds is not None and args.max_folds < 0:
        raise ValueError("--max-folds must be non-negative")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {args.seeds}")
    if args.max_consecutive_failures < 1:
        raise ValueError("--max-consecutive-failures must be positive")

    task = load_task_contract(args.task_dir)
    diseases = list(args.diseases or task.diseases)
    if len(set(diseases)) != len(diseases):
        raise ValueError(f"Duplicate diseases are not allowed: {diseases}")
    unknown = sorted(set(diseases) - set(task.diseases))
    if unknown:
        raise ValueError(f"Diseases outside task contract: {unknown}")

    expected_payload = task.manifest.get("compact_payload_fingerprint")
    actual_payload = task_payload_fingerprint(task.task_dir)
    if actual_payload != expected_payload:
        raise ValueError(
            f"Task payload fingerprint mismatch: expected={expected_payload} observed={actual_payload}"
        )
    expected_corpus_hash = task.manifest.get("input_fingerprints", {}).get("corpus_sha256")
    if not expected_corpus_hash:
        raise ValueError("Task manifest does not pin corpus SHA-256")
    actual_corpus_hash = sha256_file(task.corpus_path)
    if actual_corpus_hash != expected_corpus_hash:
        raise ValueError(
            f"Task corpus hash mismatch: expected={expected_corpus_hash} observed={actual_corpus_hash}"
        )

    adata = ad.read_h5ad(task.corpus_path, backed="r")
    try:
        n_all_folds = validate_task_folds(task)
        validate_task_corpus_alignment(task, adata)
        if (len(task.diseases) != 11 or n_all_folds != 69) and not args.allow_noncanonical_task:
            raise ValueError(
                f"Canonical task must contain 11 diseases/69 folds, got "
                f"{len(task.diseases)}/{n_all_folds}; pass --allow-noncanonical-task "
                "only for a separately fingerprinted task contract"
            )
        assets = _load_assets(
            Path(args.model_dir), Path(args.tokenizer), Path(args.phylogeny), adata
        )

        selected_folds = []
        for disease in diseases:
            selected_folds.extend(task.iter_folds(disease))
        if args.max_folds is not None:
            selected_folds = selected_folds[: args.max_folds]
        for spec in selected_folds:
            outer_y = task.labels(spec.disease, spec.train_idx)
            legacy_inner_split(
                spec.train_idx,
                outer_y,
                args.validation_fraction,
                args.inner_split_seed,
            )
        print(
            f"Contract PASS: task={task.contract_id} all_folds={n_all_folds} "
            f"selected_folds={len(selected_folds)} seeds={args.seeds}"
        )
        print(
            f"Assets PASS: model={assets['hashes']['model'][:12]} "
            f"tokenizer={assets['hashes']['tokenizer'][:12]} "
            f"phylogeny={assets['hashes']['phylogeny'][:12]}"
        )
        if args.dry_run:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing MGM finetune CPU fallback")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Formal MGM worker requires exactly one visible GPU; bind one device with "
                f"CUDA_VISIBLE_DEVICES (observed {torch.cuda.device_count()})"
            )
        cuda_runtime = {
            "torch_cuda_version": torch.version.cuda,
            "visible_device_count": torch.cuda.device_count(),
            "visible_device_name": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

        output_dir = Path(args.output_dir).resolve()
        cache_root = Path(args.cache_dir).resolve()
        work_root = Path(args.work_dir).resolve()
        refuse_fixed_asset_output(output_dir)
        runner_path = Path(__file__).resolve()
        support_sources = [
            EVAL_HUB / "utils" / name
            for name in [
                "assets.py",
                "data_loader.py",
                "metrics.py",
                "run_manifest.py",
                "unified_io.py",
            ]
        ]
        mgm_sources = [
            MGM_ROOT / "mgm/src/MicroCorpus.py",
            MGM_ROOT / "mgm/src/utils.py",
            MGM_ROOT / "mgm/CLI/main.py",
            MGM_ROOT / "mgm/CLI/CLI_utils.py",
            MGM_ROOT / "mgm/CLI/main_finetune.py",
            MGM_ROOT / "mgm/CLI/main_predict.py",
            MGM_ROOT / "mgm/resources/config.ini",
        ]
        source_snapshot = source_fingerprint([runner_path, *support_sources, *mgm_sources])
        contract = {
            "protocol_id": task.contract_id,
            "noncanonical_task_acknowledged": bool(args.allow_noncanonical_task),
            "task_dir": str(task.task_dir),
            "task_manifest_sha256": sha256_file(task.task_dir / "label_manifest.json"),
            "task_payload_fingerprint": actual_payload,
            "corpus": str(task.corpus_path),
            "corpus_sha256": actual_corpus_hash,
            "positive_class": "case=1",
            "model_name": MODEL_NAME,
            "variant": VARIANT,
            "diseases": diseases,
            "seeds": args.seeds,
            "execution_scope": {
                "max_folds": args.max_folds,
                "smoke_only": args.max_folds is not None,
                "max_consecutive_failures": args.max_consecutive_failures,
            },
            "inner_validation": {
                "method": "legacy_random_permutation",
                "fraction": args.validation_fraction,
                "seed": args.inner_split_seed,
                "model_seed_independent": True,
            },
            "tokenization": {
                "input": "layers[counts]",
                "normalization": "relative abundance then V3 per-genus z-score",
                "ordering": "descending normalized abundance among present genera",
                "max_len": MAX_LEN,
            },
            "model_dir": str(assets["model_dir"]),
            "model_weights_sha256": assets["hashes"]["model"],
            "model_config_sha256": assets["hashes"]["config"],
            "tokenizer": str(assets["tokenizer_path"]),
            "tokenizer_sha256": assets["hashes"]["tokenizer"],
            "phylogeny": str(assets["phylogeny_path"]),
            "phylogeny_sha256": assets["hashes"]["phylogeny"],
            "result_schema_version": 2,
            "sample_predictions": True,
            "runner": str(runner_path),
            "runner_sha256": source_snapshot[str(runner_path)],
            "support_source_sha256": {
                str(path): source_snapshot[str(path.resolve())]
                for path in [*support_sources, *mgm_sources]
            },
            "cuda_runtime": cuda_runtime,
            "environment": environment_fingerprint(
                [
                    "anndata",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "scipy",
                    "torch",
                    "transformers",
                ]
            ),
        }
        assert_source_fingerprint(source_snapshot)
        manifest_path = ensure_run_manifest(output_dir, contract)
        run_manifest_sha256 = sha256_file(manifest_path)
        cache_root.mkdir(parents=True, exist_ok=True)
        work_root.mkdir(parents=True, exist_ok=True)

        failures = []
        written = skipped = 0
        consecutive_failures = 0
        started = time.time()
        for spec in selected_folds:
            assert_source_fingerprint(source_snapshot)
            cache_dir, cache_manifest = prepare_fold_cache(
                cache_root,
                task,
                spec,
                adata,
                assets,
                args.validation_fraction,
                args.inner_split_seed,
                _json_sha256({
                    "runner": source_snapshot[str(runner_path)],
                    "micro_corpus": source_snapshot[
                        str((MGM_ROOT / "mgm/src/MicroCorpus.py").resolve())
                    ],
                }),
                contract["task_manifest_sha256"],
                actual_payload,
                actual_corpus_hash,
            )
            inner_train_idx, _, _, _ = legacy_inner_split(
                spec.train_idx,
                task.labels(spec.disease, spec.train_idx),
                args.validation_fraction,
                args.inner_split_seed,
            )
            y_test = task.labels(spec.disease, spec.test_idx)
            for seed in args.seeds:
                metrics_path, predictions_path, completion_path = make_task_artifact_paths(
                    output_dir,
                    MODEL_NAME,
                    VARIANT,
                    seed,
                    spec.disease,
                    spec.fold,
                    spec.test_study,
                )
                identity = {
                    "protocol_id": task.contract_id,
                    "model_name": MODEL_NAME,
                    "variant": VARIANT,
                    "seed": seed,
                    "disease": spec.disease,
                    "fold": spec.fold,
                    "test_study": spec.test_study,
                }
                with fold_artifact_lock(completion_path):
                    state = fold_artifact_state(
                        metrics_path,
                        predictions_path,
                        completion_path,
                        run_manifest_sha256,
                        identity,
                    )
                    if state == "complete":
                        skipped += 1
                        print(
                            f"SKIP complete seed={seed} {spec.disease} "
                            f"fold={spec.fold:02d} test={spec.test_study}",
                            flush=True,
                        )
                        continue
                    if state == "incomplete" and not args.overwrite:
                        raise RuntimeError(
                            f"Incomplete fold transaction: {metrics_path}; use --overwrite "
                            "only after inspecting the matching run contract"
                        )
                    if args.overwrite:
                        completion_path.unlink(missing_ok=True)

                    work_dir = (
                        work_root
                        / spec.disease
                        / f"fold_{spec.fold:02d}"
                        / f"seed{seed}"
                        / f"attempt_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                    )
                    try:
                        print(
                            f"RUN seed={seed} {spec.disease} fold={spec.fold:02d} "
                            f"test={spec.test_study}",
                            flush=True,
                        )
                        score_pos, fit_summary, diagnostics = _fit_one(
                            cache_dir,
                            cache_manifest,
                            work_dir,
                            assets["model_dir"],
                            seed,
                        )
                        if len(score_pos) != len(spec.test_idx):
                            raise ValueError(
                                f"Prediction length mismatch: {len(score_pos)} != {len(spec.test_idx)}"
                            )
                        metrics = compute_unified_metrics(y_test, score_pos)
                        result = {
                            "schema_version": 2,
                            "protocol_id": task.contract_id,
                            "model_name": MODEL_NAME,
                            "variant": VARIANT,
                            "seed": seed,
                            "disease": spec.disease,
                            "fold": spec.fold,
                            "test_study": spec.test_study,
                            "n_train": int(len(inner_train_idx)),
                            "n_test": int(len(spec.test_idx)),
                            "n_feat": EXPECTED_N_GENERA,
                            **metrics,
                        }
                        save_fold_metrics(result, metrics_path, schema_version=2)
                        save_fold_predictions(
                            spec.test_idx, y_test, score_pos, predictions_path
                        )
                        _copy_diagnostics(
                            diagnostics, fit_summary, metrics_path.parent / "diagnostics"
                        )
                        save_fold_completion(
                            metrics_path,
                            predictions_path,
                            completion_path,
                            run_manifest_sha256,
                            identity,
                        )
                        written += 1
                        consecutive_failures = 0
                        print(
                            f"DONE seed={seed} {spec.disease} fold={spec.fold:02d} "
                            f"AUROC={metrics['auroc']:.4f}",
                            flush=True,
                        )
                        shutil.rmtree(work_dir)
                    except Exception as exc:
                        failure = {
                            **identity,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "work_dir": str(work_dir),
                        }
                        failures.append(failure)
                        consecutive_failures += 1
                        print(f"FAIL {failure}", file=sys.stderr, flush=True)
                        if consecutive_failures >= args.max_consecutive_failures:
                            _atomic_json(output_dir / "failures.json", {"failures": failures})
                            raise RuntimeError(
                                "Stopping after "
                                f"{consecutive_failures} consecutive MGM transaction failures"
                            )

        assert_source_fingerprint(source_snapshot)
        if sha256_file(assets["weights_path"]) != assets["hashes"]["model"]:
            raise RuntimeError("MGM-5DB checkpoint changed during execution")
        if failures:
            _atomic_json(output_dir / "failures.json", {"failures": failures})
            raise RuntimeError(f"MGM-5DB run ended with {len(failures)} failed transactions")
        (output_dir / "failures.json").unlink(missing_ok=True)
        print(
            f"ALL DONE written={written} skipped={skipped} "
            f"elapsed={time.time() - started:.1f}s output={output_dir}",
            flush=True,
        )
    finally:
        adata.file.close()


if __name__ == "__main__":
    main()
