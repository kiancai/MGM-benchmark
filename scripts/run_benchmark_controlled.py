"""
MGM 受控 benchmark runner：单折执行。

给定 train/val/test 索引文件，完整执行 MGM 的训练和评估流程：
  1. convert_h5ad → 生成 MGM 格式的 abundance CSV + labels CSV
  2. mgm construct → 生成 corpus pkl
  3. mgm finetune  → 从 pretrained_model 出发，在 train 上微调，val 上 early stop
  4. mgm predict   → 在 test 上推理
  5. 读取 y_score.csv → softmax → 统一指标计算 → 保存 fold-level 结果

本脚本是单折原语，Stage 3 LOO 的循环由 run_second_finetune.py 调用。

Usage:
  # Stage 2: A/B 微调，不评估 C（--test-indices 省略则跳过 predict）
  python benchmarks/MGM-benchmark/scripts/run_benchmark_controlled.py \\
      --h5ad       data/processed/microbiome_dataset_benchmark.h5ad \\
      --train-indices data/processed/splits/split_group_A.npy \\
      --val-indices   data/processed/splits/split_group_B.npy \\
      --label-field   Is_Healthy_benchmark \\
      --label-values  False True \\
      --seed 42 \\
      --output-dir benchmarks/MGM-benchmark/output/stage2/seed42

  # Stage 3 LOO fold（run_second_finetune.py 会用到）
  python benchmarks/MGM-benchmark/scripts/run_benchmark_controlled.py \\
      --h5ad       data/processed/microbiome_dataset_benchmark.h5ad \\
      --train-indices /tmp/fold_train.npy \\
      --val-indices   /tmp/fold_val.npy \\
      --test-indices  /tmp/fold_test.npy \\
      --pretrained-model benchmarks/MGM-benchmark/output/stage2/seed42/model \\
      --label-field Is_Healthy_benchmark \\
      --label-values False True \\
      --seed 42 \\
      --disease-group COPD \\
      --test-study PRJNA1108737 \\
      --val-study PRJNA282010 \\
      --output-dir benchmarks/MGM-benchmark/output/stage3/seed42
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special

# 将 evaluation-hub 加入路径以使用统一指标和 IO 工具
_BENCH_ROOT = Path(__file__).resolve().parents[2]  # benchmarks/
_EVAL_HUB = _BENCH_ROOT / "evaluation-hub"
sys.path.insert(0, str(_EVAL_HUB))

from utils.metrics import compute_unified_metrics
from utils.unified_io import save_fold_metrics, make_result_filename

# 将 MGM 包路径加入以使用 convert_h5ad 函数
_MGM_ROOT = Path(__file__).resolve().parents[1]  # benchmarks/MGM-benchmark/
sys.path.insert(0, str(_MGM_ROOT))

from scripts.convert_h5ad import _build_genus_counts, _save_split


def _run_mgm_cmd(args_list: list[str], cwd: Path) -> None:
    """调用 MGM CLI，失败时抛出异常。

    注意：MGM 的 mgm/CLI/main.py 把模块级 main() 注释掉了，靠 console script
    `mgm`（entry point mgm.CLI:main）做入口，所以 `python -m mgm.CLI.main` 只会
    import 模块而不执行（静默返回 0、不干活）。这里用 `-c` 显式调 main()，
    既真正执行、又保证用当前 env 的 python（不依赖 console script 是否在 PATH）。
    """
    cmd = [sys.executable, "-c", "from mgm.CLI import main; main()"] + args_list
    print(f"  Running: mgm {' '.join(args_list)}")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"MGM command failed (exit {result.returncode}): mgm {' '.join(args_list)}")


def _convert_indices_to_mgm(
    h5ad_path: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray | None,
    label_field: str,
    label_values: list[str],
    work_dir: Path,
) -> None:
    """
    将 h5ad 的三个 split 转换为 MGM 所需的 abundance CSV + labels CSV。

    对 label_values 做字符串化处理（True/False → "True"/"False"），
    并过滤掉无有效标签（NaN / 不在 label_values 内）的样本。
    """
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")

    # 先把 train/val/test 样本并集子集化到内存，再构 genus 矩阵 ——
    # 避免对全量 (~1.83M × 8114) backed counts 直接 toarray（会 OOM ~118GB）。
    # 下游每折只有几百~几千样本，子集后内存无压力。
    split_items = [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ]
    all_idx = np.unique(np.concatenate([idx for _, idx in split_items if idx is not None]))
    sub = adata[all_idx].to_memory()
    counts_df = _build_genus_counts(sub)  # index = 子集样本 ID（含全部 train/val/test）

    for split_name, indices in split_items:
        if indices is None:
            continue

        sample_ids = adata.obs_names[indices]
        raw_labels = adata.obs[label_field].iloc[indices]

        # 标准化为字符串，过滤非有效标签（NaN 等）
        valid_mask = raw_labels.isin([True, False]) | raw_labels.isin(label_values)
        if not valid_mask.all():
            n_dropped = (~valid_mask).sum()
            print(f"  WARNING: dropping {n_dropped} samples with invalid {label_field} in {split_name}")

        valid_ids = sample_ids[valid_mask.values]
        valid_labels = raw_labels[valid_mask.values].astype(str)

        # abundance CSV（taxa 为行，samples 为列）
        split_counts = counts_df.loc[valid_ids].T
        abu_path = work_dir / f"{split_name}_abundance.csv"
        split_counts.to_csv(str(abu_path))

        # labels CSV
        label_path = work_dir / f"{split_name}_labels.csv"
        valid_labels.to_frame(name=label_field).to_csv(str(label_path))

        print(f"  {split_name}: {len(valid_ids)} samples → {abu_path.name}, {label_path.name}")


def _read_y_score_as_proba(y_score_path: Path, label_values: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    读取 MGM predict 输出的 y_score.csv，转换为 softmax 概率。

    y_score.csv 格式：index=sample_id, columns=标签名（如 "True"/"False"）。
    返回 (y_proba, ordered_classes)，y_proba 形状为 (n_samples, n_classes)。
    """
    df = pd.read_csv(y_score_path, index_col=0)
    # 按 label_values 顺序排列列（保证 0=False, 1=True）
    str_values = [str(v) for v in label_values]
    available = [c for c in str_values if c in df.columns]
    if len(available) < len(str_values):
        missing = set(str_values) - set(df.columns)
        raise ValueError(f"y_score.csv missing columns: {missing}. Available: {list(df.columns)}")

    logits = df[str_values].values.astype(float)
    y_proba = scipy.special.softmax(logits, axis=1)
    return y_proba, str_values


def run_controlled(
    h5ad_path: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray | None,
    label_field: str,
    label_values: list,
    seed: int,
    output_dir: Path,
    pretrained_model: str | None,
    disease_group: str | None,
    test_study: str | None,
    val_study: str | None,
    variant: str | None = None,
) -> dict | None:
    """
    执行单折完整 MGM 流程，返回 fold-level 指标 dict（无 test 时返回 None）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mgm_root = Path(__file__).resolve().parents[1]

    # ─── Step 1: 数据转换 ───────────────────────────────────────────────────
    print("Step 1: Converting h5ad to MGM format...")
    _convert_indices_to_mgm(
        h5ad_path, train_indices, val_indices, test_indices,
        label_field, label_values, output_dir
    )

    # ─── Step 2: construct corpus ──────────────────────────────────────────
    print("Step 2: Constructing corpus...")
    for split in ["train", "val"] + (["test"] if test_indices is not None else []):
        abu = output_dir / f"{split}_abundance.csv"
        pkl = output_dir / f"{split}_corpus.pkl"
        if abu.exists():
            _run_mgm_cmd(["construct", "-i", str(abu), "-o", str(pkl)], cwd=mgm_root)

    # ─── Step 3: finetune ─────────────────────────────────────────────────
    print("Step 3: Finetuning...")
    model_path = pretrained_model or str(
        mgm_root / "mgm" / "resources" / "general_model"
    )
    model_out = str(output_dir / "model")
    log_out = str(output_dir / "log")

    finetune_args = [
        "finetune",
        "--train-corpus", str(output_dir / "train_corpus.pkl"),
        "--val-corpus",   str(output_dir / "val_corpus.pkl"),
        "-l",             str(output_dir / "train_labels.csv"),
        "--val-labels",   str(output_dir / "val_labels.csv"),
        "-m",             model_path,
        "-o",             model_out,
        "-H",             log_out,
        "--seed",         str(seed),
    ]
    _run_mgm_cmd(finetune_args, cwd=mgm_root)

    # ─── Step 4: predict on test ──────────────────────────────────────────
    if test_indices is None:
        print("No test indices provided, skipping predict step.")
        return None

    print("Step 4: Predicting on test set...")
    test_pred_dir = str(output_dir / "test_predictions")
    predict_args = [
        "predict",
        "-i", str(output_dir / "test_corpus.pkl"),
        "-m", model_out,
        "-l", str(output_dir / "test_labels.csv"),
        "-o", test_pred_dir,
        "-E",
        "--seed", str(seed),
    ]
    _run_mgm_cmd(predict_args, cwd=mgm_root)

    # ─── Step 5: 读取结果，计算统一指标 ──────────────────────────────────────
    print("Step 5: Computing unified metrics...")
    y_score_path = Path(test_pred_dir) / "y_score.csv"
    if not y_score_path.exists():
        raise FileNotFoundError(f"y_score.csv not found at {y_score_path}")

    y_proba, _ = _read_y_score_as_proba(y_score_path, label_values)

    # 读取 test 真实标签（以字符串形式存储，转为 int）
    test_labels_df = pd.read_csv(output_dir / "test_labels.csv", index_col=0)
    str_values = [str(v) for v in label_values]
    label_to_id = {v: i for i, v in enumerate(str_values)}
    y_true = test_labels_df.iloc[:, 0].astype(str).map(label_to_id).values

    metrics = compute_unified_metrics(y_true, y_proba)

    # V5 统一 schema（evaluation-hub/utils/unified_io.RESULT_COLUMNS）：用 disease 键，
    # 去掉旧的 disease_group / val_study（CC-LOO 无 val study）。
    # variant 区分两条对比线：A=pretrained（MGM 原 ckpt 迁移）/ B=retrained（我们语料重训）。
    result = {
        "model_name": "MGM",
        "variant": variant,
        "seed": seed,
        "disease": disease_group or "ALL",
        "test_study": test_study or "unknown",
        "n_test": len(y_true),
        **metrics,
    }

    # 保存 fold-level 结果
    fname = make_result_filename(
        "MGM", variant, seed,
        disease_group or "ALL",
        test_study or "unknown",
    )
    save_fold_metrics(result, output_dir / fname)

    auroc_str = f"{metrics['auroc']:.4f}" if metrics["auroc"] is not None else "NA"
    print(
        f"  macro_f1={metrics['macro_f1']:.4f} auroc={auroc_str} "
        f"acc={metrics['accuracy']:.4f}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="MGM 受控 benchmark runner（单折）")
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--train-indices", required=True)
    parser.add_argument("--val-indices", required=True)
    parser.add_argument("--test-indices", default=None)
    parser.add_argument("--label-field", default="Is_Healthy_benchmark")
    parser.add_argument("--label-values", nargs="+", default=["False", "True"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--pretrained-model",
        default=None,
        help="MGM 模型目录（Stage 3 传入 Stage 2 输出的 model/；默认使用 MGM 自带预训练权重）",
    )
    parser.add_argument("--disease-group", default=None, help="病种组名称（用于结果文件命名）")
    parser.add_argument("--test-study", default=None, help="test study ID（用于结果文件命名）")
    parser.add_argument("--val-study", default=None, help="val study ID（记录在结果中）")
    parser.add_argument(
        "--variant", default=None,
        help="方案变体标识，写入结果 schema 与文件名以区分两条对比线："
             "A=pretrained（MGM 原 ckpt 迁移）/ B=retrained（我们语料/词表两阶段 DAPT 重训）",
    )
    args = parser.parse_args()

    train_indices = np.load(args.train_indices)
    val_indices = np.load(args.val_indices)
    test_indices = np.load(args.test_indices) if args.test_indices else None

    run_controlled(
        h5ad_path=args.h5ad,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        label_field=args.label_field,
        label_values=args.label_values,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        pretrained_model=args.pretrained_model,
        disease_group=args.disease_group,
        test_study=args.test_study,
        val_study=args.val_study,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
