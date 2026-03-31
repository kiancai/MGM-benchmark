"""
将 MiCoFormer 的 h5ad 数据转换为 MGM 可用的 CSV 格式。

输出：
  - {split}_abundance.csv : taxa 为行、samples 为列的计数矩阵（MGM 期望的输入格式）
  - {split}_labels.csv    : index=样本ID、单列=标签值
  - coverage_report.txt   : h5ad genera vs MGM phylogeny.csv 的覆盖率报告

用法示例（单次分割）：
  python scripts/convert_h5ad.py \\
      --h5ad ../../data/processed/microbiome_dataset.h5ad \\
      --train-indices splits/fold_0_train.npy \\
      --val-indices   splits/fold_0_val.npy \\
      --label-field Phenotype --label-values Health Disease \\
      --output-dir output/fold_0/

用法示例（kfold 自动检测）：
  python scripts/convert_h5ad.py \\
      --h5ad ../../data/processed/microbiome_dataset.h5ad \\
      --kfold-dir splits/ \\
      --label-field Phenotype --label-values Health Disease \\
      --output-dir output/
"""

import argparse
import os
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _load_phylogeny_genera(phylogeny_path: str) -> set:
    """从 MGM 的 phylogeny.csv 加载已知 genus 集合。"""
    phylo = pd.read_csv(phylogeny_path, index_col=0)
    return set(phylo.index)


def _build_genus_counts(adata: ad.AnnData) -> pd.DataFrame:
    """
    从 AnnData 提取 genus 级别的原始计数矩阵。

    返回 DataFrame: rows=samples, columns=g__GenusName, values=int counts
    """
    # 取原始计数
    from scipy import sparse

    counts_mat = adata.layers["counts"]
    if sparse.issparse(counts_mat):
        counts_mat = counts_mat.toarray()

    genus_names = adata.var["Genus"].values

    counts_df = pd.DataFrame(
        counts_mat, index=adata.obs_names, columns=genus_names
    )

    # 去除无效 genus（空值、NaN、__UNK__ 等非 g__ 开头的）
    valid_mask = counts_df.columns.str.match(r"^g__[A-Za-z0-9_]+$")
    dropped = counts_df.columns[~valid_mask].tolist()
    if dropped:
        print(f"Dropped {len(dropped)} invalid genus entries: {dropped[:10]}...")
    counts_df = counts_df.loc[:, valid_mask]

    # 同 genus 多 taxa 合并（sum）
    counts_df = counts_df.T.groupby(level=0).sum().T

    return counts_df


def _check_coverage(
    genus_set: set, phylogeny_genera: set, output_dir: str
) -> None:
    """检查 h5ad genera 与 MGM phylogeny 的覆盖率，输出报告。"""
    matched = genus_set & phylogeny_genera
    unmatched = genus_set - phylogeny_genera

    report_lines = [
        "=== Genus Coverage Report ===",
        f"Total genera in h5ad: {len(genus_set)}",
        f"Total genera in MGM phylogeny: {len(phylogeny_genera)}",
        f"Matched: {len(matched)} ({100 * len(matched) / len(genus_set):.1f}%)",
        f"Unmatched (will be dropped by MGM): {len(unmatched)}",
    ]
    if unmatched:
        report_lines.append(f"Unmatched genera: {sorted(unmatched)}")

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = os.path.join(output_dir, "coverage_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"Coverage report saved to {report_path}")


def _save_split(
    counts_df: pd.DataFrame,
    labels_series: pd.Series,
    indices: np.ndarray,
    split_name: str,
    output_dir: str,
) -> None:
    """保存单个 split 的丰度 CSV 和标签 CSV。"""
    sample_ids = counts_df.index[indices]

    # 丰度 CSV：taxa 为行、samples 为列（MGM 读入后会 .T）
    split_counts = counts_df.loc[sample_ids].T  # [genera, samples]
    abu_path = os.path.join(output_dir, f"{split_name}_abundance.csv")
    split_counts.to_csv(abu_path)
    print(f"  {split_name} abundance: {split_counts.shape[1]} samples x {split_counts.shape[0]} genera -> {abu_path}")

    # 标签 CSV：index=样本ID, 列=标签
    split_labels = labels_series.loc[sample_ids]
    label_path = os.path.join(output_dir, f"{split_name}_labels.csv")
    split_labels.to_frame().to_csv(label_path)
    print(f"  {split_name} labels: {len(split_labels)} samples -> {label_path}")


def _detect_kfold_files(kfold_dir: str) -> list:
    """自动检测 kfold 目录中的 fold 文件，返回 [(fold_i, train_path, val_path), ...]。"""
    folds = []
    fold_pattern = re.compile(r"fold_(\d+)_train\.npy")
    for fname in sorted(os.listdir(kfold_dir)):
        m = fold_pattern.match(fname)
        if m:
            fold_idx = int(m.group(1))
            train_path = os.path.join(kfold_dir, fname)
            val_path = os.path.join(kfold_dir, f"fold_{fold_idx}_val.npy")
            if os.path.exists(val_path):
                folds.append((fold_idx, train_path, val_path))
    return folds


def main():
    parser = argparse.ArgumentParser(
        description="Convert MiCoFormer h5ad to MGM-compatible CSV format"
    )
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--label-field", default="Phenotype", help="Label column in obs (default: Phenotype)"
    )
    parser.add_argument(
        "--label-values", nargs="+", default=None,
        help="Accepted label values (others become NaN and are excluded). If not set, use all values."
    )
    parser.add_argument(
        "--phylogeny", default=None,
        help="Path to MGM phylogeny.csv (default: auto-detect from package)"
    )

    # 分割方式：单次 or kfold
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--kfold-dir", help="Directory containing fold_*_train.npy / fold_*_val.npy")
    split_group.add_argument("--train-indices", help="Path to train indices .npy file (single split mode)")

    parser.add_argument("--val-indices", help="Path to val indices .npy file")
    parser.add_argument("--test-indices", help="Path to test indices .npy file")

    args = parser.parse_args()

    # --- 加载数据 ---
    print(f"Loading h5ad: {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    print(f"  Samples: {adata.n_obs}, Taxa: {adata.n_vars}")

    # --- 构建 genus 计数矩阵 ---
    counts_df = _build_genus_counts(adata)
    print(f"  Genus-level counts: {counts_df.shape[0]} samples x {counts_df.shape[1]} genera")

    # --- 覆盖率检查 ---
    if args.phylogeny is None:
        # 尝试自动找到 phylogeny.csv
        script_dir = Path(__file__).resolve().parent
        default_phylo = script_dir.parent / "mgm" / "resources" / "phylogeny.csv"
        if default_phylo.exists():
            args.phylogeny = str(default_phylo)
        else:
            print("Warning: phylogeny.csv not found, skipping coverage check")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.phylogeny:
        phylo_genera = _load_phylogeny_genera(args.phylogeny)
        _check_coverage(set(counts_df.columns), phylo_genera, args.output_dir)

    # --- 构建标签 ---
    labels_series = adata.obs[args.label_field].copy()
    if args.label_values:
        # 只保留指定的标签值
        mask = labels_series.isin(args.label_values)
        labels_series[~mask] = np.nan
        print(f"  Label filter: kept {mask.sum()} / {len(mask)} samples with values in {args.label_values}")

    # --- 按分割导出 ---
    if args.kfold_dir:
        folds = _detect_kfold_files(args.kfold_dir)
        if not folds:
            raise FileNotFoundError(f"No fold files found in {args.kfold_dir}")
        print(f"  Detected {len(folds)} folds")

        for fold_idx, train_path, val_path in folds:
            fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)
            print(f"\n--- Fold {fold_idx} ---")

            train_idx = np.load(train_path)
            val_idx = np.load(val_path)

            _save_split(counts_df, labels_series, train_idx, "train", fold_dir)
            _save_split(counts_df, labels_series, val_idx, "val", fold_dir)
    else:
        # 单次分割模式
        train_idx = np.load(args.train_indices)
        _save_split(counts_df, labels_series, train_idx, "train", args.output_dir)

        if args.val_indices:
            val_idx = np.load(args.val_indices)
            _save_split(counts_df, labels_series, val_idx, "val", args.output_dir)

        if args.test_indices:
            test_idx = np.load(args.test_indices)
            _save_split(counts_df, labels_series, test_idx, "test", args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
