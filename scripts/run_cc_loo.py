"""V5 CC-LOO 全量驱动器：对 10 病 × 每病所有 LOO fold × seed 跑 MGM benchmark。

套 run_benchmark_controlled.run_controlled 单折原语；两个方案通用：
  方案 A（--variant A）：--pretrained-model 留空 → 用 MGM 自带 general_model（原 ckpt 迁移）
  方案 B（--variant B）：--pretrained-model 指向我们语料/词表两阶段 DAPT 重训的 ckpt

split 映射（对齐 MiCoFormer inner split + MGM 早停范式）：
  train = fold_XX_inner_train.npy  （实际微调数据）
  val   = fold_XX_inner_val.npy    （MGM early stop / load_best_model_at_end）
  test  = fold_XX_test.npy         （留出 study，评估）
标签 = Role_<disease>，label_values = [control, case]（case=index1=正类，与传统 ML 对齐）。

结果写 evaluation-hub 统一 schema（unified_io），可与传统 ML / MiCoFormer 并表。
断点续跑：已存在结果文件的 (disease, fold, seed) 跳过。

用法（方案 A）：
  PYTHONNOUSERSITE=1 /home/cml_lab/anaconda3/envs/caiqy_mgm_bench/bin/python \
      benchmarks/MGM-benchmark/scripts/run_cc_loo.py \
      --h5ad data/gg2/MCFCorpusV2.gg2.labeled.h5ad \
      --splits-dir data/gg2/splits/cc_loo \
      --output-dir benchmarks/MGM-benchmark/output/cc_loo_v5 \
      --variant A --seeds 42 52 62
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

# 让 run_benchmark_controlled 可被 import（它内部会把 evaluation-hub 加入 sys.path）
_MGM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MGM_ROOT))
from scripts.run_benchmark_controlled import run_controlled  # noqa: E402

DISEASES = ["COVID", "Asthma", "CRS", "COPD", "RSV", "TB", "HIV", "Influenza", "RTI", "LatentTB"]


def _list_folds(disease_dir: Path) -> list[str]:
    """返回 disease 目录下所有 fold 序号（按 fold_XX_test.npy 检测）。"""
    folds = []
    for p in sorted(disease_dir.glob("fold_*_test.npy")):
        m = re.match(r"fold_(\d+)_test\.npy", p.name)
        if m:
            folds.append(m.group(1))
    return folds


def main():
    ap = argparse.ArgumentParser(description="MGM V5 CC-LOO 全量驱动器（方案 A/B 通用）")
    ap.add_argument("--h5ad", required=True, help="labeled h5ad（含 Role_<disease>）")
    ap.add_argument("--splits-dir", required=True, help="cc_loo splits 根目录")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variant", required=True, choices=["A", "B"],
                    help="A=MGM 原 ckpt 迁移 / B=我们语料/词表重训")
    ap.add_argument("--pretrained-model", default=None,
                    help="方案 B 必填：我们重训的 MGM ckpt 目录；方案 A 留空用 general_model")
    ap.add_argument("--diseases", nargs="+", default=DISEASES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    args = ap.parse_args()

    if args.variant == "B" and not args.pretrained_model:
        ap.error("--variant B 需要 --pretrained-model（我们重训的 ckpt）")

    splits_root = Path(args.splits_dir)
    out_root = Path(args.output_dir)

    n_done = n_run = n_fail = 0
    for disease in args.diseases:
        ddir = splits_root / disease
        if not ddir.is_dir():
            print(f"[skip] no splits dir for {disease}")
            continue
        label_field = f"Role_{disease}"
        for fold in _list_folds(ddir):
            train_npy = ddir / f"fold_{fold}_train.npy"
            inner_train = ddir / f"fold_{fold}_inner_train.npy"
            inner_val = ddir / f"fold_{fold}_inner_val.npy"
            test = ddir / f"fold_{fold}_test.npy"
            if not (test.exists() and (inner_train.exists() or train_npy.exists())):
                print(f"[skip] {disease} fold {fold}: missing split files")
                continue
            # inner split：优先现成 inner_*（与 MiCoFormer 调 epoch 对齐）；否则（COVID/LatentTB
            # 无 inner）从 train.npy 现切，固定 seed=0 → 跨 model seed 一致、可复现。
            # MGM 早停用 val 的 CE loss（非 AUROC），故 inner_val 不强求双类。
            if inner_train.exists() and inner_val.exists():
                tr_idx, va_idx = np.load(inner_train), np.load(inner_val)
            else:
                full = np.load(train_npy)
                perm = np.random.RandomState(0).permutation(len(full))
                n_val = max(1, int(round(len(full) * 0.15)))
                va_idx, tr_idx = full[perm[:n_val]], full[perm[n_val:]]
            te_idx = np.load(test)
            for seed in args.seeds:
                fold_out = out_root / f"variant_{args.variant}" / disease / f"fold_{fold}_seed{seed}"
                # 断点续跑：已有结果文件则跳过
                if list(fold_out.glob("MGM_*_metrics.csv")):
                    n_done += 1
                    print(f"[done] {disease} fold {fold} seed {seed}")
                    continue
                print(f"\n=== {disease} fold {fold} seed {seed} (variant {args.variant}) ===")
                try:
                    run_controlled(
                        h5ad_path=args.h5ad,
                        train_indices=tr_idx,
                        val_indices=va_idx,
                        test_indices=te_idx,
                        label_field=label_field,
                        label_values=["control", "case"],
                        seed=seed,
                        output_dir=fold_out,
                        pretrained_model=args.pretrained_model,
                        disease_group=disease,
                        test_study=f"fold{fold}",
                        val_study=None,
                        variant=args.variant,
                    )
                    n_run += 1
                except Exception as e:
                    # 一折失败不中断整轮；未写 metrics，故下次断点续跑会重试
                    print(f"[FAIL] {disease} fold {fold} seed {seed}: {type(e).__name__}: {e}")
                    n_fail += 1

    print(f"\nFinished. ran={n_run}, skipped(done)={n_done}, failed={n_fail}")


if __name__ == "__main__":
    main()
