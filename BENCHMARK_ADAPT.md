# MiCoFormer Benchmark 适配说明

本 fork 在原版 MGM 基础上增加了与 MiCoFormer 项目 benchmark 对接的**胶水代码**，
**不改动 `mgm/` 包的核心算法**（保持上游行为，公平对比）。
完整对比协议 / 词表错配公平性 / 两条线设计见 MiCoFormer 工作区
`.claude/docs/benchmarks/mgm_benchmark.md`。

## 改动一览（相对上游 HUST-NingKang-Lab/MGM）
- `requirements.txt`：`accelerate` → `0.21.0`（0.20.1 在目标机装不通）。
- `scripts/convert_h5ad.py`：h5ad → MGM CSV 桥接（上游无）。
- `scripts/run_benchmark_controlled.py`：单折受控 runner（h5ad+索引 → convert → construct → finetune → predict → 统一指标）。
  - V5 结果 schema（`disease` 键）、`--variant` 区分两条对比线、`_convert_indices_to_mgm` 子集化避 OOM、`_run_mgm_cmd` 用 `-c "from mgm.CLI import main; main()"` 入口（原 `-m mgm.CLI.main` 因 `main()` 被注释而静默不执行）。
- `scripts/run_cc_loo.py`：10 病 × LOO fold × seed 全量驱动器（A/B 通用）。
- `mgm/` 包：**未改动核心算法**。

## 两条对比线
- **A（原 ckpt 迁移）**：用自带 `general_model` 直接微调。词表错配下只覆盖人体队列约六成丰度——这是 as-published 的固有局限，报告必带覆盖率。
- **B（同范式重训）**：用目标语料词表 + 两阶段预训练重训后微调（`construct` streaming 改造待做，避免百万样本 OOM）。

## 环境（`caiqy_mgm_bench`）
```bash
conda create -n caiqy_mgm_bench python=3.10 -y
# requirements.txt 已含 accelerate==0.21.0
pip install .
pip install --only-binary=:all: "anndata>=0.11"   # 用预编译 wheel，避免 h5py 源码编译（缺系统 HDF5）
```

## 跑法（方案 A）
```bash
PYTHONNOUSERSITE=1 python scripts/run_cc_loo.py \
    --h5ad   <labeled.h5ad>  --splits-dir <cc_loo_dir> \
    --output-dir <out>  --variant A  --seeds 42 52 62
```
方案 B：`--variant B --pretrained-model <我们重训的 ckpt 目录>`。
