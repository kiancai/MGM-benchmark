#!/bin/bash
# =============================================================================
# MGM Benchmark 端到端运行脚本
#
# 用法:
#   bash scripts/run_benchmark.sh <h5ad_path> <split_dir> <output_dir> [label_field] [label_values...]
#
# 示例 (kfold):
#   bash scripts/run_benchmark.sh \
#       ../../data/processed/microbiome_dataset.h5ad \
#       ../../data/splits/kfold/ \
#       output/kfold/ \
#       Phenotype Health Disease
#
# 示例 (OOD, 单次分割):
#   bash scripts/run_benchmark.sh \
#       ../../data/processed/microbiome_dataset.h5ad \
#       ../../data/splits/ood/ \
#       output/ood/ \
#       Phenotype Health Disease
# =============================================================================

set -euo pipefail

H5AD="${1:?Usage: $0 <h5ad> <split_dir> <output_dir> [label_field] [label_values...]}"
SPLIT_DIR="${2:?}"
OUTPUT_DIR="${3:?}"
LABEL_FIELD="${4:-Phenotype}"
shift 4 2>/dev/null || true
LABEL_VALUES=("$@")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# --- 判断 kfold 还是 OOD 模式 ---
if ls "$SPLIT_DIR"/fold_*_train.npy &>/dev/null; then
    MODE="kfold"
    echo "=== Detected kfold mode ==="
elif [ -f "$SPLIT_DIR/train.npy" ]; then
    MODE="ood"
    echo "=== Detected OOD mode ==="
else
    echo "Error: cannot detect split mode from $SPLIT_DIR"
    exit 1
fi

# --- 构建 label-values 参数 ---
LV_ARGS=""
if [ ${#LABEL_VALUES[@]} -gt 0 ]; then
    LV_ARGS="--label-values ${LABEL_VALUES[*]}"
fi

# --- Step 1: 转换数据 ---
echo ""
echo "=== Step 1: Converting h5ad to MGM format ==="

if [ "$MODE" = "kfold" ]; then
    python scripts/convert_h5ad.py \
        --h5ad "$H5AD" \
        --kfold-dir "$SPLIT_DIR" \
        --label-field "$LABEL_FIELD" $LV_ARGS \
        --output-dir "$OUTPUT_DIR"

    # --- Step 2 & 3: 逐 fold 构建 corpus + finetune + predict ---
    for FOLD_DIR in "$OUTPUT_DIR"/fold_*/; do
        FOLD_NAME=$(basename "$FOLD_DIR")
        echo ""
        echo "=== Processing $FOLD_NAME ==="

        # 构建 corpus
        echo "  Building train corpus..."
        mgm construct -i "$FOLD_DIR/train_abundance.csv" -o "$FOLD_DIR/train_corpus.pkl"
        echo "  Building val corpus..."
        mgm construct -i "$FOLD_DIR/val_abundance.csv" -o "$FOLD_DIR/val_corpus.pkl"

        # finetune
        echo "  Finetuning..."
        mgm finetune \
            --train-corpus "$FOLD_DIR/train_corpus.pkl" \
            --val-corpus "$FOLD_DIR/val_corpus.pkl" \
            -l "$FOLD_DIR/train_labels.csv" \
            --val-labels "$FOLD_DIR/val_labels.csv" \
            -o "$FOLD_DIR/model/" \
            -H "$FOLD_DIR/log/"

        # predict + evaluate
        echo "  Evaluating on val set..."
        mgm predict \
            -i "$FOLD_DIR/val_corpus.pkl" \
            -m "$FOLD_DIR/model/" \
            -l "$FOLD_DIR/val_labels.csv" \
            -o "$FOLD_DIR/predictions/" \
            -E
    done

else
    # OOD 模式
    python scripts/convert_h5ad.py \
        --h5ad "$H5AD" \
        --train-indices "$SPLIT_DIR/train.npy" \
        --val-indices "$SPLIT_DIR/val.npy" \
        --test-indices "$SPLIT_DIR/test.npy" \
        --label-field "$LABEL_FIELD" $LV_ARGS \
        --output-dir "$OUTPUT_DIR"

    # 构建 corpus
    echo ""
    echo "=== Step 2: Building corpus ==="
    for SPLIT in train val test; do
        if [ -f "$OUTPUT_DIR/${SPLIT}_abundance.csv" ]; then
            echo "  Building $SPLIT corpus..."
            mgm construct -i "$OUTPUT_DIR/${SPLIT}_abundance.csv" -o "$OUTPUT_DIR/${SPLIT}_corpus.pkl"
        fi
    done

    # finetune
    echo ""
    echo "=== Step 3: Finetuning ==="
    mgm finetune \
        --train-corpus "$OUTPUT_DIR/train_corpus.pkl" \
        --val-corpus "$OUTPUT_DIR/val_corpus.pkl" \
        -l "$OUTPUT_DIR/train_labels.csv" \
        --val-labels "$OUTPUT_DIR/val_labels.csv" \
        -o "$OUTPUT_DIR/model/" \
        -H "$OUTPUT_DIR/log/"

    # predict + evaluate
    echo ""
    echo "=== Step 4: Evaluating ==="
    for SPLIT in val test; do
        if [ -f "$OUTPUT_DIR/${SPLIT}_corpus.pkl" ]; then
            echo "  Evaluating on $SPLIT set..."
            mgm predict \
                -i "$OUTPUT_DIR/${SPLIT}_corpus.pkl" \
                -m "$OUTPUT_DIR/model/" \
                -l "$OUTPUT_DIR/${SPLIT}_labels.csv" \
                -o "$OUTPUT_DIR/${SPLIT}_predictions/" \
                -E
        fi
    done
fi

echo ""
echo "=== Done! ==="
