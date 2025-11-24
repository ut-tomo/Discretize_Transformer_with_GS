#!/bin/bash

# 小規模実験スクリプト
# Transformer と Discretized Transformer の比較

echo "=== Starting Small-Scale Experiment ==="
echo "Date: $(date)"
echo ""

# 実験設定
NEPOCH=50
NSEQ=500
NMAX=10
D_MODEL=64
N_HEADS=2
N_LAYERS=2
D_FF=256
SEED=42

# ログディレクトリの設定
LOG_DIR="logs/experiment_small"

# 1. 通常のTransformerで学習
echo "=== Training Standard Transformer ==="
python train.py \
    --model_type transformer \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_layers $N_LAYERS \
    --d_ff $D_FF \
    --dropout 0.1 \
    --nchar 3 \
    --nmin 1 \
    --nmax $NMAX \
    --nepoch $NEPOCH \
    --nseq $NSEQ \
    --lr 0.001 \
    --seed $SEED \
    --log_dir $LOG_DIR

# 最新のtransformerモデルのログディレクトリを取得
TRANSFORMER_LOG=$(ls -dt ${LOG_DIR}/transformer_* 2>/dev/null | head -n 1)

if [ -n "$TRANSFORMER_LOG" ]; then
    echo ""
    echo "=== Testing Standard Transformer Generalization ==="
    python test.py \
        --model_type transformer \
        --checkpoint "${TRANSFORMER_LOG}/best_model.pt" \
        --config "${TRANSFORMER_LOG}/config.json" \
        --nmin 2 \
        --nmax 50 \
        --ntest 100 \
        --seed 10 \
        --output "${TRANSFORMER_LOG}/generalization_results.json"
fi

echo ""
echo "=== Training Discretized Transformer ==="
# 2. 離散化Transformerで学習
python train.py \
    --model_type discretized \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_layers $N_LAYERS \
    --d_ff $D_FF \
    --dropout 0.1 \
    --temp_initial 5.0 \
    --temp_final 0.1 \
    --nchar 3 \
    --nmin 1 \
    --nmax $NMAX \
    --nepoch $NEPOCH \
    --nseq $NSEQ \
    --lr 0.001 \
    --seed $SEED \
    --log_dir $LOG_DIR

# 最新のdiscretizedモデルのログディレクトリを取得
DISCRETIZED_LOG=$(ls -dt ${LOG_DIR}/discretized_* 2>/dev/null | head -n 1)

if [ -n "$DISCRETIZED_LOG" ]; then
    echo ""
    echo "=== Testing Discretized Transformer Generalization ==="
    python test.py \
        --model_type discretized \
        --checkpoint "${DISCRETIZED_LOG}/best_model.pt" \
        --config "${DISCRETIZED_LOG}/config.json" \
        --nmin 2 \
        --nmax 50 \
        --ntest 100 \
        --seed 10 \
        --output "${DISCRETIZED_LOG}/generalization_results.json"
fi

echo ""
echo "=== Experiment Completed ==="
echo "Results saved in $LOG_DIR/"
echo ""
echo "Summary:"
if [ -n "$TRANSFORMER_LOG" ]; then
    echo "  Standard Transformer:"
    echo "    - Training log: ${TRANSFORMER_LOG}/train_log.json"
    echo "    - Generalization: ${TRANSFORMER_LOG}/generalization_results.json"
fi
if [ -n "$DISCRETIZED_LOG" ]; then
    echo "  Discretized Transformer:"
    echo "    - Training log: ${DISCRETIZED_LOG}/train_log.json"
    echo "    - Generalization: ${DISCRETIZED_LOG}/generalization_results.json"
fi
