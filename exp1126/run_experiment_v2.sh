#!/bin/bash

# Discretized Transformer実験スクリプト v2 (exp1126)
# より浅いモデル + 高い学習率でのテスト
# 日付: 2025-11-26

echo "=== Discretized Transformer Training (exp1126-v2 - Shallow+Fast) ==="
echo "Date: $(date)"
echo ""

# 実験設定 v2 (浅いモデル + 高い学習率)
NEPOCH=150
NSEQ=1000
NMIN=1
NMAX=20
D_MODEL=128       # 元の設定を維持
N_HEADS=4         # 元の設定を維持
N_LAYERS=4        # レイヤー数を増加 (2 → 4)
D_FF=512          # 元の設定を維持
TEMP_INITIAL=5.0  # 元の設定を維持
TEMP_FINAL=0.1    # 最終温度を低く
LR=0.002          # 学習率を上げる (0.001 → 0.002)
SEED=43           # 異なるシードを使用
LOG_DIR="discretized_logs/exp1126"

mkdir -p $LOG_DIR

echo "=== Configuration v2 ==="
echo "Model: Discretized Transformer"
echo "Epochs: $NEPOCH"
echo "Sequences per epoch: $NSEQ"
echo "Sequence length: $NMIN - $NMAX"
echo "Model dimension: $D_MODEL"
echo "Attention heads: $N_HEADS"
echo "Layers: $N_LAYERS"
echo "Feed-forward dimension: $D_FF"
echo "Learning rate: $LR"
echo "Temperature: $TEMP_INITIAL → $TEMP_FINAL"
echo "Seed: $SEED"
echo ""

# 学習実行
echo "=== Training Started ==="
python train_noscale.py \
    --model_type discretized \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_layers $N_LAYERS \
    --d_ff $D_FF \
    --dropout 0.1 \
    --temp_initial $TEMP_INITIAL \
    --temp_final $TEMP_FINAL \
    --nchar 3 \
    --nmin $NMIN \
    --nmax $NMAX \
    --nepoch $NEPOCH \
    --nseq $NSEQ \
    --lr $LR \
    --seed $SEED \
    --log_dir $LOG_DIR 2>&1 | tee ${LOG_DIR}/training_v2.log

# 最新のモデルディレクトリを取得
MODEL_DIR=$(ls -dt ${LOG_DIR}/discretized_* 2>/dev/null | head -n 1)

if [ -n "$MODEL_DIR" ]; then
    echo ""
    echo "=== Testing Generalization (up to length 60) ==="
    python test.py \
        --model_type discretized \
        --checkpoint "${MODEL_DIR}/best_model.pt" \
        --config "${MODEL_DIR}/config.json" \
        --nmin 1 \
        --nmax 60 \
        --ntest 200 \
        --seed 10 \
        --output "${MODEL_DIR}/generalization_results.json" 2>&1 | tee ${LOG_DIR}/testing_v2.log
    
    echo ""
    echo "=== Results Summary ==="
    echo "Model directory: $MODEL_DIR"
    echo "Training log: ${LOG_DIR}/training_v2.log"
    echo "Testing log: ${LOG_DIR}/testing_v2.log"
fi

echo ""
echo "=== Experiment v2 Completed ==="
date
