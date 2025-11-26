#!/bin/bash

# Discretized Transformer実験スクリプト v3 (exp1126)
# より緩やかな温度スケジューリング + 中程度の学習率
# 日付: 2025-11-26

echo "=== Discretized Transformer Training (exp1126-v3 - Smooth Temp) ==="
echo "Date: $(date)"
echo ""

# 実験設定 v3 (緩やかな温度スケジューリング)
NEPOCH=150
NSEQ=1000
NMIN=1
NMAX=20
D_MODEL=192       # 中間的なサイズ
N_HEADS=6         # 中間的なヘッド数
N_LAYERS=3        # 3層
D_FF=768          # 中間的なFF次元
TEMP_INITIAL=4.0  # 適度な初期温度
TEMP_FINAL=0.3    # 適度な最終温度
LR=0.0008         # 中程度の学習率
SEED=44           # 異なるシードを使用
LOG_DIR="discretized_logs/exp1126"

mkdir -p $LOG_DIR

echo "=== Configuration v3 ==="
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
    --log_dir $LOG_DIR 2>&1 | tee ${LOG_DIR}/training_v3.log

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
        --output "${MODEL_DIR}/generalization_results.json" 2>&1 | tee ${LOG_DIR}/testing_v3.log
    
    echo ""
    echo "=== Results Summary ==="
    echo "Model directory: $MODEL_DIR"
    echo "Training log: ${LOG_DIR}/training_v3.log"
    echo "Testing log: ${LOG_DIR}/testing_v3.log"
fi

echo ""
echo "=== Experiment v3 Completed ==="
date
