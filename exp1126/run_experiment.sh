#!/bin/bash

# Discretized Transformer実験スクリプト (exp1126)
# 改善版: より深いモデル、調整された学習率
# 日付: 2025-11-26

echo "=== Discretized Transformer Training (exp1126 - Improved) ==="
echo "Date: $(date)"
echo ""

# 実験設定 (改善版)
NEPOCH=150        # エポック数を増加
NSEQ=1000
NMIN=1
NMAX=20
D_MODEL=256       # モデル次元を増加 (128 → 256)
N_HEADS=8         # ヘッド数を増加 (4 → 8)
N_LAYERS=3        # レイヤー数を増加 (2 → 3)
D_FF=1024         # FF次元を増加 (512 → 1024)
TEMP_INITIAL=3.0  # 初期温度を下げる (5.0 → 3.0)
TEMP_FINAL=0.5    # 最終温度を上げる (0.1 → 0.5)
LR=0.0005         # 学習率を下げる (0.001 → 0.0005)
SEED=42
LOG_DIR="discretized_logs/exp1126"

# ログディレクトリ作成
mkdir -p $LOG_DIR

echo "=== Configuration ==="
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
    --log_dir $LOG_DIR 2>&1 | tee ${LOG_DIR}/training.log

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
        --output "${MODEL_DIR}/generalization_results.json" 2>&1 | tee ${LOG_DIR}/testing.log
    
    echo ""
    echo "=== Results Summary ==="
    echo "Model directory: $MODEL_DIR"
    echo "Training log: ${LOG_DIR}/training.log"
    echo "Testing log: ${LOG_DIR}/testing.log"
    echo "Config: ${MODEL_DIR}/config.json"
    echo "Best model: ${MODEL_DIR}/best_model.pt"
    echo "Generalization results: ${MODEL_DIR}/generalization_results.json"
else
    echo "Error: Model directory not found"
    exit 1
fi

echo ""
echo "=== Experiment Completed ==="
date
