#!/bin/bash

# Discretized Transformer実験スクリプト (固定長学習)
# 日付: 2025-11-25
# 学習長: 20固定
# 汎化テスト: 60まで

echo "=== Discretized Transformer Training (Fixed Length) ==="
echo "Date: $(date)"
echo "Log directory: discretized_logs/exp1125"
echo ""

# 実験設定
NEPOCH=100
NSEQ=1000
NMIN=1
NMAX=20      # 固定長
D_MODEL=128
N_HEADS=4
N_LAYERS=2
D_FF=512
TEMP_INITIAL=5.0
TEMP_FINAL=0.1
SEED=42
LOG_DIR="discretized_logs/exp1125"

# 学習実行
echo "=== Training Discretized Transformer ==="
echo "Temperature scheduling: $TEMP_INITIAL → $TEMP_FINAL"
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
    --lr 0.001 \
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
    echo "Generalization results: ${MODEL_DIR}/generalization_results.json"
else
    echo "Error: Model directory not found"
    exit 1
fi

echo ""
echo "=== Experiment Completed ==="
date
