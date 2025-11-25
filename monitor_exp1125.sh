#!/bin/bash
# exp1125実験のモニタリングスクリプト

echo "=== Experiment 1125 Monitor ==="
echo "Date: $(date)"
echo ""

# tmuxセッションの確認
echo "=== tmux Sessions ==="
tmux list-sessions 2>/dev/null | grep exp1125 || echo "No exp1125 sessions found"
echo ""

# Transformer実験の状況
echo "=== Transformer Experiment ==="
if tmux has-session -t transformer_exp1125 2>/dev/null; then
    echo "Status: Running"
    echo "Latest output:"
    tmux capture-pane -t transformer_exp1125 -p | tail -15
else
    echo "Status: Finished or not started"
    if [ -f "transformer_logs/exp1125/training.log" ]; then
        echo "Training log (last 10 lines):"
        tail -10 transformer_logs/exp1125/training.log
    fi
fi
echo ""

# Discretized Transformer実験の状況
echo "=== Discretized Transformer Experiment ==="
if tmux has-session -t discretized_exp1125 2>/dev/null; then
    echo "Status: Running"
    echo "Latest output:"
    tmux capture-pane -t discretized_exp1125 -p | tail -15
else
    echo "Status: Finished or not started"
    if [ -f "discretized_logs/exp1125/training.log" ]; then
        echo "Training log (last 10 lines):"
        tail -10 discretized_logs/exp1125/training.log
    fi
fi
echo ""

# ログディレクトリの確認
echo "=== Log Directories ==="
echo "Transformer logs:"
ls -lth transformer_logs/exp1125/ 2>/dev/null | head -5
echo ""
echo "Discretized logs:"
ls -lth discretized_logs/exp1125/ 2>/dev/null | head -5
echo ""

echo "=== Quick Commands ==="
echo "  tmux attach -t transformer_exp1125   # Transformerセッションにアタッチ"
echo "  tmux attach -t discretized_exp1125   # Discretizedセッションにアタッチ"
echo "  tail -f transformer_logs/exp1125/training.log   # Transformerログを監視"
echo "  tail -f discretized_logs/exp1125/training.log   # Discretizedログを監視"
echo "  bash monitor_exp1125.sh   # このスクリプトを再実行"
echo ""
