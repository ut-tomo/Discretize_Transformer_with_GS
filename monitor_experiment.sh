#!/bin/bash
# 実験の進捗をモニタリングするスクリプト

echo "=== Transformer Experiment Monitor ==="
echo "Date: $(date)"
echo ""

# tmuxセッションの確認
echo "=== tmux Session Status ==="
tmux list-sessions | grep transformer_experiment
echo ""

# 最新の実験ログを表示
echo "=== Latest Experiment Output (last 30 lines) ==="
tmux capture-pane -t transformer_experiment -p | tail -30
echo ""

# ログディレクトリの確認
echo "=== Log Directory Contents ==="
ls -lth logs/experiment_small/ 2>/dev/null | head -10
echo ""

# 実行中のPythonプロセスを確認
echo "=== Running Python Processes ==="
ps aux | grep -E "python.*train.py" | grep -v grep
echo ""

echo "=== Quick Access Commands ==="
echo "  tmux attach -t transformer_experiment  # セッションにアタッチ"
echo "  tmux kill-session -t transformer_experiment  # セッションを終了"
echo "  bash monitor_experiment.sh  # このスクリプトを再実行"
echo ""
