# Experiment 1125 - Fixed Length Training

## 実験概要

カリキュラム学習を行わず、固定長（長さ20）で学習を行う実験。
Transformerモデルの並列性を活かすため、下手なスケジューリングを避けた一般的な学習手法を採用。

## 実験設定

### 共通設定
- **学習長**: 固定長20 (nmin=1, nmax=20)
- **エポック数**: 100
- **バッチサイズ**: 1000シーケンス/エポック
- **最適化**: Adam (lr=0.001)
- **シード**: 42
- **汎化テスト**: 長さ1〜60

### モデルアーキテクチャ
- **d_model**: 128
- **n_heads**: 4
- **n_layers**: 2
- **d_ff**: 512
- **dropout**: 0.1

### Discretized Transformer固有の設定
- **温度スケジューリング**: 5.0 → 0.1 (線形)

## 実験の実施

### 1. Standard Transformer
- **tmuxセッション**: `transformer_exp1125`
- **ログディレクトリ**: `transformer_logs/exp1125/`
- **モデル保存先**: `transformer_logs/exp1125/transformer_YYYYMMDD_HHMMSS/`

```bash
# セッションにアタッチ
tmux attach -t transformer_exp1125

# ログを監視
tail -f transformer_logs/exp1125/training.log
```

### 2. Discretized Transformer
- **tmuxセッション**: `discretized_exp1125`
- **ログディレクトリ**: `discretized_logs/exp1125/`
- **モデル保存先**: `discretized_logs/exp1125/discretized_YYYYMMDD_HHMMSS/`

```bash
# セッションにアタッチ
tmux attach -t discretized_exp1125

# ログを監視
tail -f discretized_logs/exp1125/training.log
```

## モニタリング

```bash
# 実験全体の状況を確認
bash monitor_exp1125.sh

# 個別のセッションを確認
tmux list-sessions | grep exp1125
```

## 期待される結果

1. **学習の安定性**: カリキュラム学習なしでも安定した学習が可能か
2. **汎化性能**: 固定長20で学習したモデルがどこまで長い系列に汎化できるか
3. **モデル比較**: 通常のTransformerとDiscretized Transformerの性能差

## ファイル構成

```
Discretize_Transformer_with_GS/
├── train_noscale.py                    # カリキュラムなしの学習スクリプト
├── transformer_logs/exp1125/
│   ├── run_experiment.sh               # Transformer実験スクリプト
│   ├── training.log                    # 学習ログ
│   ├── testing.log                     # テストログ
│   └── transformer_YYYYMMDD_HHMMSS/   # モデルとログ
│       ├── best_model.pt
│       ├── final_model.pt
│       ├── config.json
│       ├── train_log.json
│       └── generalization_results.json
└── discretized_logs/exp1125/
    ├── run_experiment.sh               # Discretized実験スクリプト
    ├── training.log                    # 学習ログ
    ├── testing.log                     # テストログ
    └── discretized_YYYYMMDD_HHMMSS/   # モデルとログ
        ├── best_model.pt
        ├── final_model.pt
        ├── config.json
        ├── train_log.json
        └── generalization_results.json
```

## 開始時刻

- Transformer: 2025-11-25 12:21:07 JST
- Discretized: 2025-11-25 12:21:16 JST

推定完了時刻: 約2〜3時間後
