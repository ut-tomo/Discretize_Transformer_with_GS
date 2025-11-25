# TransformerのAttentionの離散化

## ディレクトリ構造
```
discretized_transformer_with_GS/
    ├── models/
    │   ├── discretized_transformer.py  # Gumbel-Softmax離散化モデル
    │   └── transformer.py              # 通常のTransformer
    ├── train.py           # 学習スクリプト
    ├── test.py            # 汎化性能テストスクリプト
    ├── rc_task.py         # Reverse Copyタスクのデータ生成
    ├── utils.py           # ユーティリティ関数
    └── run_experiment.sh  # 実験実行スクリプト
```

## タスク
**Reverse Copy Task**: w + '#' + w' の形式
- w: ランダムな文字列（デリミタ'#'を除く文字を使用）
- '#': デリミタ（文字 'a' に対応）
- w': w の逆順
- 例: `nchar=3` の場合、`'bbc' + 'a' + 'cbb'` → `'bbcacbb'`

学習長のカリキュラム: 初期長さ3から開始し、最大20までスケジューリング

## パラメータ説明
- `nchar`: 文字の種類数（デリミタ 'a' を含む。例: nchar=3 → {'a', 'b', 'c'}）
- `nmax`: デリミタ前後の片側の最大長さ（全体の長さは 2*nmax+1）
- `nmin`: デリミタ前後の片側の最小長さ
