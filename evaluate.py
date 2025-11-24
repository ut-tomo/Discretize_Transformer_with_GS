"""
デリミタ(#)以降の部分のみ評価
全一致で正解にカウント
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import os
import json
from models.transformer import Transformer
from models.discretized_transformer import DiscretizedTransformer
from rc_task import generate_batch_sequences, string_to_tensor, tensor_to_string
from utils import load_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_length(model, n, args, device, num_sequences=200):
    """特定の長さnでの評価"""
    model.eval()
    total_correct = 0
    total_count = 0
    
    with torch.no_grad():
        # 長さnの固定シーケンスを生成
        sequences = generate_batch_sequences(num_sequences, n + 1, n, args.nchar)
        
        for seq in sequences:
            input_tensor = string_to_tensor(seq, args.nchar).unsqueeze(0).to(device)
            logits = model(input_tensor)
            
            # デリミタ以降の部分のみ評価
            delimiter_idx = seq.index('a')  # デリミタの位置
            predicted = logits[0, delimiter_idx:-1, :].argmax(dim=-1)
            target = input_tensor[0, delimiter_idx + 1:]
            
            # 全一致で正解にカウント
            if torch.all(predicted == target):
                total_correct += 1
            total_count += 1
    
    accuracy = total_correct / total_count
    return accuracy


def evaluate_all_lengths(model, args, device, min_len=1, max_len=20, num_sequences=200):
    """複数の長さで評価してレポートを作成"""
    model.eval()
    results = []
    
    print(f"Evaluating model on lengths from {min_len} to {max_len}...")
    
    for n in range(min_len, max_len + 1):
        accuracy = evaluate_length(model, n, args, device, num_sequences)
        results.append({
            'length': n,
            'accuracy': accuracy,
            'num_sequences': num_sequences
        })
        print(f"Length {n}: Accuracy = {accuracy:.4f}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Transformer on Reverse Copy Task')
    
    # モデル設定
    parser.add_argument('--model_type', type=str, default='transformer',
                       choices=['transformer', 'discretized'],
                       help='Type of model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    
    # 評価設定
    parser.add_argument('--min_len', type=int, default=1,
                       help='Minimum sequence length to evaluate')
    parser.add_argument('--max_len', type=int, default=20,
                       help='Maximum sequence length to evaluate')
    parser.add_argument('--num_sequences', type=int, default=200,
                       help='Number of sequences per length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 出力設定
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for evaluation results')
    
    args = parser.parse_args()
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    # 設定の読み込み
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 必要なパラメータをargsに追加
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    # モデルの初期化
    if args.model_type == 'transformer':
        model = Transformer(
            nchar=args.nchar,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout
        ).to(device)
    elif args.model_type == 'discretized':
        model = DiscretizedTransformer(
            nchar=args.nchar,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            temperature=args.temperature
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # チェックポイントの読み込み
    epoch, best_val_loss = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {epoch} (best val loss: {best_val_loss:.4f})")
    
    # 評価実行
    results = evaluate_all_lengths(model, args, device, 
                                   args.min_len, args.max_len, 
                                   args.num_sequences)
    
    # 結果の保存
    output_data = {
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'config': config,
        'evaluation_settings': {
            'min_len': args.min_len,
            'max_len': args.max_len,
            'num_sequences': args.num_sequences,
            'seed': args.seed
        },
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nEvaluation results saved to {args.output}")
    
    # 結果のサマリー
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    print(f"\nAverage accuracy across all lengths: {avg_accuracy:.4f}")

    