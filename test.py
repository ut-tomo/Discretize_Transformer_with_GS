import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import json
import os
from models.transformer import Transformer
from models.discretized_transformer import DiscretizedTransformer
from rc_task import generate_reverse_copy_sequence, string_to_tensor
from utils import load_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_generalization(model, nchar, device, nmin=2, nmax=150, ntest=200, seed=10):

    model.eval()
    set_seed(seed)
    
    results = {
        'length': [],
        'accuracy': [],
        'perplexity': [],
        'num_correct': [],
        'num_total': []
    }
    
    with torch.no_grad():
        # 各シーケンス長でテスト
        for seq_len in range(nmin, nmax):
            correct_sequences = 0
            total_log_loss = 0.0
            total_tokens = 0
            
            # この長さでntest個のシーケンスを生成してテスト
            for _ in range(ntest):
                seq_str = generate_reverse_copy_sequence(seq_len, nchar)
                input_ids = string_to_tensor(seq_str, nchar).unsqueeze(0).to(device) 
                outputs = model(input_ids)
                
                predictions = outputs.argmax(dim=-1).squeeze(0)  # (seq_len,)
                
                targets = input_ids.squeeze(0)[1:]  # (seq_len-1,)
                preds_for_eval = predictions[:-1]  # (seq_len-1,)
                
                # デリミタの位置を見つける（最初の'a'）
                delimiter_pos = seq_str.index('a')
                # デリミタ以降を評価
                eval_start = delimiter_pos
                eval_end = len(seq_str) - 1
                
                if eval_start < eval_end:
                    # デリミタ以降の予測が全て正しいかチェック
                    eval_targets = targets[eval_start:eval_end]
                    eval_preds = preds_for_eval[eval_start:eval_end]
                    
                    if torch.all(eval_preds == eval_targets):
                        correct_sequences += 1
                    
                    # 全トークンのlog lossを計算
                    log_probs = torch.nn.functional.log_softmax(outputs.squeeze(0)[:-1], dim=-1)
                    token_log_loss = -log_probs[range(len(targets)), targets]
                    # log10に変換
                    token_log_loss = token_log_loss / np.log(10)
                    total_log_loss += token_log_loss.sum().item()
                    total_tokens += len(targets)
            
            # この長さでの精度とPerplexityを計算
            accuracy = correct_sequences / ntest
            perplexity = 10 ** (total_log_loss / total_tokens) if total_tokens > 0 else float('inf')
            
            results['length'].append(seq_len)
            results['accuracy'].append(accuracy)
            results['perplexity'].append(perplexity)
            results['num_correct'].append(correct_sequences)
            results['num_total'].append(ntest)
            
            print(f"Length {seq_len:3d}: Accuracy = {accuracy:6.2%} ({correct_sequences}/{ntest}), "
                  f"Perplexity = {perplexity:8.4f}")
    
    return results


def print_generalization_summary(results):
    """汎化結果のサマリーを表示"""
    print("\n" + "="*70)
    print("GENERALIZATION TEST SUMMARY")
    print("="*70)
    
    lengths = results['length']
    accuracies = results['accuracy']
    
    # 完全な汎化範囲を見つける
    perfect_up_to = 0
    for length, acc in zip(lengths, accuracies):
        if acc >= 0.99: 
            perfect_up_to = length
        else:
            break
    
    print(f"Perfect generalization (≥99%) up to length: {perfect_up_to}")
    
    # 精度範囲ごとの分布
    ranges = [
        (0.95, 1.00, "Excellent (95-100%)"),
        (0.75, 0.95, "Good (75-95%)"),
        (0.50, 0.75, "Moderate (50-75%)"),
        (0.00, 0.50, "Poor (<50%)")
    ]
    
    for min_acc, max_acc, label in ranges:
        count = sum(1 for acc in accuracies if min_acc <= acc < max_acc)
        if count > 0:
            lengths_in_range = [l for l, a in zip(lengths, accuracies) if min_acc <= a < max_acc]
            print(f"{label:20s}: {count:3d} lengths (e.g., {lengths_in_range[:5]})")
    
    print("="*70)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test length generalization of Transformer on Reverse Copy Task')
    
    # モデル設定
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['transformer', 'discretized'],
                       help='Type of model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    
    # テスト設定
    parser.add_argument('--nmin', type=int, default=2,
                       help='Minimum sequence length to test')
    parser.add_argument('--nmax', type=int, default=150,
                       help='Maximum sequence length to test')
    parser.add_argument('--ntest', type=int, default=200,
                       help='Number of test sequences per length')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    
    # 出力設定
    parser.add_argument('--output', type=str, default='generalization_results.json',
                       help='Output file for generalization results')
    
    args = parser.parse_args()
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
            temperature=1.0  # 評価時は温度1.0
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # チェックポイントの読み込み
    epoch, best_val_loss = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Testing length generalization from {args.nmin} to {args.nmax-1}\n")
    
    # 汎化テスト実行
    results = evaluate_generalization(
        model, 
        args.nchar, 
        device, 
        args.nmin, 
        args.nmax, 
        args.ntest, 
        args.seed
    )
    
    # サマリー表示
    print_generalization_summary(results)
    
    # 結果の保存
    output_data = {
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'config': config,
        'test_settings': {
            'nmin': args.nmin,
            'nmax': args.nmax,
            'ntest': args.ntest,
            'seed': args.seed
        },
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Generalization test results saved to {args.output}")
    
    # 統計情報
    avg_accuracy = np.mean(results['accuracy'])
    median_accuracy = np.median(results['accuracy'])
    print(f"\nOverall Statistics:")
    print(f"  Average accuracy: {avg_accuracy:.4f}")
    print(f"  Median accuracy:  {median_accuracy:.4f}")
