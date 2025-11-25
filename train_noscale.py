import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import argparse
import os 
import json
from datetime import datetime
from models.transformer import Transformer
from models.discretized_transformer import DiscretizedTransformer
from rc_task import generate_batch_sequences, string_to_tensor, tensor_to_string
from utils import save_checkpoint, save_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, optimizer, args, device):
    """1エポックの学習 - シンプルな実装"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0
    
    # 固定長のバッチを生成
    sequences = generate_batch_sequences(args.nseq, args.nmax, args.nmin, args.nchar)
    
    for seq in sequences:
        # 入力と正解ラベルを準備
        input_tensor = string_to_tensor(seq, args.nchar).unsqueeze(0).to(device)
        
        # Forward pass
        logits = model(input_tensor)
        
        # 損失計算（各位置で次のトークンを予測）
        loss = 0
        seq_len = input_tensor.size(1)
        for i in range(seq_len - 1):
            loss += F.cross_entropy(logits[0, i, :], input_tensor[0, i + 1])
        loss = loss / (seq_len - 1)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 正解率の計算（デリミタ以降の部分のみ）
        delimiter_idx = seq.index('a')
        predicted = logits[0, delimiter_idx:-1, :].argmax(dim=-1)
        target = input_tensor[0, delimiter_idx + 1:]
        
        if torch.all(predicted == target):
            total_correct += 1
        total_count += 1
    
    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count
    
    return avg_loss, accuracy


def validate(model, args, device, num_sequences=1000):
    """検証セットでの評価"""
    model.eval()
    total_correct = 0
    total_count = 0
    
    with torch.no_grad():
        sequences = generate_batch_sequences(num_sequences, args.nmax, args.nmin, args.nchar)
        
        for seq in sequences:
            input_tensor = string_to_tensor(seq, args.nchar).unsqueeze(0).to(device)
            logits = model(input_tensor)
            
            # デリミタ以降の正解率
            delimiter_idx = seq.index('a')
            predicted = logits[0, delimiter_idx:-1, :].argmax(dim=-1)
            target = input_tensor[0, delimiter_idx + 1:]
            
            if torch.all(predicted == target):
                total_correct += 1
            total_count += 1
    
    accuracy = total_correct / total_count
    return accuracy


def train(args, seed, log_dir):
    """メイン学習ループ - スケジューリングなし"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    
    print(f"Training on device: {device}")
    print(f"Fixed sequence length: nmin={args.nmin}, nmax={args.nmax}")
    print(f"Model: {args.model_type}")
    print(f"Architecture: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    
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
            temperature=args.temp_initial
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # パラメータ数を表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    best_epoch = 0
    train_log = []
    
    for epoch in range(args.nepoch):
        # 温度スケジューリング（discretizedモデルの場合）
        temperature = None
        if args.model_type == 'discretized' and hasattr(model, 'set_temperature'):
            temperature = args.temp_initial - (args.temp_initial - args.temp_final) * (epoch / max(args.nepoch - 1, 1))
            model.set_temperature(temperature)
        
        train_loss, train_acc = train_epoch(model, optimizer, args, device)
        val_acc = validate(model, args, device, num_sequences=200)
        
        # ログ出力
        log_str = f"Epoch {epoch + 1:3d}/{args.nepoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        if temperature is not None:
            log_str += f" | Temp: {temperature:.4f}"
        print(log_str)
        
        train_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'temperature': temperature
        })
        
        # ベストモデルの保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_checkpoint(model, optimizer, epoch, 1 - best_val_acc, 
                          os.path.join(log_dir, 'best_model.pt'))
            print(f"  → Best model saved (Val Acc: {val_acc:.4f})")
    
    # 最終モデルの保存
    save_checkpoint(model, optimizer, args.nepoch - 1, 1 - best_val_acc,
                   os.path.join(log_dir, 'final_model.pt'))
    
    # 学習ログの保存
    with open(os.path.join(log_dir, 'train_log.json'), 'w') as f:
        json.dump(train_log, f, indent=4)
    
    print()
    print("="*70)
    print(f"Training completed.")
    print(f"Best validation accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print("="*70)
    
    return best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer on Reverse Copy Task (No Curriculum)')
    
    parser.add_argument('--model_type', type=str, default='transformer', 
                       choices=['transformer', 'discretized'],
                       help='Type of model to train')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--temp_initial', type=float, default=5.0,
                       help='Initial temperature for Gumbel-Softmax (discretized model only)')
    parser.add_argument('--temp_final', type=float, default=0.1,
                       help='Final temperature for Gumbel-Softmax (discretized model only)')
    
    parser.add_argument('--nchar', type=int, default=3, help='Number of characters (including delimiter)')
    parser.add_argument('--nmin', type=int, default=1, help='Minimum sequence length')
    parser.add_argument('--nmax', type=int, default=20, help='Maximum sequence length (fixed)')
    
    # 学習設定
    parser.add_argument('--nepoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--nseq', type=int, default=1000, help='Number of sequences per epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # ログ設定
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    
    args = parser.parse_args()
    
    # ログディレクトリの作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'{args.model_type}_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    save_config(args, os.path.join(log_dir, 'config.json'))
    
    train(args, args.seed, log_dir)
