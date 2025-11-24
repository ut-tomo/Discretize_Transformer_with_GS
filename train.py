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


def train_epoch(model, optimizer, args, device, nmax, is_first_epoch=False):
    """1エポックの学習"""
    print(f"Starting train_epoch with nmax={nmax}, nseq={args.nseq}", flush=True)
    model.train()
    total_loss = 0
    total_correct = 0  # シーケンス単位の正解数
    total_count = 0  # シーケンス数
    
    # バッチを生成
    sequences = generate_batch_sequences(args.nseq, nmax, args.nmin, args.nchar)
    
    for seq in sequences:
        # 入力と正解ラベルを準備
        input_tensor = string_to_tensor(seq, args.nchar).unsqueeze(0).to(device)  # (1, seq_len)
        
        # Forward pass
        logits = model(input_tensor)  # (1, seq_len, nchar)
        
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
        delimiter_idx = seq.index('a')  # デリミタの位置
        predicted = logits[0, delimiter_idx:-1, :].argmax(dim=-1)  # (seq_len_after_delimiter,)
        target = input_tensor[0, delimiter_idx + 1:]  # (seq_len_after_delimiter,)
        
        if torch.all(predicted == target):
            total_correct += 1
        total_count += 1
    
    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count
    
    return avg_loss, accuracy


def validate(model, args, device, nmax, nmin, num_sequences=1000):
    """検証セットでの評価"""
    model.eval()
    total_correct = 0
    total_count = 0
    
    with torch.no_grad():
        sequences = generate_batch_sequences(num_sequences, nmax, nmin, args.nchar)
        
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
    """メイン学習ループ"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    
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
            temperature=args.temp_initial  # 初期温度を使用
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    best_model_state = None
    
    # 学習ログ
    train_log = []
    
    for epoch in range(args.nepoch):
        nmax = max(min(epoch + 3, args.nmax), 3)  # 学習長は3以上で徐々に増やすスケジューリング
        
        # 温度スケジューリング（discretizedモデルの場合）
        if args.model_type == 'discretized' and hasattr(model, 'set_temperature'):
            # 温度を線形に下げる: temp_initial -> temp_final
            temperature = args.temp_initial - (args.temp_initial - args.temp_final) * (epoch / (args.nepoch - 1))
            model.set_temperature(temperature)
            print(f"Epoch {epoch + 1}: temperature={temperature:.4f}")

        train_loss, train_acc = train_epoch(model, optimizer, args, device, nmax)
        val_acc = validate(model, args, device, nmax, args.nmin, num_sequences=200)
        
        print(f"Epoch {epoch + 1}/{args.nepoch} | nmax={nmax} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        train_log.append({
            'epoch': epoch + 1,
            'nmax': nmax,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'temperature': temperature if args.model_type == 'discretized' else None
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            save_checkpoint(model, optimizer, epoch, 1 - best_val_acc, 
                          os.path.join(log_dir, 'best_model.pt'))
    
    save_checkpoint(model, optimizer, args.nepoch - 1, 1 - best_val_acc,
                   os.path.join(log_dir, 'final_model.pt'))
    
    with open(os.path.join(log_dir, 'train_log.json'), 'w') as f:
        json.dump(train_log, f, indent=4)
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer on Reverse Copy Task')
    
    parser.add_argument('--model_type', type=str, default='transformer', 
                       choices=['transformer', 'discretized'],
                       help='Type of model to train')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--temperature', type=float, default=5.0, 
                       help='Initial temperature for Gumbel-Softmax (discretized model only, deprecated: use --temp_initial)')
    parser.add_argument('--temp_initial', type=float, default=5.0,
                       help='Initial temperature for Gumbel-Softmax (discretized model only)')
    parser.add_argument('--temp_final', type=float, default=0.1,
                       help='Final temperature for Gumbel-Softmax (discretized model only)')
    
    parser.add_argument('--nchar', type=int, default=3, help='Number of characters (including delimiter)')
    parser.add_argument('--nmin', type=int, default=1, help='Minimum sequence length')
    parser.add_argument('--nmax', type=int, default=20, help='Maximum sequence length')
    
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
        