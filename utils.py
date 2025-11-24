"""
Utility functions for the Discretized Transformer project
"""
import torch
import os
import json


def save_checkpoint(model, optimizer, epoch, best_val_loss, filepath):
    """モデルのチェックポイントを保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """チェックポイントを読み込み"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_val_loss']


def save_config(args, filepath):
    """設定を JSON ファイルに保存"""
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)


def load_config(filepath):
    """設定を JSON ファイルから読み込み"""
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config
