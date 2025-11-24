"""
Reverse Copy Task Generator
w#w' 形式の文字列を生成するタスク
"""
import random
import torch


def generate_reverse_copy_sequence(n, nchar):
    # w を生成（#を除いた文字を使用）
    # nchar=3の場合: 'a'がデリミタ、'b','c'が使用可能な文字
    if nchar < 2:
        raise ValueError("nchar must be at least 2 (1 for delimiter, 1+ for content)")
    
    available_chars = nchar - 1  # デリミタを除いた使用可能な文字数
    w = ''.join([chr(ord('a') + random.randint(1, available_chars)) for _ in range(n)])
    
    # w' (反転) を生成
    w_reverse = w[::-1]
    
    # w#w' の形式で結合（#は 'a' = 文字0に対応）
    sequence = w + 'a' + w_reverse
    
    return sequence


def generate_batch_sequences(batch_size, nmax, nmin, nchar):
    """
    バッチサイズ分のReverse Copy シーケンスを生成
    
    Args:
        batch_size: バッチサイズ
        nmax: 最大長さ（#の前後）
        nmin: 最小長さ（#の前後）
        nchar: 文字の種類数
    
    Returns:
        文字列のリスト
    """
    sequences = []
    for _ in range(batch_size):
        n = random.randint(nmin, nmax - 1)
        seq = generate_reverse_copy_sequence(n, nchar)
        sequences.append(seq)
    return sequences


def string_to_tensor(s, nchar):
    return torch.LongTensor([ord(c) - ord('a') for c in s])


def tensor_to_string(tensor):
    return ''.join([chr(int(x) + ord('a')) for x in tensor])
