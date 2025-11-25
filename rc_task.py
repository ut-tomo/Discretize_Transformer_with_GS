"""
Reverse Copy Task Generator
w#w' 形式の文字列を生成するタスク
"""
import random
import torch


def generate_reverse_copy_sequence(n, nchar):
    """
    Reverse Copy Task のシーケンスを生成: w + 'a' + w[::-1]
    
    Args:
        n: デリミタ前後の片側の長さ（全体の長さは 2n+1）
        nchar: 文字の種類数（デリミタを含む）
               例: nchar=3 → 'a'(デリミタ), 'b', 'c'
    
    Returns:
        w + 'a' + w' の形式の文字列（w'はwの逆順）
        例: n=3, nchar=3 → 'bbcacbb' (w='bbc', delimiter='a', w'='cbb')
    """
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
        nmax: 最大長さ（デリミタ前後の片側の長さ）
              注意: 実際の生成範囲は [nmin, nmax] (両端含む)
        nmin: 最小長さ（デリミタ前後の片側の長さ）
        nchar: 文字の種類数（デリミタ 'a' を含む）
    
    Returns:
        文字列のリスト。各文字列は w + 'a' + w[::-1] の形式
        例: nchar=3の場合、'bbc' + 'a' + 'cbb' = 'bbcacbb'
    """
    sequences = []
    for _ in range(batch_size):
        n = random.randint(nmin, nmax)  # nmaxを含むように修正
        seq = generate_reverse_copy_sequence(n, nchar)
        sequences.append(seq)
    return sequences


def string_to_tensor(s, nchar):
    return torch.LongTensor([ord(c) - ord('a') for c in s])


def tensor_to_string(tensor):
    return ''.join([chr(int(x) + ord('a')) for x in tensor])
