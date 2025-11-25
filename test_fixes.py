#!/usr/bin/env python3
"""
修正内容の検証スクリプト
"""

import sys
from rc_task import generate_batch_sequences, generate_reverse_copy_sequence

def test_nmax_parameter():
    """nmax パラメータが両端を含むか検証"""
    print("=== Testing nmax parameter ===")
    
    # nmax=5 で複数回生成し、長さ5が生成されるか確認
    nchar = 3
    nmin = 5
    nmax = 5
    batch_size = 100
    
    sequences = generate_batch_sequences(batch_size, nmax, nmin, nchar)
    
    lengths = []
    for seq in sequences:
        # デリミタを見つける
        delimiter_idx = seq.index('a')
        w_length = delimiter_idx
        lengths.append(w_length)
    
    print(f"Generated {len(sequences)} sequences with nmin={nmin}, nmax={nmax}")
    print(f"Lengths found: {set(lengths)}")
    print(f"Min length: {min(lengths)}, Max length: {max(lengths)}")
    
    if max(lengths) == nmax and min(lengths) == nmin:
        print("✓ PASS: nmax is inclusive (both endpoints included)")
        return True
    else:
        print("✗ FAIL: nmax behavior incorrect")
        return False


def test_reverse_copy_task():
    """Reverse Copy タスクが正しく生成されるか検証"""
    print("\n=== Testing Reverse Copy Task ===")
    
    nchar = 3
    n = 5
    
    # いくつかサンプルを生成
    for i in range(5):
        seq = generate_reverse_copy_sequence(n, nchar)
        delimiter_idx = seq.index('a')
        
        w = seq[:delimiter_idx]
        w_prime = seq[delimiter_idx + 1:]
        
        print(f"Sample {i+1}: {seq}")
        print(f"  w = {w}, w' = {w_prime}, w[::-1] = {w[::-1]}")
        
        if w[::-1] == w_prime:
            print("  ✓ Correct reverse")
        else:
            print("  ✗ Incorrect reverse")
            return False
    
    print("✓ PASS: All sequences are correct reverse copies")
    return True


def test_sequence_format():
    """シーケンスのフォーマットを検証"""
    print("\n=== Testing Sequence Format ===")
    
    nchar = 3
    n = 4
    seq = generate_reverse_copy_sequence(n, nchar)
    
    print(f"Generated sequence: {seq}")
    print(f"Total length: {len(seq)} (expected: {2*n + 1} = {2*n+1})")
    
    # 文字の種類を確認
    chars = set(seq)
    print(f"Characters used: {chars}")
    
    # デリミタが含まれているか
    if 'a' in seq:
        print("✓ Delimiter 'a' found")
    else:
        print("✗ Delimiter 'a' not found")
        return False
    
    # デリミタが1つだけか
    if seq.count('a') == 1:
        print("✓ Exactly one delimiter")
    else:
        print(f"✗ Multiple delimiters found: {seq.count('a')}")
        return False
    
    # 長さが正しいか
    if len(seq) == 2*n + 1:
        print("✓ Correct total length")
        return True
    else:
        print(f"✗ Incorrect length: {len(seq)} != {2*n+1}")
        return False


if __name__ == "__main__":
    results = []
    
    results.append(test_nmax_parameter())
    results.append(test_reverse_copy_task())
    results.append(test_sequence_format())
    
    print("\n" + "="*50)
    if all(results):
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
