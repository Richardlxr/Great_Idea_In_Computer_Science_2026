# -*- coding: utf-8 -*-
# 纯 Python 版本：不用 numpy（OJ 环境可能没装），矩阵用二维列表表示
import sys
import math


def matmul(A, B):
    """矩阵乘法：A 是 n×k，B 是 k×m，返回 n×m"""
    n, k, m = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(m)] for i in range(n)]


def transpose(M):
    return [list(col) for col in zip(*M)]


def softmax_rows(x):
    """逐行 softmax：先减每行最大值，再取指数、归一化"""
    result = []
    for row in x:
        mx = max(row)
        e = [math.exp(v - mx) for v in row]  # exp(-inf) = 0
        s = sum(e)
        result.append([v / s for v in e])
    return result


def project(x, W_Q, W_K, W_V):
    """线性投影：Q = X W_Q, K = X W_K, V = X W_V"""
    return matmul(x, W_Q), matmul(x, W_K), matmul(x, W_V)


def attention(Q, K, V, causal=False):
    n = len(Q)
    d_k = len(Q[0])

    # S = Q K^T / sqrt(d_k)
    scale = math.sqrt(d_k)
    S = [[v / scale for v in row] for row in matmul(Q, transpose(K))]

    # 因果掩码：j > i 的位置设为 -inf（复制一份，输出的 S 不含掩码）
    if causal:
        masked = [[S[i][j] if j <= i else float("-inf") for j in range(n)] for i in range(n)]
        A = softmax_rows(masked)
    else:
        A = softmax_rows(S)

    O = matmul(A, V)
    return S, A, O


def read_matrix(data, pos, rows, cols):
    """从 data[pos:] 取 rows*cols 个数，按行优先摆成 rows×cols 矩阵；返回矩阵和下一个位置"""
    M = [[float(data[pos + i * cols + j]) for j in range(cols)] for i in range(rows)]
    return M, pos + rows * cols


def coutMatrix(M):
    for row in M:
        print(" ".join("%.10f" % x for x in row))


def main():
    # 一次读入全部输入，按空白（空格、换行都算）切开，得到一个字符串列表
    data = sys.stdin.read().split()

    # 前 5 个是整数：n d_model d_k d_v causal
    n, d_model, d_k, d_v, causal = [int(t) for t in data[:5]]
    pos = 5  # 下一个要读的数在 data 里的下标

    X, pos = read_matrix(data, pos, n, d_model)
    W_Q, pos = read_matrix(data, pos, d_model, d_k)
    W_K, pos = read_matrix(data, pos, d_model, d_k)
    W_V, pos = read_matrix(data, pos, d_model, d_v)

    Q, K, V = project(X, W_Q, W_K, W_V)
    S, A, O = attention(Q, K, V, causal=bool(causal))
    coutMatrix(Q)
    coutMatrix(K)
    coutMatrix(V)
    coutMatrix(S)
    coutMatrix(A)
    coutMatrix(O)


if __name__ == "__main__":
    main()
