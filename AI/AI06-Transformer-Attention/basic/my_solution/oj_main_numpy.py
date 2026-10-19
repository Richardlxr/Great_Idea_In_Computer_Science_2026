# 原版（numpy 实现）。OJ 上 RE，提交用的是纯 Python 版 oj_main.py
import sys

import numpy as np


def softmax_rows(x):
    """逐行 softmax：先减每行最大值，再取指数、归一化"""
    max_column = x.max(axis=1, keepdims=True)  # shape:(n,1)
    x = x - max_column  # (shape: n, m)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def project(x, W_Q, W_K, W_V):
    """
    线性投影：Q = X W_Q, K = X W_K, V = X W_V
    X 一行一个token,行向量为这个token的特征向量
    """
    assert x.ndim == 2
    assert W_Q.shape[0] == W_K.shape[0] == W_V.shape[0] == x.shape[1]
    assert W_Q.shape[1] == W_K.shape[1]
    token_num = x.shape[0]
    token_feature = x.shape[1]
    assert token_num > 0
    assert token_feature > 0
    assert W_Q.shape[0] == token_feature
    assert W_K.shape[0] == token_feature
    assert W_V.shape[0] == token_feature

    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V

    return Q, K, V


def causal_mask(n):
    """生成一个 n x n 的因果掩码矩阵"""
    mask = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mask[i, j] = -np.inf
    return mask


def attention(Q, K, V, causal=False):
    assert Q.shape == K.shape, "Q, K 必须形状相同"
    assert V.shape[0] == Q.shape[0], "V 的token数量必须相同,但是特征维度可以不一样"
    token_num = Q.shape[0]
    token_features = Q.shape[1]

    # 打分是Q的每一行去和K的每一行点积
    S = Q @ K.T / np.sqrt(token_features)  # (token_num, token_num)

    # S(i,j) 表示 Q 的第 i 行和 K 的第 j 行的点积

    # 对打分进行 softmax 归一化
    if causal:
        A = softmax_rows(S + causal_mask(token_num))  # (token_num, token_num)
    else:
        A = softmax_rows(S)  # (token_num, token_num)

    # A(i,j) 表示 Q 的第 i 行和 K 的第 j 行的注意力权重

    # 输出是 V 的加权和,此处可以把V写成行向量的形式
    O = A @ V  # (token_num, token_features)

    # O 的第 i 行 = 用 A 的第 i 行当权重，把 V 的所有行加权平均

    return S, A, O


def read_matrix(data, pos, rows, cols):
    """从 data[pos:] 取 rows*cols 个数，按行优先摆成 rows×cols 矩阵；返回矩阵和下一个位置"""
    nums = [float(t) for t in data[pos : pos + rows * cols]]
    M = np.array(nums).reshape(rows, cols)
    return M, pos + rows * cols


def coutMatrix(M):
    for row in M:
        print(" ".join(f"{x:.10f}" for x in row))


def main():
    # 一次读入全部输入，按空白（空格、换行都算）切开，得到一个字符串列表
    data = sys.stdin.read().split()

    # 前 5 个是整数：n d_model d_k d_v causal
    n, d_model, d_k, d_v, causal = (int(t) for t in data[:5])
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
