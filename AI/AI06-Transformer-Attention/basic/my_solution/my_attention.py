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


def check(A, O, V, causal, tol=1e-6):
    """检查注意力结果的基本性质"""
    assert A.shape[0] == A.shape[1] == V.shape[0] == O.shape[0], "A 必须是 n×n"
    assert O.shape[1] == V.shape[1], "O 必须是 n×d_v"
    assert np.all(A >= 0), "权重出现负数"
    assert np.allclose(A.sum(axis=1), 1, atol=tol), "存在行和不为 1 的行"
    if causal:
        assert np.allclose(A[np.triu_indices_from(A, k=1)], 0, atol=tol), (
            "被屏蔽的未来位置权重不为 0"
        )
    print(f"check passed (causal={causal})\n")


def show(name, M):
    """打印矩阵的名字、形状和数值"""
    print(f"{name}  shape = {M.shape}")
    print(np.round(M, 6))
    print()


def experiment_q5():
    print("========== 实验 1：Q5 手算核对 ==========")
    I = np.eye(2)
    Q, K, V = project(I, I, I, I)
    for causal in [False, True]:
        S, A, O = attention(Q, K, V, causal=causal)
        show("S", S)
        show(f"A (causal={causal})", A)
        show(f"O (causal={causal})", O)
        check(A, O, V, causal)

    expected = np.array([[0.669762, 0.330238], [0.330238, 0.669762]])
    _, A, _ = attention(Q, K, V, causal=False)
    print("与任务书数值一致:", np.allclose(A, expected, atol=1e-6))
    print()


def experiment_single():
    print("========== 实验 2：只有一个位置 ==========")
    I = np.eye(2)
    X = np.array([[2.0, 3.0]])
    Q, K, V = project(X, I, I, I)
    S, A, O = attention(Q, K, V, causal=False)
    show("A", A)
    check(A, O, V, False)
    print("O==V:", np.allclose(O, V))
    print()


def experiment_causality():
    print("========== 实验 3：因果性（只改最后一个位置）==========")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 3))  # 4 个 token，每个 3 维
    W_Q = rng.standard_normal((3, 2))
    W_K = rng.standard_normal((3, 2))
    W_V = rng.standard_normal((3, 2))

    X2 = X.copy()
    X2[-1] = [9.0, 9.0, 9.0]  # 只改最后一个 token

    for causal in [True, False]:
        Q, K, V = project(X, W_Q, W_K, W_V)
        _, _, O1 = attention(Q, K, V, causal=causal)
        Q2, K2, V2 = project(X2, W_Q, W_K, W_V)
        _, _, O2 = attention(Q2, K2, V2, causal=causal)

        # 前 3 行（除最后一个 token 外）输出的最大绝对差
        diff = np.max(np.abs(O1[:-1] - O2[:-1]))
        print(f"causal={causal}: 前 3 个位置输出的最大变化 = {diff:.3e}")
    print()


def experiment_mine():
    """自己设计的小例子：n=3, d_model=2"""
    print("========== 实验 4：自己的例子 ==========")
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    W_Q = np.array([[1.0, 0.0], [0.0, 2.0]])
    W_K = np.array([[1.0, 1.0], [0.0, 1.0]])
    W_V = np.array([[1.0, 2.0], [0.0, 1.0]])
    Q, K, V = project(X, W_Q, W_K, W_V)
    show("Q", Q)
    show("K", K)
    show("V", V)
    for causal in [False, True]:
        S, A, O = attention(Q, K, V, causal=causal)
        show("S", S)
        show(f"A (causal={causal})", A)
        show(f"O (causal={causal})", O)
        check(A, O, V, causal)

    _, A, O = attention(Q, K, V, causal=True)
    print(
        "与手算的结果一样:",
        np.allclose(A[1], [0.5, 0.5, 0.0]) and np.allclose(O[1], [0.5, 1.5]),
    )
    print()


def plot_heatmap(A, title, filename):
    """把注意力矩阵画成热力图并保存"""
    import matplotlib

    matplotlib.use("Agg")  # 只保存图片、不弹窗
    import matplotlib.pyplot as plt

    n = A.shape[0]
    fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.2 * n + 1.5))
    im = ax.imshow(A, cmap="viridis", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            color = "black" if A[i, j] > 0.6 else "white"
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", color=color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("key position j")
    ax.set_ylabel("query position i")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"热力图已保存到 {filename}")


def main():
    np.set_printoptions(precision=6, suppress=True)
    experiment_q5()
    experiment_single()
    experiment_causality()
    experiment_mine()

    print("========== 随机小矩阵 + 热力图 ==========")
    rng = np.random.default_rng(42)
    n, d_model, d_k, d_v = 5, 4, 3, 3
    X = rng.standard_normal((n, d_model))
    W_Q = rng.standard_normal((d_model, d_k))
    W_K = rng.standard_normal((d_model, d_k))
    W_V = rng.standard_normal((d_model, d_v))
    Q, K, V = project(X, W_Q, W_K, W_V)
    show("Q", Q)
    show("K", K)
    show("V", V)
    for causal in [False, True]:
        S, A, O = attention(Q, K, V, causal=causal)
        show("S", S)
        show(f"A (causal={causal})", A)
        show(f"O (causal={causal})", O)
        check(A, O, V, causal)
        name = "causal" if causal else "nomask"
        plot_heatmap(A, f"Attention A ({name}, random weights)", f"heatmap_{name}.png")


if __name__ == "__main__":
    main()
