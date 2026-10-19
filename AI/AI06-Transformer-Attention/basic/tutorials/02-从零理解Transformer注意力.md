# 教程二：从零理解 Transformer 注意力（边学边做，完成 Basic）

> **怎么用这份教程**：从头往下读，每读完一个概念，后面紧跟一个 **🛠 任务**，立刻去做，做完再往下读。
> 每个任务里都有 **🐍 本任务用到的语法**，Python 语法边做边学，**不需要先学完教程一**。
> 教程一（[Python 基础语法](01-Python基础语法.md)）当字典用，遇到不懂的语法再去查对应章节。
>
> 读完、做完这份教程，Basic 的全部提交内容（源码、OJ 3280、矩阵日志、热力图、Q1–Q5）就完成了。
> 预计用时：10–14 小时。

---

## 总览：一共 4 个文件

```text
AI06-Transformer-Attention/basic/my_solution/
├── my_attention.py   ★ 主程序（提交）：所有核心函数 + 检查 + 实验 + 画图，逐个任务往里加
├── oj_main.py        ★ OJ 程序（提交）：读标准输入、调用核心函数、按格式输出
├── answers.md        ★ 问答（提交）：Q1–Q5，读到对应概念时就写
├── playground.py       草稿（不提交）：试代码、跑测试，随便改、随时清空
│
├── log.txt           （运行生成）矩阵日志
├── heatmap_nomask.png（运行生成）热力图：无掩码
└── heatmap_causal.png（运行生成）热力图：因果掩码
```

`my_attention.py` 最终会包含这些函数，**每个任务写一个**：

| 函数                           | 作用                   | 在哪个任务写            |
| ------------------------------ | ---------------------- | ----------------------- |
| `softmax_rows(S)`              | 数值稳定的逐行 softmax | 任务 3                  |
| `project(X, W_Q, W_K, W_V)`    | 投影出 Q、K、V         | 任务 4                  |
| `attention(Q, K, V, causal)`   | 算 S、A、O             | 任务 5，任务 8 补上掩码 |
| `causal_mask(n)`               | 因果掩码               | 任务 8                  |
| `show(name, M)`                | 打印形状和数值         | 任务 10                 |
| `check(A, O, V, causal)`       | 验收性质检查           | 任务 10                 |
| `experiment_*()`               | 三个验收实验           | 任务 11                 |
| `plot_heatmap(...)` / `main()` | 画图、串起全部流程     | 任务 12                 |

## 任务地图

| 任务 | 位置                                 | 做什么                                         | 产出                        | 用时   |
| ---- | ------------------------------------ | ---------------------------------------------- | --------------------------- | ------ |
| 0    | [第 0 章](#0-开工准备)               | 建目录、建文件、跑通参考代码                   | 4 个空文件                  | 15 min |
| 1    | [第 2 章](#2-token-和向量q1)         | 写 Q1                                          | answers.md                  | 15 min |
| 2    | [第 3 章](#3-数学准备点积与矩阵乘法) | 第一次写 NumPy：点积、Q Kᵀ                     | playground.py               | 30 min |
| 3    | [第 4 章](#4-softmax把分数变成权重)  | 实现 `softmax_rows`                            | my_attention.py             | 45 min |
| 4    | [第 5 章](#5-从加权平均到-qkvq2)     | 实现 `project`，写 Q2                          | my_attention.py、answers.md | 45 min |
| 5    | [第 6 章](#6-完整公式与形状)         | 实现 `attention`（先不带掩码）                 | my_attention.py             | 30 min |
| 6    | [第 7 章](#7-为什么要除以-d_k)       | 验证缩放的作用                                 | playground.py               | 15 min |
| 7    | [第 8 章](#8-softmax-沿哪一维q3)     | 对比 axis，写 Q3                               | answers.md                  | 20 min |
| 8    | [第 9 章](#9-因果掩码q4)             | 实现 `causal_mask`，给 attention 加掩码，写 Q4 | my_attention.py、answers.md | 1 h    |
| 9    | [第 10 章](#10-手算两个完整例子q5)   | 纸笔手算，程序核对，写 Q5                      | answers.md                  | 1 h    |
| 10   | [第 11 章](#11-打印与性质检查)       | `show` 和 `check`                              | my_attention.py             | 45 min |
| 11   | [第 12 章](#12-三个验收实验)         | 三个验收实验                                   | my_attention.py             | 1 h    |
| 12   | [第 13 章](#13-热力图与主函数)       | 热力图 + `main`                                | my_attention.py、png        | 45 min |
| 13   | [第 14 章](#14-矩阵日志与自己的例子) | 生成日志、加一个自己的例子                     | log.txt                     | 45 min |
| 14   | [第 15 章](#15-oj-3280)              | OJ 程序                                        | oj_main.py                  | 1 h    |
| 15   | [第 16 章](#16-整理-q1q5-并提交)     | 整理问答，对照清单提交                         | 全部                        | 1 h    |

---

## 0. 开工准备

### 🛠 任务 0：建目录、建文件、跑通参考代码

**第 1 步：确认 Python 环境**

```bash
python3 -c "import numpy, matplotlib; print('OK')"
```

输出 `OK` 就没问题。如果报 `No module named 'numpy'`，说明当前的 `python3` 不是装了 NumPy 的那个（你的电脑上有 Homebrew 和 miniconda 两个 Python），改用 `python` 再试一次；之后所有命令都用能输出 OK 的那个。

**第 2 步：跑一下课程给的参考代码**

```bash
cd ~/Vault/Code/Learning/Great_Idea_In_Computer_Science_2026/AI/AI06-Transformer-Attention
python3 attention.py
```

现在看不懂输出没关系，读完第 10 章你会全部看懂。

**第 3 步：建工作目录和 4 个文件**

```bash
mkdir -p basic/my_solution
cd basic/my_solution
touch my_attention.py oj_main.py answers.md playground.py
```

**第 4 步：在 `answers.md` 里先写好框架**

```markdown
# AI6 Transformer Basic 问答

## Q1 token 和向量是什么关系？

## Q2 Q、K、V 各参与哪一步？

## Q3 softmax 应沿哪一维计算？

## Q4 因果掩码屏蔽什么？

## Q5 Q=K=V 为 2×2 单位阵，d_k=2，A 和 O 是多少？
```

> 🐍 **本任务用到的命令**
>
> - `mkdir -p 目录`：建目录；`cd 目录`：进入目录；`touch 文件名`：建空文件。
> - `python3 文件.py`：运行 Python 文件。Python 不用编译，直接运行源码。
> - **之后所有命令都在 `basic/my_solution/` 目录下运行**，否则 `playground.py` 会找不到 `my_attention.py`。

---

## 1. 我们要解决什么问题

看这句话：

> 小猫追着毛线球跑，因为**它**很好奇。

人一眼就知道"它"指的是"小猫"。但对计算机来说，在处理"它"这个位置时，需要**从句子里其他位置拿信息**，而且不同位置的重要程度不一样："小猫"很重要，"跑"没那么重要。

**注意力机制（Attention）要回答的就是：序列中的每一个位置，应该从其他每一个位置读取多少信息？**

一点历史：

- 早期的序列模型（RNN）像读书一样从左到右逐字处理，信息只能一步步往后传，句子一长，前面的信息就容易"忘掉"，而且没法并行计算。
- 2017 年的论文 _Attention Is All You Need_ 提出了 **Transformer**：去掉逐步传递，让**每个位置直接和所有位置交换信息**，交换的方式就是注意力。今天的 GPT、BERT 等大模型都建立在这个结构之上。

Basic 只做 Transformer 里**最核心的一个零件**：一次**单头缩放点积注意力**（Scaled Dot-Product Attention）。

---

## 2. token 和向量（Q1）

### 2.1 token：序列的基本单元

计算机先把文本切成一个个小单元，叫 **token**。怎么切取决于"分词器"（tokenizer）：

| 原文         | 可能的切法           | 说明                   |
| ------------ | -------------------- | ---------------------- |
| 我爱学习     | `我` `爱` `学习`     | 按词                   |
| 我爱学习     | `我` `爱` `学` `习`  | 按字                   |
| unbelievable | `un` `believ` `able` | 子词：一个词被切成几块 |

所以 **token 不一定恰好是一个汉字或一个单词**，它只是"序列里的一个单元"。

### 2.2 向量：token 的数字表示

神经网络只会做数值运算，所以每个 token 要变成一串数字，也就是一个**向量**。常见做法是查表（叫"词嵌入"，embedding）：

```text
词表:   我 → 0    爱 → 1    学习 → 2

嵌入表 E（每行是一个 token 的向量，d_model = 4）:
  行0 (我):   [ 0.2, -1.0,  0.5,  0.1]
  行1 (爱):   [ 0.9,  0.3, -0.2,  0.7]
  行2 (学习): [-0.4,  0.8,  0.6, -0.3]
```

"我 爱 学习"这 3 个 token 就变成一个 **3×4 的矩阵 X**，**每一行对应一个位置**：

```text
      d_model = 4 列
X = [[ 0.2, -1.0,  0.5,  0.1],   ← 位置 0
     [ 0.9,  0.3, -0.2,  0.7],   ← 位置 1
     [-0.4,  0.8,  0.6, -0.3]]   ← 位置 2
     n = 3 行
```

真实模型里，嵌入表是训练出来的，含义相近的词向量会比较接近。**Basic 不需要训练**，直接用手写或固定随机种子生成的向量当 X 就行。

> 📌 **记住两个符号**：`n` = 序列长度（多少个位置），`d_model` = 每个位置向量的长度。X 的形状是 `n × d_model`。

### 🛠 任务 1：写 Q1

在 `answers.md` 的 Q1 下面，用自己的话写 3–5 句，要包含：

- token 是什么，为什么不一定等于一个字或一个词；
- token 怎么变成向量；
- n 个 token 组成的 X 是什么形状，每行代表什么；
- 本项目中的向量是手写或随机的，不代表真实语义。

先写个草稿，第 16 章最后统一润色。

---

## 3. 数学准备：点积与矩阵乘法

### 3.1 点积：衡量两个向量有多"像"

两个等长向量的点积 = 对应位置相乘再求和：

```text
a = [1, 2, 3]
b = [4, 0, -1]
a · b = 1×4 + 2×0 + 3×(-1) = 4 + 0 - 3 = 1
```

几何意义：`a · b = |a| |b| cos θ`。

| 两个向量的关系 | 点积     |
| -------------- | -------- |
| 方向相同       | 大的正数 |
| 互相垂直       | 0        |
| 方向相反       | 负数     |

所以**点积可以当作"相似度/匹配度"的分数**，这是注意力的数学基础。

### 3.2 矩阵乘法 = 批量点积

`C = A @ B` 中，`C[i][j]` = **A 的第 i 行** 与 **B 的第 j 列** 的点积。

形状规则：`(n × k) @ (k × m) → (n × m)`，中间的 k 必须相等。

### 3.3 关键技巧：`Q @ K.T` 一次算出所有两两点积

设 Q 和 K 都是 n×d 矩阵（每行是一个位置的向量）。`K.T` 是 d×n，它的**第 j 列**就是 K 的**第 j 行**。所以：

```text
(Q @ K.T)[i][j] = Q 的第 i 行 · K 的第 j 行
```

**一次矩阵乘法得到 n×n 个点积**，也就是"每个位置对每个位置"的匹配分数。

### 3.4 矩阵乘法 = 批量加权求和

反过来看 `O = A @ V`（A 是 n×n，V 是 n×d_v）：

```text
O 的第 i 行 = A[i][0] × (V 的第 0 行) + A[i][1] × (V 的第 1 行) + ... + A[i][n-1] × (V 的第 n-1 行)
```

也就是说：**A 的第 i 行给出一组系数，O 的第 i 行就是用这组系数把 V 的所有行加权求和**。第 5 章会用到这一点。

### 🛠 任务 2：第一次写 NumPy（playground.py）

把下面的代码写进 `playground.py`（**建议手敲，不要复制**，敲一遍就熟悉语法了），然后运行 `python3 playground.py`：

```python
import numpy as np

# 1. 点积
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 0.0, -1.0])
print("a · b =", a @ b)

# 2. 一个 3×2 的矩阵：3 个位置，每个位置 2 维
Q = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])
K = Q
print("Q 的形状:", Q.shape)
print("K.T 的形状:", K.T.shape)

# 3. 一次算出所有两两点积
S = Q @ K.T
print("Q @ K.T 的形状:", S.shape)
print(S)

# 4. 对比：* 是逐元素相乘，不是矩阵乘法
print(Q * K)
```

**预期输出**：

```text
a · b = 1.0
Q 的形状: (3, 2)
K.T 的形状: (2, 3)
Q @ K.T 的形状: (3, 3)
[[1. 0. 1.]
 [0. 1. 1.]
 [1. 1. 2.]]
[[1. 0.]
 [0. 1.]
 [1. 1.]]
```

**做完自查**：手算 `S[2][2]` 和 `S[0][1]`，说出它们分别是哪两个向量的点积，和程序输出对得上吗？

> 🐍 **本任务用到的语法**
>
> - `import numpy as np`：导入 NumPy 库，起别名 `np`（相当于 C++ 的 `#include`）。之后用 `np.xxx` 调用它的功能。
> - `# ...`：注释，相当于 C++ 的 `//`。
> - **不用声明类型**：`a = ...` 直接赋值，没有 `int`、`double`。
> - `np.array([[1.0, 0.0], [0.0, 1.0]])`：用**嵌套列表**创建矩阵，外层列表的每个元素是一行。括号内可以随意换行。
> - `print(x, y)`：输出多个值，自动用空格隔开、末尾换行。
> - `.shape`：矩阵的形状，是一个元组 `(行数, 列数)`。**调试时最常用，遇到 bug 先打印 shape。**
> - `@` 是矩阵乘法，`.T` 是转置；⚠️ `*` 是逐元素相乘，是最常见的错误来源。
> - 浮点数打印成 `1.` 就是 `1.0` 的意思。

---

## 4. softmax：把分数变成权重

### 4.1 定义

给一组任意实数 `z = [z_0, z_1, ..., z_{m-1}]`：

```text
softmax(z)_j = e^{z_j} / (e^{z_0} + e^{z_1} + ... + e^{z_{m-1}})
```

它保证输出：

1. **每个数都大于 0**（指数永远为正）
2. **总和为 1**（除以了总和）
3. **保持大小顺序**：原来的分数越大，权重越大

所以 softmax 的输出可以理解成一组"概率"或者"分配比例"。

### 4.2 手算一个例子

`z = [2.0, 1.0, 0.1]`：

```text
e^2.0 = 7.389    e^1.0 = 2.718    e^0.1 = 1.105    总和 = 11.212
softmax = [7.389/11.212, 2.718/11.212, 1.105/11.212]
        = [0.659, 0.242, 0.099]
```

softmax 会把差距**放大**：分数 2.0 和 1.0 只差 1，但权重差了将近 3 倍（因为 e¹ ≈ 2.718）。

### 4.3 数值稳定：先减最大值

如果分数很大，比如 `z = [1001, 1002, 1003]`，`e^1003` 会超出 double 能表示的范围，变成 `inf`，再做 `inf / inf` 就得到 `nan`（不是一个数），程序就算坏了。

**解决办法**：每个数先减去这组数里的最大值，再做 softmax。**结果完全不变**，因为：

```text
e^{z_j - c} / Σ e^{z_k - c} = (e^{z_j} · e^{-c}) / (Σ e^{z_k} · e^{-c}) = e^{z_j} / Σ e^{z_k}
```

分子分母同时乘以 `e^{-c}`，约掉了。减去最大值后，最大的那一项变成 `e^0 = 1`，其余都在 (0, 1] 之间，绝不会溢出。

```text
softmax([1, 2, 3])          = [0.090031, 0.244728, 0.665241]
softmax([1001, 1002, 1003]) = [0.090031, 0.244728, 0.665241]   ← 减去 1003 后变成 [-2, -1, 0]，结果一样
```

> 📌 这是任务书和 OJ 的**硬性要求**：softmax 必须先减去该行最大值。

### 4.4 负无穷的作用

如果某一项是 `-∞`：`e^{-∞} = 0`，它在 softmax 后的权重就**恰好是 0**，而其他项的权重仍然加起来等于 1。这就是第 9 章因果掩码的原理。

```text
softmax([0.5, -∞, 0.5]) = [0.5, 0, 0.5]
```

### 4.5 对矩阵的每一行做 softmax

注意力里要处理的是一个矩阵 S，**每一行单独做一次 softmax**（第 8 章解释原因）。NumPy 不用写循环，靠 `axis` 和"广播"一次完成：

```python
S.max(axis=1, keepdims=True)   # 每行的最大值，形状 (n, 1)
S - 上面的结果                   # 每行减去自己那一行的最大值
```

- `axis=1`：沿着列的方向压缩，**每一行得到一个结果**。（`axis=0` 则是每一列一个结果。）
- `keepdims=True`：结果保持成 `(n, 1)` 的列向量，而不是 `(n,)`。
- **广播**：`(n, m) - (n, 1)` 时，NumPy 会自动把 `(n, 1)` 沿列方向复制成 `(n, m)` 再相减，效果就是"每行减去自己的那个数"。

> ⚠️ **一定要加 `keepdims=True`**。不加的话形状是 `(n,)`，在方阵上不会报错，而是**按列**去减，悄悄算错。

### 🛠 任务 3：实现 `softmax_rows`（my_attention.py）

**任务**：在 `my_attention.py` 里写：

```python
import numpy as np


def softmax_rows(S):
    """逐行 softmax：先减每行最大值，再取指数、归一化"""
    # TODO: 你来写，3 行左右
```

步骤：① 每行减去该行最大值；② 用 `np.exp` 取指数；③ 每行除以该行的和（`sum(axis=1, keepdims=True)`），并 `return` 结果。

**测试**：清空 `playground.py`，写入下面的代码并运行：

```python
import numpy as np
from my_attention import softmax_rows

np.set_printoptions(precision=6, suppress=True)
print(softmax_rows(np.array([[2.0, 1.0, 0.1]])))
print(softmax_rows(np.array([[1001.0, 1002.0, 1003.0]])))
print(softmax_rows(np.array([[0.5, -np.inf, 0.5]])))
print(softmax_rows(np.array([[1.0, 2.0],
                             [5.0, 5.0]])))
```

**预期输出**（不能有任何 overflow 警告）：

```text
[[0.659001 0.242433 0.098566]]
[[0.090031 0.244728 0.665241]]
[[0.5 0.  0.5]]
[[0.268941 0.731059]
 [0.5      0.5     ]]
```

<details>
<summary>卡住了再看：参考实现</summary>

```python
def softmax_rows(S):
    """逐行 softmax：先减每行最大值，再取指数、归一化"""
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)
```

</details>

> 🐍 **本任务用到的语法**
>
> - **定义函数**：`def 函数名(参数):`，冒号结尾，函数体**缩进 4 个空格**。Python 没有大括号，**缩进就是代码块**，缩进不对会直接报错。不写参数类型和返回类型。
> - `"""..."""`：函数的文档字符串，相当于函数注释，可写可不写。
> - `return 值`：返回结果。
> - 函数之间空两行是 Python 的书写惯例（不空也能运行）。
> - `from my_attention import softmax_rows`：从同目录的 `my_attention.py` 导入函数（相当于 `#include` 自己的头文件）。**每次修改 my_attention.py 后，重新运行 playground.py 即可。**
> - `np.exp(S)`：对**每个元素**取指数，不用写循环。NumPy 的 `+ - * /` 都是这样逐元素运算的。
> - `-np.inf`：负无穷。
> - `np.set_printoptions(precision=6, suppress=True)`：打印时保留 6 位小数、不用科学计数法。
> - ⚠️ `S = S - ...` 会创建新数组，不影响调用者传进来的 S；而 `S -= ...` 会**原地修改**外面的数组。Python 中数组传参类似 C++ 传引用，所以这里写 `S = S - ...`。

---

## 5. 从"加权平均"到 Q/K/V（Q2）

我们一步一步把注意力"发明"出来。

### 5.1 第一步：每个位置读取所有位置的平均

最简单的"从其他位置拿信息"：每个位置的输出 = 所有位置向量的平均。

```text
输出_i = (1/n) × (x_0 + x_1 + ... + x_{n-1})
```

问题是**每个位置的输出完全一样**，而且所有位置被看成同等重要，这显然不对。"它"应该更关注"小猫"。

### 5.2 第二步：按相似度分配权重

让位置 i 根据"自己和位置 j 有多像"来决定从 j 拿多少：

```text
分数   s_ij = x_i · x_j                 （点积 = 相似度）
权重   a_i  = softmax([s_i0, s_i1, ...]) （变成和为 1 的比例）
输出_i = Σ_j a_ij × x_j                  （加权平均）
```

这已经是注意力的雏形了。但还有一个问题：**同一个向量 x 同时承担了三种不同的角色**：

- 作为"提问者"：我想找什么样的信息？
- 作为"被匹配者"：别人凭什么来找我？
- 作为"内容提供者"：被找到后，我交出什么信息？

这三件事不应该被绑在一起。比如"它"需要找的是"一个名词/主语"，而它自己作为内容提供的是"代词"的信息，二者不是一回事。而且用 x_i · x_j 打分是对称的（i 找 j 的分数 = j 找 i 的分数），但"它找小猫"和"小猫找它"的需求明显不同。

### 5.3 第三步：用三个矩阵把三种角色分开

给每个位置的向量分别乘上三个不同的矩阵，得到三个新向量：

```text
q_i = x_i W_Q     Query（查询）：我在找什么
k_i = x_i W_K     Key（键）：    我能被怎样找到
v_i = x_i W_V     Value（值）：  我实际提供的内容
```

**图书馆类比**：

- 你带着一个**搜索词**（Query）去图书馆；
- 每本书的书脊上贴着**标签**（Key）；
- 你拿搜索词和每个标签比对，算出匹配度（q · k），越匹配越想看；
- 最后你得到的是这些书的**正文内容**（Value）按匹配度加权混合的结果。

用 Q/K/V 重写第二步：

```text
分数   s_ij = q_i · k_j          ← Q 和 K 负责"算匹配"
权重   a_i  = softmax(s_i)       ← 归一化成比例
输出_i = Σ_j a_ij × v_j          ← 用权重对 V 加权求和
```

> 📌 **Q2 的答案核心**：Q 和 K 用来计算匹配分数（点积），softmax 后得到权重，再用权重对 V 做加权求和。
> **Q 和 K 的维数必须相同**（否则没法做点积），V 的维数可以不同。

W_Q、W_K、W_V 在真实模型中是训练出来的参数；**Basic 中直接手写或随机生成**。

所有位置写成矩阵后，投影就是三次矩阵乘法：

```text
Q = X @ W_Q     (n × d_model) @ (d_model × d_k) → (n × d_k)
K = X @ W_K     (n × d_model) @ (d_model × d_k) → (n × d_k)
V = X @ W_V     (n × d_model) @ (d_model × d_v) → (n × d_v)
```

### 🛠 任务 4：实现 `project`，写 Q2

**任务**：在 `my_attention.py` 里添加：

```python
def project(X, W_Q, W_K, W_V):
    """线性投影：Q = X W_Q, K = X W_K, V = X W_V"""
    # TODO: 1. 检查形状   2. 返回 Q, K, V
```

任务书要求"检查形状和非空序列"，用 `assert` 检查：

- X 是二维的（`X.ndim == 2`），且 n ≥ 1；
- 三个 W 的行数都等于 d_model；
- W_Q 和 W_K 的列数相同（都是 d_k）。

**测试**（清空 playground.py）：

```python
import numpy as np
from my_attention import project

X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])            # n=3, d_model=2
W_Q = np.ones((2, 4))
W_K = np.ones((2, 4))
W_V = np.ones((2, 5))
Q, K, V = project(X, W_Q, W_K, W_V)
print(Q.shape, K.shape, V.shape)     # 应输出: (3, 4) (3, 4) (3, 5)

# 故意传错形状，应该报 AssertionError 并显示你写的提示信息
project(X, np.ones((3, 4)), W_K, W_V)
```

最后一行**应该报错**，看到错误提示里有你写的信息就说明检查生效了。测试完把这行删掉。

再在 `answers.md` 的 **Q2** 下写草稿：Q、K、V 从哪来，各参与哪一步计算，为什么 Q 和 K 维数必须相同。

<details>
<summary>卡住了再看：参考实现</summary>

```python
def project(X, W_Q, W_K, W_V):
    """线性投影：Q = X W_Q, K = X W_K, V = X W_V"""
    assert X.ndim == 2, "X 必须是二维矩阵"
    n, d_model = X.shape
    assert n >= 1, "序列不能为空"
    assert W_Q.shape[0] == d_model, "W_Q 行数必须等于 d_model"
    assert W_K.shape[0] == d_model, "W_K 行数必须等于 d_model"
    assert W_V.shape[0] == d_model, "W_V 行数必须等于 d_model"
    assert W_Q.shape[1] == W_K.shape[1], "Q 和 K 的维数 d_k 必须相同"
    return X @ W_Q, X @ W_K, X @ W_V
```

</details>

> 🐍 **本任务用到的语法**
>
> - `assert 条件, "提示信息"`：条件为假时报错并显示提示，相当于 C++ 的 `assert`，但可以带信息。
> - `n, d_model = X.shape`：**解包**。`X.shape` 是元组 `(3, 2)`，左边写两个变量就分别拿到 3 和 2。
> - `X.shape[0]`：元组取下标，即行数；`X.shape[1]` 是列数。
> - `X.ndim`：维数，矩阵是 2。
> - `np.ones((2, 4))`：全 1 矩阵。⚠️ **注意是双层括号**，里面的 `(2, 4)` 是一个表示形状的元组。`np.zeros((n, n))` 同理。
> - **返回多个值**：`return a, b, c` 实际返回一个元组；调用时 `Q, K, V = project(...)` 解包接收。
> - `==` 比较，`and` / `or` / `not` 是逻辑运算（C++ 的 `&&` / `||` / `!`）。

---

## 6. 完整公式与形状

把所有位置写成矩阵，循环就变成了矩阵运算：

```text
① 投影    Q = X W_Q      K = X W_K      V = X W_V
② 打分    S = Q Kᵀ / √d_k
③ 归一化  A = softmax(S + mask)          ← 按行做
④ 汇总    O = A V
```

### 形状表（务必熟记）

| 矩阵     | 形状          | 含义                                                 |
| -------- | ------------- | ---------------------------------------------------- |
| X        | n × d_model   | n 个位置的输入向量                                   |
| W_Q, W_K | d_model × d_k | 投影矩阵                                             |
| W_V      | d_model × d_v | 投影矩阵                                             |
| Q, K     | n × d_k       | 每行是一个位置的 query / key                         |
| V        | n × d_v       | 每行是一个位置的 value                               |
| S        | n × n         | S[i][j] = 位置 i 的 query 和位置 j 的 key 的匹配分数 |
| mask     | n × n         | 0 或 −∞                                              |
| A        | n × n         | A[i][j] = 位置 i 从位置 j 读取的比例，每行和为 1     |
| O        | n × d_v       | 每行是一个位置的输出                                 |

### 形状推导（每一步都要能自己推出来）

```text
X (n×d_model) @ W_Q (d_model×d_k)  →  Q (n×d_k)
Q (n×d_k)     @ Kᵀ (d_k×n)         →  S (n×n)
softmax 不改变形状                  →  A (n×n)
A (n×n)       @ V (n×d_v)          →  O (n×d_v)
```

### 流程图

```text
           ┌── @ W_Q ──→ Q ──┐
           │                 ├─→ Q Kᵀ / √d_k ─→ S ─→ (+mask) ─→ 按行 softmax ─→ A ──┐
  X ───────┼── @ W_K ──→ K ──┘                                                     ├─→ A V ─→ O
           │                                                                       │
           └── @ W_V ──→ V ────────────────────────────────────────────────────────┘
```

mask 第 9 章再讲，这一步先实现不带掩码的版本。

### 🛠 任务 5：实现 `attention`（先不带掩码）

**任务**：在 `my_attention.py` 里添加：

```python
def attention(Q, K, V, causal=False):
    """缩放点积注意力，返回打分 S、权重 A、输出 O"""
    # TODO:
    # 1. 从 Q.shape 拿到 n 和 d_k；检查 K 形状与 Q 相同、V 的行数等于 n
    # 2. S = Q @ K.T / np.sqrt(d_k)
    # 3. A = softmax_rows(S)          （causal 参数先不管，任务 8 再实现）
    # 4. O = A @ V
    # 5. 返回 S, A, O
```

> 为什么要返回 S？因为 OJ 要求输出 S。

**测试**（清空 playground.py）：

```python
import numpy as np
from my_attention import project, attention

np.set_printoptions(precision=6, suppress=True)
I = np.eye(2)                       # 2×2 单位阵
Q, K, V = project(I, I, I, I)
S, A, O = attention(Q, K, V)
print("S =\n", S)
print("A =\n", A)
print("O =\n", O)
print("每行和:", A.sum(axis=1))
```

**预期输出**：

```text
S =
 [[0.707107 0.      ]
 [0.       0.707107]]
A =
 [[0.669762 0.330238]
 [0.330238 0.669762]]
O =
 [[0.669762 0.330238]
 [0.330238 0.669762]]
每行和: [1. 1.]
```

这正是任务书第 4 节给出的"无掩码"数字。**到这里你已经实现了注意力的核心计算。**

<details>
<summary>卡住了再看：参考实现（无掩码版）</summary>

```python
def attention(Q, K, V, causal=False):
    """缩放点积注意力，返回打分 S、权重 A、输出 O"""
    n, d_k = Q.shape
    assert K.shape == (n, d_k), "K 的形状必须与 Q 相同"
    assert V.shape[0] == n, "V 的行数必须等于 n"
    S = Q @ K.T / np.sqrt(d_k)
    A = softmax_rows(S)
    O = A @ V
    return S, A, O
```

</details>

> 🐍 **本任务用到的语法**
>
> - `causal=False`：**默认参数**，调用时不传就是 False。调用时可以按名字传：`attention(Q, K, V, causal=True)`，可读性更好。
> - `np.sqrt(x)`：开方；`np.eye(2)`：2×2 单位阵。
> - `K.shape == (n, d_k)`：元组可以直接比较是否相等。
> - 一个函数可以调用同文件里其他函数（这里调用了 `softmax_rows`），定义顺序不影响，只要调用时已经定义过就行。
> - `"\n"` 是换行；`A.sum(axis=1)` 是每行的和。

---

## 7. 为什么要除以 √d_k

### 7.1 问题：维度越高，点积越大

如果 q 和 k 的每个分量都是均值 0、方差 1 的随机数，那么点积 `q·k = Σ q_t k_t` 是 d_k 项之和，**方差等于 d_k，标准差等于 √d_k**。实际模拟 10000 次的结果：

| d_k | 点积的标准差 | 除以 √d_k 后的标准差 |
| --- | ------------ | -------------------- |
| 2   | 1.42         | 1.01                 |
| 16  | 4.01         | 1.00                 |
| 64  | 8.02         | 1.00                 |
| 512 | 22.60        | 1.00                 |

### 7.2 后果：softmax 变得"极端"

分数差距大时，softmax 几乎把全部权重给最大的那一项：

```text
softmax([16, 8, 0])     = [0.999665, 0.000335, 0.000000]   ← 几乎是 one-hot
softmax([16, 8, 0] / 8) = softmax([2, 1, 0])
                        = [0.665241, 0.244728, 0.090031]   ← 分布平缓得多
```

权重太极端有两个坏处：每个位置几乎只看一个位置；训练时梯度接近 0，模型学不动。

### 7.3 解决：除以 √d_k

除以 √d_k 后，分数的标准差回到 1 左右，与维度无关。这就是"**缩放**点积注意力"中"缩放"二字的由来。

> 📌 注意是 `√d_k`（Q/K 的维数），不是 d_model，也不是 d_v。

### 🛠 任务 6：亲眼看看缩放的作用（playground.py，15 分钟）

清空 playground.py，运行：

```python
import numpy as np
from my_attention import softmax_rows

np.set_printoptions(precision=6, suppress=True)
s = np.array([[16.0, 8.0, 0.0]])
print("不缩放:", softmax_rows(s))
print("除以 8:", softmax_rows(s / 8))

rng = np.random.default_rng(0)        # 固定随机种子
for d in [2, 16, 64, 512]:
    q = rng.standard_normal((10000, d))
    k = rng.standard_normal((10000, d))
    dots = (q * k).sum(axis=1)        # 10000 组 q·k
    print(f"d_k={d:4d}  点积标准差={dots.std():6.2f}  缩放后={(dots / np.sqrt(d)).std():.2f}")
```

输出应该和 7.1、7.2 的两张表一致。

> 🐍 **本任务用到的语法**
>
> - `for d in [2, 16, 64, 512]:`：遍历列表里的每个元素（C++ 的 `for (auto d : {...})`），循环体同样靠缩进。
> - `f"...{表达式}..."`：**f-string**，大括号里直接写变量或表达式。`{d:4d}` 是宽度 4 的整数，`{x:.2f}` 是保留 2 位小数，相当于 printf 的 `%4d`、`%.2f`。
> - `np.random.default_rng(0)`：固定种子的随机数生成器，每次运行结果一样。`standard_normal((行, 列))` 生成标准正态随机矩阵。
> - `(q * k).sum(axis=1)`：这里正是 `*` 的正确用法：先逐元素相乘，再按行求和，得到每一行的点积。

---

## 8. softmax 沿哪一维（Q3）

S 是 n×n 矩阵：**第 i 行**是"位置 i 的 query 和所有 key 的分数"。

我们要回答的问题是"**位置 i 该如何把注意力分配给各个位置**"，所以应该**对每一行**单独归一化，使每一行加起来等于 1。

```text
             key 0   key 1   key 2
query 0  [  s00     s01     s02  ]  ← 对这一行做 softmax：位置 0 的分配方案
query 1  [  s10     s11     s12  ]  ← 对这一行做 softmax：位置 1 的分配方案
query 2  [  s20     s21     s22  ]  ← 对这一行做 softmax：位置 2 的分配方案
```

在 NumPy 里就是 `axis=1`（配合 `keepdims=True`）。

**如果沿列做会怎样？** 每一列加起来为 1，意思变成"位置 j 被所有人读取的总量为 1"，这不是我们想要的；之后 `A @ V` 的第 i 行也就不再是一个加权平均。

> 📌 **Q3 的答案核心**：对每个 query 的所有候选 key 做归一化，在本项目的矩阵约定下就是**对每一行**做 softmax。

### 🛠 任务 7：对比两种 axis，写 Q3

在 playground.py 里运行，观察行和、列和的区别：

```python
import numpy as np

np.set_printoptions(precision=6, suppress=True)
S = np.array([[1.0, 2.0, 3.0],
              [0.0, 0.0, 5.0],
              [2.0, 2.0, 2.0]])

E = np.exp(S - S.max(axis=1, keepdims=True))
A_row = E / E.sum(axis=1, keepdims=True)    # 正确：按行
E = np.exp(S - S.max(axis=0, keepdims=True))
A_col = E / E.sum(axis=0, keepdims=True)    # 错误：按列

print("按行：行和", A_row.sum(axis=1), " 列和", A_row.sum(axis=0))
print("按列：行和", A_col.sum(axis=1), " 列和", A_col.sum(axis=0))
```

按行的版本每行和都是 1；按列的版本每列和是 1，行和却不是。

然后在 `answers.md` 的 **Q3** 下写草稿：沿哪一维、为什么、代码里怎么体现（`axis=1`）。

---

## 9. 因果掩码（Q4）

### 9.1 为什么需要它

像 GPT 这样的模型是**一个字一个字往后生成**的：写第 i 个字时，后面的字还不存在。为了让训练和生成保持一致，训练时也必须规定：**第 i 个位置只能看到位置 0..i，不能偷看 j > i 的"未来"位置**。

否则模型在训练时"抄答案"（直接看到下一个字），到真正生成时就不会了。

### 9.2 怎么做

构造一个 n×n 的掩码矩阵：**允许的位置填 0，禁止的位置（严格上三角，j > i）填 −∞**。

```text
n = 4 时：
mask = [[ 0, -∞, -∞, -∞],     ← 位置 0 只能看位置 0
        [ 0,  0, -∞, -∞],     ← 位置 1 能看 0, 1
        [ 0,  0,  0, -∞],     ← 位置 2 能看 0, 1, 2
        [ 0,  0,  0,  0]]     ← 位置 3 能看全部
```

**在 softmax 之前**把它加到 S 上：`A = softmax(S + mask)`。

- 允许的位置加 0，分数不变；
- 禁止的位置变成 −∞，softmax 后权重恰好为 0；
- 每行剩下的权重自动重新归一化，**行和仍然是 1**。

**对角线要保留**（i 可以看自己）。这保证每一行至少有一个有限的分数，否则一整行都是 −∞，softmax 会算出 `0/0 = nan`。

### 9.3 ❌ 错误做法：先 softmax，再把未来位置置零

用第 10.2 节的 3×3 例子，第 1 行（位置 1）的分数是 `[0, 0.7071, 0.7071]`：

```text
✅ 正确：softmax([0, 0.7071, -∞])        = [0.330238, 0.669762, 0]          行和 = 1
❌ 错误：softmax([0, 0.7071, 0.7071])    = [0.197776, 0.401112, 0.401112]
         再把未来位置置零                  = [0.197776, 0.401112, 0]          行和 = 0.598888 ✗
```

错误做法的行和不为 1，输出 O 就不再是加权平均，整体被"缩小"了。

> 📌 **Q4 的答案核心**：第 i 行不能访问 j > i 的未来位置；在 softmax **之前**把这些位置的分数设为 −∞（加上掩码），保留对角线。

### 🛠 任务 8：实现 `causal_mask`，给 `attention` 加上掩码，写 Q4

**第 1 步**：在 `my_attention.py` 里添加：

```python
def causal_mask(n):
    """因果掩码：严格上三角为 -inf，其余为 0"""
    # TODO: 先建 n×n 全 0 矩阵，再用双重循环把 j > i 的位置设成 -np.inf
```

**第 2 步**：修改任务 5 写的 `attention`，把 `A = softmax_rows(S)` 改成：causal 为真时对 `S + causal_mask(n)` 做 softmax，否则对 S 做。

> ⚠️ **不要写 `S = S + mask`**。OJ 要求输出的 S 是**不含掩码**的 `Q Kᵀ / √d_k`，所以直接把 `S + causal_mask(n)` 传给 softmax，保持 S 本身不变。

**测试**（清空 playground.py）：

```python
import numpy as np
from my_attention import causal_mask, attention

np.set_printoptions(precision=6, suppress=True)
print(causal_mask(3))
print(causal_mask(1))

I = np.eye(2)
S, A, O = attention(I, I, I, causal=True)
print("S（不含掩码）=\n", S)
print("A（因果）=\n", A)
```

**预期输出**：

```text
[[  0. -inf -inf]
 [  0.   0. -inf]
 [  0.   0.   0.]]
[[0.]]
S（不含掩码）=
 [[0.707107 0.      ]
 [0.       0.707107]]
A（因果）=
 [[1.       0.      ]
 [0.330238 0.669762]]
```

最后在 `answers.md` 的 **Q4** 下写草稿：屏蔽什么、什么时候屏蔽（softmax 前）、怎么屏蔽（加 −∞）、为什么不能先 softmax 再置零（引用 9.3 的数字）。

<details>
<summary>卡住了再看：参考实现</summary>

```python
def causal_mask(n):
    """因果掩码：严格上三角为 -inf，其余为 0"""
    mask = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mask[i, j] = -np.inf
    return mask


def attention(Q, K, V, causal=False):
    """缩放点积注意力，返回打分 S、权重 A、输出 O"""
    n, d_k = Q.shape
    assert K.shape == (n, d_k), "K 的形状必须与 Q 相同"
    assert V.shape[0] == n, "V 的行数必须等于 n"
    S = Q @ K.T / np.sqrt(d_k)
    if causal:
        A = softmax_rows(S + causal_mask(n))
    else:
        A = softmax_rows(S)
    O = A @ V
    return S, A, O
```

`causal_mask` 的等价一行写法：`np.triu(np.full((n, n), -np.inf), k=1)`（`triu` 取上三角，`k=1` 表示不含对角线）。

</details>

> 🐍 **本任务用到的语法**
>
> - `for i in range(n):` 就是 `for (int i = 0; i < n; i++)`；`range(i + 1, n)` 是 i+1 到 n−1（**左闭右开**）。
> - `mask[i, j] = ...`：给矩阵第 i 行第 j 列赋值（也可写 `mask[i][j]`）。
> - `if 条件:` / `else:`，冒号加缩进；`else if` 在 Python 里写作 `elif`。
> - `if causal:` 直接判断布尔值，不用写 `== True`。

---

## 10. 手算两个完整例子（Q5）

### 🛠 任务 9：纸笔手算 → 程序核对 → 写 Q5

**拿出纸笔，先盖住答案自己算，再对照。** 这一章是 Q5 的标准答案过程，也是理解整个注意力最好的方式。

### 10.1 例一：Q5（2×2 单位阵）

已知：`Q = K = V = I = [[1,0],[0,1]]`，`d_k = 2`。

**第 1 步：算 S**

```text
Q Kᵀ = I · I = [[1, 0],
                [0, 1]]
√d_k = √2 ≈ 1.414214
S = [[1/√2,    0   ],    =  [[0.707107, 0       ],
     [  0,   1/√2 ]]         [0,        0.707107]]
```

**第 2 步：无掩码，按行 softmax**

第 0 行 `[0.707107, 0]`：

```text
先减最大值 0.707107 → [0, -0.707107]
e^0 = 1,  e^-0.707107 = 0.493069
总和 = 1.493069
A[0] = [1/1.493069, 0.493069/1.493069] = [0.669762, 0.330238]
```

第 1 行 `[0, 0.707107]` 是对称的：`A[1] = [0.330238, 0.669762]`。

```text
A（无掩码）= [[0.669762, 0.330238],
             [0.330238, 0.669762]]
```

**第 3 步：因果掩码**

```text
S + mask = [[0.707107,   -∞    ],
            [0,        0.707107]]
第 0 行：softmax([0.707107, -∞]) = [1, 0]    ← 只能看自己
第 1 行：不受影响，仍为 [0.330238, 0.669762]

A（因果）= [[1,        0       ],
           [0.330238, 0.669762]]
```

**第 4 步：算 O**

V 是单位阵，`O = A · I = A`，所以两种情况下 O 都等于对应的 A。

**解释第一行的变化**：无掩码时，位置 0 有约 33% 的信息来自位置 1（未来位置）；加上因果掩码后，位置 0 看不到位置 1，只能 100% 读取自己。第二行本来就只涉及位置 0 和 1（都不是位置 1 的"未来"），所以不变。

### 10.2 例二：3×3 带因果掩码

已知：`X = [[1,0],[0,1],[1,1]]`，`W_Q = W_K = W_V = I`（2×2），所以 `Q = K = V = X`，`d_k = 2`，开启因果掩码。

**S：**

```text
Q Kᵀ = [[1, 0, 1],          S = Q Kᵀ / √2 = [[0.707107, 0,        0.707107],
        [0, 1, 1],                           [0,        0.707107, 0.707107],
        [1, 1, 2]]                           [0.707107, 0.707107, 1.414214]]
```

**加掩码后按行 softmax：**

```text
第 0 行：[0.707107, -∞, -∞]       → [1, 0, 0]
第 1 行：[0, 0.707107, -∞]        → [0.330238, 0.669762, 0]          （同例一）
第 2 行：[0.707107, 0.707107, 1.414214]，减去最大值 1.414214：
        [-0.707107, -0.707107, 0] → e: [0.493069, 0.493069, 1]，总和 1.986138
        → [0.248255, 0.248255, 0.503490]
```

**O = A V：**

```text
O[0] = 1 × [1,0]                                         = [1, 0]
O[1] = 0.330238 × [1,0] + 0.669762 × [0,1]               = [0.330238, 0.669762]
O[2] = 0.248255 × [1,0] + 0.248255 × [0,1] + 0.503490 × [1,1]
     = [0.248255 + 0.503490, 0.248255 + 0.503490]        = [0.751745, 0.751745]
```

### 任务 9 要做的事

1. **纸笔**：独立算出例一（两种情况）和例二，和上面对照。
2. **程序核对**：清空 playground.py，运行下面的代码，输出应与手算一致：

   ```python
   import numpy as np
   from my_attention import project, attention

   np.set_printoptions(precision=6, suppress=True)
   X = np.array([[1.0, 0.0],
                 [0.0, 1.0],
                 [1.0, 1.0]])
   W = np.eye(2)
   Q, K, V = project(X, W, W, W)
   S, A, O = attention(Q, K, V, causal=True)
   print("S =\n", S)
   print("A =\n", A)
   print("O =\n", O)
   ```

   再运行一次 `python3 ../../attention.py`（课程参考代码），它第 3 个实验的"O 原输入"也是这组数。现在你能看懂它的全部输出了。

3. **写 Q5**：把例一的完整手算过程（S → 逐行 softmax → A → O，两种情况）和"第一行为什么变化"的解释写进 `answers.md`。

---

## 11. 打印与性质检查

到这里核心计算已经全部完成。从这一章开始，是把它变成**可验收的提交**。

任务书的验收项：**与手算相符、行和为 1、非负、被屏蔽位置为 0、形状正确、单位置时 O=V、开掩码后只改未来输入时较早位置的输出不变**。容差 `1e-6`。

### 🛠 任务 10：添加 `show` 和 `check`

**第 1 步**：添加打印工具（任务书要求"打印 Q/K/V、S、A、O 的形状和数值"），这个直接照抄：

```python
def show(name, M):
    """打印矩阵的名字、形状和数值"""
    print(f"{name}  shape = {M.shape}")
    print(np.round(M, 6))
    print()
```

**第 2 步**：写 `check(A, O, V, causal, tol=1e-6)`，用 `assert` 检查：

- A 是 n×n，O 是 n×d_v；
- A 所有元素 ≥ 0；
- A 每行和与 1 的差不超过 tol；
- 如果 causal，A 的严格上三角全为 0；
- 全部通过后打印一行"检查通过"。

**测试**（清空 playground.py）：

```python
import numpy as np
from my_attention import project, attention, show, check

I = np.eye(2)
Q, K, V = project(I, I, I, I)
for causal in [False, True]:
    S, A, O = attention(Q, K, V, causal=causal)
    show("A", A)
    check(A, O, V, causal)

# 故意造一个坏的 A，check 应该报错
bad = np.array([[0.5, 0.4],
                [0.3, 0.7]])
check(bad, bad, V, False)
```

前面应打印两次"检查通过"，最后一行应报 `AssertionError: 存在行和不为 1 的行`（或你自己写的提示）。

<details>
<summary>卡住了再看：参考实现</summary>

```python
def check(A, O, V, causal, tol=1e-6):
    """检查注意力结果的基本性质"""
    n = A.shape[0]
    assert A.shape == (n, n), "A 必须是 n×n"
    assert O.shape == (n, V.shape[1]), "O 必须是 n×d_v"
    assert np.all(A >= 0), "权重出现负数"
    assert np.allclose(A.sum(axis=1), 1.0, atol=tol), "存在行和不为 1 的行"
    if causal:
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        assert np.all(A[upper] == 0), "被屏蔽的未来位置权重不为 0"
    print(f"[检查通过] n={n}, causal={causal}: 形状正确、非负、行和为 1"
          + ("、未来位置为 0" if causal else ""))
```

</details>

> 🐍 **本任务用到的语法**
>
> - `A >= 0`：对每个元素比较，得到一个**布尔矩阵**；`np.all(...)` 判断是否全为 True，`np.any(...)` 判断是否有 True。
> - `np.allclose(a, b, atol=1e-6)`：浮点数近似相等（浮点数不要用 `==` 比较）。`a` 是数组、`b` 是一个数时，会拿每个元素和 b 比。
> - `np.triu(..., k=1)`：取严格上三角；`dtype=bool` 让矩阵元素是布尔值。
> - `A[upper]`：**布尔下标**，取出 upper 为 True 的那些位置的元素。
> - `np.round(M, 6)`：保留 6 位小数。
> - `"a" if 条件 else "b"`：三目运算符，即 C++ 的 `条件 ? "a" : "b"`。字符串可以用 `+` 拼接。
> - 一行太长时，在括号内可以直接换行。

---

## 12. 三个验收实验

### 🛠 任务 11：在 my_attention.py 里写三个实验函数

每个实验写成一个函数，最后在任务 12 的 `main()` 里统一调用。

**实验 1：Q5 手算核对** `experiment_q5()`

- 用 `I = np.eye(2)` 投影出 Q、K、V；
- 对 `causal` 取 False 和 True 两种情况：算出 S、A、O，用 `show` 打印，用 `check` 检查；
- 最后用 `np.allclose` 和任务书给的数字 `[[0.669762, 0.330238], [0.330238, 0.669762]]` 比较，打印是否一致。

**实验 2：只有一个位置时 O = V** `experiment_single()`

- `X = np.array([[2.0, 3.0]])`（1×2），W 用单位阵；
- 打印 A，打印 `np.allclose(O, V)` 的结果。
- **原理**：n = 1 时 S 是 1×1，softmax 一个数的结果必然是 1，所以 `O = 1 × V = V`。

**实验 3：因果性（只改未来，前面不变）** `experiment_causality()`

- 用固定种子随机生成 X（4×3）和三个 W；
- 复制一份 X 得到 X2，**只改最后一个位置**：`X2[-1] = [9.0, 9.0, 9.0]`；
- 分别在 causal=True 和 causal=False 下算 O1（用 X）和 O2（用 X2），打印前 3 个位置输出的最大绝对误差 `np.max(np.abs(O1[:-1] - O2[:-1]))`；
- **预期**：开启掩码时误差为 0（前面的位置看不到最后一个位置）；关闭掩码时误差明显大于 0。

**测试**（清空 playground.py）：

```python
from my_attention import experiment_q5, experiment_single, experiment_causality
import numpy as np

np.set_printoptions(precision=6, suppress=True)
experiment_q5()
experiment_single()
experiment_causality()
```

用参考实现里的种子（`default_rng(0)`），实验 3 应输出：

```text
causal=True: 前 3 个位置输出的最大变化 = 0.000e+00
causal=False: 前 3 个位置输出的最大变化 = 4.993e+00
```

<details>
<summary>卡住了再看：参考实现</summary>

```python
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
        print()
    expected = np.array([[0.669762, 0.330238],
                         [0.330238, 0.669762]])
    _, A, _ = attention(Q, K, V)
    print("与任务书数值一致:", np.allclose(A, expected, atol=1e-6))
    print()


def experiment_single():
    print("========== 实验 2：只有一个位置 ==========")
    X = np.array([[2.0, 3.0]])
    W = np.eye(2)
    Q, K, V = project(X, W, W, W)
    S, A, O = attention(Q, K, V)
    show("A", A)
    print("O == V:", np.allclose(O, V))
    print()


def experiment_causality():
    print("========== 实验 3：因果性（只改最后一个位置）==========")
    rng = np.random.default_rng(0)
    n, d_model, d_k, d_v = 4, 3, 2, 2
    X = rng.standard_normal((n, d_model))
    W_Q = rng.standard_normal((d_model, d_k))
    W_K = rng.standard_normal((d_model, d_k))
    W_V = rng.standard_normal((d_model, d_v))

    X2 = X.copy()
    X2[-1] = [9.0, 9.0, 9.0]          # 只改最后一个位置

    for causal in [True, False]:
        _, _, O1 = attention(*project(X, W_Q, W_K, W_V), causal=causal)
        _, _, O2 = attention(*project(X2, W_Q, W_K, W_V), causal=causal)
        diff = np.max(np.abs(O1[:-1] - O2[:-1]))
        print(f"causal={causal}: 前 {n-1} 个位置输出的最大变化 = {diff:.3e}")
    print()
```

</details>

> 🐍 **本任务用到的语法**
>
> - ⚠️ `X2 = X.copy()`：**必须用 copy**。Python 里 `X2 = X` 不会拷贝，只是给同一个数组起了第二个名字（类似 C++ 的引用 `auto& X2 = X;`），改 X2 就会把 X 也改掉，实验就白做了。
> - **负下标**：`X2[-1]` 是最后一行，`X[-2]` 是倒数第二行。
> - **切片**：`O1[:-1]` 是"除最后一行以外的所有行"；`a[1:3]` 是第 1、2 行（左闭右开）；`a[:2]` 是前两行。
> - `_, A, _ = attention(...)`：`_` 是"这个返回值我不要"的惯用写法。
> - `attention(*project(...), causal=causal)`：`*` 是**解包**，把 `project` 返回的 `(Q, K, V)` 拆成三个参数传进去。
> - `n, d_model, d_k, d_v = 4, 3, 2, 2`：一行给多个变量赋值。
> - `{diff:.3e}`：科学计数法保留 3 位小数。
> - `np.abs`：逐元素绝对值；`np.max`：所有元素的最大值。

> （选做）**实验 4：打乱顺序**。无掩码时把输入的行打乱（`X[[2, 0, 1]]`），输出也会按同样的方式打乱。这说明注意力本身**不知道位置顺序**，所以完整的 Transformer 需要"位置编码"（Advance 内容）。`attention.py` 的第 4 个实验演示了这一点。

---

## 13. 热力图与主函数

热力图把矩阵 A 画成色块：颜色越亮，权重越大。行是 query（谁在读），列是 key（被读的位置）。

### 🛠 任务 12：添加 `plot_heatmap` 和 `main`，让 my_attention.py 能独立运行

**第 1 步**：画图代码不是本课程的重点，**直接照抄**下面的函数到 `my_attention.py`：

```python
def plot_heatmap(A, title, filename):
    """把注意力矩阵画成热力图并保存"""
    import matplotlib
    matplotlib.use("Agg")              # 只保存图片、不弹窗
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
```

**第 2 步**：在文件**最末尾**写 `main()`，并加上程序入口：

```python
def main():
    np.set_printoptions(precision=6, suppress=True)
    experiment_q5()
    experiment_single()
    experiment_causality()

    # TODO: 用 np.random.default_rng(42) 随机生成 n=5, d_model=4, d_k=3, d_v=3 的 X 和三个 W，
    #       投影后对 causal = False / True 各算一次 attention，
    #       用 show 打印 Q、K、V、S、A、O，用 check 检查，
    #       再用 plot_heatmap 分别保存为 heatmap_nomask.png 和 heatmap_causal.png


if __name__ == "__main__":
    main()
```

**运行**：

```bash
python3 my_attention.py
```

目录下应出现两张图。**怎么看图**：

- 无掩码的图：每一格都可能有颜色；
- 因果掩码的图：**右上三角全为 0（深色）**，第 0 行只有对角线那一格是 1。

<details>
<summary>卡住了再看：main 的参考实现</summary>

```python
def main():
    np.set_printoptions(precision=6, suppress=True)
    experiment_q5()
    experiment_single()
    experiment_causality()

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
```

</details>

> 🐍 **本任务用到的语法**
>
> - `if __name__ == "__main__": main()`：**程序入口**，相当于 C++ 的 `int main()`。直接运行 `python3 my_attention.py` 时会执行 `main()`；被 playground.py 或 oj_main.py `import` 时**不会**执行。所以 my_attention.py 既能单独运行，又能被别的文件当工具库用。
> - 函数内部也可以 `import`。
> - 标签用英文是为了避免中文显示成方框；想用中文可以在 `import matplotlib.pyplot as plt` 后加 `plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]`。

> ⚠️ 任务书特别强调：**随机权重的热力图只说明运算过程，不能说明模型"理解了语义"**。报告里描述图时要注意措辞，比如写"在这组随机参数下，位置 3 对位置 0 的权重最大"，而不是"位置 3 关注了位置 0 的语义"。

---

## 14. 矩阵日志与自己的例子

### 🛠 任务 13：生成日志，加一个自己设计的例子

**第 1 步：加一个自己的例子**。任务书要求"先复现数字，再用自己的小矩阵说明过程"。在 `my_attention.py` 里加一个 `experiment_mine()`（记得在 `main()` 里调用）：

- 自己手写一个 n=3、d_model=2 的 X 和三个 2×2 的 W（数字简单些，比如 0、1、2）；
- 程序算出 S、A、O 并打印；
- **纸笔手算其中一行**的 A 和 O，和程序输出对照，过程写进报告。

**第 2 步：生成日志**

```bash
python3 my_attention.py | tee log.txt
```

`tee` 会把输出同时显示在屏幕上并写进 `log.txt`。检查日志里是否包含：Q/K/V/S/A/O 的形状和数值、每次检查的"检查通过"、三个实验的结论。

---

## 15. OJ 3280

### 题目格式回顾

- 输入第一行：`n d_model d_k d_v causal`；
- 接着依次是 X、W_Q、W_K、W_V，全部**按行优先**排列的浮点数；
- 输出：依次输出 Q、K、V、S、A、O 的所有元素（按行优先）；
- 误差容限：绝对 1e-6 加相对 2e-5。

### 🛠 任务 14：写 oj_main.py

**先确认两件事**（看 OJ 网站说明）：OJ 支持 Python 吗？装了 NumPy 吗？

- 都支持：按下面的方法写；
- 不支持 NumPy：要用 Python 列表加循环自己实现矩阵乘法，或者改用 C++ 提交（逻辑完全一样，你已经全部理解了）。

**任务**：在 `oj_main.py` 里实现：

1. 一次性读入所有输入：`data = sys.stdin.read().split()`；
2. 前 5 个数是 `n d_model d_k d_v causal`；
3. 依次读 X（n×d_model）、W_Q（d_model×d_k）、W_K（d_model×d_k）、W_V（d_model×d_v），建议写一个 `read_matrix` 辅助函数；
4. 调用 `project` 和 `attention`；
5. 依次输出 Q、K、V、S、A、O，每个数保留 10 位小数。

要点：

- 按行优先读入正好对应 `np.array(列表).reshape(rows, cols)`；
- 输出的 S **不含掩码**，所以输出里不会出现 `-inf`；
  > ⚠️ **题面这里有歧义**：公式写的是 `S = QKᵀ/√d_k`、`A = softmax(S + mask)`，按公式 S 不含掩码；但"注意事项 2"又写了"在 softmax 之前加入 S_ij = −∞"。**先按不含掩码提交**。如果 `causal=0` 的数据全对、`causal=1` 的数据错了，再试着输出加了掩码的 S（Python 打印 −∞ 会得到 `-inf`），或者问助教确认。
- **OJ 一般只能提交一个文件**：提交时把 `softmax_rows`、`causal_mask`、`project`、`attention` 四个函数**复制进 oj_main.py**，删掉 `from my_attention import ...` 这一行。

**本地测试**：新建 `test_q5.txt`（Q5 的数据，开启因果掩码）：

```text
2 2 2 2 1
1 0
0 1
1 0
0 1
1 0
0 1
1 0
0 1
```

运行：

```bash
python3 oj_main.py < test_q5.txt
```

预期输出（Q、K、V、S、A、O 各 2 行）：

```text
1.0000000000 0.0000000000
0.0000000000 1.0000000000
1.0000000000 0.0000000000
0.0000000000 1.0000000000
1.0000000000 0.0000000000
0.0000000000 1.0000000000
0.7071067812 0.0000000000
0.0000000000 0.7071067812
1.0000000000 0.0000000000
0.3302384507 0.6697615493
1.0000000000 0.0000000000
0.3302384507 0.6697615493
```

再自己造几组边界数据：`n = 1`；`d_k ≠ d_v`（比如第一行写 `3 4 2 5 0`，后面跟对应个数的数字）；数值很大的输入（检查 softmax 稳定性）。

<details>
<summary>卡住了再看：参考实现（多文件版本）</summary>

```python
import sys
import numpy as np
from my_attention import project, attention   # 单文件提交时：删掉这行，把四个核心函数复制到这里


def read_matrix(data, pos, rows, cols):
    """从 data[pos:] 读取 rows*cols 个数，按行优先摆成矩阵；返回矩阵和新位置"""
    M = np.array([float(t) for t in data[pos:pos + rows * cols]]).reshape(rows, cols)
    return M, pos + rows * cols


def print_matrix(M):
    for row in M:
        print(" ".join(f"{x:.10f}" for x in row))


def main():
    data = sys.stdin.read().split()
    n, d_model, d_k, d_v, causal = (int(t) for t in data[:5])
    pos = 5
    X, pos = read_matrix(data, pos, n, d_model)
    W_Q, pos = read_matrix(data, pos, d_model, d_k)
    W_K, pos = read_matrix(data, pos, d_model, d_k)
    W_V, pos = read_matrix(data, pos, d_model, d_v)

    Q, K, V = project(X, W_Q, W_K, W_V)
    S, A, O = attention(Q, K, V, causal=(causal == 1))
    for M in (Q, K, V, S, A, O):
        print_matrix(M)


if __name__ == "__main__":
    main()
```

</details>

> 🐍 **本任务用到的语法**
>
> - `import sys` + `sys.stdin.read()`：读入全部标准输入；`.split()` 按任意空白（空格、换行）切成字符串列表。效果类似 C++ 的 `while (cin >> s)`，不用关心换行在哪。
> - `int("3")`、`float("1.5")`：字符串转数字。
> - **列表推导式**：`[float(t) for t in 列表]`，对每个元素做转换得到新列表，相当于一个简写的 for 循环。
> - **切片**：`data[pos:pos + k]` 取出第 pos 到 pos+k−1 个元素。
> - `.reshape(rows, cols)`：把一维数组按行优先摆成矩阵。
> - `" ".join(字符串列表)`：用空格把字符串拼起来；`f"{x:.10f}"` 保留 10 位小数。
> - `for row in M:` 遍历矩阵的每一行；`for M in (Q, K, V, S, A, O):` 遍历元组里的每个矩阵。
> - `python3 oj_main.py < test_q5.txt`：把文件内容当标准输入喂给程序（和 C++ 一样）。

---

## 16. 整理 Q1–Q5 并提交

### 🛠 任务 15：润色 answers.md，对照清单提交

把前面写的草稿整理成完整回答，尽量配上本项目里的具体数字或实验结果。下面是每题的**要点**，检查有没有漏：

**Q1：token 和向量是什么关系？**

- token 是序列被切分后的基本单元，由分词器决定，不一定恰好是一个汉字或一个单词（可能是子词、字符等）；
- 模型不能直接处理 token，需要通过嵌入表等方式把每个 token 映射成 d_model 维向量；
- 长度为 n 的序列就变成 n × d_model 的矩阵 X，每行对应一个位置；
- 本项目中向量是手写或随机固定的，不代表真实语义。

**Q2：Q、K、V 各参与哪一步？**

- Q、K、V 都由 X 乘以各自的投影矩阵得到；
- Q 与 K 做点积（`S = QKᵀ/√d_k`）得到位置之间的匹配分数，因此二者维数必须相同；
- 分数经 softmax 归一化成权重 A；
- 用 A 对 V 的各行加权求和得到输出 `O = AV`；
- 可以用"查询—标签—内容"的检索类比说明。

**Q3：softmax 应沿哪一维计算？**

- 沿每一行（`axis=1`）；
- 原因：第 i 行是第 i 个 query 对所有候选 key 的分数，需要把这些候选的权重归一化成和为 1；
- 实现时先减去每行最大值，保证数值稳定；可以引用任务 7 中按列做时行和不为 1 的现象。

**Q4：因果掩码屏蔽什么？**

- 屏蔽第 i 行中 j > i 的未来位置（严格上三角），保留对角线；
- 在 softmax **之前**给这些位置加 −∞，softmax 后权重为 0，且每行和仍为 1；
- 不能先 softmax 再置零，那样行和小于 1（第 9.3 节的数字：0.598888）；
- 意义：自回归生成时不能看到尚未生成的内容；用实验 3 的结果佐证（掩码下前面位置的输出变化为 0）。

**Q5：Q=K=V 为 2×2 单位阵，d_k=2，A 和 O 是多少？**

- 写出第 10.1 节的完整计算过程：S → 逐行 softmax → A → O；
- 无掩码：`A = O = [[0.669762, 0.330238], [0.330238, 0.669762]]`；
- 因果掩码：`A = O = [[1, 0], [0.330238, 0.669762]]`；
- 解释：因为 V = I，所以 O = A；第一行从"读 33% 的位置 1"变成"只读自己"，第二行不受影响。

### 提交清单

- [ ] `my_attention.py`：`softmax_rows`、`project`、`attention`、`causal_mask`、`show`、`check`、三个实验 + 自己的例子、`plot_heatmap`、`main`
- [ ] `oj_main.py`：通过 OJ 3280
- [ ] `log.txt`：Q/K/V/S/A/O 的形状和数值、检查结果、实验结论
- [ ] `heatmap_nomask.png`、`heatmap_causal.png`
- [ ] `answers.md`：Q1–Q5（Q5 附完整手算过程），自己例子的手算核对
- [ ] `playground.py` 是草稿，**不用提交**

---

## 附录 A：常见错误清单

遇到 bug 先打印各个矩阵的 `.shape`，再对照这张表：

| 错误                                | 后果                                 | 正确做法                              |
| ----------------------------------- | ------------------------------------ | ------------------------------------- |
| 用 `*` 做矩阵乘法                   | 形状报错或结果全错                   | 矩阵乘法用 `@`                        |
| softmax 忘记减最大值                | 大数时出现 `inf` / `nan`             | 先 `S - S.max(axis=1, keepdims=True)` |
| `max`/`sum` 忘了 `keepdims=True`    | 报错，或方阵时**静默算错**（按列减） | 按行操作一律加 `keepdims=True`        |
| softmax 用 `axis=0`                 | 列和为 1，含义错误                   | `axis=1`                              |
| 除以 `sqrt(d_model)` 或 `sqrt(d_v)` | 数值不对                             | 除以 `sqrt(d_k)`                      |
| 先 softmax 再把未来位置置零         | 行和 < 1                             | softmax 之前加 `-inf`                 |
| 掩码把对角线也屏蔽了                | 第 0 行全是 `-inf`，得到 `nan`       | 只屏蔽严格上三角 `j > i`              |
| OJ 输出的 S 加了掩码                | 输出含 `-inf`，答案错误              | S 输出不含掩码                        |
| `X2 = X` 后修改 X2                  | 原来的 X 也被改掉，实验结论错误      | `X2 = X.copy()`                       |
| `np.zeros(n, n)`                    | 报错                                 | `np.zeros((n, n))`，形状要用元组      |
| 输出精度不够（如 `.3f`）            | 超出 1e-6 容差                       | 用 `.10f`                             |
| `IndentationError`                  | 无法运行                             | 缩进统一用 4 个空格，不要混用 Tab     |
| `ModuleNotFoundError: my_attention` | 导入失败                             | 在 `basic/my_solution/` 目录下运行    |

## 附录 B：自测题（交之前过一遍）

1. X 是 6×16，W_Q 是 16×8，W_V 是 16×4。S、A、O 的形状分别是什么？
2. `softmax([3, 3, 3])` 等于多少？`softmax([0, -∞, -∞])` 呢？
3. 如果把 S 的每个元素都加上 100，A 会变吗？为什么？
4. 如果把 S 的每个元素都乘以 2，A 会变得更"平均"还是更"集中"？
5. n = 4 开启因果掩码时，A 中一共有多少个元素被强制为 0？
6. 为什么"只有一个位置时 O = V"？
7. 开启因果掩码时，修改第 0 个位置的输入，哪些位置的输出可能改变？

<details>
<summary>答案</summary>

1. Q、K 是 6×8，V 是 6×4；S 和 A 都是 6×6；O 是 6×4。
2. `[1/3, 1/3, 1/3]`；`[1, 0, 0]`。
3. 不变。softmax 对整行加同一个常数不变（分子分母同乘 e¹⁰⁰ 后约掉），这也是"减最大值"技巧成立的原因。
4. 更集中。分数差距被放大，softmax 更接近 one-hot（类比第 7 章：除以 √d_k 是为了让它更平缓）。
5. 严格上三角共 `n(n-1)/2 = 6` 个。
6. S 是 1×1，softmax 一个数的结果必然是 1，所以 O = 1 · V = V。
7. 所有位置都可能改变，因为每个位置 i ≥ 0 都能看到位置 0。因果掩码只保证"改后面不影响前面"，不保证"改前面不影响后面"。

</details>

## 附录 C：延伸阅读

- [NumPy 入门](https://numpy.org/doc/stable/user/absolute_beginners.html)：数组形状、转置、矩阵乘法
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)：先读 _Scaled Dot-Product Attention_ 一节，对照本教程第 6 章
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：原论文第 3.2.1 节（Advance 阶段精读）

完成 Basic 后，Advance 会在此基础上加入**多头注意力、位置编码、前馈网络、残差连接和 LayerNorm**，组成一个完整的 Transformer 块。
