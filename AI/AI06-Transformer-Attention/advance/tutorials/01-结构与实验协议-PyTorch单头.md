# 阶段 1：读懂 Transformer 结构 + 实验协议 + PyTorch 单头（第 1 周）

> **怎么用**：和 basic 教程 02 一样，从头往下读，读到 **🛠 任务** 就去做，做完再往下读。
> 每个任务后面有 **🔥 本任务用到的 PyTorch**，不用先学 PyTorch。
>
> **本阶段对应**：G1（原始 Transformer 结构）、G2（实验协议），以及 G3 的第一步：单头对拍。
> **做完你会有**：一张结构图初稿、一篇阅读笔记、一份所有实验共用的 `protocol.py`、
> 一个能直接扩展成多头的 `attn.py`，以及一张"PyTorch 版 = 你的 NumPy 版"的误差表。
> 预计用时：8–12 小时。

---

## 总览：本阶段的文件

```text
AI06-Transformer-Attention/advance/my_solution/
├── protocol.py           ★ G2 实验协议：尺寸、种子、容差、固定输入（以后每个实验都 import 它）
├── attn.py               ★ 注意力核心（PyTorch）：本阶段单头，阶段 2 扩成多头
├── check_single_head.py  ★ 与 basic 的 NumPy 版对拍，输出误差表
├── notes/
│   └── 阅读笔记1-AttentionIsAllYouNeed.md   ★ G1 问答 + 阅读笔记（算两篇笔记中的第一篇）
├── figures/
│   └── transformer结构图.png               ★ 结构图初稿（阶段 6 出终稿）
├── logs/
│   └── stage1.txt        （运行生成）对拍日志
└── playground.py           草稿，不提交
```

## 任务地图

| 任务 | 位置 | 做什么 | 产出 | 用时 |
| --- | --- | --- | --- | --- |
| 0 | [第 0 章](#0-开工准备) | 确认 torch、建目录 | 空文件 | 15 min |
| 1 | [第 1 章](#1-原始-transformer-全景g1) | 读论文 §3，画结构图，写 G1 笔记 | 笔记、结构图 | 3 h |
| 2 | [第 2 章](#2-pytorch张量--能求导能上-gpu-的-ndarray) | 把 NumPy 操作翻译成 PyTorch | playground.py | 45 min |
| 3 | [第 3 章](#3-单头注意力的-pytorch-版) | 写 `softmax_lastdim`、`causal_blocked`、`attention` | attn.py | 1 h |
| 4 | [第 4 章](#4-实验协议g2) | 写实验协议 | protocol.py | 45 min |
| 5 | [第 5 章](#5-对拍让-basic-当标准答案) | 和 basic 对拍 | check_single_head.py、logs/stage1.txt | 1 h |
| 6 | [第 6 章](#6-为什么对拍要用-float64) | 看 float32 的误差有多大 | playground.py | 15 min |
| 7 | [第 7 章](#7-交叉注意力把-g1-学的用起来) | 亲手算一次交叉注意力 | check_single_head.py、笔记 | 45 min |
| 8 | [第 8 章](#8-阶段-1-checkpoint) | 对照清单收尾 | — | 30 min |

---

## 0. 开工准备

### 🛠 任务 0：确认 torch、建目录

torch 已经用 `uv add torch` 加进 `pyproject.toml` 了。在 `AI06-Transformer-Attention` 目录下运行：

```bash
uv run python -c "import torch; print(torch.__version__)"
mkdir -p advance/my_solution/notes advance/my_solution/figures advance/my_solution/logs
touch advance/my_solution/{protocol.py,attn.py,check_single_head.py,playground.py}
```

**预期**：打印出版本号（类似 `2.15.0`），不报错。

> 以后所有命令都在 `AI06-Transformer-Attention` 目录下用 `uv run python advance/my_solution/xxx.py` 运行，和 basic 一样。

---

## 1. 原始 Transformer 全景（G1）

basic 里你实现的是**一次**注意力。2017 年的原始 Transformer 是一个**翻译模型**（英文→德文），
注意力在里面用了**三次、三种用法**。先把全景看清，后面每个阶段都在实现这张图的某个零件。

### 1.1 两个塔：编码器和解码器

```text
           源句 "I love you"                          目标句前缀 "<bos> Ich liebe"
                  │                                              │
          词向量 + 位置编码                                词向量 + 位置编码
                  │                                              │
      ┌───────────▼───────────┐                      ┌───────────▼────────────┐
      │  ① 自注意力（无因果）   │                      │  ② 带因果掩码的自注意力  │
      │  + 残差 & LayerNorm    │                      │  + 残差 & LayerNorm     │
      │  FFN                   │   memory            │  ③ 交叉注意力           │
      │  + 残差 & LayerNorm    │ ──(n_src×d_model)──▶ │    Q 来自解码器         │
      └───────────┬───────────┘    给每一层解码器     │    K、V 来自 memory     │
             × N = 6 层                               │  + 残差 & LayerNorm     │
                                                     │  FFN                    │
                                                     │  + 残差 & LayerNorm     │
                                                     └───────────┬────────────┘
                                                            × N = 6 层
                                                                 │
                                                       Linear + softmax → 下一个词的概率
```

- **编码器**（Encoder）：读整句源语言。每层两个子层：**自注意力** + **FFN**。
  源句是完整已知的，每个位置可以看所有位置，所以**不加因果掩码**（只在有 padding 时屏蔽填充位）。
- **解码器**（Decoder）：一个词一个词地生成目标语言。每层三个子层：
  **带因果掩码的自注意力** + **交叉注意力** + **FFN**。
- 每个子层外面都包着 **残差 + LayerNorm**（阶段 3 实现）。原论文参数：`d_model=512, h=8, d_ff=2048, N=6`。

### 1.2 三种注意力，区别只在 Q/K/V 从哪来

你 basic 写的公式一个字都不用改，变的只是**输入从哪来、加什么掩码**：

| 用法 | Q 来自 | K、V 来自 | 掩码 | A 的形状 |
| --- | --- | --- | --- | --- |
| ① 编码器自注意力 | 源句 | 源句 | 无（或 padding） | n_src × n_src |
| ② 解码器自注意力 | 目标前缀 | 目标前缀 | **因果** | n_tgt × n_tgt |
| ③ 交叉注意力 | **解码器**当前层 | **编码器输出 memory** | 无（或源句 padding） | **n_tgt × n_src** |

注意 ③：**A 不再是方阵**。第 i 行表示"正在生成的第 i 个目标词，去源句的哪些词里读信息"——
翻译 "liebe" 时，这一行的权重应该主要落在 "love" 上。这就是"对齐"。

> 任务 7 你会亲手算一次 ③，并发现你 basic 代码里有一个 assert 挡住了它。

### 1.3 训练和生成：同一个模型，两种跑法

- **训练**（teacher forcing）：目标句整句已知。解码器输入是**右移一位**的目标句
  `<bos> Ich liebe dich`，要预测的是 `Ich liebe dich <eos>`。
  有了因果掩码，第 i 个位置只能看 ≤ i 的输入，所以**所有位置可以一次并行算完**，每个位置都不会偷看答案。
- **生成**：答案未知。先输入 `<bos>` 得到第 1 个词，把它接到输入后面，再算第 2 个……**只能一步一步来**。

这个"训练并行、生成逐步"的区别，阶段 4 会用数值实验证明，阶段 5 会真的用上。

### 1.4 post-norm 和 pre-norm

```text
原论文（post-norm）：  x = LayerNorm(x + Sublayer(x))
任务书（pre-norm）：   x = x + Sublayer(LayerNorm(x))
```

pre-norm 的残差通路上没有任何变换，深层更好训练，GPT-2 以后的模型基本都用它。
**我们的实现用 pre-norm**，报告和结构图里必须写清楚这一点（任务书要求）。

### 1.5 三大家族：今天的模型都是这张图的一部分

| 家族 | 用到哪部分 | 掩码 | 代表 | 擅长 |
| --- | --- | --- | --- | --- |
| Encoder-only | 只用编码器 | 双向（无因果） | BERT | 理解、完形填空、分类 |
| Decoder-only | 只用解码器，**去掉交叉注意力** | 因果 | GPT 系列 | 续写、生成 |
| Encoder-Decoder | 两个都用 | 编码器双向，解码器因果 | 原始 Transformer、T5 | 翻译、摘要 |

**阶段 5 的写诗模型是 Decoder-only**：只有 ② 和 FFN，没有编码器、也没有 ③。

### 🛠 任务 1：读论文，画结构图，写 G1 笔记

**阅读**（[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，只读这些）：

- Figure 1（结构图）和 §3.1 Encoder and Decoder Stacks
- §3.2.3 Applications of Attention in our Model（正好对应上面的三种用法）
- §3.3 Position-wise Feed-Forward Networks、§3.5 Positional Encoding 先**扫一眼**，阶段 3 再细读
- 对照 [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) 的 "Model Architecture" 一节看代码长什么样（看不懂的代码先跳过）

**画结构图**（`figures/transformer结构图.png`，手画拍照或 draw.io 都行），必须包含：

1. 编码器、解码器两个塔，标出 `× N`
2. 三种注意力，每个都标出 **Q/K/V 来自哪里、用什么掩码**
3. 残差和 LayerNorm 的位置，写明"原论文 post-norm，本项目实现 pre-norm"
4. 箭头上标形状：`n_src × d_model`、`n_tgt × d_model`，以及交叉注意力的 `A: n_tgt × n_src`

**写笔记**（`notes/阅读笔记1-AttentionIsAllYouNeed.md`），回答这 6 个问题，每个 3–6 句：

```markdown
# 阅读笔记 1：Attention Is All You Need（G1）

## 1 编码器一层有哪几个子层？解码器一层呢？
## 2 三种注意力的 Q/K/V 各来自哪里？各用什么掩码？A 是什么形状？
## 3 为什么解码器自注意力必须加因果掩码，编码器却不用？
## 4 训练时解码器的输入是什么？为什么训练能并行，生成却要逐步？
## 5 post-norm 和 pre-norm 的公式分别是什么？本项目用哪一种？
## 6 BERT、GPT、T5 各属于哪个家族？阶段 5 的写诗模型用到原始结构的哪些部分？

## 读后疑问（写下没搞懂的，后面阶段回来补答）
```

> 第 6 问和"读后疑问"不用一次写完美。阶段 3、5 做完，你会回来改。

---

## 2. PyTorch 张量 = 能求导、能上 GPU 的 ndarray

你已经会 NumPy 了，PyTorch 的 `Tensor` 基本就是**换了名字的 ndarray**，多了两个能力：

1. **能放到 GPU 上**算（`.to("cuda")`）——阶段 5 用 4060Ti 训练靠它；
2. **能自动求导**（`requires_grad`）——阶段 5 训练靠它，前四个阶段用不到。

### 2.1 对照表（遇到就查）

| 要做的事 | NumPy（你会的） | PyTorch | 注意 |
| --- | --- | --- | --- |
| 从列表建 | `np.array([[1., 2.]])` | `torch.tensor([[1., 2.]])` | |
| 单位阵 | `np.eye(2)` | `torch.eye(2)` | |
| 随机正态 | `rng.standard_normal((n, d))` | `torch.randn(n, d, generator=g)` | 种子见第 4 章 |
| 矩阵乘法 | `A @ B` | `A @ B` | 一样 |
| 转置最后两维 | `A.T` | `A.transpose(-2, -1)` | **3 维以上别用 `.T`** |
| 每行最大值 | `x.max(axis=1, keepdims=True)` | `x.amax(dim=-1, keepdim=True)` | `axis→dim`，`keepdims→keepdim` |
| 指数 / 绝对值 | `np.exp(x)` / `np.abs(x)` | `torch.exp(x)` / `x.abs()` | |
| 近似相等 | `np.allclose(a, b)` | `torch.allclose(a, b)` | |
| 最大误差（变成 Python 数） | `np.abs(a - b).max()` | `(a - b).abs().max().item()` | `.item()` 取出标量 |
| 上三角 | `np.triu(M, k=1)` | `M.triu(diagonal=1)` | |
| 互相转换 | `t.numpy()` | `torch.from_numpy(a)` | |
| **默认小数类型** | **float64** | **float32** | 第 6 章专门讲 |

### 2.2 为什么到处写 `-1`、`-2`

basic 里 S 是 `n × n`，你写 `axis=1`。阶段 2 会多出 batch 和 head 两维，S 变成 `(B, h, n, n)`，
"每一行"变成了 `dim=3`。如果一开始就写 **`dim=-1`（最后一维）** 和 **`transpose(-2, -1)`（最后两维）**，
同一份代码对 2 维、3 维、4 维都对，**阶段 2 一行都不用改**。这是本阶段最重要的写法习惯。

### 🛠 任务 2：把 NumPy 操作翻译成 PyTorch（playground.py）

**任务**：清空 `playground.py`，输入并运行：

```python
import torch

# 1. 默认类型：注意和 NumPy 不一样
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("a.dtype =", a.dtype)
b = torch.tensor([[1, 2], [3, 4]])
print("b.dtype =", b.dtype)

# 2. 矩阵乘法和转置
print("a @ a.T =\n", a @ a.transpose(-2, -1))

# 3. 3 维张量：2 个 3×4 的矩阵叠在一起
x = torch.arange(24.0).reshape(2, 3, 4)
print("x.shape =", x.shape)
print("转置最后两维 ->", x.transpose(-2, -1).shape)
print("x.T 却是 ->", x.T.shape)
print("每行最大值 ->", x.amax(dim=-1, keepdim=True).shape)

# 4. max 和 amax 的区别
print("x.max(dim=-1) 返回 ->", type(x.max(dim=-1)))

# 5. 整数张量：exp 会自动变成小数，但"写回"整数张量照样截断（basic 的坑）
print("torch.exp(b).dtype =", torch.exp(b).dtype)
b[0] = torch.tensor([0.7, 0.3])
print("写回后 b =\n", b)
```

**预期输出**（另外还会在终端看到一条 `UserWarning: The use of x.T on tensors of dimension other than 2 ...`，那正是第 3 点要说的）：

```text
a.dtype = torch.float32
b.dtype = torch.int64
a @ a.T =
 tensor([[ 5., 11.],
        [11., 25.]])
x.shape = torch.Size([2, 3, 4])
转置最后两维 -> torch.Size([2, 4, 3])
x.T 却是 -> torch.Size([4, 3, 2])
每行最大值 -> torch.Size([2, 3, 1])
x.max(dim=-1) 返回 -> <class 'torch.return_types.max'>
torch.exp(b).dtype = torch.float32
写回后 b =
 tensor([[0, 0],
        [3, 4]])
```

**看懂这 5 件事**：

1. 带小数点建的是 **float32**，不是 NumPy 的 float64；不带小数点建的是 int64。
2. `@` 和 NumPy 一样；转置写 `transpose(-2, -1)`。
3. `(2, 3, 4)` 转置最后两维得 `(2, 4, 3)`：**前面的维度不动，只对每个矩阵各自转置**。阶段 2 的多头就靠这个。
   而 `x.T` 把**所有维度倒过来**变成 `(4, 3, 2)`——不报错，只给一条警告，结果却完全不是你要的。
4. `x.max(dim=-1)` 返回的是 **(最大值, 位置)** 两样东西，不能直接拿去相减；要最大值就用 `amax`。
5. 整数张量 `exp` 会自动得到 float32；但把小数**写回**整数张量 `b[0] = ...` 依旧悄悄截断成 0，和 basic 里 `softmax_rows` 的坑一模一样。
   所以本阶段的函数都写成"算出新张量再 return"，不往输入里写回。

> 🔥 **本任务用到的 PyTorch**
>
> - `torch.arange(24.0)`：0 到 23 的一维张量（写 `24.0` 让它是小数类型）；`.reshape(2, 3, 4)` 改形状，和 NumPy 一样。
> - `x.shape` 打印出来是 `torch.Size([2, 3, 4])`，用法和元组一样，可以 `n, d = x.shape`。
> - `b[0] = ...`：给第 0 行整体赋值，和 NumPy 一样；目标张量是什么类型，写进去就变成什么类型。

---

## 3. 单头注意力的 PyTorch 版

公式和 basic 完全一样：

```text
S = Q Kᵀ / √d_k
A = softmax(S 中被屏蔽的位置换成 -∞)，沿最后一维
O = A V
```

和 basic 有三处**有意的改变**，都是为后面的阶段铺路：

| 改变 | basic | 现在 | 为什么 |
| --- | --- | --- | --- |
| 维度写法 | `axis=1`、`K.T` | `dim=-1`、`K.transpose(-2, -1)` | 阶段 2 直接支持 `(B, h, n, d)` |
| 掩码参数 | `causal=True/False` | `blocked`：**True = 屏蔽** 的布尔矩阵 | 阶段 2 还要 padding 掩码；交叉注意力的 A 不是方阵 |
| 形状检查 | `Q.shape == K.shape` | 只要求 `d_k` 相同、K 和 V 行数相同 | 交叉注意力里 Q 和 K 行数不同（任务 7） |

### 3.1 掩码为什么叫 `blocked`，不叫 `mask`

阶段 2 你要和两个官方实现对拍，它们的布尔掩码**意思正好相反**：

- `F.scaled_dot_product_attention`：True = **可以看**
- `nn.MultiheadAttention`：True = **不能看**

任务书专门提醒"核对双方掩码 True/False 的含义"。最好的办法是**变量名直接说出意思**：
我们的约定是 `blocked[i, j] == True` 表示第 i 个查询**不能看**第 j 个键。以后看到 `blocked` 就不会搞反。

### 3.2 用 `masked_fill` 代替"加 -∞ 矩阵"

basic 里你是 `S + causal_mask(n)`，mask 里放 0 和 -∞。PyTorch 有更直接的写法：

```python
S.masked_fill(blocked, float("-inf"))   # blocked 为 True 的位置换成 -∞，其余不变
```

因果掩码就是"严格上三角为 True"：

```python
torch.ones(n, n, dtype=torch.bool).triu(diagonal=1)
# n=3 时：
# [[False,  True,  True],
#  [False, False,  True],
#  [False, False, False]]
```

### 🛠 任务 3：写 attn.py

**任务**：在 `attn.py` 里写三个函数。

```python
import torch


def softmax_lastdim(S):
    """沿最后一维的数值稳定 softmax（basic 的 softmax_rows 换成 dim=-1）"""
    # TODO: 减最大值（amax, keepdim=True）→ exp → 除以和（keepdim=True）


def causal_blocked(n, device=None):
    """n×n 因果掩码，True = 屏蔽：第 i 行屏蔽 j > i，保留对角线"""
    # TODO: 一行搞定，见 3.2


def attention(Q, K, V, blocked=None):
    """缩放点积注意力。Q:(..., n_q, d_k)  K:(..., n_k, d_k)  V:(..., n_k, d_v)
    blocked: 可广播到 (..., n_q, n_k) 的布尔张量，True = 屏蔽；None = 不屏蔽
    返回 S（屏蔽前的打分）、A、O"""
    # TODO:
    # 1. d_k = Q.shape[-1]
    # 2. assert：K 的最后一维 == d_k；K 和 V 的倒数第二维（键的个数）相同
    # 3. S = Q @ K.transpose(-2, -1) / d_k ** 0.5
    # 4. 有 blocked 就 masked_fill 成 -inf（存到新变量，S 本身保留屏蔽前的值）
    # 5. A = softmax_lastdim(...)，O = A @ V
    # 6. return S, A, O
```

> `device=None` 先照抄，作用是"掩码建在哪个设备上"，阶段 5 上 GPU 时才用到。

**测试**（清空 playground.py）：

```python
import torch
from attn import attention, causal_blocked

torch.set_printoptions(precision=6, sci_mode=False)
I = torch.eye(2, dtype=torch.float64)

S, A, O = attention(I, I, I)
print("A 无掩码 =\n", A)
S, A, O = attention(I, I, I, blocked=causal_blocked(2))
print("A 因果 =\n", A)
print("causal_blocked(4) =\n", causal_blocked(4))

# 立刻用上 dim=-1 的好处：两份输入叠起来一次算
Q3 = torch.stack([I, 2 * I])          # (2, 2, 2)
S, A, O = attention(Q3, Q3, Q3, blocked=causal_blocked(2))
print("叠起来后 A.shape =", A.shape)
print("A[1] =\n", A[1])
```

**预期输出**：

```text
A 无掩码 =
 tensor([[0.669762, 0.330238],
        [0.330238, 0.669762]], dtype=torch.float64)
A 因果 =
 tensor([[1.000000, 0.000000],
        [0.330238, 0.669762]], dtype=torch.float64)
causal_blocked(4) =
 tensor([[False,  True,  True,  True],
        [False, False,  True,  True],
        [False, False, False,  True],
        [False, False, False, False]])
叠起来后 A.shape = torch.Size([2, 2, 2])
A[1] =
 tensor([[1.000000, 0.000000],
        [0.055807, 0.944193]], dtype=torch.float64)
```

前两个矩阵正是任务书 Q5 的数字。最后一段说明：**你什么都没改，函数已经能一次处理一批矩阵了**，
而且 `(2, 2)` 的掩码自动套用到了每一份上（这叫**广播**，阶段 2 细讲）。
`A[1]` 是输入 `2I` 的结果：Q、K 都放大 2 倍，`S` 就放大 4 倍，所以第二行从 Q5 的 0.67 变成了 0.94，更"尖"。

<details>
<summary>卡住了再看：参考实现</summary>

```python
def softmax_lastdim(S):
    S = S - S.amax(dim=-1, keepdim=True)
    e = torch.exp(S)
    return e / e.sum(dim=-1, keepdim=True)


def causal_blocked(n, device=None):
    return torch.ones(n, n, dtype=torch.bool, device=device).triu(diagonal=1)


def attention(Q, K, V, blocked=None):
    d_k = Q.shape[-1]
    assert K.shape[-1] == d_k, "Q 和 K 的最后一维 d_k 必须相同"
    assert K.shape[-2] == V.shape[-2], "K 和 V 的键个数 n_k 必须相同"
    S = Q @ K.transpose(-2, -1) / d_k ** 0.5
    S_masked = S if blocked is None else S.masked_fill(blocked, float("-inf"))
    A = softmax_lastdim(S_masked)
    O = A @ V
    return S, A, O
```

</details>

> 🔥 **本任务用到的 PyTorch**
>
> - `dtype=torch.bool`：布尔张量；`dtype=torch.float64`：双精度。
> - `torch.stack([a, b])`：把两个形状相同的张量在最前面叠一维，`(2,2)` 两个 → `(2,2,2)`。
> - `x if 条件 else y`：Python 的三元表达式，相当于 C++ 的 `条件 ? x : y`。
> - `d_k ** 0.5`：开方（`**` 是幂）。d_k 是 Python 整数，不需要 `torch.sqrt`。

---

## 4. 实验协议（G2）

G2 要求"固定小矩阵、种子和结构实验协议"。意思是：**以后所有实验，用的尺寸、随机数、精度、容差都从同一个地方来**，
这样阶段 4 说"有位置和无位置的输出差了 0.3"时，别人能复现，你也能确认差别来自结构，而不是来自换了一组随机数。

### 4.1 协议里要固定什么

| 项 | 取值 | 理由 |
| --- | --- | --- |
| `D_MODEL, N_HEADS, D_HEAD` | 32, 4, 8 | 任务书建议值；`D_HEAD = D_MODEL // N_HEADS` |
| `LENGTHS` | (1, 2, 8) | 任务书要求至少测长度 1、2、8 |
| `DTYPE` | `torch.float64` | 对拍时误差小到可以忽略（第 6 章） |
| `TOL` | 1e-6 | 任务书给的容差 |
| `SEED_W, SEED_X` | 0, 1 | **参数和输入用两个不同的种子**，见 4.2 |

### 4.2 为什么参数和输入要分开种子

如果只用一个随机数发生器，先生成 X 再生成 W：

```text
make(n=2)：发生器先吐 2×32 个数给 X，接下来的数给 W
make(n=8)：发生器先吐 8×32 个数给 X，接下来的数给 W   ← W 变了！
```

那么"长度 2 和长度 8 的结果不同"就分不清是长度造成的还是**权重换了**造成的。
所以：**权重只由 `SEED_W` 决定，与 n 无关；输入只由 `SEED_X` 和 n 决定**。
这就是"控制变量"——阶段 4 的干预实验整个建立在这上面。

### 4.3 为什么权重要乘 `1/√d_model`

Q 的每个元素是 32 个 `x·w` 的和。若 x、w 都是标准正态，和的方差是 32，数值会偏大，softmax 容易饱和（basic 第 7 章讲过同样的道理）。
权重乘 `1/√32` 后，每项方差变成 1/32，32 项加起来方差约为 1，Q/K/V 的数值保持在正常范围。

### 🛠 任务 4：写 protocol.py

**任务**：

```python
"""实验协议（G2）：所有实验共用的尺寸、种子、容差和固定输入"""
import torch

D_MODEL = 32
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 8
LENGTHS = (1, 2, 8)
TOL = 1e-6
DTYPE = torch.float64
SEED_W = 0  # 参数种子
SEED_X = 1  # 输入种子


def get_device():
    """有 CUDA 用 CUDA（4060Ti），其次 Apple 的 MPS，否则 CPU。阶段 5 训练时用"""
    # TODO: torch.cuda.is_available() / torch.backends.mps.is_available()


def make_weights(d_model=D_MODEL, d_k=D_HEAD, d_v=D_HEAD, seed=SEED_W):
    """固定的 W_Q, W_K, W_V，只由 seed 决定，乘 1/sqrt(d_model)"""
    # TODO:
    # g = torch.Generator().manual_seed(seed)
    # 按 W_Q、W_K、W_V 的顺序各生成一次：torch.randn(行, 列, generator=g, dtype=DTYPE) * d_model ** -0.5
    # 顺序必须是 Q、K、V，否则数字和预期输出对不上


def make_x(n, d_model=D_MODEL, seed=SEED_X):
    """固定的输入 X:(n, d_model)，只由 seed 和 n 决定"""
    # TODO
```

**测试**（playground.py）：

```python
import torch
from protocol import make_weights, make_x, get_device

torch.set_printoptions(precision=6, sci_mode=False)
W_Q, W_K, W_V = make_weights()
print(W_Q.shape, W_K.shape, W_V.shape, W_Q.dtype)
print("W_Q[0,:4] =", W_Q[0, :4])
X8 = make_x(8)
print("X8[0,:4] =", X8[0, :4])
print("调两次一样:", torch.equal(make_x(8), make_x(8)), torch.equal(make_weights()[0], make_weights()[0]))
print("换种子不一样:", not torch.equal(make_x(8), make_x(8, seed=2)))
print("device =", get_device())
```

**预期输出**：

```text
torch.Size([32, 8]) torch.Size([32, 8]) torch.Size([32, 8]) torch.float64
W_Q[0,:4] = tensor([-0.408427, -0.065982, -0.187528,  0.176690], dtype=torch.float64)
X8[0,:4] = tensor([-0.311290, -0.713030, -0.729068, -0.299202], dtype=torch.float64)
调两次一样: True True
换种子不一样: True
device = mps
```

`W_Q[0,:4]` 和 `X8[0,:4]` 的数字必须一字不差。对不上，先查生成顺序和 `dtype`。
`device` 在 Mac 上是 `mps`，到了 4060Ti 上会是 `cuda`。

> 🔥 **本任务用到的 PyTorch**
>
> - `torch.Generator().manual_seed(seed)`：一个**独立的**随机数发生器，传给 `randn(..., generator=g)`。
>   不用全局的 `torch.manual_seed`，就不会被别处的随机调用打乱顺序。相当于 basic 的 `np.random.default_rng(seed)`。
> - `torch.equal(a, b)`：形状相同且每个元素**完全**相等（不是近似）。
> - `d_model ** -0.5` 就是 `1/√d_model`。

---

## 5. 对拍：让 basic 当标准答案

OJ 选手都知道"对拍"：写一个新的程序，和一个确定正确的程序跑同样的输入，比较输出。
你的 basic NumPy 版已经通过了手算核对和 OJ，**它就是标准答案**。PyTorch 版和它在所有长度、有无掩码下都一致，才算过关。
这也是任务书阶段 1–2 的 checkpoint："单头对照"。

要从 advance 的代码里 import basic 的 `my_attention.py`，得告诉 Python 去哪个目录找：

```python
import sys
from pathlib import Path

BASIC_DIR = Path(__file__).resolve().parents[2] / "basic" / "my_solution"
sys.path.insert(0, str(BASIC_DIR))
import my_attention as np_attn   # basic 的 NumPy 版
```

`Path(__file__).resolve()` 是当前文件的绝对路径，`.parents[0]` 是 `my_solution/`，`[1]` 是 `advance/`，`[2]` 是 `AI06-Transformer-Attention/`。

### 🛠 任务 5：写 check_single_head.py

**任务**：对每个 `n ∈ LENGTHS`、每种 `causal ∈ (False, True)`：

1. 用 `make_weights()`、`make_x(n)` 拿到固定输入，用 PyTorch 算 `Q = X @ W_Q`……再调 `attention`；
2. 把 X 和 W 用 `.numpy()` 转成 ndarray，调 basic 的 `np_attn.project` 和 `np_attn.attention(..., causal=causal)`；
3. 算 A 和 O 的最大绝对误差，打印一行表格；
4. 最后打印"全部通过 / 有失败"。

```python
import sys
from pathlib import Path

import numpy as np
import torch

from attn import attention, causal_blocked
from protocol import LENGTHS, TOL, make_weights, make_x

BASIC_DIR = Path(__file__).resolve().parents[2] / "basic" / "my_solution"
sys.path.insert(0, str(BASIC_DIR))
import my_attention as np_attn  # noqa: E402


def compare_with_basic():
    print("== 单头对拍：PyTorch(attn.py) vs NumPy(basic) ==")
    print(f"{'n':>3} {'causal':>7} {'max|ΔA|':>10} {'max|ΔO|':>10}  pass")
    W_Q, W_K, W_V = make_weights()
    all_ok = True
    for n in LENGTHS:
        for causal in (False, True):
            # TODO: PyTorch 算一遍，NumPy 算一遍，求误差，打印一行，更新 all_ok
            pass
    print("全部通过" if all_ok else "有失败！")
    print()


if __name__ == "__main__":
    compare_with_basic()
```

运行并保存日志：

```bash
uv run python advance/my_solution/check_single_head.py | tee advance/my_solution/logs/stage1.txt
```

**预期输出**：

```text
== 单头对拍：PyTorch(attn.py) vs NumPy(basic) ==
  n  causal    max|ΔA|    max|ΔO|  pass
  1   False   0.00e+00   0.00e+00  True
  1    True   0.00e+00   0.00e+00  True
  2   False   0.00e+00   0.00e+00  True
  2    True   0.00e+00   0.00e+00  True
  8   False   1.11e-16   2.22e-16  True
  8    True   1.11e-16   4.44e-16  True
全部通过
```

误差是 1e-16 这个量级，就是 float64 的舍入误差，可以认为**完全相等**。
n=1 的误差恰好是 0：只有一个位置时 A 就是 `[[1.]]`，O 就是 V（basic 验收过的性质）。
n=2 也是 0，是因为计算步骤少、两边舍入方式恰好一样；n=8 时求和项多了，才出现最后一位的差别。

<details>
<summary>卡住了再看：循环体</summary>

```python
            X = make_x(n)
            Q, K, V = X @ W_Q, X @ W_K, X @ W_V
            blocked = causal_blocked(n) if causal else None
            _, A_t, O_t = attention(Q, K, V, blocked)

            Qn, Kn, Vn = np_attn.project(X.numpy(), W_Q.numpy(), W_K.numpy(), W_V.numpy())
            _, A_n, O_n = np_attn.attention(Qn, Kn, Vn, causal=causal)

            err_A = np.abs(A_t.numpy() - A_n).max()
            err_O = np.abs(O_t.numpy() - O_n).max()
            ok = err_A < TOL and err_O < TOL
            all_ok = all_ok and ok
            print(f"{n:>3} {str(causal):>7} {err_A:10.2e} {err_O:10.2e}  {ok}")
```

</details>

> 🔥 **本任务用到的 PyTorch / Python**
>
> - `t.numpy()`：CPU 上的张量转成 ndarray（共享内存，不复制）。GPU 上的张量要先 `.cpu()`。
> - `f"{err_A:10.2e}"`：宽 10、保留 2 位的科学计数法；`{n:>3}` 右对齐宽 3。
> - `tee`：命令行工具，把输出同时打印到屏幕和写进文件。
> - `if __name__ == "__main__":` 和 basic 一样：直接运行时才执行，被 import 时不执行。

---

## 6. 为什么对拍要用 float64

PyTorch 默认 float32（约 7 位有效数字），NumPy 默认 float64（约 16 位）。阶段 5 训练时我们**会**用 float32（快、省显存，4060Ti 擅长），
但**对拍和干预实验用 float64**，否则你分不清 1e-7 的差别是 bug 还是舍入。

### 🛠 任务 6：看 float32 的误差有多大（playground.py，15 分钟）

```python
import torch
from attn import attention, causal_blocked
from protocol import make_weights, make_x

W_Q, W_K, W_V = make_weights()
X = make_x(8)
Q, K, V = X @ W_Q, X @ W_K, X @ W_V
_, _, O64 = attention(Q, K, V, causal_blocked(8))
_, _, O32 = attention(Q.float(), K.float(), V.float(), causal_blocked(8))
print("float32 与 float64 的最大差 =", (O32.double() - O64).abs().max().item())
```

**预期输出**（只需量级一致）：

```text
float32 与 float64 的最大差 = 2.4223471362105897e-07
```

float32 的误差约 2e-7，**离任务书的容差 1e-6 只差几倍**。模型一深、数一大，就可能误报失败。
结论写进笔记："对拍和干预实验一律 float64 + CPU；训练用 float32 + GPU"。

> 🔥 `.float()` 转 float32，`.double()` 转 float64。

---

## 7. 交叉注意力：把 G1 学的用起来

回到 1.2 的表格第 ③ 行：Q 来自解码器的 3 个目标位置，K、V 来自编码器的 5 个源位置。

```text
Q: (3, 8)   K: (5, 8)   V: (5, 8)
S = Q Kᵀ:  (3, 8) @ (8, 5) = (3, 5)     ← 不是方阵
O = A V:   (3, 5) @ (5, 8) = (3, 8)     ← 行数跟 Q 走，每个目标位置得到一个新向量
```

### 🛠 任务 7：亲手算一次交叉注意力

**任务**：在 `check_single_head.py` 里加一个函数 `cross_attention_demo()`，并在 `__main__` 里调用它：

```python
def cross_attention_demo():
    print("== 交叉注意力：3 个目标位置读 5 个源位置 ==")
    W_Q, W_K, W_V = make_weights()
    X_tgt = make_x(3, seed=3)   # 解码器这边：3 个目标位置
    X_src = make_x(5, seed=4)   # 编码器输出 memory：5 个源位置
    # TODO:
    # 1. Q 从 X_tgt 投影；K、V 从 X_src 投影
    # 2. 调 attention（不加掩码）
    # 3. 打印 Q、K、V、S、A、O 的形状，和 A 的每行和
    # 4. 再把同样的 Q、K、V（.numpy()）交给 basic 的 np_attn.attention，用 try/except AssertionError 打印报错
    print()
```

**预期输出**（接在任务 5 的表格后面）：

```text
== 交叉注意力：3 个目标位置读 5 个源位置 ==
Q: (3, 8)
K: (5, 8)
V: (5, 8)
S: (3, 5)
A: (3, 5)
O: (3, 8)
A 每行和: tensor([1.0000, 1.0000, 1.0000], dtype=torch.float64)
basic 版报错 AssertionError: Q, K 必须形状相同
```

然后在笔记里回答（写进"第 2 问"下面就行）：

1. basic 的哪一行 assert 挡住了交叉注意力？它检查的条件在**自注意力**里为什么总是成立？
2. `causal_blocked(n)` 只能造方阵。交叉注意力需要因果掩码吗？为什么？（提示：编码器的源句是不是完整已知的？）
3. 如果源句末尾有 2 个 padding，`blocked` 应该长什么形状、哪些位置是 True？

> 第 3 问就是阶段 2 要实现的 padding 掩码，先想清楚。

<details>
<summary>卡住了再看：参考实现</summary>

```python
def cross_attention_demo():
    print("== 交叉注意力：3 个目标位置读 5 个源位置 ==")
    W_Q, W_K, W_V = make_weights()
    X_tgt = make_x(3, seed=3)
    X_src = make_x(5, seed=4)
    Q = X_tgt @ W_Q
    K, V = X_src @ W_K, X_src @ W_V
    S, A, O = attention(Q, K, V)
    for name, M in [("Q", Q), ("K", K), ("V", V), ("S", S), ("A", A), ("O", O)]:
        print(f"{name}: {tuple(M.shape)}")
    print("A 每行和:", A.sum(dim=-1))
    try:
        np_attn.attention(Q.numpy(), K.numpy(), V.numpy())
    except AssertionError as e:
        print("basic 版报错 AssertionError:", e)
    print()
```

</details>

> 🔥 `for name, M in [("Q", Q), ...]`：遍历一个"(名字, 矩阵)"对的列表，每次拆成两个变量，相当于 C++ 的结构化绑定 `for (auto [name, M] : v)`。

---

## 8. 阶段 1 checkpoint

对照任务书"第 1–2 周：形状表与单头对照"，本阶段完成**单头对照**，形状表在阶段 2 做多头时完成。

- [ ] `uv run python advance/my_solution/check_single_head.py` 输出 6 行全 `True`，外加交叉注意力的形状
- [ ] `logs/stage1.txt` 已生成
- [ ] `attn.py` 里没有 `axis=1`、没有 `.T`，掩码参数叫 `blocked`，True = 屏蔽
- [ ] `protocol.py` 的权重只依赖 `SEED_W`，输入只依赖 `SEED_X` 和 n；`W_Q[0,:4]` 与预期一致
- [ ] 结构图初稿：两个塔、三种注意力的 Q/K/V 来源与掩码、post-norm vs pre-norm、形状
- [ ] 笔记 6 问 + 任务 7 的 3 个小问 + 读后疑问
- [ ] 能不看资料说出：三种注意力的区别只在**输入从哪来、加什么掩码**；训练为什么能并行

做完告诉我，我写阶段 2（多头注意力）的教程。

---

## 附录：本阶段常见错误

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `TypeError: unsupported operand ... 'torch.return_types.max'` | 用了 `x.max(dim=-1)`，它返回 (值, 位置) | 改用 `x.amax(dim=-1, keepdim=True)` |
| 对拍误差约 1e-7，不是 1e-16 | 某处是 float32（如 `torch.tensor([...])` 没指定 dtype） | 统一用 `DTYPE`，或 `.double()` |
| `W_Q[0,:4]` 和预期对不上 | W_Q/W_K/W_V 生成顺序不同，或少乘了 `d_model ** -0.5` | 按 Q、K、V 的顺序，同一个 generator |
| `ModuleNotFoundError: my_attention` | `parents[2]` 写错，或 `sys.path.insert` 写在 import 之后 | 先 insert 再 import；打印 `BASIC_DIR` 看看 |
| `RuntimeError: expected m1 and m2 to have the same dtype, but got: float != double` | float32 和 float64 张量相乘 | 两边统一 dtype |
| A 某一行全是 `nan` | 这一行所有位置都被屏蔽，softmax 算成 0/0 | 本阶段不会出现；阶段 2 padding 时专门处理 |
| `UserWarning: The use of x.T on tensors of dimension other than 2`，形状不对 | 对 3 维张量用了 `.T`，它把所有维度倒过来 | 改 `transpose(-2, -1)` |
