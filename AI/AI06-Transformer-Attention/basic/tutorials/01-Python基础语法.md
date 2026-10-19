# 教程一：给 C++ 选手的 Python 基础语法

> 目标：学完能读懂 `attention.py`，能独立写出 Basic 的代码（含 OJ 读写、NumPy 矩阵运算、画热力图）。
> 每个例子都可以直接复制运行，`# 输出:` 后面是运行结果。
> **赶时间的话不用先学这份**：直接做[教程二](02-从零理解Transformer注意力.md)，每个任务都会讲用到的语法。这份教程当字典用，遇到不懂的再按目录查。

---

## 目录

0. [怎么运行 Python](#0-怎么运行-python)
1. [变量与基本类型](#1-变量与基本类型)
2. [运算符](#2-运算符)
3. [字符串与 print](#3-字符串与-print)
4. [条件判断与缩进](#4-条件判断与缩进)
5. [循环](#5-循环)
6. [列表 list](#6-列表-list)
7. [元组、字典、集合](#7-元组字典集合)
8. [函数](#8-函数)
9. [引用语义：Python 和 C++ 最大的不同](#9-引用语义python-和-c-最大的不同)
10. [模块、import 与程序入口](#10-模块import-与程序入口)
11. [assert 与异常](#11-assert-与异常)
12. [读取输入（OJ 必备）](#12-读取输入oj-必备)
13. [NumPy：本项目的核心工具](#13-numpy本项目的核心工具)
14. [matplotlib：画热力图](#14-matplotlib画热力图)
15. [综合练习](#15-综合练习)
16. [C++ ↔ Python 速查表](#16-c--python-速查表)

---

## 0. 怎么运行 Python

Python 是**解释型**语言：不用编译，直接运行源文件。

```bash
python3 hello.py        # 运行一个文件（相当于 g++ hello.cpp && ./a.out）
python3                 # 进入交互模式（REPL），一行一行试代码，输入 exit() 退出
python3 -m pip install numpy matplotlib   # 安装第三方库
```

`hello.py`：

```python
print("Hello, Transformer!")
```

> ⚠️ **环境提示**：你的电脑上可能有不止一个 Python（Homebrew 的和 miniconda 的）。
> 如果运行时报 `ModuleNotFoundError: No module named 'numpy'`，先用 `which python3` 看看用的是哪个，
> 然后用同一个解释器安装：`python3 -m pip install -r requirements.txt`，或者直接用 `python`（miniconda 的那个已经装好了 numpy）。

**交互模式小技巧**：在 REPL 里直接输入表达式就会显示它的值，不用写 `print`：

```python
>>> 3 * 7
21
>>> type(3.0)
<class 'float'>
```

---

## 1. 变量与基本类型

Python 变量**不用声明类型**，赋值时自动确定；同一个变量还可以换成别的类型（动态类型）。

```python
a = 10          # int（整数，没有溢出！可以无限大）
b = 3.14        # float（相当于 C++ 的 double）
c = "hello"     # str（字符串，单引号双引号都行）
d = True        # bool（注意首字母大写：True / False）
e = None        # 空值，相当于 C++ 的 nullptr

print(type(a), type(b), type(c), type(d), type(e))
# 输出: <class 'int'> <class 'float'> <class 'str'> <class 'bool'> <class 'NoneType'>

big = 2 ** 100  # 整数不会溢出
print(big)
# 输出: 1267650600228229401496703205376
```

**类型转换**（和 C++ 的强制转换类似，但写法是函数调用）：

```python
print(int("42") + 1)      # 输出: 43
print(float("2.5") * 2)   # 输出: 5.0
print(int(3.99))          # 输出: 3   （截断，不是四舍五入）
print(str(7) + "个")       # 输出: 7个
```

**多重赋值 / 交换**：

```python
x, y = 1, 2
x, y = y, x      # 交换，不需要 tmp
print(x, y)      # 输出: 2 1
```

---

## 2. 运算符

```python
print(7 / 2)     # 输出: 3.5    ⚠️ / 永远是浮点除法（和 C++ 不同！）
print(7 // 2)    # 输出: 3      整除（向下取整）
print(-7 // 2)   # 输出: -4     ⚠️ 向下取整，C++ 的 -7/2 是 -3
print(7 % 3)     # 输出: 1
print(2 ** 10)   # 输出: 1024   乘方（C++ 用 pow）
print(9 ** 0.5)  # 输出: 3.0    开方

# 逻辑运算用英文单词
print(True and False)   # 输出: False   （C++: &&）
print(True or False)    # 输出: True    （C++: ||）
print(not True)         # 输出: False   （C++: !）

# 比较可以连写
x = 5
print(1 < x < 10)       # 输出: True   相当于 1 < x && x < 10

# 自增：没有 ++ 和 --
i = 0
i += 1
print(i)                # 输出: 1
```

---

## 3. 字符串与 print

### 3.1 print 的用法

```python
print("a", "b", "c")             # 输出: a b c      （默认用空格分隔）
print("a", "b", "c", sep=",")    # 输出: a,b,c
print("不换行", end="")           # end 默认是 "\n"
print("!")                        # 输出: 不换行!
```

### 3.2 f-string：最好用的格式化（相当于 printf）

在字符串前加 `f`，大括号里直接写表达式：

```python
name = "attention"
n = 3
pi = 3.14159265
print(f"{name} 有 {n} 个位置")      # 输出: attention 有 3 个位置
print(f"{pi:.6f}")                # 输出: 3.141593   （保留 6 位小数，同 printf("%.6f")）
print(f"{pi:10.3f}|")             # 输出:      3.142|  （宽度 10）
print(f"{n * 2 + 1}")             # 输出: 7          （可以放表达式）
```

### 3.3 字符串常用操作

```python
s = "1.0 2.0  3.0\n"
print(s.split())              # 输出: ['1.0', '2.0', '3.0']   按任意空白切分（OJ 读入用这个）
print(" ".join(["a", "b"]))   # 输出: a b                     用空格把列表拼起来
print(len("hello"))           # 输出: 5
print("hello"[0], "hello"[-1])  # 输出: h o    负下标从末尾数
```

---

## 4. 条件判断与缩进

**Python 没有大括号，缩进就是代码块。** 冒号 `:` 开启一个块，块内统一缩进 4 个空格。

```python
x = 15
if x > 10:
    print("大于 10")
    print("还在 if 里面")
elif x > 5:              # C++ 的 else if
    print("大于 5")
else:
    print("其他")
print("已经出了 if")
# 输出:
# 大于 10
# 还在 if 里面
# 已经出了 if
```

> ⚠️ 缩进错误会直接报错（`IndentationError`）。不要 Tab 和空格混用，编辑器设成"Tab 插入 4 个空格"。

**三目运算符**：

```python
x = 7
s = "奇数" if x % 2 == 1 else "偶数"   # C++: x%2==1 ? "奇数" : "偶数"
print(s)   # 输出: 奇数
```

**空块用 `pass`**（相当于 C++ 的 `{}`）：

```python
if x > 0:
    pass   # 以后再写
```

---

## 5. 循环

### 5.1 for + range

```python
for i in range(3):          # i = 0, 1, 2       C++: for(int i=0;i<3;i++)
    print(i, end=" ")
print()
# 输出: 0 1 2

for i in range(2, 5):       # i = 2, 3, 4       左闭右开
    print(i, end=" ")
print()
# 输出: 2 3 4

for i in range(10, 0, -3):  # 步长 -3
    print(i, end=" ")
print()
# 输出: 10 7 4 1
```

**双重循环**（因果掩码就要用到）：

```python
n = 3
for i in range(n):
    for j in range(i + 1, n):    # j 从 i+1 开始：严格上三角
        print(f"({i},{j})", end=" ")
print()
# 输出: (0,1) (0,2) (1,2)
```

### 5.2 直接遍历元素

```python
words = ["我", "爱", "AI"]
for w in words:                 # C++: for (auto w : words)
    print(w, end=" ")
print()
# 输出: 我 爱 AI

for i, w in enumerate(words):   # 同时拿到下标和元素
    print(i, w)
# 输出:
# 0 我
# 1 爱
# 2 AI

a = [1, 2, 3]
b = [10, 20, 30]
for x, y in zip(a, b):          # 两个列表并排遍历
    print(x + y, end=" ")
print()
# 输出: 11 22 33
```

### 5.3 while、break、continue

```python
k = 0
while True:
    k += 1
    if k % 2 == 0:
        continue
    if k > 7:
        break
    print(k, end=" ")
print()
# 输出: 1 3 5 7
```

---

## 6. 列表 list

列表相当于 C++ 的 `vector`，但可以混装不同类型。

```python
v = [3, 1, 4]
v.append(1)          # push_back
print(v)             # 输出: [3, 1, 4, 1]
print(len(v))        # 输出: 4         size()
print(v[0], v[-1])   # 输出: 3 1       v[-1] 是最后一个
v[1] = 100
print(v)             # 输出: [3, 100, 4, 1]
print(sum(v), max(v), min(v))   # 输出: 108 100 1
print(sorted(v))     # 输出: [1, 3, 4, 100]
print(4 in v)        # 输出: True      查找元素是否存在
```

### 6.1 切片 `[start:stop:step]`（左闭右开）

```python
v = [0, 1, 2, 3, 4, 5]
print(v[1:4])    # 输出: [1, 2, 3]
print(v[:2])     # 输出: [0, 1]        从头开始
print(v[3:])     # 输出: [3, 4, 5]     到末尾
print(v[::2])    # 输出: [0, 2, 4]     步长 2
print(v[::-1])   # 输出: [5, 4, 3, 2, 1, 0]   反转
```

### 6.2 列表推导式（非常常用）

```python
squares = [x * x for x in range(5)]
print(squares)                 # 输出: [0, 1, 4, 9, 16]

evens = [x for x in range(10) if x % 2 == 0]
print(evens)                   # 输出: [0, 2, 4, 6, 8]

nums = [float(t) for t in "1 2.5 3".split()]
print(nums)                    # 输出: [1.0, 2.5, 3.0]
```

### 6.3 二维列表与一个经典陷阱

```python
n, m = 2, 3
good = [[0] * m for _ in range(n)]   # ✅ 正确写法：每行是独立的列表
bad  = [[0] * m] * n                 # ❌ 错误：n 行其实是同一个列表！

good[0][0] = 1
bad[0][0] = 1
print(good)   # 输出: [[1, 0, 0], [0, 0, 0]]
print(bad)    # 输出: [[1, 0, 0], [1, 0, 0]]   ← 两行一起变了
```

> 原因见第 9 节"引用语义"。实际做矩阵运算时我们用 NumPy，不用二维列表。`_` 是"我不关心这个变量"的惯用名。

---

## 7. 元组、字典、集合

```python
# 元组 tuple：不可修改的列表，用圆括号
shape = (3, 4)
rows, cols = shape        # 解包
print(rows, cols)         # 输出: 3 4
# shape[0] = 5            # ❌ 报错：元组不能改

# 字典 dict：相当于 C++ 的 unordered_map
vocab = {"我": 0, "爱": 1}
vocab["AI"] = 2
print(vocab["爱"])         # 输出: 1
print("你" in vocab)       # 输出: False
for k, v in vocab.items():
    print(k, v, end="; ")
print()
# 输出: 我 0; 爱 1; AI 2;

# 集合 set：相当于 unordered_set
s = {1, 2, 2, 3}
print(s)                  # 输出: {1, 2, 3}
```

> NumPy 数组的 `.shape` 就是一个元组，所以 `n, d = X.shape` 这种写法会经常出现。

---

## 8. 函数

```python
def add(a, b):
    """返回 a + b（三引号里是文档字符串，相当于函数注释）"""
    return a + b

print(add(2, 3))          # 输出: 5
print(add("ab", "cd"))    # 输出: abcd   （没有类型限制，能 + 就行）
```

### 8.1 默认参数与关键字参数

```python
def attention_demo(n, causal=False):     # causal 默认是 False
    return f"n={n}, causal={causal}"

print(attention_demo(3))                 # 输出: n=3, causal=False
print(attention_demo(3, True))           # 输出: n=3, causal=True
print(attention_demo(3, causal=True))    # 输出: n=3, causal=True   ← 推荐：按名字传参，可读性好
```

### 8.2 返回多个值

```python
def min_max(v):
    return min(v), max(v)    # 实际上返回一个元组

lo, hi = min_max([3, 1, 4])
print(lo, hi)                # 输出: 1 4
```

`attention.py` 里的 `return A, A @ V` 和 `A, O = attention(Q, K, V)` 就是这个用法。

### 8.3 类型标注（可选，只是提示，不强制）

```python
def scale(x: float, k: float = 2.0) -> float:
    return x * k

print(scale(1.5))   # 输出: 3.0
```

---

## 9. 引用语义：Python 和 C++ 最大的不同

在 C++ 中 `vector<int> b = a;` 会**拷贝**一份。
在 Python 中 `b = a` **不拷贝**，只是给同一个对象多起了一个名字（类似 C++ 的引用 `auto& b = a;`）。

```python
a = [1, 2, 3]
b = a           # b 和 a 指向同一个列表
b[0] = 99
print(a)        # 输出: [99, 2, 3]   ← a 也变了！

c = a.copy()    # 真正拷贝
c[0] = 0
print(a, c)     # 输出: [99, 2, 3] [0, 2, 3]
```

**函数传参也是这样**：在函数里修改列表或 NumPy 数组的内容，外面也能看到。

```python
def modify(v):
    v[0] = -1

x = [1, 2, 3]
modify(x)
print(x)        # 输出: [-1, 2, 3]
```

> int、float、str、tuple 是**不可变**的，`x += 1` 实际上是创建了一个新对象，所以不会出现上面的问题。
> list、dict、NumPy 数组是**可变**的，要特别小心。
> `attention.py` 第 3 个实验写 `X2 = X.copy()` 再修改 `X2`，就是为了保证原来的 `X` 不被改掉。

---

## 10. 模块、import 与程序入口

```python
import math                 # 导入标准库模块
print(math.sqrt(2))         # 输出: 1.4142135623730951

import numpy as np          # 导入并起别名（约定俗成都叫 np）
from math import pi, exp    # 只导入部分名字
print(round(pi, 4), exp(0)) # 输出: 3.1416 1.0
```

**自己的文件也能 import**：同一目录下有 `my_attention.py`，在另一个文件里写 `from my_attention import attention` 就能用它里面的函数。

**程序入口**：

```python
def main():
    print("程序从这里开始")

if __name__ == "__main__":   # 只有直接运行本文件时才为真
    main()
```

直接 `python3 my_attention.py` 时会执行 `main()`；被别的文件 `import` 时不会执行。
这样同一个文件既能当工具库，也能单独运行。

---

## 11. assert 与异常

```python
def check_shape(n, d):
    assert n >= 1, "序列不能为空"            # 条件为假时报错并显示信息
    if d <= 0:
        raise ValueError(f"维度必须为正，得到 {d}")   # 主动抛异常（C++: throw）

check_shape(3, 2)    # 正常
try:
    check_shape(0, 2)
except AssertionError as err:                 # C++: catch
    print("捕获到:", err)
# 输出: 捕获到: 序列不能为空
```

Basic 任务要求"检查形状和非空序列"，用 `assert` 就够了。

---

## 12. 读取输入（OJ 必备）

OJ 输入是一堆用空白分隔的数字。最稳妥的写法是**一次性全部读入，再按顺序取**（不用管换行在哪）：

```python
import sys

def main():
    data = sys.stdin.read().split()   # 所有输入切成字符串列表
    pos = 0                           # 当前读到第几个

    def next_int():
        nonlocal pos                  # 修改外层函数的变量需要 nonlocal
        val = int(data[pos])
        pos += 1
        return val

    n = next_int()
    m = next_int()
    nums = [float(t) for t in data[pos:pos + n * m]]   # 一次取 n*m 个
    pos += n * m
    print(n, m, nums)

if __name__ == "__main__":
    main()
```

测试方法（Shell 里用管道或文件喂输入）：

```bash
echo "2 3  1 2 3 4 5 6" | python3 read_demo.py
# 输出: 2 3 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

python3 read_demo.py < input.txt      # 从文件读入
```

> 也可以用 `input()` 读一行（相当于 `getline`），但数据多、格式乱的时候不如上面的写法方便。

---

## 13. NumPy：本项目的核心工具

NumPy 提供高效的多维数组 `ndarray`，矩阵运算一行就能写完，底层是 C 实现，速度很快。

```python
import numpy as np
np.set_printoptions(precision=6, suppress=True)   # 打印保留 6 位小数，不用科学计数法
```

### 13.1 创建数组

```python
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
print(X.shape)       # 输出: (2, 3)       ← 最重要的属性：(行数, 列数)
print(X.ndim)        # 输出: 2            维数
print(X.dtype)       # 输出: float64      元素类型（相当于 double）

print(np.zeros((2, 3)))   # 全 0
print(np.ones((2, 2)))    # 全 1
print(np.eye(2))          # 单位阵
# 输出:
# [[1. 0.]
#  [0. 1.]]
print(np.full((2, 2), 7.0))  # 全部填 7
print(np.arange(6))          # 输出: [0 1 2 3 4 5]   类似 range
```

### 13.2 reshape：一维数据摆成矩阵（OJ 读入用）

OJ 的矩阵是"按行优先"给出的一串数，这正好是 `reshape` 默认的顺序：

```python
flat = np.array([1, 2, 3, 4, 5, 6], dtype=float)
M = flat.reshape(2, 3)
print(M)
# 输出:
# [[1. 2. 3.]
#  [4. 5. 6.]]
print(M.reshape(-1))   # 输出: [1. 2. 3. 4. 5. 6.]   -1 表示"自动算"，展平回一维
```

### 13.3 下标与切片

```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(M[0, 2])     # 输出: 3            第 0 行第 2 列（也可写 M[0][2]）
print(M[1])        # 输出: [4 5 6]      第 1 行
print(M[:, 0])     # 输出: [1 4 7]      第 0 列（: 表示这一维全取）
print(M[:2])       # 输出: 前两行
print(M[[2, 0, 1]])  # 按给定顺序重排行（attention.py 打乱顺序实验用的就是这个）
# 输出:
# [[7 8 9]
#  [1 2 3]
#  [4 5 6]]
```

> ⚠️ NumPy 切片得到的是**视图**（view），修改切片会改到原数组。需要独立副本时用 `.copy()`。

### 13.4 ⚠️ `@` 和 `*` 的区别（最容易错的地方）

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(A @ B)    # 矩阵乘法 ✅
# 输出:
# [[19 22]
#  [43 50]]

print(A * B)    # 逐元素相乘（对应位置相乘），不是矩阵乘法！
# 输出:
# [[ 5 12]
#  [21 32]]

print(A.T)      # 转置
# 输出:
# [[1 3]
#  [2 4]]
```

矩阵乘法的形状规则：`(n×k) @ (k×m) = (n×m)`，中间的 k 必须相等，否则报错：

```python
X = np.ones((3, 2))
W = np.ones((2, 4))
print((X @ W).shape)    # 输出: (3, 4)
# W @ X  → 报错 ValueError: matmul: ... mismatch   因为 (2×4)@(3×2) 中间维度 4≠3
```

### 13.5 逐元素运算

对数组做数学运算，会作用到**每一个元素**上，不需要写循环：

```python
x = np.array([0.0, 1.0, 2.0])
print(x + 1)          # 输出: [1. 2. 3.]
print(x * 10)         # 输出: [ 0. 10. 20.]
print(np.exp(x))      # 输出: [1.       2.718282 7.389056]
print(np.sqrt(4.0))   # 输出: 2.0
print(np.exp(-np.inf))  # 输出: 0.0   ← 负无穷取指数等于 0，因果掩码的原理
```

### 13.6 axis：沿哪个方向汇总（理解 softmax 的关键）

```python
S = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
print(S.sum())          # 输出: 21.0          全部求和
print(S.sum(axis=0))    # 输出: [5. 7. 9.]    沿行方向压缩 → 每一列一个结果
print(S.sum(axis=1))    # 输出: [ 6. 15.]     沿列方向压缩 → 每一行一个结果
print(S.max(axis=1))    # 输出: [3. 6.]       每行最大值
```

记忆方法：**`axis=k` 就是把第 k 维"压扁"消掉**。形状 (2,3) 的数组，`axis=1` 消掉第 1 维（长度 3 的那维），剩下形状 (2,)，也就是每行一个数。

### 13.7 keepdims 与广播（broadcasting）

想让每一行都减去自己那一行的最大值：

```python
S = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 9.0]])
row_max = S.max(axis=1, keepdims=True)   # keepdims=True 保留被压扁的维度
print(row_max.shape)    # 输出: (2, 1)     ← 是列向量，不是 (2,)
print(row_max)
# 输出:
# [[3.]
#  [9.]]
print(S - row_max)      # (2,3) - (2,1)：自动把 (2,1) 沿列方向"复制"成 (2,3) 再相减
# 输出:
# [[-2. -1.  0.]
#  [-5. -4.  0.]]
```

**广播规则**：从最后一维往前比对，每一维要么相等，要么其中一个是 1（长度为 1 的那一维会被自动复制）。

如果不加 `keepdims`，得到形状 `(2,)`，`(2,3) - (2,)` 会按最后一维对齐，`3≠2`，直接报错。
（更隐蔽的情况：如果矩阵是方阵，形状 `(n,)` 不会报错，但会**按列**去减，悄悄算错！所以按行操作时一定要加 `keepdims=True`。）

### 13.8 比较、布尔数组、近似相等

```python
A = np.array([[0.5, 0.5],
              [1.0, 0.0]])
print(A >= 0)                  # 逐元素比较，得到布尔数组
# 输出:
# [[ True  True]
#  [ True  True]]
print(np.all(A >= 0))          # 输出: True    全部为真？
print(np.any(A == 0))          # 输出: True    有任何一个为真？

print(0.1 + 0.2 == 0.3)                      # 输出: False   浮点误差
print(np.isclose(0.1 + 0.2, 0.3))            # 输出: True
print(np.allclose(A.sum(axis=1), 1.0, atol=1e-6))   # 输出: True   每行和是否都≈1
print(np.max(np.abs(A - A.T)))               # 输出: 0.5    最大绝对误差
```

### 13.9 上三角、负无穷

```python
n = 3
print(np.triu(np.ones((n, n)), k=1))   # k=1：严格上三角（不含对角线）
# 输出:
# [[0. 1. 1.]
#  [0. 0. 1.]
#  [0. 0. 0.]]

mask = np.zeros((n, n))
mask[np.triu(np.ones((n, n), dtype=bool), k=1)] = -np.inf   # 布尔下标赋值
print(mask)
# 输出:
# [[  0. -inf -inf]
#  [  0.   0. -inf]
#  [  0.   0.   0.]]
```

这种写法和 `attention.py` 里用双重循环构造掩码的结果完全一样。初学时写循环更直观，两种都可以。

### 13.10 随机数（固定种子，保证可复现）

```python
rng = np.random.default_rng(42)       # 固定种子
X = rng.standard_normal((3, 4))       # 3×4 的标准正态随机矩阵
print(X.shape)                        # 输出: (3, 4)
# 每次运行，只要种子相同，得到的 X 就完全一样
```

### 13.11 打印数组的格式控制（OJ 输出）

```python
M = np.array([[1/3, 2/3], [1.0, 0.0]])
print(" ".join(f"{x:.10f}" for x in M.reshape(-1)))
# 输出: 0.3333333333 0.6666666667 1.0000000000 0.0000000000
```

`f"{x:.10f}" for x in ...` 叫生成器表达式，和列表推导式一样，只是省略了方括号。

---

## 14. matplotlib：画热力图

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1.0, 0.0, 0.0],
              [0.33, 0.67, 0.0],
              [0.25, 0.25, 0.50]])

fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(A, cmap="viridis", vmin=0, vmax=1)  # 把矩阵画成色块
for i in range(A.shape[0]):                         # 在每个格子里写数值
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color="white")
ax.set_xlabel("key j（被读取的位置）")
ax.set_ylabel("query i（正在读取的位置）")
ax.set_title("Attention weights A")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("heatmap.png", dpi=150)    # 保存成图片
plt.show()                             # 弹出窗口显示（不需要可以删掉）
```

> 如果中文显示成方框，在开头加：
> `plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]`（macOS 自带这个字体），
> 或者把标签改成英文。

---

## 15. 综合练习

做完这些练习，就可以开始写 Basic 了。先自己写，再看答案。

**练习 1**：写函数 `row_sums(M)`，用两重循环（不用 NumPy 的 sum）返回二维列表每行的和。

**练习 2**：用 NumPy 写 `matmul_loop(A, B)`，用三重循环实现矩阵乘法，并用 `np.allclose(matmul_loop(A, B), A @ B)` 验证。

**练习 3**：给定 `S = np.array([[1.0, 2.0], [3.0, 3.0]])`，用 NumPy 一行算出"每行除以该行的和"的结果。

**练习 4**：从标准输入读取 `n d`，再读 `n*d` 个浮点数组成矩阵 X，输出 `X @ X.T`（按行优先，空格分隔）。

<details>
<summary>点击查看参考答案</summary>

```python
import sys
import numpy as np

# 练习 1
def row_sums(M):
    result = []
    for row in M:
        s = 0
        for x in row:
            s += x
        result.append(s)
    return result

print(row_sums([[1, 2], [3, 4]]))   # 输出: [3, 7]

# 练习 2
def matmul_loop(A, B):
    n, k = A.shape
    k2, m = B.shape
    assert k == k2, "形状不匹配"
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for t in range(k):
                C[i, j] += A[i, t] * B[t, j]
    return C

A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
print(np.allclose(matmul_loop(A, B), A @ B))   # 输出: True

# 练习 3
S = np.array([[1.0, 2.0], [3.0, 3.0]])
print(S / S.sum(axis=1, keepdims=True))
# 输出:
# [[0.333333 0.666667]     （未设置 printoptions 时会显示更多位小数）
#  [0.5      0.5     ]]

# 练习 4（单独存成一个文件运行：echo "2 2 1 0 0 1" | python3 ex4.py）
def main():
    data = sys.stdin.read().split()
    n, d = int(data[0]), int(data[1])
    X = np.array([float(t) for t in data[2:2 + n * d]]).reshape(n, d)
    G = X @ X.T
    print(" ".join(f"{x:.6f}" for x in G.reshape(-1)))

# if __name__ == "__main__":
#     main()
```

</details>

---

## 16. C++ ↔ Python 速查表

| 功能 | C++ | Python |
|---|---|---|
| 输出 | `cout << a << " " << b << endl;` | `print(a, b)` |
| 格式化 | `printf("%.6f\n", x);` | `print(f"{x:.6f}")` |
| 读全部输入 | `while (cin >> x)` | `data = sys.stdin.read().split()` |
| 代码块 | `{ }` | `:` + 缩进 |
| 注释 | `// ...` | `# ...` |
| 逻辑 | `&& \|\| !` | `and or not` |
| 整除 | `a / b`（整型） | `a // b` |
| 乘方 | `pow(a, b)` | `a ** b` |
| 自增 | `i++` | `i += 1` |
| 循环 | `for (int i = 0; i < n; i++)` | `for i in range(n):` |
| 范围遍历 | `for (auto x : v)` | `for x in v:` |
| 动态数组 | `vector<int> v; v.push_back(1);` | `v = []; v.append(1)` |
| 大小 | `v.size()` | `len(v)` |
| 哈希表 | `unordered_map<string,int>` | `dict` |
| 空指针 | `nullptr` | `None` |
| 布尔 | `true / false` | `True / False` |
| 函数 | `int f(int x) { return x; }` | `def f(x): return x` |
| 断言 | `assert(cond);` | `assert cond, "信息"` |
| 抛异常 | `throw runtime_error("...")` | `raise ValueError("...")` |
| 拷贝 | `auto b = a;` | `b = a.copy()` |
| 引用 | `auto& b = a;` | `b = a` |
| 主函数 | `int main()` | `if __name__ == "__main__": main()` |
| 矩阵乘法 | 三重循环 | `A @ B` |
| 转置 | 循环交换 | `A.T` |

---

**下一步** → 阅读 [教程二：从零理解 Transformer 注意力](02-从零理解Transformer注意力.md)
