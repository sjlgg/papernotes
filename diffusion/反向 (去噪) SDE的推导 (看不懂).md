好——下面给出**严格而完整**的反向（去噪）SDE的推导。推导分两部分：先给出一般带状态依赖扩散矩阵的结论（更一般、更正规），再把它化简到你关心的常见情形（扩散系数为标量时间函数 $g(t)$ 的情形），最后把结果与你起初的式子对应起来并讨论若干细节与注记。

为保证严格性，我会在每一步写出偏导/散度的代数变换与所用的偏微分方程（Fokker–Planck / Kolmogorov 前向方程），并指明何处使用了对 $x$ 的可微性假设与边界条件（如密度在无穷处衰减以消去边界项）。

---

### 设定（记号）

* $x_t\in\mathbb{R}^d$。
* 前向 Ito SDE（从 $t=0$ 到 $t=T$）：
  $$
  dx_t = f(x_t,t)dt + G(x_t,t)dW_t,
  $$
  其中 $f:\mathbb{R}^d\times[0,T]\to\mathbb{R}^d$，$G:\mathbb{R}^d\times[0,T]\to\mathbb{R}^{d\times m}$，$W_t$ 是标准 $m$-维 Wiener 过程。
* 定义扩散矩阵
  $$
  a(x,t) := G(x,t)G(x,t)^\top \in\mathbb{R}^{d\times d},
  $$
  它是对称半正定矩阵。
* 令 $p_t(x)$ 为 $x_t$ 的边际密度：$p_t(x)=\mathrm{Law}(x_t)$ 的概率密度（假设存在且足够光滑）。
* 我们要找与前向过程时间反向对应的 SDE：若定义反向时间 $s:=T-t$，令逆序过程 $y_s := x_{T-s}$，则希望写成
  $$
  dy_s = f_{\text{rev}}(y_s,s)ds + \widetilde G(y_s,s)d\widetilde W_s
  $$
  并求出 $f_{\text{rev}}$ 与 $\widetilde G$（结论是 $\widetilde G(y,s)=G(y,T-s)$，即扩散项形式不变，漂移要修正）。

---

### 一步：前向 Fokker–Planck 方程（Kolmogorov 前向方程）

对前向 SDE，边际密度 $p_t(x)$ 满足 Fokker–Planck 方程（写成散度形式）：
$$
\boxed{
\partial_t p_t(x)
=- \nabla_x\cdot\bigl(f(x,t)p_t(x)\bigr)
  +\tfrac{1}{2}\sum_{i,j}\partial_{x_i}\partial_{x_j}\bigl( a_{ij}(x,t)p_t(x)\bigr).}
$$
更紧凑地写为
$$
\partial_t p_t = -\nabla\cdot(f p_t) + \tfrac12 \nabla\cdot\nabla\cdot(a p_t),
$$
其中 $\nabla\cdot\nabla\cdot(a p) := \sum_{i,j}\partial_{x_i}\partial_{x_j}(a_{ij} p)$。

（证明和导出是标准的 Ito → Fokker–Planck 推导，这里不重复。）

---

### 二步：引入时间反演的密度

定义反向时间 $s=T-t$，并定义反向密度
$$
q_s(x) := p_{T-s}(x).
$$
则
$$
\partial_s q_s(x) = -\partial_t p_{T-s}(x).
$$
把前向 FP 在时间点 $t=T-s$ 代入：
$$
\partial_s q_s(x)
= -\bigl[-\nabla\cdot(f(x,T-s)p_{T-s}(x)) + \tfrac12 \nabla\cdot\nabla\cdot(a(x,T-s)p_{T-s}(x))\bigr].
$$
即
$$
\partial_s q_s(x)
= \nabla\cdot\bigl(f(x,T-s) q_s(x)\bigr) - \tfrac12 \nabla\cdot\nabla\cdot\bigl(a(x,T-s) q_s(x)\bigr).
$$
将 $a(x,T-s)$ 简记为 $a^{\leftarrow}(x,s)$，$f(x,T-s)$ 为 $f^{\leftarrow}(x,s)$（只是记号），于是
$$
\boxed{\quad
\partial_s q_s = \nabla\cdot\bigl(f^{\leftarrow} q_s\bigr) - \tfrac12 \nabla\cdot\nabla\cdot\bigl(a^{\leftarrow} q_s\bigr).
\quad}
$$

---

### 三步：要求反向过程满足的 Fokker–Planck 形式

如果反向（时间递增的）过程 $y_s$ 是由 Ito SDE
$$
dy_s = f_{\mathrm{rev}}(y_s,s)ds + \widetilde G(y_s,s) d\widetilde W_s
$$
生成的，并用 $\widetilde a(x,s)=\widetilde G\widetilde G^\top$ 表示其扩散矩阵，则该反向过程的密度 $q_s$ 必须满足（前向 FP 形式）：
$$
\partial_s q_s = -\nabla\cdot\bigl(f_{\mathrm{rev}} q_s\bigr) + \tfrac12 \nabla\cdot\nabla\cdot\bigl(\widetilde a, q_s\bigr).
$$
但我们在第二步得到的是另一个表达式，因此要求两者等价。把第二步的等式写成“标准前向 FP”的形式（移动项的符号）：
$$
-\nabla\cdot(f_{\mathrm{rev}} q_s) + \tfrac12 \nabla\cdot\nabla\cdot(\widetilde a q_s)
=
\nabla\cdot(f^{\leftarrow} q_s) - \tfrac12 \nabla\cdot\nabla\cdot(a^{\leftarrow} q_s).
$$
将右边移到左边，得
$$
-\nabla\cdot(f_{\mathrm{rev}} q_s) - \nabla\cdot(f^{\leftarrow} q_s) + \tfrac12 \nabla\cdot\nabla\cdot(\widetilde a q_s) + \tfrac12 \nabla\cdot\nabla\cdot(a^{\leftarrow} q_s)=0.
$$
重排以解出 $f_{\mathrm{rev}}$。通常我们要求反向过程与前向过程共享相同的局部噪声结构（这是自然且常用的选择），即取
$$
\boxed{ \widetilde a(x,s) = a(x,T-s) = a^{\leftarrow}(x,s). }
$$
也就是 $\widetilde G(x,s)=G(x,T-s)$（任意关于时间的矩阵扩散在时间反向下保持相同的扩散矩阵）。在这一选择下，扩散二阶导数那两项相互抵消，于是方程化为
$$
-\nabla\cdot(f_{\mathrm{rev}} q_s) = \nabla\cdot(f^{\leftarrow} q_s).
$$
两边同为散度项，展开得到
$$
\nabla\cdot\bigl( (f_{\mathrm{rev}} + f^{\leftarrow}) q_s \bigr)=0.
$$
更具体地，展开散度项：
$$
(f_{\mathrm{rev}}(x,s) + f^{\leftarrow}(x,s))\cdot \nabla q_s(x) + (\nabla\cdot(f_{\mathrm{rev}} + f^{\leftarrow}))(x,s)q_s(x) = 0.
$$
但我们可以更直接地通过代数变形求出 $f_{\mathrm{rev}}$。把等式写为
$$
-\nabla\cdot(f_{\mathrm{rev}} q_s) = \nabla\cdot(f^{\leftarrow} q_s)
\quad\Longrightarrow\quad
\nabla\cdot\bigl( (f_{\mathrm{rev}} + f^{\leftarrow}) q_s \bigr)=0.
$$
在一般情形下，对任意 $q_s$ 使上式成立的解为
$$
f_{\mathrm{rev}}(x,s) = - f^{\leftarrow}(x,s) + \frac{C(x,s)}{q_s(x)},
$$
其中 $C$ 是一个散度为零的向量场（$\nabla\cdot C =0$）。但我们要用更明确定量的方式还原出 $f_{\mathrm{rev}}$ 与 $f$、$a$、$p$ 的关系；回到更早的步骤，不妨不在此处直接抵消扩散项，而是在没有事先假定 $\widetilde a=a^{\leftarrow}$ 的一般等式中解出 $f_{\mathrm{rev}}$。

更直接的常用推导方式是：从第二步的表达式
$$
\partial_s q_s = \nabla\cdot(f^{\leftarrow} q_s) - \tfrac12 \nabla\cdot\nabla\cdot(a^{\leftarrow} q_s)
$$
将右端写成标准前向 FP 的形式 $-\nabla\cdot(\cdot) + \frac12 \nabla\cdot\nabla\cdot(\cdot)$。也就是说，等价要求
$$
-\nabla\cdot\bigl(f_{\mathrm{rev}} q_s\bigr) + \tfrac12 \nabla\cdot\nabla\cdot\bigl(a^{\leftarrow} q_s\bigr)
=
\nabla\cdot(f^{\leftarrow} q_s) - \tfrac12 \nabla\cdot\nabla\cdot\bigl(a^{\leftarrow} q_s\bigr).
$$
于是把第二项移项：
$$
-\nabla\cdot(f_{\mathrm{rev}} q_s)
=
\nabla\cdot(f^{\leftarrow} q_s) - \nabla\cdot\nabla\cdot\bigl(a^{\leftarrow} q_s\bigr).
$$
移负号并取负：
$$
\nabla\cdot(f_{\mathrm{rev}} q_s)
=
- \nabla\cdot(f^{\leftarrow} q_s) + \nabla\cdot\nabla\cdot\bigl(a^{\leftarrow} q_s\bigr).
$$
因此（以散度算子外提）
$$
\nabla\cdot\Bigl( f_{\mathrm{rev}} q_s + f^{\leftarrow} q_s - \nabla\cdot\bigl(a^{\leftarrow} q_s\bigr)\Bigr) = 0.
$$
在通常的光滑性与无边界贡献条件下（例如 (p) 在无穷远衰减足够快或域为有界且无流出边界），由散度为零并结合自然边界条件，可以推出括号内向量场等于零，从而得到**点值等式**
$$
\boxed{
f_{\mathrm{rev}}(x,s) q_s(x)
= - f^{\leftarrow}(x,s) q_s(x) + \nabla\cdot\bigl(a^{\leftarrow}(x,s) q_s(x)\bigr).
}
$$
将 $q_s(x)=p_{T-s}(x)$ 代回，并记 $f^{\leftarrow}(x,s)=f(x,T-s)$，$a^{\leftarrow}(x,s)=a(x,T-s)$，得到
$$
f_{\mathrm{rev}}(x,s)
= -f(x,T-s) + \frac{1}{p_{T-s}(x)}\nabla_x\cdot\bigl( a(x,T-s) p_{T-s}(x)\bigr).
$$

把 $\nabla\cdot(a p)$ 展开：
$$
\frac{1}{p}\nabla\cdot(a p)
= \frac{1}{p}\bigl[(\nabla\cdot a)p + a\nabla p\bigr]
= \nabla\cdot a + a\frac{\nabla p}{p},
$$
其中 $(\nabla\cdot a)_i := \sum_j \partial_{x_j} a_{ij}$（这是对矩阵按行取散度，结果为向量）；并使用 $\nabla p/p = \nabla \log p$。因此
$$
f_{\mathrm{rev}}(x,s)
= -f(x,T-s) + \nabla\cdot a(x,T-s) + a(x,T-s)\nabla\log p_{T-s}(x).
$$

换回以 $t$ 为变量（设 $t=T-s$，反向过程用 $t$ 表示时得到经典形式）：对于反向时间写成与 $t$ 同步的形式（即把 $s$ → $t$），可以得到**反向时刻 $t$ 的漂移**（常见写法）：

若
$$
\boxed{
dx_t = f(x_t,t)dt + G(x_t,t)dW_t}
$$
则反向 SDE 为
$$
\boxed{
dx_t = \Bigl(f(x_t,t) - a(x_t,t)\nabla_x\log p_t(x_t) - \nabla_x\cdot a(x_t,t)\Bigr)dt + G(x_t,t)d\overline W_t.}
$$
注意这里我们把“正向→反向时间翻转”的符号安排成和常见文献一致（见下注）。上式中

* $a=G G^\top$；
* $(\nabla\cdot a)_i = \sum_j \partial_{x_j} a_{ij}$；
* $d\overline W_t$ 是沿反向时间的 Wiener 过程（在反向时间的自然滤波下仍是标准 Wiener 过程）。

> 这是一般情形的**精确**公式。若 $G$ 依赖 $x$，则 $\nabla\cdot a$ 一项通常不为零；若 $G$ 仅依赖时间或为 $g(t)I$ 的形式，则该项为零或简化为标量乘子形式。

---

### 四步：化简到你给出的常见情形

你关心的常见去噪扩散模型情形通常为：

* 扩散项与空间无关，只与时间有关，且对所有坐标各向同性：
  $$
  G(x,t) = g(t)I_d \quad\Longrightarrow\quad a(x,t)=g(t)^2 I_d.
  $$
* 此时 $\nabla\cdot a(x,t)$（按行散度）为零，因为 $a_{ij} = g(t)^2 \delta_{ij}$ 不依赖 $x$。于是上式中 $\nabla\cdot a = 0$。
* 因而反向漂移简化为
  $$
  f_{\mathrm{rev}}(x,t) = f(x,t) - g(t)^2 \nabla_x\log p_t(x).
  $$
  于是反向 Ito SDE 写为
  $$
  \boxed{
  dx_t = \bigl[f(x_t,t) - g(t)^2\nabla_x\log p_t(x_t)\bigr]dt + g(t)d\overline w_t}
  $$
  这与你最初写的公式完全一致（只是把记号规范化为 $p_t$ 与 $g(t)$）。这就是**常见的 score-based / denoising reverse SDE** 的标准形式。

---

### 备注、边界条件与严格性说明

1. 我们的推导关键点在于把前向 Fokker–Planck 用时间反转变换 $q_s(x)=p_{T-s}(x)$ 写出，再与一个假想的“反向过程”产生的 Fokker–Planck 比较，从而解出反向漂移。在这个过程中需要：

   * $p_t(x)$ 与 $a(x,t)$ 有足够的光滑性（至少可微并且相关导数可交换）；
   * 在无穷远或域边界上，概率流项应衰减到零，以便从“散度等于零”推导出点值等式（或在有界域上需指定适当边界条件）。
     这些条件在随机微分方程和 Fokker–Planck 理论里是标准假设，可见 Anderson (1982)、Pavliotis 等教材。
2. 当 $G$ 依赖 $x$ 时，$\nabla\cdot a$ 项不能忽略；许多文献中在写“反向漂移 $= f - a\nabla\log p$”时默认 $a$ 为常数或仅为时间函数，从而省略了 $\nabla\cdot a$ 项——这是常见但需要注意的假设。
3. 关于 Wiener 过程的反向：严格地说，若用 $y_s=x_{T-s}$ 则存在相应的反向过滤与反向布朗运动 $\widetilde W_s$。构造上有标准方法证明 $\widetilde W$ 在反向时间下是标准 Wiener；所以在 SDE 书写上我们用 $d\overline W_t$ 表示“反向时间的标准 Wiener 增量”。
4. 与概率流 ODE（deterministic probability flow）的关系：把反向 SDE 的噪声项去掉并把 (1) 换为 $\tfrac12$（见文献化约），可以得到决定性的概率流 ODE：
   $$
   \frac{dx_t}{dt} = f(x_t,t) - \tfrac{1}{2} a(x_t,t)\nabla\log p_t(x_t) - \tfrac{1}{2}\nabla\cdot a(x_t,t),
   $$
   在 $a=g(t)^2 I$ 时进一步降为 $dx/dt = f - \tfrac12 g^2 \nabla\log p$。这与常见的概率流 ODE（Song et al., 2021 等）一致。

---

### 总结（把关键结论再强调一次）

* 一般（状态相关扩散）形式：
若
  $$
  \boxed{
    dx_t=f(x_t,t)dt+G(x_t,t)dW_t,\quad a=GG^\top,}
$$
则
$$
\boxed{
dx_t = \bigl[f(x_t,t) - a(x_t,t)\nabla\log p_t(x_t) - \nabla\cdot a(x_t,t)\bigr]dt + G(x_t,t)d\overline W_t.
}
$$
* 常见各向同性仅时间依赖扩散 (G=g(t)I) 的情形（即你的情形）：
$$
\boxed{
dx_t = \bigl[f(x_t,t) - g(t)^2\nabla_x\log p_t(x_t)\bigr]dt + g(t)d\overline w_t.}
$$

---

如果你愿意，我可以：

* （A）把上面的推导写成更“严格”的数学证明形式（给出假设精确陈述：$p\in C^{2,1}$ 等，引用具体定理与文献如 Anderson (1982) 或 Nelson 的时间反演理论）；
* 或者（B）把推导改写为分量索引形式（用 $i,j$ 下标逐项展开并显示每一步的微分算子代数），便于做手工验证；
* 或者（C）把上面的结论与常见 score-based 文献（例如 Song & Ermon 的 score-SDE / probability flow ODE）中的写法逐句对应并说明换算关系（比如为什么概率流 ODE 有 $\tfrac12$ 因子）。

你更想要哪种后续深入？我会直接按你选项把内容补全，不再问额外确认（以便你能无缝继续学习）。
