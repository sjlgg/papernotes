# 常微分方程 ODE

常微分方程（ODE, Ordinary Differential Equation）的标准形式之一确实就是：

$$
dx_t=f_t(x_t)dt \tag{1}
$$ 

或者更常见地写作：

$$ \frac{dx_t}{dt}=f_t(x_t) $$

其中，  
$x_t \in \mathbb{R}^n$ 表示系统在时间 $t$ 时刻的状态（可能是标量或向量）。  
$f_t(x_t)$ 表示在时间 $t$ 时刻，状态的变化率（导数），它可能依赖于当前状态 $x_t$ 以及时间 $t$。

# 随机微分方程 SDE

$$
dx_t=f_t(x_t)dt + g_t(x_t)dW_t
$$

其中多了一个随机项 $g_t(x_t)dW_t$，$W_t \sim \mathcal{N}(0,t)$  是布朗运动或叫做 wiener 过程，满足 $W_0=0$。 
这是，系统的变化既有确定性部分 (由 $f_t$ 控制)，也有随机扰动部分 (由 $g_t$ 控制)。

> 布朗运动的增量服从 $\mathcal{N}(0,\Delta t)$，且相互独立。  
整个过程 $W_t$ 是一个连续随机过程，不是独立的高斯变量族。

在扩散模型中，

$$
\boxed{dx_t=f_t(x_t)dt + g_tdW_t} \tag{2}
$$

其中，$g_t$ 是时间的函数。

**从布朗运动（Wiener 过程）的定义出发**

设 $W_t$ 是标准布朗运动（标准 Wiener 过程），它有两条关键性质：

1. **独立增量**：对任意 $0\le s<t$，增量 $W_t-W_s$ 与过去无关；不同不相交区间的增量相互独立。
2. **正态增量且方差等于时间差**：
$$
\boxed{W_{t+\Delta t}-W_t \sim \mathcal{N}(0,\Delta t)} \tag{3}
$$

这第二条就是核心：增量是均值 0、方差 $\Delta t$ 的高斯分布。

**写成标准正态乘以尺度**

如果随机量 $Z\sim\mathcal{N}(0,1)$，那么对任意常数 $a$ 有 $aZ\sim\mathcal{N}(0,a^2)$。把它套到上一条：

令 $\varepsilon_t\sim\mathcal{N}(0,1)$，取 $a=\sqrt{\Delta t}$，则
$$
\sqrt{\Delta t}\varepsilon_t \sim \mathcal{N}\bigl(0,(\sqrt{\Delta t})^2\bigr)=\mathcal{N}(0,\Delta t) \tag{4}
$$
因此可以写
$$
W_{t+\Delta t}-W_t = \sqrt{\Delta t}\varepsilon_t,
\quad \varepsilon_t\sim\mathcal{N}(0,1) \tag{5}
$$

<!-- 这就是代数层面的“推导”——只是用了一条标准事实：方差随时间线性放大，所以把标准正态乘以 $\sqrt{\Delta t}$。 -->

把上式视为微小时间步长的表示，当 $\Delta t\to 0$ 我们常把增量记作微分形式：
$$
dW_t \equiv W_{t+dt}-W_t \approx \sqrt{dt}\varepsilon_t,
\quad \varepsilon_t\sim\mathcal{N}(0,1). \tag{6}
$$
这给出了常用的“物理直觉”表示：布朗运动的微小增量的典型规模是 $O(\sqrt{dt})$，不是 $O(dt)$。

> 严格意义上 $dW_t$ 是随机测度增量，不是普通微分。  
$dW_t=\sqrt{dt}\varepsilon$ 只是“分布等价”的写法，用于数值近似。

将等式 6 代入等式 2：

$$
\begin{align*}
x_{t+\Delta t}-x_t &= dx_t = f_t(x_t)dt + g_tdW_t \\
&= f_t(x_t)dt + g_t\sqrt{\Delta t}\varepsilon_t
\end{align*}
$$
$$
\Rightarrow \boxed{x_{t+\Delta t} = x_t + f_t(x_t)dt + g_t\sqrt{\Delta t}\varepsilon_t} \tag{7}
$$
这就是 Euler–Maruyama 离散化形式。

## DDPM 加噪中的 SDE

对照 DDPM 加噪公式：

$$
x_{t+1}=\sqrt{1-\beta_t}x_t+\sqrt{\beta_t}\varepsilon \tag{8}
$$

DDPM 的前向加噪过程可以视作扩散 SDE 的离散近似形式（在步长趋于 0 的极限下与 SDE 等价）。

> 严格地说：  
DDPM 的加噪过程是离散时间的马尔可夫链；  
它可以被视作一个离散化的 SDE（扩散过程的离散近似）；  
只有在 $\beta_t \to 0$，步长 $\to 0$ 的连续极限下，才得到真正的 SDE。

> 包括 score matching 和 flow matching 都可以往 SDE 上靠。

# SDE 去噪

目标：$p(x_t|x_{t+\Delta t})$，即已知 $x_{t+\Delta t}$ 求 $x_t$。

由 (7) 知 $x_{t+\Delta t}$ 在前向 SDE 中的分布 $p(x_{t+\Delta t}|x_t) \sim \mathcal{N}(x_t + f_t(x_t)dt,\ g_t^2dt)$

由贝叶斯公式：

$$
\begin{align*}
&p(x_t|x_{t+\Delta t})=\frac{p(x_{t+\Delta t}|x_t)p(x_t)}{p(x_{t+\Delta t})} \\
&=\exp\Big(-\frac{(x_{t+\Delta t}-x_t-f_t(x_t)\Delta t)^2}{2g_t^2 \Delta t}+\log p(x_t)-\log p(x_{t+\Delta t})\Big) \tag{9}
\end{align*}
$$

如何分析？进行变换！  
泰勒展开？朗之万动力学采样公式：$\nabla_x \log p(x) $  
在 $x_0$ 处泰勒展开公式：
$$
f(x)=f(x_0)+(x-x_0)f'(x_0)+\cdots \tag{10}
$$
在本段内容中，把 $x_{t+\Delta t}$ 作为 (10) 中的 $x$，把 $x_t$ 作为 $x_0$，进行一阶泰勒展开：
$$
\log p(x_{t+\Delta t})=\log p(x_t)+(x_{t+\Delta t}-x_t)\nabla_x \log p(x_t) + (t+\Delta t - t) \nabla_t \log p(x_t) + \cdots
$$
代入 (9) 中，得：

$$
\begin{align*}
p(x_t|x_{t+\Delta t})=\exp\Big(&-\frac{(x_{t+\Delta t}-x_t-f_t(x_t)\Delta t)^2}{2g_t^2 \Delta t} \\
&- (x_{t+\Delta t}-x_t)\nabla_x \log p(x_t) \\
&- \Delta t \nabla_t \log p(x_t)\Big) \tag{11}
\end{align*}
$$

其中 $\Delta t \rightarrow 0$

$$
p(x_t|x_{t+\Delta t})=\exp\Big(-\frac{[x_{t+\Delta t}-x_t-\big(f_t(x_t) - g_t^2 \nabla_x \log p(x_t) \big)\Delta t]^2}{2g_t^2 \Delta t} \Big) \tag{10}
$$

> tag 9 是怎么推导到 tag 10 的？
>
> — 它就是用**配方平方（completing the square）**把 (9) 中的线性项吸收到平方项里，然后利用 $\Delta t\to0$ 丢掉高阶小量（这些项可以被归入归一化常数或在极限下变为 1）。下面一步步做代数变形（不引入新变量，全部用题中已有符号）。
>
> 记
> $$
B:=x_{t+\Delta t}-x_t-f_t(x_t)\Delta t,
\qquad s:=\nabla_x\log p(x_t),
\qquad \sigma^2:=g_t^2\Delta t.
> $$
>（只是为了书写方便，后面不会再引入新符号。）
>
>原式指数的负号内项可写为
>$$
\frac{B^2}{2g_t^2\Delta t}+(x_{t+\Delta t}-x_t)s+\Delta t \nabla_t\log p(x_t).
>$$
>将 ((x_{t+\Delta t}-x_t)) 用 (B+f_t(x_t)\Delta t) 展开：
>$$
\frac{B^2}{2g_t^2\Delta t}+Bs+f_t(x_t)\Delta t \cdot s+\Delta t \nabla_t\log p(x_t).
>$$
>
>现在对前两项 $\dfrac{B^2}{2g_t^2\Delta t}+Bs$ 做配方平方。利用恒等式（对向量与标量同样成立）
>$$
\frac{B^2}{2\sigma^2}+B\cdot s
=\frac{(B+\sigma^2 s)^2}{2\sigma^2}-\frac{\sigma^2|s|^2}{2},
>$$
>其中此处 (\sigma^2=g_t^2\Delta t)。代入得
>$$
\begin{aligned}
&\frac{B^2}{2g_t^2\Delta t}+B,s
= \frac{\big(B+g_t^2\Delta t s\big)^2}{2g_t^2\Delta t}
-\frac{g_t^2\Delta t |s|^2}{2}.
\end{aligned}
>$$
>
>于是整个指数（负号内）变为
>$$
\frac{\big(B+g_t^2\Delta t  s\big)^2}{2g_t^2\Delta t}
-\frac{g_t^2\Delta t |s|^2}{2}
+f_t(x_t)\Delta t; s+\Delta t \nabla_t\log p(x_t).
>$$
>
>注意第一项是以 (1/\Delta t) 级别的大项（决定条件分布的高斯形状），而后面三项都是 (O(\Delta t))。把第一项展开回原变量：
>$$
B+g_t^2\Delta t \cdot s
= x_{t+\Delta t}-x_t-f_t(x_t)\Delta t + g_t^2\Delta t \cdot s
= x_{t+\Delta t}-x_t -\big(f_t(x_t)-g_t^2 s\big)\Delta t .
>$$
>
>因此概率密度的指数部分可写为
>$$
-\frac{\big[x_{t+\Delta t}-x_t-(f_t(x_t)-g_t^2\nabla_x\log p(x_t))\Delta t\big]^2}{2g_t^2\Delta t}
;+; O(\Delta t).
>$$
>
>在极限 $\Delta t\to0$ 下，指数中的 $O(\Delta t)$ 项对条件概率的形状（以 $1/\Delta t$ 尺度主导的高斯核）贡献为一个近似为 (1) 的乘子或可被归入归一化常数，因此可以忽略，从而得到公式 (10)：
>$$
p(x_t|x_{t+\Delta t})
\propto \exp\Big(-\frac{[x_{t+\Delta t}-x_t-(f_t(x_t)-g_t^2\nabla_x\log p(x_t))\Delta t]^2}{2g_t^2\Delta t}\Big),
>$$
>并在规范化后得到与 (10) 等价的表达。
>
>—— 总结：关键步骤是对以 (B) 为变量的二次+线性项配方平方，再利用 $\Delta t\to0$ 将 $O(\Delta t)$ 项（归一化或次要因子）忽略。

然后由于 $\Delta t \to 0$，为了从 $t+\Delta t$ 求 $t$ 的信息，所以把 (10) 中部分 $t$ 项写为 $t+\Delta t$：

$$
p(x_t|x_{t+\Delta t})=\exp\Big(-\frac{[x_t-\Big(x_{t+\Delta t}-\big(f_{t+\Delta t}(x_{t+\Delta t}) - g_{t+\Delta t}^2 \nabla_x \log p(x_{t+\Delta t}) \big)\Delta t\Big)]^2}{2g_{t+\Delta t}^2 \Delta t} \Big) 
$$

则

$$
x_t = x_{t+\Delta t}-\big[f_{t+\Delta t}(x_t) - g_{t+\Delta t}^2 \nabla_x \log p(x_{t+\Delta t}) \big]\Delta t + g_{t+\Delta t} \sqrt{\Delta t} \varepsilon
$$

其中 $\sqrt{\Delta t} \varepsilon = \Delta w$

$$
x_{t+\Delta t} - x_t = \big[f_{t+\Delta t}(x_t) - g_{t+\Delta t}^2 \nabla_x \log p(x_{t+\Delta t}) \big]\Delta t - g_{t+\Delta t} \Delta w
$$

由 $\Delta t \to 0$，得：

$$
\Rightarrow \quad \boxed{dx_t = \big[f_{t}(x_t) - g_{t}^2 \nabla_x \log p(x_{t}) \big]dt + g_{t} d\overline{w}} \tag{12}
$$

另一种写法：
$$
\boxed{%
dx_t = \bigl[f(x_t,t) - g(t)^2\nabla_x \log p_t(x_t)\bigr]dt + g(t)d\overline{w}_t
}
$$

其中 $\overline{w}$ 是逆向 wiener 过程。
其中 $\nabla_x \log p(x_{t})$ 即为 score function。
这就是扩散模型与 score-based 生成模型在连续形式下的核心联系。

> 对应的概率流（deterministic probability flow）ODE 去掉噪声项为
>$$
\boxed{%
\frac{d x_t}{dt} = f(x_t,t) - \tfrac{1}{2}g(t)^2\nabla_x\log p_t(x_t)
}
>$$
>这个 ODE 在分布演化上與上面的 SDE 保持相同的边际分布路径，但为确定性积分常用于生成样本的可逆映射。


## 在实际算法中（用可学习的 score）

通常我们并不知道真实的 $\nabla_x\log p_t(x)$，所以用一个参数化网络 $s_\theta(x,t)$ 来近似：
$$
\nabla_x\log p_t(x) \approx s_\theta(x,t).
$$
把它代入反向 SDE 或概率流 ODE：
$$
dx_t = \bigl[f(x_t,t) - g(t)^2 s_\theta(x_t,t)\bigr]dt + g(t)d\overline w_t, \\
\quad\text{或}\quad
\frac{dx_t}{dt}=f(x_t,t)-\tfrac12 g(t)^2 s_\theta(x_t,t).
$$


# 扩散模型 前向加噪公式的 SDE 表达形式

> SDE 变为 pfODE 从而实现跳步。

SDE 前向公式：
$$
\boxed{dx_t=f_t(x_t)dt + g_tdW_t} \tag{13}
$$
即：
$$
dx_t=f(x_t, t)dt + g(t)dW_t,\\
dW_t=\sqrt{dt}\varepsilon,\\
\varepsilon \sim \mathcal{N}(0,I)
$$

离散化：

$$
x_t=x_{t-\Delta t} + \boxed{\phantom{x}}\ \Delta t + \boxed{\phantom{x}}\ \Delta W, \quad \Delta W=\sqrt{\Delta t}\varepsilon \tag{14}
$$

## 1. DDPM

前向 (加噪) 公式：
$$
x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\varepsilon \tag{15}
$$
$$
\begin{align*}
x(t)&=\sqrt{1-\beta(t)}x(t-1)+\sqrt{\beta(t)}\varepsilon \\
&=\sqrt{1-\beta(t)}x(t-1)+\sqrt{\frac{\beta(t)}{\Delta t}}\sqrt{\Delta t}\varepsilon \tag{16}
\end{align*}
$$

把 $\sqrt{1-x}$ 在 $x=0$ 处泰勒展开，
$$
\sqrt{1-x}=1-\frac{1}{2\sqrt{1-0}}(x-0)=1-\frac{x}{2}
$$
代入 (15)，
$$
\begin{align*}
x(t)&=[1-\frac{\beta(t)}{2}]x(t-1)+\sqrt{\frac{\beta(t)}{\Delta t}}\sqrt{\Delta t}\varepsilon \\
&=x(t-1)-\frac{\beta(t)}{2}x(t-1)+\sqrt{\frac{\beta(t)}{\Delta t}}\sqrt{\Delta t}\varepsilon
\end{align*}
$$
此时与 (14) 对齐，令 $\Delta t=1$，
$$
\begin{align*}
x(t)&=x(t-\Delta t)-\frac{\beta(t)}{2}x(t-\Delta t)+\sqrt{\frac{\beta(t)}{\Delta t}}\Delta W \\
x(t)-x(t-\Delta t)&=-\frac{\beta(t)}{2 \Delta t}x(t-\Delta t)\Delta t+\sqrt{\frac{\beta(t)}{\Delta t}}\Delta W
\end{align*}
$$
令 $\frac{\beta(t)}{ \Delta t}=\overline\beta(t)$，
$$
\begin{align*}
x(t)-x(t-\Delta t)&=-\frac{\overline\beta(t)}{2}x(t-\Delta t)\Delta t+\sqrt{\overline\beta(t)}\Delta W \\
dx(t)&=-\frac{\overline\beta(t)}{2}x(t)dt+\sqrt{\overline\beta(t)}d W
\end{align*}
$$
此时已经可与 (13) 对齐，$f(x_t,t)=-\frac{\overline\beta(t)}{2}x(t)$，$g(t)=\sqrt{\overline\beta(t)}$

## 2. Score Matching

加噪公式：
$$
\begin{align*}
x(t)&=x+\sigma(t)\varepsilon, \quad x\sim P_{data} \\
x(t+\Delta t)&=x+\sigma(t+\Delta t)\varepsilon
\end{align*}
$$

分布：
$$
x(t)\sim \mathcal{N}(x,\ \sigma^2(t)I) \\
x(t+\Delta t) \sim \mathcal{N}(x,\ \sigma^2(t+\Delta t)I)
$$

从 $x(t) \to x(t+\Delta t)$ 需要加上分布 $\mathcal{N}\Big(0,\ \big[\sigma^2(t+\Delta t)-\sigma^2(t)\big]I\Big)$ :

$$
\begin{align*}
x(t+\Delta t)&=x(t) + \sqrt{\sigma^2(t+\Delta t)-\sigma^2(t)}\varepsilon \\
x(t+\Delta t) - x(t) &=  \sqrt{\frac{\sigma^2(t+\Delta t)-\sigma^2(t)}{\Delta t}}\sqrt{\Delta t}\varepsilon \\
dx(t)&=\sqrt{\frac{d\sigma^2{t}}{dt}}dW
\end{align*}
$$

此时已经可与 (13) 对齐，即：
$$
f(x_t,t)=0, \quad g(t)=\sqrt{\frac{d\sigma^2{t}}{dt}}
$$

对应一致性模型里的：
$$
f(x_t,t)=0, \quad g(t)=\sqrt{2t}
$$

## 3. 从 SDE 前向公式到扩散模型 加噪公式

要求加噪的均值 $\mu$ 和方差 $\sigma$：

$$
p(x_t|x_0)\sim \mathcal{N}(x_t;\ \mu,\ \sigma^2 I)
$$


SDE 前向公式：
$$
\boxed{dx_t=f(x_t, t)dt + g(t)dW_t}\ ,\\
dW_t=\sqrt{dt}\varepsilon,\\
\varepsilon \sim \mathcal{N}(0,I)
$$

Ito 过程，引理：$\forall \phi(x_t,t)$
$$
d\phi=\frac{\partial \phi}{\partial t}dt+\sum_i \frac{\partial \phi}{\partial x_i}dx_i+\frac{1}{2}\sum_{ij}\frac{\partial^2 \phi}{\partial x_i \partial y_i}dx_i dx_j
$$

其中 $x_t$ 表示向量，$x_i,x_j,\cdots$ 是标量，表示 $x_t$ 中的一个元素。

$dx_i, dx_j$ 也是随机变量，

$$
dx_i=f_i(x_t,t)dt+g(t)dW \\
dx_j=f_j(x_t,t)dt+g(t)dW \\
dx_idy_i=f_i(x_t,t)f_j(x_t,t)(dt)^2+f_i(x_t,t)dt\cdot g(t)dW \\
+f_j(x_t,t)dt\cdot g(t)dW + g^2(t)(dW)^2
$$

其中 第一项 $\approx 0$，

求第四项 $(dW)^2$ 均值和方差

$$
dW=\sqrt{dt}\varepsilon
$$

$$
\because E[(dW)^2]=E[dt\varepsilon^2]=dtE[\varepsilon^2] \\
\because D(\varepsilon)=E[\varepsilon^2]-E^2[\varepsilon] \\
\therefore E[(dW)^2]=dt
$$

$$
\because D[(dW)^2]=D[dt\varepsilon^2]=(dt)^2D[\varepsilon^2]=0 \\
\because (dt)^2\approx 0 \\
\therefore D[(dW)^2]=0
$$
得到 $(dW)^2 $ 均值为 $dt$, 方差为 0。
> 此处通过矩母函数可以求出 $D[\varepsilon^2]=2$，正常要用二次变差和均方收敛推上面的均值和方差

求第二、三项 $dt\cdot dW$，  
由于 
$$
dW=\sqrt{dt}\varepsilon
$$
所以
$$
dt \cdot dW \approx 0 
$$

所以综上：
$$
dx_idy_i= {g^2(t)}dt
$$

则 
$$
d\phi=\frac{\partial \phi}{\partial t}dt+\sum_i [\frac{\partial \phi}{\partial x_i}f_i(x_t,t)dt]+\sum_{i}[\frac{\partial \phi}{\partial x_i}g(t)dW_t] \\
+\frac{1}{2}\sum_{ij}[\frac{\partial^2 \phi}{\partial x_i \partial x_j}g^2(t)dt]
$$

求

$$
dE[\phi]=E[\frac{\partial \phi}{\partial t}dt]+\sum_i E[\frac{\partial \phi}{\partial x_i}f_i(x_t,t)dt]+\sum_{i}E[\frac{\partial \phi}{\partial x_i}g(t)dW_t] \\
+\frac{1}{2}\sum_{ij}E[\frac{\partial^2 \phi}{\partial x_i \partial x_j}g^2(t)dt]
$$

其中，$dW_t$ 与其它项无关，所以

$$
E[\frac{\partial \phi}{\partial x_i}g(t)dW_t]=E[\frac{\partial \phi}{\partial x_i}g(t)]E[dW_t]
$$

其中，$E[dW_t]=E[\sqrt{d_t}\varepsilon]=0$，所以

$$
dE[\phi]=E[\frac{\partial \phi}{\partial t}dt]+\sum_i E[\frac{\partial \phi}{\partial x_i}f_i(x_t,t)dt] 
+\frac{1}{2}\sum_{ij}E[\frac{\partial^2 \phi}{\partial x_i \partial x_j}g^2(t)dt]
$$
$$
\frac{dE[\phi]}{dt}=E[\frac{\partial \phi}{\partial t}]+\sum_i E[\frac{\partial \phi}{\partial x_i}f_i(x_t,t)] 
+\frac{1}{2}\sum_{ij}E[\frac{\partial^2 \phi}{\partial x_i \partial x_j}g^2(t)]
$$


而 Ito 过程 $\phi$ 与 $t$ 无关，所以
$$
\frac{\partial \phi}{\partial t}=0
$$

前面在 Ito 过程，提到过：引理：$\forall \phi(x_t,t)$，此处尝试令 $\phi(x_t,t)=x_t$，其中 $x_t$ 的每个分量是 $x_u$。所以

$$
\frac{\partial \phi}{\partial x_i}=\frac{\partial x_u}{\partial x_i}=
\begin{cases}
1, & i=u \\
0, & i\neq u
\end{cases}
$$
$$
\frac{\partial^2 \phi}{\partial x_i \partial x_j}=0
$$

所以
$$
\frac{dE[\phi]}{dt}=\frac{dE[x_u]}{dt}=E[f_u(x_t,t)]
$$

令均值向量 $m(t)=E[x_t]$，

$$
\boxed{
\frac{dm}{dt}=E[f(x_t,t)] \tag{17}
}
$$

协方差：
$
\phi(x_t,t)=x_ux_v-m_t(t)m_v(t)
$

$$
\begin{align*}
\frac{\partial \phi}{\partial t}&=-E[f_u(x_t,t)]m_v(t)-m_u(t)E[f_u(x_t,t)] \\

\frac{\partial \phi}{\partial x_i}&\Rightarrow
\begin{cases}
\frac{\partial \phi}{\partial x_u}=x_v, & i=u \\
\frac{\partial \phi}{\partial x_v}=x_u, & i=v \\    
\end{cases} \\

\frac{\partial^2 \phi}{\partial x_i \partial x_j}&\Rightarrow
\begin{cases}
\frac{\partial^2 \phi}{\partial x_u \partial x_v}=1, & i=u,j=v \\
\frac{\partial^2 \phi}{\partial x_v \partial x_u}=1, & i=v,j=u \\
\end{cases}

\end{align*}

$$

> 个人感觉二阶导这里如果 $i=j$ 那二阶导是 0，up主没说这个事。

$$
\frac{dE[\phi]}{dt}=E[(x_v-m_v(t))f_u(x_t,t)]+E[(x_u-m_u(t))f_v(x_t,t)]+E[g^2(t)]
$$

令协方差 $P(t)=E[(x_t-m(t))(x_t-m(t))^\top]$

$$
\boxed{
\frac{dP}{dt}=E[f(x_t,t)(x_t-m)^\top]+E[(x_t-m)f(x_t,t)^\top+E[g^2(t)] \tag{18}
}
$$

所以推导出 $p(x_t)$ 的均值和方差的微分形式，有什么用？

