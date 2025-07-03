# A new age in protein design empowered by deep learning

## INTRODUCTION

酶的作用：  
酶反应 enzymatic reactions, 转运 transport, 免疫反应 immune response, and 信号转导 signal transduction。

> 蛋白质有通常地 defined 3D structure 作为天然构象（native conformation），但<mark> 同样的分子通常会有其他的构象 </mark>。

蛋白质设计领域，包括：设计新型酶、小分子 binders、蛋白 binders、和免疫原 immunogens。

从头设计蛋白质包括：designing （1）sequence from structure（inverse protein folding）,
（2）sequence from function,（3）structure from function

## 图神经网络

邻域信息。

## 受物理启发的归纳偏差 Physics-inspired inductive biases

基本的归纳偏差：  
（1）选择输入（即序列、结构、表面surface 等）和  
（2）输入把数据传递给算法。

> 例如，序列信息可以编码为单串氨基酸（single string of amino acids）、多序列比对（multiple sequence alignment，MSA）或位置特定评分矩阵（position-specific scoring matrix，PSSM），  
>
> 而结构信息的选项包括成对距离图（pairwise distance maps）、离散体素网格（discrete voxel grids）、具有原子分辨率的图（graphs with atomic resolution）或粗粒度图（coarsegrained graphs）。
>
> 另一个例子是MaSIF框架采用的表面表示（surface），以解释蛋白质之间的相互作用主要涉及它们的表面。

物理和几何启发的（Physics-
and geometry-inspired）归纳偏差有助于提高神经网络的数据效率和泛化能力。

### invariant or equivariant  
等变是这样一个概念：

对输入做某种变换（比如旋转），模型输出也以相同的方式变换，结果的“几何关系”保持一致。

数学上，如果变换是 $T$，模型是 $f$，那我们说 $f$ 是 $T$-等变的，当：
$$f(T(x))=T(f(x))$$

例子：SchNet，
AlphaFold2 的 invariant point attention (IPA) 模块。


## 语言模型，transformers 和注意力

protein language models (PLMs) 自监督学习  
$\Rightarrow$ inverse folding and protein structure prediction

transformer 一个应用例子是 AlphaFold2 的 Evoformer，两个 transformer 分别处理 MSA 和 pair representations 的信息。

## 深度生成模型 Deep generative models

### VAEs

真实目标：最大化 log-likelihood，即 $p(x)=\mathrm{ELBO}+\mathrm{KL}(q(z|x)\| p(z|x))$  

实际优化目标：$\mathrm{ELBO}=\mathbb{E}_{q(z\vert x)}[\log p(x\vert z)]-\mathrm{KL}(q(q|x)\|p(z))$，因为 ELBO 是 $p(x)$ 下界。并且因为它只依赖于   
我们自己设计的近似分布（编码器）$p(x∣z)$、  
事先设定的先验分布（如标准高斯）$p(z)$ 和   
解码器生成的分布（可训练）$q(z∣x)$，  
这些都可以计算或采样。

两个目标之间的误差：$\mathrm{KL}(q(z|x)\| p(z|x))$ 不可计算（因为我们不知道真实的后验 
$p(z\vert x)$），所以要优化 ELBO。

### GANs

### Diffusion models

## PROTEIN DESIGN POWERED BY DEEP LEARNING

### Structure to sequence

a protein can be represented as 2D matrices of distances, angles and dihedrals (二面体) for every residue pair and the differences between these inter-residue features can be used to define an optimization objective

损失函数通常由   
(1) structural loss 结构损失 (constrained hallucination 约束幻觉),   
(2) structural stability loss 稳定性损失 (unconstrained hallucination 无约束幻觉),   
比如 AlphaFold2 的 local distance difference test (<mark>pLDDT</mark>) 每个残基的结构准确性评分（0~100）, predicted template modeling (<mark>pTM</mark>) 结构整体相似度预测分数, and predicted aligned error (<mark>pAE</mark>) 预测的配对残基之间的误差  
(3) or a combination of both。

RoseTTAfold, AlphaFold2, ESMFold, OmegaFold

> hallucination 在这里不是“幻觉”，而是从零开始设计蛋白质序列

三种序列优化方法：  
Markov chain Monte Carlo (MCMC), Gradient-based approaches, combined the
MCMC and gradient-based strategies

由于设计的蛋白质可能 表面暴露的疏水性残留物，体外成功率低，溶解性差。所以可能需要下游进一步优化，比如使用 Rosetta or ProteinMPNN。

逆折叠 inverse folding   
比如 ProteinMPNN 基于图神经网络。  
逆折叠的模型共识：基于局部环境预测残基类型，使用完全由刚性蛋白质结构（通常是晶体结构）。
> 这些用于训练模型的蛋白质结构，是通过实验（尤其是 X 射线晶体学）得到的、静止的、高精度结构，是一种“冻结定格”的蛋白质快照  

最近的方法基于图的，强烈几何先验。

使用 AlphaFold2 生成数据集。

### Function to structure

通过优化选定的结构来设计新的蛋白质，这些结构可以执行特定的任务或表现出特定的特性，如酶活性、与目标分子的结合亲和力、溶解度或稳定性。

蛋白质设计子任务的全部范围，如从头骨架 de novo backbones、小分子或蛋白质binders small molecule or protein binders、对称低聚物 symmetric oligomers、表位特异性抗体 epitope-specific antibodies、和基序支持支架 motif-supporting scaffolds

成对距离图 (pairwise distance maps) 来解决的，这是一种方便且关键的旋转/翻译不变蛋白质表示

早期方法的一个共同问题是无效生成接触图，即无法嵌入3D空间的不一致的欧几里得距离矩阵。由于神经网络的输出域通常没有足够的限制，模型必须纯粹从数据中学习所有几何约束，这通常会导致不完美的结果，并最终阻碍3D模型的重建。

尽管<mark>扩散模型</mark>已经成功地应用于在各种环境中生成小分子，但由于蛋白质结构中的原子数量要多得多，将这种技术应用于蛋白质设计更具挑战性。迄今为止，基于扩散的蛋白质设计方法通过仅设计蛋白质的一部分或通过对更粗粒度的蛋白质表示进行操作来克服这一问题。**最广泛采用的选择是 $C\alpha$ 原子以及相应的残基类型作为基本构建块**。在大多数情况下，这种表示还被赋予了*基于N-$C\alpha$-C主链原子的全局残基方向*的概念，这些主链原子形成了一个局部参考帧。

扩散模型中，positions, orientations, and amino acid types 配备了合适的扩散过程和损失函数。

残基的 位置 positions（3D 坐标 Coordinates）“加一个高斯噪声” → 模拟扩散过程。  
残基的 朝向 orientations（怎么转的）角度 angular features 是周期的，比如：
0° 和 360° 是一样的，所以你不能像处理实数一样随便加噪声，不然会导致跳过边界。所以对于角度，比如方向朝向、扭转角，要用特殊的数学结构来建模，比如：
圆环（S¹） 或 球面（S²）
在这些流形（manifolds）或 groups 上加噪声是数学上比较精细的操作。  
氨基酸种类 Amino acid types 是离散的分类特征（20种），所以通常用的是离散扩散。  
氨基酸的“侧链”通常会被省略，等到后面的“序列设计步骤”再加回来；只有在少数情况下，它们会用专门设计的扩散机制来建模，这个机制通常是围绕它们的**扭转角（torsional angles）**来建模的。

<mark>ProtDiff</mark> 通过用“粒子滤波”的思路的采样流程能够从一个只“无条件训练”的模型中，生成满足某个条件的样本的结构，比如：托住某个关键功能片段的蛋白质骨架，而不需要重新训练模型。

ProtDiff 的核心做法可以总结成三步：  
① 初始时采样多个“粒子”（结构候选）
从噪声中采样出多个结构候选，不带条件。  
② 在每一步，计算这些粒子对条件的“匹配程度”
比如：预测的结构是否包住了目标功能位点（motif）？  
匹配越好 → 给的“权重”越高  
匹配差的 → 权重低，甚至被丢弃  
③ 根据这些权重，选择最好的粒子继续向前生成
就像筛选出“有潜力的设计”，下一轮继续细化。

<mark>RFdiffusion</mark> 结合 RoseTTAfold 和 ProteinMPNN，实现：  
【1】Stable monomer design（稳定的单体蛋白设计）  
【2】Symmetric oligomer design（对称的寡聚体设计）。  
“Oligomer” 是由多个蛋白单元（monomer）组成的复合体，例如二聚体（2个）、四聚体（4个）  
“Symmetric” = 它们之间有对称关系（比如围成一圈、镜像对称）  
【3】Scaffold design for functional motifs or enzyme active sites。  
“Scaffold” 是蛋白质的整体结构框架，  
“Functional motif” 是负责功能的小结构单元，比如结合位点、活性残基。  
【4】Protein binder design（结合蛋白设计）设计出一个蛋白质能够特异性地结合另一个目标蛋白，类似于“抗体”识别病毒、配体结合受体。

> 对训练好的RoseTTAfold模型进行结构预测和自调节（一种在每个去噪步骤中将先前的预测作为额外输入提供给神经网络的技术）微调有助于提高性能。

> 对于 binder design 和 scaffolding applications，separate conditional RFdiffusion 模型被训练来生成给定一组固定残基（interface hotspots, catalytic site, and input motif）的蛋白质结构。

最常见的性能度量是“自洽性 self-consistency 评估”（也叫 round-trip 评估），先根据生成的骨架结构设计出蛋白序列，然后用一个**从头预测结构的算法（ab initio folding algorithm）**把这个序列折叠回去，看看预测出的结构是不是和最初生成的骨架一致。但实验验证是滞后的。

### Function to sequence

ProGen

## Methodological challenges

一个更广泛的问题是机器学习方法的泛化和分布外推。由于监督学习算法从根本上将复杂的非线性函数拟合到提供的数据点，随着我们远离训练示例，它们的不确定性会大幅增加。  
Roney和Ovchinnikov讨论了克服这一障碍的可能补救措施之一，其中表明AlphaFold2已经学会了一种近似的生物物理能量函数，可用于从其主要序列预测蛋白质结构，从而超出可用的进化数据进行推广。  
> Roney and Ovchinnikov: State-of-the-art estimation of protein model accuracy using alphafold.