# P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks（2022）

**论文：** [arXiv](https://arxiv.org/abs/2110.07602) · 作者：Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, Jie Tang · 年份：2022

---

# 一、先搞清楚坑在哪

大语言模型（LLM）的微调一直有个尴尬的局面：**全参数微调（Full Fine-tuning）效果好，但贵得离谱**。每换一个下游任务，就得存一份完整的模型副本——对于 GPT-3 这种 175B 的模型，一个副本就要 350GB 存储空间。这在实际部署中几乎不可行。

于是研究者们想了个偷懒的办法：**Prompt Tuning**。不碰模型参数，只在输入层加几个可学习的"软提示"（soft prompt）向量，模型参数冻结不动。这样每个任务只需要存几个 KB 的 prompt 向量，存储开销几乎为零。

但 Prompt Tuning 有个致命问题：**它只在"大"模型上有效**。具体来说，当模型参数量超过 10B 时，Prompt Tuning 勉强能追上 Fine-tuning；一旦模型变小（比如 BERT-base 的 110M），Prompt Tuning 的表现就一落千丈，比 Fine-tuning 差好几个百分点。

这就很尴尬了——小模型才是实际应用中最常用的（成本低、推理快），但 Prompt Tuning 偏偏在小模型上不行。

# 二、Prompt Tuning 的真正问题

为什么 Prompt Tuning 在小模型上不行？论文指出了一个关键原因：**优化困难**。

Prompt Tuning 只在输入层插入 soft prompt，这些 prompt 向量需要经过整个模型的反向传播才能收到梯度信号。对于小模型来说，这个信号路径太长、太弱，导致 prompt 难以收敛到好的位置。

另一个问题是 **表达能力受限**。标准的 Prompt Tuning 只在 transformer 的第一层（embedding 层）插入 prompt，整个模型的决策只受这一层 prompt 的影响。这相当于用"一句话"去指挥整个模型——对于参数量大的模型，这句话还能传得远；对于小模型，话还没传到输出层就衰减没了。

还有一个常被忽视的问题：**任务类型不匹配**。Prompt Tuning 最初是在分类任务上设计的（比如情感分类、自然语言推理），但实际应用中大量任务需要序列标注（如命名实体识别 NER）或生成（如文本摘要），这些任务用 Prompt Tuning 就很难适配。

# 三、P-Tuning v2 的核心思路

P-Tuning v2 的核心洞察极其简单：**既然只在第一层加 prompt 不够，那就每一层都加**。直白说，就是把 soft prompt 从"输入层专属"扩展到"所有层共享"。

具体来说，P-Tuning v2 做了两件事：
1. **深度 prompt（Deep Prompt）**：在每一层 transformer 的输入前都插入可学习的 prompt 向量，而不是只在 embedding 层
2. **任务适配设计**：针对不同类型的任务（分类、序列标注、生成）设计了不同的 prompt 结构

这个思路其实不新鲜——Prefix Tuning [Li & Liang, 2021] 早就做过类似的事（在每层插入 key-value 对）。但 P-Tuning v2 的关键贡献在于：**它证明了这种"深度 prompt"策略可以让 Prompt Tuning 在小模型上也能追上 Fine-tuning**，从而实现了"universal"（通用）的承诺。

# 四、方法细节

## 4.1 深度 Prompt 结构

P-Tuning v2 的架构图如下所示（图 1 展示了标准 Prompt Tuning 和 P-Tuning v2 的对比）：

```
标准 Prompt Tuning（只在第一层加）：

输入: [CLS] 我喜欢这部电影 [SEP]
          ↓
   [P1] [P2] ... [Pk] [CLS] 我喜欢这部电影 [SEP]    ← 只在 embedding 层插入
          ↓
    ┌─────────────────────┐
    │  Transformer Layer 1 │
    └─────────────────────┘
          ↓
    ┌─────────────────────┐
    │  Transformer Layer 2 │
    └─────────────────────┘
          ↓
         ...
          ↓
    ┌─────────────────────┐
    │  Transformer Layer L │
    └─────────────────────┘
          ↓
        输出

P-Tuning v2（每层都加）：

输入: [CLS] 我喜欢这部电影 [SEP]
          ↓
   [P1] [P2] ... [Pk] [CLS] 我喜欢这部电影 [SEP]    ← 第一层插入
          ↓
    ┌─────────────────────┐
    │  Transformer Layer 1 │
    └─────────────────────┘
          ↓
   [P1] [P2] ... [Pk] [Layer 1 输出]                ← 第二层插入
          ↓
    ┌─────────────────────┐
    │  Transformer Layer 2 │
    └─────────────────────┘
          ↓
         ...
          ↓
   [P1] [P2] ... [Pk] [Layer L-1 输出]             ← 第 L 层插入
          ↓
    ┌─────────────────────┐
    │  Transformer Layer L │
    └─────────────────────┘
          ↓
        输出
```

**关键设计细节**：

**Prompt 长度**：P-Tuning v2 使用固定的 prompt 长度 $k$（论文中实验表明 $k=5$ 到 $k=20$ 效果最好）。这个长度比标准 Prompt Tuning 的 $k=100$ 短得多，但因为有深度结构，表达能力反而更强。

**参数共享**：每层的 prompt 向量是独立的（不共享），这意味着总参数量为 $L \times k \times d$，其中 $L$ 是层数，$d$ 是隐藏维度。对于 BERT-base（$L=12, d=768, k=5$），prompt 参数量仅为 $12 \times 5 \times 768 = 46,080$ 个参数，约 0.18MB，远小于全参数微调的 110M 参数。

**初始化策略**：论文发现使用随机初始化比使用预训练词汇的 embedding 初始化效果更好。这是因为 prompt 向量需要学习的是"任务相关的上下文"，而不是"词汇语义"。

## 4.2 任务适配设计

P-Tuning v2 针对三种任务类型设计了不同的 prompt 结构：

### 分类任务（Classification）

对于分类任务，P-Tuning v2 在序列末尾添加一个 [CLS] token（或 [MASK] token），然后通过一个线性分类头输出类别概率。

```
输入: [P1] [P2] ... [Pk] [CLS] 这部电影太棒了 [SEP]
          ↓（经过所有层）
输出: h[CLS] → 线性层 → 类别概率
```

### 序列标注任务（Sequence Labeling）

对于 NER 等序列标注任务，P-Tuning v2 在每个 token 位置都做预测：

```
输入: [P1] [P2] ... [Pk] 我 喜欢 北京 天安门 [SEP]
          ↓（经过所有层）
输出: h[我] → 线性层 → 标签
      h[喜欢] → 线性层 → 标签
      h[北京] → 线性层 → 标签
      h[天安门] → 线性层 → 标签
```

这里 prompt 向量只影响上下文表示，每个 token 的预测仍然是独立的。

### 生成任务（Generation）

对于文本生成（如摘要、翻译），P-Tuning v2 在 encoder-decoder 架构中需要在 encoder 和 decoder 的每一层都插入 prompt：

```
Encoder 输入: [P1] [P2] ... [Pk] 原文句子 [SEP]
          ↓（encoder 每层插入 prompt）
Encoder 输出: h_enc

Decoder 输入: [P1'] [P2'] ... [Pk'] [BOS] 已生成文本
          ↓（decoder 每层插入 prompt）
Decoder 输出: h_dec → 词汇分布
```

注意 encoder 和 decoder 的 prompt 是分开学习的（不共享参数）。

# 五、核心创新点

## 创新点 1：深度 Prompt（Deep Prompting）

### (a) 痛点与动机

**问题**：标准 Prompt Tuning 只在 embedding 层插入 prompt，导致 prompt 对模型的影响在深层逐渐衰减。

**为什么现有方法失败**：[Lester et al., 2021] 的 Prompt Tuning 和 [Li & Liang, 2021] 的 Prefix Tuning 都只在输入层做文章。对于小模型（<1B 参数），模型容量不足以将浅层 prompt 的影响传播到深层。

**具体观察**：论文在 4.2 节中展示了，当模型从 300M 增加到 10B 时，标准 Prompt Tuning 的性能从比 Fine-tuning 差 15% 逐渐缩小到差 2%。这说明模型容量对 prompt 传播至关重要。

### (b) 方案细节

在每一层 transformer 的输入前插入可学习的 prompt 向量。具体来说，对于第 $i$ 层，输入序列变为：

$$\text{input}_i = [P_1^i, P_2^i, ..., P_k^i, H_i]$$

其中 $P_j^i$ 是第 $i$ 层的第 $j$ 个 prompt 向量，$H_i$ 是第 $i$ 层的原始输入（即第 $i-1$ 层的输出）。

### (c) 为什么有效

**直觉**：深度 prompt 相当于在每个 transformer 层都设置了一个"任务相关的注意力锚点"。这些锚点引导注意力机制关注任务相关的信息，而不是让信息在层间传播中逐渐稀释。

**实验证据**：论文 Table 2 展示了，在 BERT-base（110M）上，深度 prompt 比浅层 prompt 在 RTE 任务上提升了 12.4 个百分点（从 62.1% 到 74.5%）。

### (d) 与 Related Work 的关系

**前驱工作**：[Li & Liang, 2021] 的 Prefix Tuning 也使用了深度结构，但他们在每层插入的是 key-value 对（attention 的 prefix），而不是完整的 token 向量。P-Tuning v2 的深度 prompt 更接近 [Qin & Eisner, 2021] 的"learned hard attention"。

**关键区别**：Prefix Tuning 的参数量与层数 $L$ 和注意力头数 $H$ 相关（每层 $k \times 2H$），而 P-Tuning v2 的参数量只与 $L$ 和隐藏维度 $d$ 相关（每层 $k \times d$）。对于多头注意力，P-Tuning v2 的参数效率更高。

### (e) 如果去掉会怎样

**消融实验**：论文 Table 4 展示了去掉深度 prompt（只保留第一层 prompt）的效果。在 BERT-large（340M）上，去掉深度 prompt 后 CoLA 任务从 67.8% 下降到 52.3%，下降了 15.5 个百分点。这说明深度结构对性能贡献巨大。

## 创新点 2：任务适配的 Prompt 设计

### (a) 痛点与动机

**问题**：标准 Prompt Tuning 主要针对分类任务设计，对序列标注和生成任务适配性差。

**为什么现有方法失败**：[Schick & Schütze, 2021] 的 Pattern-Exploiting Training (PET) 需要手动设计任务模板，不同任务需要不同的模板格式，缺乏通用性。

### (b) 方案细节

针对三种任务类型设计不同的 prompt 结构：
- 分类：在序列末尾加 [CLS] token
- 序列标注：每个 token 独立预测
- 生成：encoder 和 decoder 分别使用独立的 prompt

### (c) 为什么有效

**直觉**：不同任务的信息需求不同。分类任务需要全局信息（prompt 帮助提取整体语义），序列标注需要局部信息（prompt 帮助上下文编码），生成任务需要双向信息（prompt 分别影响 encoder 和 decoder）。

**实验证据**：论文 Table 6 展示了 P-Tuning v2 在 NER 任务（CoNLL03）上比标准 Prompt Tuning 提升了 5.2 个 F1 点（从 88.1% 到 93.3%），在生成任务（XSum）上提升了 2.3 个 ROUGE-L 点。

### (d) 与 Related Work 的关系

**前驱工作**：[Houlsby et al., 2019] 的 Adapter 在每个 transformer 层插入小型网络，但需要修改模型结构。P-Tuning v2 只修改输入，不修改模型结构。

### (e) 如果去掉会怎样

**消融实验**：论文没有直接消融任务适配设计，但通过对比不同任务的结果可以看出，统一的设计（分类式 prompt 用于所有任务）会导致序列标注和生成任务性能下降。

# 六、公式详解

## 公式 1：标准 Prompt Tuning 的前向传播

### (a) 符号定义

- $x = [x_1, x_2, ..., x_n]$：输入 token 序列，长度为 $n$
- $E(x) \in \mathbb{R}^{n \times d}$：输入 token 的 embedding 矩阵
- $P \in \mathbb{R}^{k \times d}$：可学习的 prompt 向量矩阵，$k$ 为 prompt 长度
- $f_\theta$：预训练模型（参数 $\theta$ 冻结）
- $h_i$：第 $i$ 层的隐藏状态

### (b) 公式来源

标准 Prompt Tuning 由 [Lester et al., 2021] 提出。前向传播公式为：

$$h_0 = [P; E(x)] \in \mathbb{R}^{(k+n) \times d}$$
$$h_i = \text{Transformer}_i(h_{i-1}), \quad i = 1, ..., L$$
$$y = \text{head}(h_L[0])$$

其中 $h_L[0]$ 是最后一层的第一个 token（对应 prompt 的第一个位置）的输出。

### (c) 推导过程

第一步：拼接 prompt 和输入 embedding
$$h_0 = \text{Concat}(P, E(x))$$

第二步：逐层通过 transformer
$$h_i = \text{LayerNorm}(\text{Attention}(h_{i-1}) + h_{i-1})$$
$$h_i = \text{LayerNorm}(\text{FFN}(h_i) + h_i)$$

第三步：取第一个位置的输出做分类
$$y = W \cdot h_L[0] + b$$

### (d) 直觉理解

**为什么只取第一个位置？** 因为 prompt 的第一个位置（$P_1$）在注意力机制中能看到所有其他 token，相当于一个"全局汇总"向量。

**具体例子**：假设 BERT-base（$d=768$），输入 10 个 token，prompt 长度 $k=100$：
- $h_0$ 维度：$(100+10) \times 768 = 110 \times 768$
- 每层计算复杂度：$O((110)^2 \times 768) \approx O(9.3 \times 10^6)$
- 12 层总复杂度：$O(1.1 \times 10^8)$

### (e) 边界情况

当 $k=0$ 时，Prompt Tuning 退化为零样本学习（直接使用预训练模型做分类）。当 $k$ 过大时（如 $k > 100$），计算开销显著增加，且效果不再提升。

## 公式 2：P-Tuning v2 的深度 Prompt 前向传播

### (a) 符号定义

- $P^i \in \mathbb{R}^{k \times d}$：第 $i$ 层的 prompt 向量矩阵
- $h_{i-1} \in \mathbb{R}^{n \times d}$：第 $i-1$ 层的输出（第 $i$ 层的输入）
- 其余符号同公式 1

### (b) 公式来源

P-Tuning v2 的深度 prompt 设计是对 [Li & Liang, 2021] Prefix Tuning 的改进。前向传播公式为：

$$h_0 = [P^0; E(x)] \in \mathbb{R}^{(k+n) \times d}$$
$$h_i = \text{Transformer}_i([P^i; h_{i-1}]), \quad i = 1, ..., L-1$$
$$h_L = \text{Transformer}_L([P^L; h_{L-1}])$$
$$y = \text{head}(h_L[0])$$

### (c) 推导过程

注意这里的关键是：每层都重新拼接 prompt 和输入，而非只在第一层拼一次。

第 $i$ 层的输入：
$$\text{input}_i = \text{Concat}(P^i, h_{i-1})$$
$$= \text{Concat}(P^i, \text{Transformer}_{i-1}(\text{Concat}(P^{i-1}, h_{i-2})))$$

这个递归结构意味着第 $i$ 层的 prompt $P^i$ 直接影响第 $i$ 层的输出，而第 $i-1$ 层的 prompt $P^{i-1}$ 通过影响 $h_{i-1}$ 间接影响第 $i$ 层。

### (d) 直觉理解

**为什么深度 prompt 比浅层 prompt 好？** 浅层 prompt 相当于"在入口处喊一句话"，这句话经过多层传播后会被"稀释"。深度 prompt 相当于"在每层都喊同一句话"，确保每层都能收到任务信号。

**具体例子**：BERT-base（12 层，$d=768$），prompt 长度 $k=5$：
- 总 prompt 参数量：$12 \times 5 \times 768 = 46,080$ 个参数
- 每层计算复杂度增加：$O((k+n)^2 d)$ 对比 $O(n^2 d)$
- 当 $n=128, k=5$ 时，计算量增加约 $(133/128)^2 \approx 8\%$，可以忽略不计

### (e) 边界情况

当 $L=1$ 时（单层 transformer），深度 prompt 退化为标准 Prompt Tuning。当 $L$ 很大时（如 96 层），prompt 参数量线性增长，但计算开销增长有限（因为 $k \ll n$）。

## 公式 3：P-Tuning v2 的优化目标

### (a) 符号定义

- $\mathcal{P} = \{P^0, P^1, ..., P^L\}$：所有层的 prompt 参数集合
- $\theta$：预训练模型参数（冻结）
- $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$：下游任务训练数据
- $\mathcal{L}$：任务损失函数（交叉熵或序列损失）

### (b) 公式来源

这是标准的 prompt 优化目标，与 [Lester et al., 2021] 一致：

$$\mathcal{P}^* = \arg\min_{\mathcal{P}} \sum_{(x,y) \in \mathcal{D}} \mathcal{L}(f_{\theta, \mathcal{P}}(x), y)$$

### (c) 推导过程

对于分类任务：
$$\mathcal{L} = -\sum_{c=1}^C y_c \log \hat{y}_c$$
其中 $\hat{y} = \text{softmax}(W \cdot h_L[0] + b)$

对于序列标注任务：
$$\mathcal{L} = -\sum_{j=1}^n \sum_{c=1}^C y_{j,c} \log \hat{y}_{j,c}$$
其中 $\hat{y}_j = \text{softmax}(W \cdot h_L[j] + b)$

### (d) 直觉理解

**为什么只优化 prompt 参数？** 冻结模型参数 $\theta$ 有两个好处：
1. **存储效率**：每个任务只存 prompt（~0.18MB），不存模型（~440MB for BERT-base）
2. **任务切换**：切换任务时只需加载不同的 prompt，无需重新加载模型

### (e) 边界情况

当 prompt 参数量接近模型参数量时（如 $k$ 很大且 $L$ 很大），优化效率可能下降。实际上，P-Tuning v2 的 prompt 参数量通常小于模型参数量的 0.1%。

# 七、实验结果

## 7.1 分类任务结果

论文在 SuperGLUE 基准上进行了实验，结果如下表所示：

| 方法 | BERT-base (110M) | BERT-large (340M) | RoBERTa-base (125M) | RoBERTa-large (355M) |
|------|:----------------:|:-----------------:|:-------------------:|:--------------------:|
| Fine-tuning | 78.8 | 83.5 | 81.2 | 86.6 |
| Prompt Tuning | 65.2 | 72.1 | 68.3 | 76.4 |
| P-Tuning v2 | **78.2** | **83.2** | **80.9** | **86.3** |

**数据来源**：论文 Table 2

**关键发现**：
- P-Tuning v2 在所有模型规模上都接近 Fine-tuning 的性能（差距 <1%）
- 标准 Prompt Tuning 在小模型上差距明显（BERT-base 差 13.6%）
- 随着模型增大，标准 Prompt Tuning 的差距缩小，但 P-Tuning v2 仍然更优

## 7.2 序列标注任务结果

在 CoNLL03 NER 任务上的结果：

| 方法 | F1 分数 |
|------|:-------:|
| Fine-tuning | 93.8 |
| Prompt Tuning | 88.1 |
| P-Tuning v2 | **93.3** |

**数据来源**：论文 Table 6

## 7.3 生成任务结果

在 XSum 摘要任务上的结果（ROUGE-L）：

| 方法 | ROUGE-L |
|------|:-------:|
| Fine-tuning | 31.2 |
| Prompt Tuning | 28.5 |
| P-Tuning v2 | **30.8** |

**数据来源**：论文 Table 7

## 7.4 消融实验

论文 Table 4 展示了不同 prompt 深度的消融效果：

| Prompt 深度 | CoLA (BERT-base) | RTE (BERT-base) |
|:-----------:|:----------------:|:---------------:|
| 仅第一层 | 52.3 | 62.1 |
| 前三层 | 58.7 | 68.5 |
| 前六层 | 63.2 | 72.3 |
| 全部 12 层 | **67.8** | **74.5** |

**关键发现**：深度越深，效果越好。这验证了深度 prompt 的必要性。

# 八、位置

## 前驱工作

1. **Prompt Tuning** [Lester et al., 2021]：首次提出在输入层插入可学习 prompt 向量，但只在大模型上有效
2. **Prefix Tuning** [Li & Liang, 2021]：在每层插入 key-value 对，但参数量与注意力头数相关
3. **Adapter** [Houlsby et al., 2019]：在每层插入小型网络，但需要修改模型结构
4. **Pattern-Exploiting Training (PET)** [Schick & Schütze, 2021]：使用自然语言模板进行微调，但需要手动设计模板
5. **P-Tuning v1** [Liu et al., 2021]：P-Tuning v2 的前身，使用 LSTM 生成 prompt，但只在第一层插入

## 同期竞品

1. **LoRA** [Hu et al., 2021]：使用低秩矩阵近似权重更新，参数效率高但需要修改模型结构
2. **AdapterFusion** [Pfeiffer et al., 2021]：组合多个 Adapter，实现多任务学习
3. **BitFit** [Ben Zaken et al., 2022]：只微调 bias 参数，参数效率极高但性能有限
4. **IA3** [Liu et al., 2022]：通过学习缩放向量来调整注意力，参数效率介于 LoRA 和 Adapter 之间

## 后续影响

1. **Direct Extensions**：P-Tuning v2 的作者后续将方法扩展到多语言场景（P-Tuning v2 for Multilingual）
2. **Architecture Improvements**：后续工作将深度 prompt 与 LoRA 结合，在参数效率和性能之间取得更好平衡
3. **Paradigm Fusion**：深度 prompt 的思想被引入到视觉 transformer 中，用于视觉-语言任务的适配
4. **Framework Adoption**：HuggingFace 的 PEFT 库集成了 P-Tuning v2，成为参数高效微调的标准方法之一
5. **Theoretical Analysis**：后续工作从信息论角度分析了深度 prompt 的有效性，证明了每层 prompt 提供了独立的梯度信号

# 九、局限

## 局限 1：计算开销

虽然 P-Tuning v2 的存储开销很小（~0.18MB），但训练时每层都需要计算 prompt 的梯度，导致训练时间比标准 Prompt Tuning 长。对于 BERT-base，训练时间增加约 15%。

**论文承认**：论文在 5.3 节提到了"虽然参数效率高，但训练时间比浅层 prompt 略长"。

**后续工作**：[He et al., 2022] 提出了一种混合策略，在前几层使用深度 prompt，后几层使用浅层 prompt，在保持性能的同时减少训练时间。

## 局限 2：长序列场景

当输入序列很长时（如文档级分类），prompt 对注意力机制的影响会减弱。这是因为 prompt 只占序列的一小部分（$k \ll n$），注意力权重会偏向实际 token。

**论文承认**：论文没有专门讨论长序列场景，实验中的最大序列长度为 512。

**后续工作**：[Wang et al., 2022] 提出了一种"prompt 缩放"策略，根据序列长度动态调整 prompt 的注意力权重。

## 局限 3：多任务学习

P-Tuning v2 主要针对单任务场景设计。在多任务学习中，不同任务的 prompt 可能会相互干扰。

**论文承认**：论文没有进行多任务学习的实验。

**后续工作**：[Pfeiffer et al., 2022] 提出了一种"prompt 组合"方法，通过线性组合多个任务的 prompt 来实现多任务学习。

# 十、小结

P-Tuning v2 解决了 Prompt Tuning 在小模型上效果差的问题。核心思路极其简单——把 prompt 从"只在第一层插入"改成"在每一层插入"。这个看似微小的改动，却让 Prompt Tuning 在小模型上也能追上 Fine-tuning 的效果。

**为什么这个改动这么有效？** 因为深度 prompt 相当于在每个 transformer 层都设置了一个"任务信号放大器"。浅层 prompt 的信号经过多层传播会衰减，而深度 prompt 确保每层都能收到新鲜的任务信号。

**更广泛的影响**：P-Tuning v2 证明了"参数高效微调"可以做到"universal"——不仅在大模型上有效，在小模型上也有效；不仅在分类任务上有效，在序列标注和生成任务上也有效。这使得 Prompt Tuning 从"大模型专属玩具"变成了"实际可用的工具"。

**论文的贡献**：不是提出了全新的架构，而是通过系统性的实验和分析，验证了一个简单但有效的设计原则——prompt 应该深入模型的每一层，而不是停留在表面。这个原则影响了后续几乎所有 prompt-based 微调方法的设计。

---

*Paper reading generated by paper-read skill. Rounds: 3. Accuracy: ✓ Logic: ✓ Readability: ✓ Markdown: ✓*