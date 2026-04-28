# Deformable DETR: Deformable Transformers for End-to-End Object Detection（2021）

**论文：** [arXiv](https://arxiv.org/abs/2010.04159) · Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai (SenseTime, USTC, CUHK) · 2021

---

## 一、先搞清楚问题在哪里

2020 年 DETR（Carion et al.）的出现给目标检测领域带来了一股新风——它第一次用 Transformer + Hungarian 匹配实现了端到端检测，彻底去掉了 anchor 生成、NMS 等手工设计组件。架构极其简洁：CNN backbone + Transformer encoder-decoder + FFN，不到 50 行推理代码。

但 DETR 有两个致命的硬伤，限制了它成为通用的检测框架：

**硬伤 1：收敛极慢。** DETR 需要 500 epoch 训练才能收敛，而 Faster R-CNN 只需要 ~109 epoch。原因在于 Transformer 的交叉注意力（cross-attention）在初始化时对特征图的所有像素几乎均匀分配注意力——每个 query 看到整张图，但不知道看哪里。训练过程需要逐步把注意力"聚焦"到有物体的区域，这个从均匀到稀疏的转变非常缓慢。

**硬伤 2：小物体检测差。** DETR 的特征图分辨率是 32 倍下采样（COCO 上典型输入 800×1333 → 特征图 25×42），小物体的信息几乎丢失了。想用高分辨率特征图？encoder 的自注意力复杂度是 $O(H^2W^2C)$，分辨率翻倍复杂度就变 4 倍，直接炸显存。

所以问题来了：有没有一种注意力机制，既能保持 Transformer 的关系建模能力，又能像卷积一样高效地在空间上采样？

## 二、DETR 的注意力机制为什么不行

论文从两个角度剖析了 DETR 的问题：

**收敛慢的根源：注意力权重的初始化问题。** 对于 self-attention 和 cross-attention，在合理的参数初始化下，$\boldsymbol{U}_m\boldsymbol{z}_q$ 和 $\boldsymbol{V}_m\boldsymbol{x}_k$ 大致服从均值 0 方差 1 的分布。当 key 的数量 $N_k$ 很大时，注意力权重 $A_{mqk} \approx 1/N_k$。这意味着**初始时每个 query 几乎均匀地看所有 key**，梯度信号极其微弱。在图像领域，key 数量（像素）轻松上万，这个问题格外严重。

**复杂度高的根源：全局注意力是 $O(N_q N_k)$。** 标准 multi-head attention 的计算复杂度：

$$\text{MultiHeadAttn}(\boldsymbol{z}_q, \boldsymbol{x}) = \sum_{m=1}^{M} \boldsymbol{W}_m \left[ \sum_{k \in \Omega_k} A_{mqk} \cdot \boldsymbol{W}'_m \boldsymbol{x}_k \right]$$

复杂度为 $O(N_qC^2 + N_kC^2 + N_qN_kC)$。图像领域 $N_q = N_k \gg C$，第三项 $O(N_qN_kC)$ 占主导。对于 encoder 的 self-attention，$N_q = N_k = H \times W$，复杂度 $O(H^2W^2C)$——这就是为什么 DETR 不能用高分辨率特征图。

## 三、Deformable DETR 的核心思路

**把可变形卷积的稀疏空间采样能力和 Transformer 的关系建模能力结合起来——用可变形注意力替代标准注意力，每个 query 只关注一组少量的关键采样点。**

```
DETR:
  query → 关注所有像素 → O(N²) 复杂度、收敛慢

Deformable DETR:
  query → 预测偏移量 → 只采样 K 个关键点 → O(NK) 复杂度、收敛快
```

三个核心设计：

1. **可变形注意力模块**（Deformable Attention）：每个 query 从特征图上预测 K 个采样点的偏移量和注意力权重，只在这 K 个点上做 attention。复杂度从 $O(H^2W^2)$ 降到 $O(HWK)$，$K \ll HW$（论文默认 K=4）。

2. **多尺度可变形注意力**：不需要 FPN，直接在多尺度特征图上做可变形 attention。每个 query 从多个尺度的特征图上各采样 K 个点，自然聚合多尺度信息。

3. **迭代框精细化**（Iterative Bounding Box Refinement）：每层 decoder 都预测框，后续层在前一层的基础上精细化。

## 四、整体结构与 forward 流程

### 整体架构

```
输入图片
    │
    ▼
CNN Backbone (ResNet-50)
    │
    ├── Level 1 (1/8) ────┐
    ├── Level 2 (1/16) ───┤
    └── Level 3 (1/32) ───┤
                          │
                          ▼
          Multi-Scale Deformable Encoder (×6)
                          │
                          ▼
          Multi-Scale Deformable Decoder (×6)
                    │          │
              Self-Attn    Cross-Attn (Deformable)
              (object queries)   (multi-scale features)
                          │
                          ▼
                  Prediction FFN
                    │      │
                class    box (refined per layer)
```

### 前向传播

1. **Backbone**（ResNet-50）：输入图片，输出三个分辨率的特征图：1/8（$C_4$）、1/16（$C_5$）、1/32（$C_5$ 再下采样）。用 1×1 卷积统一通道数到 256。

2. **多尺度特征投影**：三个尺度的特征图通过线性投影到相同维度 $d=256$：
   - Level 1: $256 \times H/8 \times W/8$
   - Level 2: $256 \times H/16 \times W/16$
   - Level 3: $256 \times H/32 \times W/32$

3. **Multi-Scale Deformable Encoder**（6 层）：每层对每个 query 位置，从所有尺度的特征图上采样 K 个点，聚合多尺度信息。输出 $256 \times (H/8 \times W/8 + H/16 \times W/16 + H/32 \times W/32)$。

4. **Decoder**（6 层）：100 个 object queries（$100 \times 256$）：
   - Self-attention：queries 之间互相通信
   - Cross-attention（可变形）：从 encoder 输出的多尺度特征图上采样 K 个点
   - 每层都通过 FFN 预测框和类别

5. **损失计算**：Hungarian 匹配 + L1 损失 + GIoU 损失

## 五、核心模块解释

### 5.1 可变形注意力模块

**要解决的问题：** 标准注意力对所有 key 计算权重，$O(N_qN_k)$ 的复杂度在图像上不可接受。同时均匀初始化的注意力权重导致收敛极慢。

**模块工作原理：** 对于每个 query，不关注所有 key，而是只关注 K 个空间位置（K << HW）。这 K 个位置是从 query 特征中预测出来的偏移量加上参考点。

**公式：**

$$\text{DeformAttn}(\boldsymbol{z}_q, \boldsymbol{p}_q, \boldsymbol{x}) = \sum_{m=1}^{M} \boldsymbol{W}_m \left[ \sum_{k=1}^{K} A_{mqk} \cdot \boldsymbol{W}'_m \boldsymbol{x}(\boldsymbol{p}_q + \Delta \boldsymbol{p}_{mqk}) \right]$$

其中：
- $\boldsymbol{z}_q$：query 特征
- $\boldsymbol{p}_q$：参考点位置（归一化坐标 [0,1]²）
- $\Delta \boldsymbol{p}_{mqk}$：第 m 个 head 第 k 个采样点相对于参考点的偏移——由 query 特征经过一个线性层预测
- $A_{mqk}$：注意力权重——由 query 特征经过另一个线性层 + softmax 预测，$\sum_{k=1}^K A_{mqk} = 1$
- $\boldsymbol{x}(\boldsymbol{p}_q + \Delta \boldsymbol{p}_{mqk})$：采样点位置的特征，通过双线性插值获取

关键性质：**复杂度降到 $O(N_q K C)$**，K=4 时比标准注意力的 $O(N_q N_k)$ 小几个数量级。

### 5.2 多尺度可变形注意力

**要解决的问题：** DETR 只用了单分辨率特征图，小物体检测差。传统方案用 FPN，但 DETR 的 attention 架构天然支持跨尺度聚合。

**模块工作原理：** 每个 query 从 L 个尺度的特征图上各采样 K 个点（共 LK 个点）：

$$\text{MSDeformAttn}(\boldsymbol{z}_q, \hat{\boldsymbol{p}}_q, \{\boldsymbol{x}^l\}_{l=1}^L) = \sum_{m=1}^{M} \boldsymbol{W}_m \left[ \sum_{l=1}^{L} \sum_{k=1}^{K} A_{mlqk} \cdot \boldsymbol{W}'_m \boldsymbol{x}^l(\phi_l(\hat{\boldsymbol{p}}_q) + \Delta \boldsymbol{p}_{mlqk}) \right]$$

其中 $\hat{\boldsymbol{p}}_q$ 是归一化到 [0,1]² 的参考点坐标，$\phi_l$ 将归一化坐标映射到第 l 层特征图的坐标空间。

复杂度：$O(N_q (L K) C)$，仍然与特征图大小无关。

### 5.3 迭代框精细化

**要解决的问题：** 单层预测的框不够精确。

**设计：** 每层 decoder 都预测边框偏移量，后续层在前一层的预测基础上做 refinement。具体来说，第 i 层的 FFN 预测偏移量 $( \Delta b_x, \Delta b_y, \Delta b_w, \Delta b_h)$，加到 $i-1$ 层的预测框上。**梯度只回传到当前层的偏移量**，前一层的结果被 stop-gradient 处理。

消融（论文 Table 2）：没有迭代精细化时 AP=45.7，加上后 AP=46.2（+0.5）。

## 六、核心公式与数学直觉

### 6.1 标准 Multi-Head Attention 回顾

$$\text{MultiHeadAttn}(\boldsymbol{z}_q, \boldsymbol{x}) = \sum_{m=1}^{M} \boldsymbol{W}_m \left[ \sum_{k \in \Omega_k} A_{mqk} \cdot \boldsymbol{W}'_m \boldsymbol{x}_k \right]$$

其中 $A_{mqk} \propto \exp\left\{\frac{\boldsymbol{z}_q^T \boldsymbol{U}_m^T \boldsymbol{V}_m \boldsymbol{x}_k}{\sqrt{C_v}}\right\}$。

复杂度分析：
- 第一项 $N_q C^2$：query 投影
- 第二项 $N_k C^2$：key 投影
- 第三项 $N_q N_k C$：注意力矩阵计算+加权求和
- 在图像中 $N_q = N_k = HW \gg C$，第三项占绝对主导

### 6.2 可变形注意力复杂度对比

| 注意力类型 | Encoder 复杂度 | Decoder Cross-Attn 复杂度 |
|-----------|---------------|--------------------------|
| 标准 (DETR) | $O(H^2W^2C)$ | $O(HWC^2 + NHWC)$ |
| 可变形 | $O(HWC^2)$ | $O(NKLC)$ |
| 节省 | 平方→线性 | 与特征图大小解耦 |

以 COCO 典型输入为例：$H=25, W=42, C=256, N=100, K=4, L=3$。
- Encoder 标准：$25^2 \times 42^2 \times 256 \approx 282M$
- Encoder 可变形：$25 \times 42 \times 256 \times K \times \text{head} \approx 25 \times 42 \times 256 \times 4 \times 8 \approx 8.6M$
- 节省约 **33 倍**

## 七、实验结果

### COCO 检测（论文 Table 1）

| 方法 | 骨干网络 | epoch | AP | AP_S | AP_M | AP_L |
|------|---------|-------|-----|------|------|------|
| DETR | ResNet-50 | 500 | 42.0 | 20.5 | 45.8 | 61.1 |
| **Deformable DETR** | ResNet-50 | **50** | **43.8** | **26.4** | **47.1** | **58.0** |
| Deformable DETR (iter) | ResNet-50 | 50 | 46.2 | 28.7 | 49.1 | 60.9 |
| Deformable DETR (iter+2stage) | ResNet-50 | 50 | **46.9** | **29.5** | **49.7** | **61.9** |

关键发现：
- **收敛速度快 10 倍**：50 epoch 达到 DETR 500 epoch 的水平，甚至更高（43.8 vs 42.0）
- **小物体大幅提升**：AP_S 从 20.5 提升到 26.4（+5.9），多尺度特征直接解决了 DETR 的小物体检测短板
- 迭代精细化 + 2-stage 进一步提升到 46.9 AP

### 消融实验（论文 Table 2）

| 配置 | AP | 说明 |
|------|-----|------|
| 基线 | 43.8 | 单尺度 + 无迭代 |
| + 多尺度 | 45.4 | +1.6，多尺度效益明显 |
| + 迭代精细化 | 46.2 | +2.4 累积 |
| + 2-stage | 46.9 | +3.1 累积 |

## 八、复杂度、效率与可扩展性分析

Deformable DETR 的核心贡献就是把检测器的复杂度瓶颈从不可扩展变成了可扩展：

| 指标 | DETR | Deformable DETR | 提升 |
|------|------|----------------|------|
| 训练 epoch | 500 | 50 | 10× 加速 |
| Encoder FLOPs (R50, 800×1333) | ~376G | ~124G | 3× 减少 |
| 小物体 AP_S | 20.5 | 26.4 | +29% |
| 最大特征图分辨率 | 1/32 | 1/8 | 16× 更多像素 |

新引入的瓶颈：K 个采样点虽然高效，但每个 query 只能看到局部区域，丢失了全局视野。不过 encoder 的 6 层堆叠和自注意力（object queries 之间）一定程度上补偿了这一点。

## 九、官方 GitHub / 源码实现

官方代码：[github.com/fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)

核心文件：

```
projects/configs/deformable_detr/
 ├── deformable_detr.py          # 模型定义
 ├── deformable_detr_r50_50ep.py # 训练配置
 └── ...
modeling/
 ├── deformable_transformer.py   # 可变形 Transformer
 │   ├── DeformableTransformer
 │   ├── DeformableTransformerEncoderLayer
 │   ├── DeformableTransformerDecoderLayer
 │   └── MSDeformAttn            # 核心模块
 └── ops/
     └── functions/
         └── ms_deform_attn.py   # CUDA 实现（Custom op）
```

核心可变形注意力实现（`MSDeformAttn`）：
```python
class MSDeformAttn(nn.Module):
    def forward(self, query, reference_points, input_flatten, ...):
        # query → 预测采样偏移和注意力权重
        sampling_offsets = self.sampling_offsets(query)  # (N, LvL*K*2)
        attention_weights = self.attention_weights(query).softmax(-1)
        
        # 双线性插值采样
        sampled = torch.nn.functional.grid_sample(
            input_flatten, sampling_locations, mode='bilinear', 
            padding_mode='zeros', align_corners=False)
        
        # 加权求和
        output = torch.einsum('...lkd,...lkd->...d', attention_weights, sampled)
        return output
```

注：多尺度可变形注意力涉及大量的双线性插值和索引操作，为了效率，官方实现用 CUDA 自定义算子（MultiScaleDeformableAttention）加速。这在纯 PyTorch 中实现起来会很慢。

## 十、Related Work 与技术谱系位置

```text
DETR (2020)
 ├── Deformable DETR (2021)   ← 本论文
 │   ├── Conditional DETR (2021)
 │   ├── DAB-DETR (2022)
 │   ├── DN-DETR (2022)
 │   └── DINO (2023)
 └── DETR 框架的后续变体
```

Deformable DETR 属于 DETR 路线上的**工程有效改进**节点。它不改变 DETR 的 set prediction 范式，而是在注意力机制的实现层面做了关键优化——用可变形采样替代全局注意力。这个改进让 DETR 从"理论可行但工程上难用"变成了"可以实际部署"。

它与同期工作 Conditional DETR（加速收敛）的关注点不同：Conditional DETR 从 query 设计的角度切入，Deformable DETR 从注意力机制角度切入。两者互补，后续的 DAB-DETR 把两者融合。

## 十一、时间线位置

- **2020.05**：DETR 发布，开启端到端检测
- **2020.10**：Deformable DETR 发布（本论文），解决收敛和小物体问题
- **2021.05**：Conditional DETR，从 query 角度加速收敛
- **2022.01**：DAB-DETR，4D anchor query + 温度调制
- **2022.03**：DN-DETR，去噪训练加速
- **2023.03**：DINO，综合 SOTA

Deformable DETR 出现在 DETR 发布后仅仅 5 个月，说明 DETR 的收敛和尺度问题是当时社区最迫切的痛点。这篇论文的速度（5 个月）也反映了这个方向的热度。

## 十二、范式意义

**工程改进节点**——不是范式转变，但把范式从"理论上可行"推进到了"实践中可用"。

具体来说：
- 注意力机制层面：从全局注意力 → 可变形稀疏注意力（借鉴可变形卷积的思想）
- 特征层面：从单尺度 → 多尺度（不需要 FPN）
- 训练层面：从 500 epoch → 50 epoch（10× 加速）

这三个改进叠加，让 DETR 从一个需要超长训练时间的概念验证变成了一个实用的检测框架。后续几乎所有 DETR 变体都继承了 Deformable DETR 的多尺度可变形注意力。

## 十三、局限性与后续方向

1. **自定义 CUDA 算子**：多尺度可变形注意力的高效实现依赖 CUDA 自定义算子（bilinear interpolation + indexing），在部署和跨平台移植时带来额外成本。后续 DINO 等简化了实现但本质没变。
2. **K 值固定**：每个 query 的采样点数 K 是固定的（默认 4），对于不同空间位置可能需要不同的 K 值。自适应 K 值是一个自然扩展方向。
3. **参考点初始化**：参考点 $\boldsymbol{p}_q$ 在 encoder 中均匀初始化（每个像素位置），在 decoder 中由 object queries 预测。2-stage 变体用 encoder 的输出预测参考点，效果最好但也最复杂。
4. **全局视野的丧失**：虽然 6 层堆叠和 self-attention 补偿了局部性，但极端情况（如超大物体跨越多尺度区域）仍然可能漏检。不过从实验结果看，AP_L 并没有下降（61.1→58.0 单尺度，61.9 多尺度+迭代）。

## 十四、小结

直白说，Deformable DETR 的价值不只是提出了可变形注意力机制，而是**把 DETR 从"500 个 epoch 的理论玩具"变成了"50 个 epoch 的实用工具"**。它用可变形卷积的思想改造了 Transformer 的注意力机制，同时引入多尺度特征，解决了 DETR 最痛的两个问题。

如果把它放进技术树，它是 DETR 分支上的**关键工程改进节点**——没有它，后面的一众 DETR 变体（DINO、DN-DETR 等）可能都跑不动。
