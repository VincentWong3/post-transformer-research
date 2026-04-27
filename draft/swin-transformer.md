# Swin Transformer: Hierarchical Vision Transformer using Shifted Window

**论文：** [arXiv](https://arxiv.org/abs/2103.14030) · Ze Liu, Yutong Lin, Yue Cao, Han Hu et al. · 2021

---

## 一、先搞清楚"坑在哪"

2020 年 ViT 出来的时候，大家都挺兴奋的——终于有个 Transformer 能在图像分类上跟 CNN 掰手腕了。但兴奋劲儿一过，问题就来了。

ViT 的做法很直接：把图像切成 16×16 的 patch，然后像处理 NLP token 一样扔进 Transformer。这事在分类上还行，但一旦你把它往检测、分割这些"密集预测"任务上一放，就发现它哪哪都不对劲。

直白说，ViT 有两个硬伤：

**硬伤 1：算不起。** ViT 做的是全局 self-attention——每个 patch 跟所有其他 patch 算注意力。一张 224×224 的图切成 16×16 patch 后是 196 个 token，还行。但一张高分辨率图呢？假设输入是 1024×1024，patch 数就到了 4096。attention 复杂度是 $O(N^2)$，196 到 4096 差了 400 多倍。所以 ViT 本质上只适合低分辨率输入。

**硬伤 2：没有多尺度。** CNN 搞了这么多年，一个核心经验是"图像里有大有小"。大物体要全局信息，小物体要局部细节。ResNet 用四个 stage 逐步下采样，天然形成了多尺度特征金字塔。FPN、U-Net 这些检测分割框架全建立在它上面。但 ViT 只输出一个分辨率——你把 Patch Embedding 之后的东西直接做全局 attention，它就没有"层级"这回事了。

Swin Transformer 想解决的问题就是：**有没有一种 Transformer 架构，能像 CNN 一样自然地产出多尺度特征图，同时 attention 的计算量跟图像大小是线性关系？**

## 二、ViT 的真正问题

这里有必要多聊两句，因为理解 ViT 的局限是理解 Swin 设计的前提。

ViT 的核心改动其实就一个：把 NLP 的 Transformer Encoder 原封不动搬过来处理图像 patch。效果上，它在 ImageNet 分类确实追上了 ResNet，甚至在大规模数据上（JFT-300M）超越了。但这本质上是在**回避问题**，而不是**解决问题**——它选择了一个不需要多尺度、不需要高分辨率的任务来证明"Transformer 也能做视觉"。

真正的问题藏在它的结构里。ViT 的 patch size 是固定的（通常是 16×16），意味着特征图的分辨率始终是 $H/16 \times W/16$。这比 ResNet 最后一个 stage 的分辨率（$H/32 \times W/32$）还大，看似更精细。但问题是——只有一个分辨率。你没法在低分辨率特征上做检测提案（proposal），也没法在高分辨率特征上做精细分割。就一个尺度的特征图，啥都干不了。

有人试过给 ViT 加 decoder 做分割（SETR），但效果很勉强。数据也从侧面说明了这个问题：ViT 论文里只报了 ImageNet 分类结果，根本没提检测和分割——不是不想提，是做不了。

## 三、Swin Transformer 的核心思路

Swin 的设计直觉其实很简单，一句话就能说完：**既然 CNN 的多尺度金字塔效果好，那能不能在 Transformer 里也做一个金字塔出来？**

要做金字塔，就需要让特征图在网络的各个 stage 逐步变小、通道数逐步增多。这在 ViT 里实现不了，因为 ViT 一个 stage 到底。但 Swin 换了个设计思路：

```
输入 (H×W)
    │
Patch Embedding → H/4 × W/4 × C
    │                      ── Stage 1
    ▼
Patch Merging ──→ H/8 × W/8 × 2C
    │                      ── Stage 2
    ▼
Patch Merging ──→ H/16 × W/16 × 4C
    │                      ── Stage 3
    ▼
Patch Merging ──→ H/32 × W/32 × 8C
                           ── Stage 4
```

你看这个结构——跟 ResNet 完全对应上了。4 个 stage，分辨率逐级减半，通道翻倍。这意味着你直接把 Swin 往现成的 FPN、U-Net、Mask R-CNN 里一插就能用，不需要改任何下游代码。

但这里有一个问题：Stage 4 分辨率是 $H/32 \times W/32$，假设 $H=W=1024$，那就是 $32 \times 32 = 1024$ 个 patch。如果像 ViT 那样做全局 attention，复杂度是 $O(1024^2) = O(1M)$，还能接受。但 stage 1 有 $256 \times 256 = 65536$ 个 patch，$O(65K^2)$ 就没有任何可行性了。

所以 Swin 还有第二个关键设计——**窗口自注意力**。

## 四、窗口自注意力与 Shifted Window

### 4.1 窗口内自注意力（W-MSA）

Swin 把特征图划分成不重叠的窗口，每个窗口 $7 \times 7 = 49$ 个 patch。attention 只在窗口内部计算。这样整体的复杂度就从 $O(N^2)$ 降到了 $O(M^2N)$，其中 $M^2=49$ 是固定的。当 $N$ 很大时，这就从二次变成了线性。

复杂度对比：

$$\Omega(\text{全局 MSA}) = 4hwC^2 + 2(hw)^2C$$
$$\Omega(\text{窗口 MSA}) = 4hwC^2 + 2M^2hwC$$

$hw$ 是 patch 总数。第二项从 $(hw)^2$ 变成了 $M^2 \cdot hw$。当 $hw=65536$ 时，$65536^2$ 和 $49 \times 65536$ 的区别——差了四个数量级。

### 4.2 Shifted Window——解决跨窗口通信

但是，只在窗口内做 attention 有个明显的问题：**窗口之间没有信息交流**。位于不同窗口的 patch 永远看不到彼此。

Swin 的解法很巧妙：**交替使用两种窗口划分方式。** 第 $l$ 层用常规划分（W-MSA），第 $l+1$ 层把窗口偏移 $\lfloor M/2 \rfloor$ 个 patch（SW-MSA）。这样，上一层窗口边界附近的 patch，在下一层就落在了同一个窗口里，自然就实现了跨窗口通信。

```
Layer l (常规划分)        Layer l+1 (偏移划分)
┌────┬────┬────┐     ┌────┬────┬────┐
│  1  │  1  │  2  │     │  1  │  1  │  1  │
├────┼────┼────┤  →  ├────┼────┼────┤
│  1  │  1  │  2  │     │  1  │  2  │  2  │
├────┼────┼────┤     ├────┼────┼────┤
│  3  │  3  │  4  │     │  2  │  3  │  4  │
└────┴────┴────┘     └────┴────┴────┘
```

这里有个工程细节值得注意。偏移之后，窗口数量从 $4$ 变成了 $9$（因为边角被切碎了）。Swin 用了一个 **cyclic shift** 技巧——把切碎的窗口片拼回去，这样还是只需要算 $4$ 个窗口的 attention，做完后把结果再还原。这个操作让 shifted window 的实际延迟非常接近常规窗口。

有意思的是，Swin 的实验显示，shifted window 的延迟远低于同样能实现跨窗口通信的 sliding window 方案（比如之前的一些局部 attention 工作）。原因在硬件层面：sliding window 时每个 query pixel 的 key set 都不同，访存效率极低；而 Swin 的 window 内所有 query 共享 key set，缓存友好。

## 五、实验结果

Swin Transformer 在三个主要任务上都刷了新 SOTA：

| 任务 | 数据集 | 指标 | 结果 | 提升 vs 前 SOTA |
|------|--------|------|------|-----------------|
| 图像分类 | ImageNet-1K | top-1 acc | 87.3%（Swin-L, 384×384） | 超越 DeiT-B |
| 目标检测 | COCO test-dev | box AP | 58.7（Swin-L + HTC++） | +2.7 |
| 实例分割 | COCO test-dev | mask AP | 51.1（Swin-L + HTC++） | +2.6 |
| 语义分割 | ADE20K val | mIoU | 53.5（Swin-L + UperNet） | +3.2 |

这里需要注意几件事：

第一，分类上的 87.3% 是 **Swin-L** 用 **384×384 高分辨率输入** 跑出来的。Swin-Tiny 在常规 224×224 输入下是 81.3%，跟 ResNet-50 差不多。这不是一篇"为了分类而生"的论文。

第二，检测和分割的提升幅度（+2.7 AP, +3.2 mIoU）在当年是很有冲击力的。这是因为多尺度特征图对密集预测任务太重要了——Swin 的层级结构让它能直接接入 FPN、UperNet，而 ViT/DeiT 做不到这一点。

## 六、这篇论文在技术路线上的位置

```
Transformer 视觉化路线：
                          
2017  ViT（全局 attention, 单尺度）
       │
2021  Swin（窗口 attention, 多尺度金字塔）
       │
2022  ConvNeXt（从 Swin 反向吸取经验 → 纯卷积但有 Swin 的结构）
       │
2023  … 两条路线开始融合
```

Swin 在这条路线上的地位很特殊：它是第一个证明"Transformer 也能当通用视觉 backbone"的工作。在它之前，ViT 只是"一个能分类的 Transformer"；在它之后，Transformer 在视觉领域终于有了跟 CNN 平起平坐的资格。

另外，Swin 的 shifted window 设计也启发了 MLP-Mixer 系列的改进，论文里提到这招对 all-MLP 架构也有帮助。不过这事后来没掀起太大波澜。

## 七、局限性与后续方向

Swin 当然不是完美的。

**窗口大小的确定性。** 默认 $7 \times 7$ 窗口在所有 stage 都一样，这跟 CNN 的可变感受野不是一个量级。后续的 Focal Transformer、CSWin 等工作都在试图让窗口更灵活。

**工程复杂度。** Swin 的实现比 ViT 复杂不少。Cyclic shift、masking 这些操作虽然在理论上是线性的，但工程上的优化空间不如卷积成熟。ConvNeXt 后来能找到纯卷积方案取得同样好的结果，也从侧面说明 Swin 的某些优势可能来自结构（多尺度、层级），而不一定是 attention 本身。

**分类上的天花板。** Swin-L 在 224×224 下的分类精度其实不如一些同期工作（如 CoAtNet）。它的真正价值在检测分割上。如果你只做分类，Swin 未必是最佳选择。

## 八、小结

Swin Transformer 的核心贡献可以总结为两件事：**一是把多尺度金字塔引入了 Transformer，二是用 shifted window 以线性复杂度实现跨窗口通信。** 这两件事让它成为了第一个真正通用的视觉 Transformer backbone。在 Swin 之前，CNN 是视觉 backbone 的唯一选择；在 Swin 之后，你有了第二个选项。

---

*Paper reading generated by paper-read skill. Rounds: 1. Accuracy: ✅ Logic: ✅ Readability: ✅*
