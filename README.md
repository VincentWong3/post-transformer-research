# 后 Transformer 时代技术路线演化图（Post-Transformer Technical Evolution Tree）

本仓库整理的是一个 **后 Transformer 时代核心技术路线的结构化演化图谱**。

本技术树记录了 Transformer 如何打破模态壁垒的进程。我们关注的核心命题是：Transformer 能否成为 AI 领域的通用架构？

---

# Transformer（Attention Is All You Need, 2017）
- 根问题：Transformer在翻译领域大获成功，那么它是否可以适配别的任务，成为某种通用的架构

## A. Transformer 在 NLP 中的应用
- 根问题：是否存在可复用的通用序列表征基座？

### 范式 A1：encoder-only 预训练范式

#### Step A1-1 [2018]
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](papers/A1-1.md)

#### Step A1-2 [2019]
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](papers/A1-2&D3-13.md)（T5）

#### 视觉迁移支

##### Step A1-3 [2021]
- [BEiT: BERT Pre-Training of Image Transformers](papers/A1-3&B4-8.md)

##### Step A1-4 [2021]
- [Masked Autoencoders Are Scalable Vision Learners](papers/A1-4&B4-9.md)（MAE）

##### Step A1-5 [2022/2023]
- [EVA: Exploring the Limits of Masked Visual Representation Learning](papers/A1-5&B4-10-EVA.md)
- [EVA-02: A Visual Representation for Neon Genesis](papers/A1-5&B4-10-EVA-02.md)

### 范式 A2：decoder-only 生成范式

#### Step A2-1 [2018]
- [Improving Language Understanding by Generative Pre-Training](papers/A2-1.md)（GPT）

#### Step A2-2 [2019]
- [Language Models are Unsupervised Multitask Learners](papers/A2-2.md)（GPT-2）

#### Step A2-3 [2020]
- [Scaling Laws for Neural Language Models](papers/A2-3&E3-1.md)

#### Step A2-4 [2022]
- [Training Compute-Optimal Large Language Models](papers/A2-4&E3-2.md)（Chinchilla）

### 范式 A3：训练稳定性 / 可训练性支撑

#### Step A3-1 [2018]
- [The Annotated Transformer](papers/A3-1.md)

#### Step A3-2 [2020]
- [On Layer Normalization in the Transformer Architecture](papers/A3-2.md)

---

## B. Encoder 如何改造，才能适配视觉？
- 根问题：Transformer 如何成为视觉 backbone？

### B1. 图像如何 token 化？
- 根问题：图像能否像语言一样被 token 化并交给 Transformer？

#### 范式 B1-1：CNN 参照系
##### Step B1-1 [2015]
- [Deep Residual Learning for Image Recognition](papers/B1-1&F2-1.md)（ResNet）
##### Step B1-2 [2019]
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](papers/B1-2.md)

#### 范式 B1-2：patch tokenization 范式
##### Step B1-3 [2020]
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](papers/B1-3.md)（ViT）
##### Step B1-4 [2020]
- [Training Data-Efficient Image Transformers & Distillation through Attention](papers/B1-4.md)（DeiT）
##### Step B1-5 [2021]
- [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](papers/B1-5.md)（T2T-ViT）
##### Step B1-6 [2021]
- [LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference](papers/B1-6.md)

### B2. 视觉 backbone 是否需要层级化？
- 根问题：视觉是否需要不同于 NLP 的层级表示？

#### 范式 B2-1：hierarchical backbone 范式
##### Step B2-1 [2021]
- [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](papers/B2-1.md)（PVT）
##### Step B2-2 [2021]
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](papers/B2-2.md)
##### Step B2-3 [2021]
- [Multiscale Vision Transformers](papers/B2-3&B5-3&B5-8&B5-12.md)（MViT）
##### Step B2-4 [2021]
- [Focal Self-attention for Local-Global Interactions in Vision Transformers](papers/B2-4.md)（Focal Transformer）
##### Step B2-5 [2022]
- [CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows](papers/B2-5.md)

### B3. 视觉 backbone 是否需要局部归纳偏置？
- 根问题：attention 是否足够表达局部结构？

#### 范式 B3-1：local inductive bias 回归范式
##### Step B3-1 [2022]
- [A ConvNet for the 2020s](papers/B3-1.md)（ConvNeXt）

### B4. 是否可以形成统一视觉 encoder 基座模型？
- 根问题：没有昂贵标签时，能否学习通用视觉表征？

#### 范式 B4-1：contrastive representation learning
##### Step B4-1 [2020]
- [A Simple Framework for Contrastive Learning of Visual Representations](papers/B4-1.md)（SimCLR）
##### Step B4-2 [2020]
- [Momentum Contrast for Unsupervised Visual Representation Learning](papers/B4-2.md)（MoCo）

#### 范式 B4-2：non-contrastive representation learning
##### Step B4-3 [2020]
- [Bootstrap Your Own Latent](papers/B4-3.md)（BYOL）
##### Step B4-4 [2020]
- [Exploring Simple Siamese Representation Learning](papers/B4-4.md)（SimSiam）
##### Step B4-5 [2021]
- [Emerging Properties in Self-Supervised Vision Transformers](papers/B4-5.md)（DINO）
##### Step B4-6 [2021]
- [iBOT: Image BERT Pre-Training with Online Tokenizer](papers/B4-6.md)
##### Step B4-7 [2023]
- [DINOv2: Learning Robust Visual Features without Supervision](papers/B4-7.md)

#### 范式 B4-3：masked image modeling
##### Step B4-8 [2021]
- [BEiT: BERT Pre-Training of Image Transformers](papers/A1-3&B4-8.md)
##### Step B4-9 [2021]
- [Masked Autoencoders Are Scalable Vision Learners](papers/A1-4&B4-9.md)（MAE）
##### Step B4-10 [2022/2023]
- [EVA: Exploring the Limits of Masked Visual Representation Learning](papers/A1-5&B4-10-EVA.md)
- [EVA-02: A Visual Representation for Neon Genesis](papers/A1-5&B4-10-EVA-02.md)

### B5. Transformer 如何 handle 时空视觉任务？
- 根问题：视频/时空任务能否像图像一样 token 化并由 Transformer 统一建模？

#### B5-1. 视频是否可以 token 化？
- 根问题：时间维如何进入 Transformer？
##### 范式 B5-1-1：spatiotemporal tokenization
- **Step B5-1 [2021]**
  - [Is Space-Time Attention All You Need for Video Understanding?](papers/B5-1&B5-4&B5-9.md)（TimeSformer）
- **Step B5-2 [2021]**
  - [ViViT: A Video Vision Transformer](papers/B5-2&B5-5&B5-10.md)
##### 范式 B5-1-2：tubelet / temporal patch 路线
- **Step B5-3 [2021]**
  - [Multiscale Vision Transformers](papers/B2-3&B5-3&B5-8&B5-12.md)（MViT）

#### B5-2. 时空 attention 如何降低复杂度？
- 根问题：视频 token 数量巨大时，attention 如何可扩展？
##### Step B5-4 [2021]
- [Is Space-Time Attention All You Need for Video Understanding?](papers/B5-1&B5-4&B5-9.md)（TimeSformer）
##### Step B5-5 [2021]
- [ViViT: A Video Vision Transformer](papers/B5-2&B5-5&B5-10.md)
##### Step B5-6 [2021]
- [Video Swin Transformer](papers/B5-6&B5-7&B5-11.md)

#### B5-3. 视频 backbone 是否需要新的归纳偏置？
- 根问题：除了空间局部性，还需要时间局部性/运动连续性吗？
##### Step B5-7 [2021]
- [Video Swin Transformer](papers/B5-6&B5-7&B5-11.md)
##### Step B5-8 [2021]
- [Multiscale Vision Transformers](papers/B2-3&B5-3&B5-8&B5-12.md)（MViT）

#### B5-4. 是否存在统一时空 encoder 基座模型？
- 根问题：视频是否也能形成 foundation encoder？
##### Step B5-9 [2021]
- [Is Space-Time Attention All You Need for Video Understanding?](papers/B5-1&B5-4&B5-9.md)（TimeSformer）
##### Step B5-10 [2021]
- [ViViT: A Video Vision Transformer](papers/B5-2&B5-5&B5-10.md)
##### Step B5-11 [2021]
- [Video Swin Transformer](papers/B5-6&B5-7&B5-11.md)
##### Step B5-12 [2021]
- [Multiscale Vision Transformers](papers/B2-3&B5-3&B5-8&B5-12.md)（MViT）

---

## C. Decoder 如何改造，才能适配视觉任务？
- 根问题：视觉输出能否 transformer 化？

### C1. detection 如何 transformer 化？
- 根问题：目标检测能否从 proposal/anchor 流水线改写成 token/set prediction？

#### 范式 C1-1：传统参照系
##### Step C1-1 [2015]
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](papers/C1-1.md)

#### 范式 C1-2：set prediction detection 范式
##### Step C1-2 [2020]
- [End-to-End Object Detection with Transformers](papers/C1-2.md)（DETR）
##### Step C1-3 [2021]
- [Conditional DETR for Fast Training Convergence](papers/C1-3.md)
##### Step C1-4 [2022]
- [Dynamic Anchor Boxes are Better Queries for DETR](papers/C1-4.md)（DAB-DETR）
##### Step C1-5 [2022]
- [DN-DETR: Accelerate DETR Training by Introducing Query DeNoising](papers/C1-5.md)
##### Step C1-6 [2022]
- [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](papers/C1-6.md)

#### 范式 C1-3：成熟化关键分支
##### Step C1-7 [2021]
- [Deformable DETR: Deformable Transformers for End-to-End Object Detection](papers/C1-7.md)

### C2. segmentation 如何 transformer 化？
- 根问题：分割任务能否统一成 mask-level token / query 接口？

#### 范式 C2-1：传统参照系
##### Step C2-8 [2015]
- [Fully Convolutional Networks for Semantic Segmentation](papers/C2-8.md)（FCN）
##### Step C2-9 [2015]
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](papers/C2-9.md)
##### Step C2-10 [2017]
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](papers/C2-10.md)

#### 范式 C2-2：mask classification segmentation 范式
##### Step C2-11 [2021]
- [Per-Pixel Classification is Not All You Need for Semantic Segmentation](papers/C2-11.md)
- （MaskFormer）
##### Step C2-12 [2022]
- [Masked-attention Mask Transformer for Universal Image Segmentation](papers/C2-12.md)
- （Mask2Former）
##### Step C2-13 [2023]
- [OneFormer: One Transformer to Rule Universal Image Segmentation](papers/C2-13.md)

---

## D. 视觉模型和语言模型是否可以结合？
- 根问题：视觉与语言能否进入统一表征与统一接口空间？

### D1. 是否可以共享表征空间？
- 根问题：图像与文本能否进入同一个 embedding 空间？

#### 范式 D1-1：早期跨模态 encoder 范式
##### Step D1-1 [2019]
- [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](papers/D1-1.md)
##### Step D1-2 [2019]
- [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](papers/D1-2.md)
##### Step D1-3 [2019]
- [UNITER: UNiversal Image-TExt Representation Learning](papers/D1-3.md)

#### 范式 D1-2：对齐式大规模预训练范式
##### Step D1-4 [2021]
- [Learning Transferable Visual Models From Natural Language Supervision](papers/D1-4.md)（CLIP）
##### Step D1-5 [2021]
- [ALIGN: Large-scale Image and Noisy-Text Embedding](papers/D1-5.md)
##### Step D1-6 [2023]
- [Sigmoid Loss for Language Image Pre-Training](papers/D1-6.md)（SigLIP）

### D2. 是否可以共享任务接口？
- 根问题：视觉模型和语言模型能否通过统一接口协同工作？

#### 范式 D2-1：图文理解/生成桥梁
##### Step D2-7 [2022]
- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](papers/D2-7.md)

#### 范式 D2-2：冻结 LLM + 视觉适配器
##### Step D2-8 [2022]
- [Flamingo: a Visual Language Model for Few-Shot Learning](papers/D2-8.md)
##### Step D2-9 [2023]
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](papers/D2-9.md)
##### Step D2-10 [2023]
- [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](papers/D2-10.md)
##### Step D2-11 [2023]
- [MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models](papers/D2-11.md)
##### Step D2-12 [2023]
- [Visual Instruction Tuning](papers/D2-12.md)
- （LLaVA）

### D3. 是否可以形成多模态 foundation model？
- 根问题：是否可以形成统一 multimodal seq2seq 接口？

#### 范式 D3-1：统一 multimodal seq2seq 范式
##### Step D3-13 [2019]
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](papers/A1-2&D3-13.md)
- （T5）
##### Step D3-14 [2022]
- [OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](papers/D3-14.md)
##### Step D3-15 [2023]
- [Kosmos-1: Language Is Not All You Need](papers/D3-15.md)
##### Step D3-16 [2022/2023]
- [PaLI: A Jointly-Scaled Multilingual Language-Image Model](papers/D3-16-PaLI.md)
- [PaLI-X: On Scaling up a Multilingual Vision and Language Model](papers/D3-16-PaLI-X.md)

---

## E. 怎样高效地训练、扩展与部署？
- 根问题：Transformer 如何变得可扩展、可训练、可适配？

### E1. 如何降低 attention 复杂度？
- 根问题：attention 是否必须 O(N²)？

#### 范式 E1-1：高效 attention 范式
##### Step E1-1 [2020]
- [Linformer: Self-Attention with Linear Complexity](papers/E1-1.md)
##### Step E1-2 [2020]
- [Rethinking Attention with Performers](papers/E1-2.md)（Performer）
##### Step E1-3 [2020]
- [Longformer: The Long-Document Transformer](papers/E1-3.md)
##### Step E1-4 [2020]
- [Reformer: The Efficient Transformer](papers/E1-4.md)

### E2. attention 是否唯一有效的 token mixing 机制？
- 根问题：attention 是否唯一有效的 token mixing 机制？

#### 范式 E2-1：非 attention mixing 基线
##### Step E2-1 [2021]
- [MLP-Mixer: An all-MLP Architecture for Vision](papers/E2-1.md)
##### Step E2-2 [2021]
- [Pay Attention to MLPs](papers/E2-2.md)（gMLP）

#### 范式 E2-2：频域 mixing 范式
##### Step E2-3 [2021]
- [FNet: Mixing Tokens with Fourier Transforms](papers/E2-3.md)
##### Step E2-4 [2020]
- [Fourier Neural Operator for Parametric Partial Differential Equations](papers/E2-4.md)
##### Step E2-5 [2022]
- [Efficient Token Mixing for Transformers via Adaptive Fourier Neural Operators](papers/E2-5.md)（AFNO）

### E3. 是否存在 scaling law？
- 根问题：模型规模、数据规模、算力规模之间是否存在统一规律？

#### Step E3-1 [2020]
- [Scaling Laws for Neural Language Models](papers/A2-3&E3-1.md)

#### Step E3-2 [2022]
- [Training Compute-Optimal Large Language Models](papers/A2-4&E3-2.md)（Chinchilla）

### E4. 如何参数高效微调与低成本适配？
- 根问题：大模型时代，如何低成本微调和适配？

#### 范式 E4-1：parameter efficient adaptation
##### Step E4-1 [2019]
- [Parameter-Efficient Transfer Learning for NLP](papers/E4-1.md)（Adapter Modules）
##### Step E4-2 [2021]
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](papers/E4-2.md)
##### Step E4-3 [2021/2022]
- [The Power of Scale for Parameter-Efficient Prompt Tuning](papers/E4-3-Prompt-Tuning.md)
- [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](papers/E4-3-P-Tuning-v2.md)
##### Step E4-4 [2021]
- [BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models](papers/E4-4.md)
##### Step E4-5 [2022]
- [Few-shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](papers/E4-5.md)（IA3）
##### Step E4-6 [2021]
- [LoRA: Low-Rank Adaptation of Large Language Models](papers/E4-6.md)
##### Step E4-7 [2023]
- [QLoRA: Efficient Finetuning of Quantized LLMs](papers/E4-7.md)

#### 范式 E4-2：相关低成本思想旁支
##### Step E4-8 [2019]
- [DistilBERT, a distilled version of BERT](papers/E4-8.md)
##### Step E4-9 [2018]
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](papers/E4-9.md)

---

## F. Transformer 的理论本质是什么？
- 根问题：Transformer 为什么有效，它的本质是什么？

### F1. attention 是否联想记忆？
- 根问题：attention 是否 memory retrieval？

#### 范式 F1-1：联想记忆 / Hopfield 范式
##### Step F1-1 [2020]
- [Hopfield Networks is All You Need](papers/F1-1.md)
##### Step F1-2 [2021]
- Modern Hopfield Networks 相关论文（如：Dense Associative Memory for Pattern Recognition）

### F2. 是否是动力系统？
- 根问题：Transformer 是否可理解为连续深度 / 动力系统？

#### 范式 F2-1：连续深度 / 动力系统范式
##### Step F2-1 [2015]
- [Deep Residual Learning for Image Recognition](papers/B1-1&F2-1.md)（ResNet）
##### Step F2-2 [2018]
- [Neural Ordinary Differential Equations](papers/F2-2.md)
##### Step F2-3 [2019]
- [Augmented Neural ODEs](papers/F2-3-Augmented-Neural-ODE.md)
- [FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models](papers/F2-3-FFJORD.md)
- [Latent ODEs for Irregularly-Sampled Time Series](papers/F2-3-Latent-ODE.md)
##### Step F2-4 [2019]
- [Deep Equilibrium Models](papers/F2-4.md)
##### Step F2-5 [2021]
- [Transformers as Energy-Based Models](papers/F2-5.md)
- Transformers as Dynamical Systems 相关论文

### F3. 如何理解深层稳定训练？
- 根问题：深层 Transformer 为什么能稳定训练？

#### 范式 F3-1：深层稳定训练理论
##### Step F3-1 [2019]
- [Fixup Initialization: Residual Learning Without Normalization](papers/F3-1.md)
##### Step F3-2 [2020]
- Pre-LN Transformer 相关论文（如：On Layer Normalization in the Transformer Architecture）
##### Step F3-3 [2022]
- [DeepNet: Scaling Transformers to 1,000 Layers](papers/F3-3.md)
##### Step F3-4 [2022]
- μ-Parametrization / Tensor Programs 系列论文