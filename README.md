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
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

#### Step A1-2 [2019]
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer （T5）

#### 视觉迁移支

##### Step A1-3 [2021]
- BEiT: BERT Pre-Training of Image Transformers

##### Step A1-4 [2021]
- Masked Autoencoders Are Scalable Vision Learners（MAE）

##### Step A1-5 [2022/2023]
- EVA: Exploring the Limits of Masked Visual Representation Learning
- EVA-02: A Visual Representation for Neon Genesis

### 范式 A2：decoder-only 生成范式

#### Step A2-1 [2018]
- Improving Language Understanding by Generative Pre-Training（GPT）

#### Step A2-2 [2019]
- Language Models are Unsupervised Multitask Learners（GPT-2）

#### Step A2-3 [2020]
- Scaling Laws for Neural Language Models

#### Step A2-4 [2022]
- Training Compute-Optimal Large Language Models（Chinchilla）

### 范式 A3：训练稳定性 / 可训练性支撑

#### Step A3-1 [2018]
- The Annotated Transformer

#### Step A3-2 [2020]
- On Layer Normalization in the Transformer Architecture

#### Step A3-3 [2020-2021]
- Pre-LN / Post-LN Transformer 相关论文（如：On Layer Normalization in the Transformer Architecture）

---

## B. Encoder 如何改造，才能适配视觉？
- 根问题：Transformer 如何成为视觉 backbone？

### B1. 图像如何 token 化？
- 根问题：图像能否像语言一样被 token 化并交给 Transformer？

#### 范式 B1-1：CNN 参照系
##### Step B1-1 [2015]
- Deep Residual Learning for Image Recognition（ResNet）
##### Step B1-2 [2019]
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

#### 范式 B1-2：patch tokenization 范式
##### Step B1-3 [2020]
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（ViT）
##### Step B1-4 [2020]
- Training Data-Efficient Image Transformers & Distillation through Attention（DeiT）
##### Step B1-5 [2021]
- Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet（T2T-ViT）
##### Step B1-6 [2021]
- LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference

### B2. 视觉 backbone 是否需要层级化？
- 根问题：视觉是否需要不同于 NLP 的层级表示？

#### 范式 B2-1：hierarchical backbone 范式
##### Step B2-1 [2021]
- Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions（PVT）
##### Step B2-2 [2021]
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
##### Step B2-3 [2021]
- Multiscale Vision Transformers（MViT）
##### Step B2-4 [2021]
- Focal Self-attention for Local-Global Interactions in Vision Transformers（Focal Transformer）
##### Step B2-5 [2022]
- CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows

### B3. 视觉 backbone 是否需要局部归纳偏置？
- 根问题：attention 是否足够表达局部结构？

#### 范式 B3-1：local inductive bias 回归范式
##### Step B3-1 [2022]
- A ConvNet for the 2020s（ConvNeXt）

### B4. 是否可以形成统一视觉 encoder 基座模型？
- 根问题：没有昂贵标签时，能否学习通用视觉表征？

#### 范式 B4-1：contrastive representation learning
##### Step B4-1 [2020]
- A Simple Framework for Contrastive Learning of Visual Representations （SimCLR）
##### Step B4-2 [2020]
- Momentum Contrast for Unsupervised Visual Representation Learning（MoCo）

#### 范式 B4-2：non-contrastive representation learning
##### Step B4-3 [2020]
- Bootstrap Your Own Latent（BYOL）
##### Step B4-4 [2020]
- Exploring Simple Siamese Representation Learning（SimSiam）
##### Step B4-5 [2021]
- Emerging Properties in Self-Supervised Vision Transformers（DINO）
##### Step B4-6 [2021]
- iBOT: Image BERT Pre-Training with Online Tokenizer
##### Step B4-7 [2023]
- DINOv2: Learning Robust Visual Features without Supervision

#### 范式 B4-3：masked image modeling
##### Step B4-8 [2021]
- BEiT: BERT Pre-Training of Image Transformers
##### Step B4-9 [2021]
- Masked Autoencoders Are Scalable Vision Learners（MAE）
##### Step B4-10 [2022/2023]
- EVA: Exploring the Limits of Masked Visual Representation Learning
- EVA-02: A Visual Representation for Neon Genesis

### B5. Transformer 如何 handle 时空视觉任务？
- 根问题：视频/时空任务能否像图像一样 token 化并由 Transformer 统一建模？

#### B5-1. 视频是否可以 token 化？
- 根问题：时间维如何进入 Transformer？
##### 范式 B5-1-1：spatiotemporal tokenization
- **Step B5-1 [2021]**
  - Is Space-Time Attention All You Need for Video Understanding?（TimeSformer）
- **Step B5-2 [2021]**
  - ViViT: A Video Vision Transformer
##### 范式 B5-1-2：tubelet / temporal patch 路线
- **Step B5-3 [2021]**
  - Multiscale Vision Transformers（MViT）

#### B5-2. 时空 attention 如何降低复杂度？
- 根问题：视频 token 数量巨大时，attention 如何可扩展？
##### Step B5-4 [2021]
- Is Space-Time Attention All You Need for Video Understanding? （TimeSformer）
##### Step B5-5 [2021]
- ViViT: A Video Vision Transformer
##### Step B5-6 [2021]
- Video Swin Transformer

#### B5-3. 视频 backbone 是否需要新的归纳偏置？
- 根问题：除了空间局部性，还需要时间局部性/运动连续性吗？
##### Step B5-7 [2021]
- Video Swin Transformer
##### Step B5-8 [2021]
- Multiscale Vision Transformers（MViT）

#### B5-4. 是否存在统一时空 encoder 基座模型？
- 根问题：视频是否也能形成 foundation encoder？
##### Step B5-9 [2021]
- Is Space-Time Attention All You Need for Video Understanding? （TimeSformer）
##### Step B5-10 [2021]
- ViViT: A Video Vision Transformer
##### Step B5-11 [2021]
- Video Swin Transformer
##### Step B5-12 [2021]
- Multiscale Vision Transformers（MViT）

---

## C. Decoder 如何改造，才能适配视觉任务？
- 根问题：视觉输出能否 transformer 化？

### C1. detection 如何 transformer 化？
- 根问题：目标检测能否从 proposal/anchor 流水线改写成 token/set prediction？

#### 范式 C1-1：传统参照系
##### Step C1-1 [2015]
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

#### 范式 C1-2：set prediction detection 范式
##### Step C1-2 [2020]
- End-to-End Object Detection with Transformers（DETR）
##### Step C1-3 [2021]
- Conditional DETR for Fast Training Convergence
##### Step C1-4 [2022]
- Dynamic Anchor Boxes are Better Queries for DETR（DAB-DETR）
##### Step C1-5 [2022]
- DN-DETR: Accelerate DETR Training by Introducing Query DeNoising
##### Step C1-6 [2022]
- DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection

#### 范式 C1-3：成熟化关键分支
##### Step C1-7 [2021]
- Deformable DETR: Deformable Transformers for End-to-End Object Detection

### C2. segmentation 如何 transformer 化？
- 根问题：分割任务能否统一成 mask-level token / query 接口？

#### 范式 C2-1：传统参照系
##### Step C2-8 [2015]
- Fully Convolutional Networks for Semantic Segmentation（FCN）
##### Step C2-9 [2015]
- U-Net: Convolutional Networks for Biomedical Image Segmentation
##### Step C2-10 [2017]
- DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

#### 范式 C2-2：mask classification segmentation 范式
##### Step C2-11 [2021]
- Per-Pixel Classification is Not All You Need for Semantic Segmentation
- （MaskFormer）
##### Step C2-12 [2022]
- Masked-attention Mask Transformer for Universal Image Segmentation
- （Mask2Former）
##### Step C2-13 [2023]
- OneFormer: One Transformer to Rule Universal Image Segmentation

---

## D. 视觉模型和语言模型是否可以结合？
- 根问题：视觉与语言能否进入统一表征与统一接口空间？

### D1. 是否可以共享表征空间？
- 根问题：图像与文本能否进入同一个 embedding 空间？

#### 范式 D1-1：早期跨模态 encoder 范式
##### Step D1-1 [2019]
- ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks
##### Step D1-2 [2019]
- LXMERT: Learning Cross-Modality Encoder Representations from Transformers
##### Step D1-3 [2019]
- UNITER: UNiversal Image-TExt Representation Learning

#### 范式 D1-2：对齐式大规模预训练范式
##### Step D1-4 [2021]
- Learning Transferable Visual Models From Natural Language Supervision（CLIP）
##### Step D1-5 [2021]
- ALIGN: Large-scale Image and Noisy-Text Embedding
##### Step D1-6 [2023]
- Sigmoid Loss for Language Image Pre-Training（SigLIP）

### D2. 是否可以共享任务接口？
- 根问题：视觉模型和语言模型能否通过统一接口协同工作？

#### 范式 D2-1：图文理解/生成桥梁
##### Step D2-7 [2022]
- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

#### 范式 D2-2：冻结 LLM + 视觉适配器
##### Step D2-8 [2022]
- Flamingo: a Visual Language Model for Few-Shot Learning
##### Step D2-9 [2023]
- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
##### Step D2-10 [2023]
- InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning
##### Step D2-11 [2023]
- MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models
##### Step D2-12 [2023]
- Visual Instruction Tuning
- （LLaVA）

### D3. 是否可以形成多模态 foundation model？
- 根问题：是否可以形成统一 multimodal seq2seq 接口？

#### 范式 D3-1：统一 multimodal seq2seq 范式
##### Step D3-13 [2019]
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- （T5）
##### Step D3-14 [2022]
- OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework
##### Step D3-15 [2023]
- Kosmos-1: Language Is Not All You Need
##### Step D3-16 [2022/2023]
- PaLI: A Jointly-Scaled Multilingual Language-Image Model
- PaLI-X: On Scaling up a Multilingual Vision and Language Model

---

## E. 怎样高效地训练、扩展与部署？
- 根问题：Transformer 如何变得可扩展、可训练、可适配？

### E1. 如何降低 attention 复杂度？
- 根问题：attention 是否必须 O(N²)？

#### 范式 E1-1：高效 attention 范式
##### Step E1-1 [2020]
- Linformer: Self-Attention with Linear Complexity
##### Step E1-2 [2020]
- Rethinking Attention with Performers（Performer）
##### Step E1-3 [2020]
- Longformer: The Long-Document Transformer
##### Step E1-4 [2020]
- Reformer: The Efficient Transformer

### E2. attention 是否唯一有效的 token mixing 机制？
- 根问题：attention 是否唯一有效的 token mixing 机制？

#### 范式 E2-1：非 attention mixing 基线
##### Step E2-1 [2021]
- MLP-Mixer: An all-MLP Architecture for Vision
##### Step E2-2 [2021]
- Pay Attention to MLPs（gMLP）

#### 范式 E2-2：频域 mixing 范式
##### Step E2-3 [2021]
- FNet: Mixing Tokens with Fourier Transforms
##### Step E2-4 [2020]
- Fourier Neural Operator for Parametric Partial Differential Equations
##### Step E2-5 [2022]
- Efficient Token Mixing for Transformers via Adaptive Fourier Neural Operators（AFNO）

### E3. 是否存在 scaling law？
- 根问题：模型规模、数据规模、算力规模之间是否存在统一规律？

#### Step E3-1 [2020]
- Scaling Laws for Neural Language Models

#### Step E3-2 [2022]
- Training Compute-Optimal Large Language Models（Chinchilla）

### E4. 如何参数高效微调与低成本适配？
- 根问题：大模型时代，如何低成本微调和适配？

#### 范式 E4-1：parameter efficient adaptation
##### Step E4-1 [2019]
- Parameter-Efficient Transfer Learning for NLP（Adapter Modules）
##### Step E4-2 [2021]
- Prefix-Tuning: Optimizing Continuous Prompts for Generation
##### Step E4-3 [2021/2022]
- The Power of Scale for Parameter-Efficient Prompt Tuning
- P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks
##### Step E4-4 [2021]
- BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models
##### Step E4-5 [2022]
- Few-shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning（IA3）
##### Step E4-6 [2021]
- LoRA: Low-Rank Adaptation of Large Language Models
##### Step E4-7 [2023]
- QLoRA: Efficient Finetuning of Quantized LLMs

#### 范式 E4-2：相关低成本思想旁支
##### Step E4-8 [2019]
- DistilBERT, a distilled version of BERT
##### Step E4-9 [2018]
- The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

---

## F. Transformer 的理论本质是什么？
- 根问题：Transformer 为什么有效，它的本质是什么？

### F1. attention 是否联想记忆？
- 根问题：attention 是否 memory retrieval？

#### 范式 F1-1：联想记忆 / Hopfield 范式
##### Step F1-1 [2020]
- Hopfield Networks is All You Need
##### Step F1-2 [2021]
- Modern Hopfield Networks 相关论文（如：Dense Associative Memory for Pattern Recognition）

### F2. 是否是动力系统？
- 根问题：Transformer 是否可理解为连续深度 / 动力系统？

#### 范式 F2-1：连续深度 / 动力系统范式
##### Step F2-1 [2015]
- Deep Residual Learning for Image Recognition（ResNet）
##### Step B2-2 [2018]
- Neural Ordinary Differential Equations
##### Step F2-3 [2019]
- Augmented Neural ODEs
- FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models
- Latent ODEs for Irregularly-Sampled Time Series
##### Step F2-4 [2019]
- Deep Equilibrium Models
##### Step F2-5 [2021]
- Transformers as Energy-Based Models
- Transformers as Dynamical Systems 相关论文

### F3. 如何理解深层稳定训练？
- 根问题：深层 Transformer 为什么能稳定训练？

#### 范式 F3-1：深层稳定训练理论
##### Step F3-1 [2019]
- Fixup Initialization: Residual Learning Without Normalization
##### Step F3-2 [2020]
- Pre-LN Transformer 相关论文（如：On Layer Normalization in the Transformer Architecture）
##### Step F3-3 [2022]
- DeepNet: Scaling Transformers to 1,000 Layers
##### Step F3-4 [2022]
- μ-Parametrization / Tensor Programs 系列论文