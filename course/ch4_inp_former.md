# Ch4: INP-Former 完整拆解 — 可学习原型与像素级异常检测

## ❓ 灵魂拷问

Ch2 的热力图在插针阵列上一片红——因为余弦距离无法区分"高频纹理"和"真正缺陷"。Ch3 解释了 DINOv2 的特征底层原理。现在的问题是：**如何在 DINOv2 之上加一层可学习的机制，让它知道"插针区域红是正常的"？**

这就是 INP-Former 的核心贡献。它不是在 DINOv2 后面硬接一个分类器，而是在 DINOv2 的中间层插入 6 个可学习的"INP 原型 token"，让模型自己去学"什么程度的偏离算异常，什么程度的偏离只是纹理"。

> 💡 **一句话总结：** Ch2 用固定规则（余弦距离）判断异常；INP-Former 用可学习参数判断异常。前者像尺子——精度固定；后者像经验丰富的质检员——能根据产品类型调整判断标准。

## 📚 前置认知

**你已经有的知识：**
- Ch1：DINOv2 输出 768 维全局特征 → 余弦距离 → 异常分数
- Ch2：DINOv2 多层 patch token → 逐位置余弦距离 → 热力图
- Ch3：DINOv2 自监督训练原理 + register token + 特征空间分布

**本章新增：**
- INP 原型 token 的交叉注意力机制
- 聚合块、瓶颈、解码器三阶段架构
- Gather Loss：INP 原型的训练目标
- 完整模型加载 + 推理 → 生成比 Ch2 更精确的热力图

## 🎯 学习目标

### 📊 INP 原型机制

- 🔴 **学习前：** 用固定的正常均值当"标准答案"，无法适应产品差异
- 🟢 **学习后：** 理解 INP token 通过交叉注意力动态匹配特征——可学习的"容错度"

### 📊 三阶段架构

- 🔴 **学习前：** 以为 INP-Former 就是 DINOv2 + 一个分类头
- 🟢 **学习后：** 理解聚合(INP→特征匹配) → 瓶颈(特征压缩) → 解码(残差展开)的完整流程

### 📊 训练 vs 推理

- 🔴 **学习前：** 不知道 INP-Former 是怎么训练的
- 🟢 **学习后：** 理解 Gather Loss 鼓励 INP token 覆盖所有正常模式

## 📖 主体内容

### 4.1 整体架构：DINOv2 中间插入三层增强

INP-Former 不改变 DINOv2 本身的参数（冻结编码器），而是在其基础上插入三个可训练模块：

```
输入图片 (448×448)
    ↓
DINOv2 编码器（冻结，不训练）
    ↓ 每层输出特征图（8层，target_layers=[2,3,4,5,6,7,8,9]）
    ├──→ ① 聚合块 (Aggregation)
    │      INP token 通过交叉注意力匹配所有特征图
    │      输出：agg_prototype — 6个已经"理解"了这张图的原型
    │
    ├──→ ② 瓶颈 (Bottleneck)
    │      特征压缩：768 → 3072 → 768（FFN）
    │
    └──→ ③ 解码器 (Decoder)
           8个 Prototype_Block，每个用 INP 原型引导特征重建
           重建残差 = 异常信号 → 热力图
```

> 💡 **类比：** DINOv2 像个翻译官，把图片翻译成"特征语言"。INP token 是 6 个"质检标准卡片"——每张卡片上写着一类正常模式（"正常纹理"、"正常反光"、"正常边缘"……）。聚合块让卡片和图片特征对对碰——匹配得上的区域是正常的，匹配不上的区域是异常。解码器把"匹配不上"的信息展开成像素级热力图。

### 4.2 INP 原型 Token：可学习的"正常标准"

INP（Image-Neutral Prototype）是 6 个 768 维的可学习向量。和 Ch2 的固定正常均值不同，它们是通过训练学习出来的。

```python
# INP 原型初始化（8层，每层6个INP token，每个768维）
INP = nn.ParameterList([
    nn.Parameter(torch.randn(6, 768))  # 第2层
    nn.Parameter(torch.randn(6, 768))  # 第3层
    ...  # 共8组，对应target_layers=[2,3,4,5,6,7,8,9]
])
```

**INP 和 Ch2 固定原型的本质区别：**

| | Ch2 固定原型 | INP 可学习原型 |
|---|---|---|
| 怎么来的 | 训练样本的特征均值 | 梯度下降训练出来的 |
| 数量 | 每位置 1 个均值向量 | 每层 6 个独立 token |
| 能否适应 | ❌ 固定不变 | ✅ 通过交叉注意力自适应 |
| 纹理容错 | ❌ 高纹理=高异常 | ✅ INP 学到了"纹理正常" |

**INP 的交叉注意力机制：**

```
交叉注意力（INP token 查询特征图）：
    Q = INP token (6 个, 每个 768 维)         ← 我关心的正常模式
    K = 特征图 token (1024 个 patch, 768 维)   ← 图片的实际特征
    V = 特征图 token                           ← 有价值的特征信息
    
    Attention(Q, K, V) = softmax(Q·K^T / √768) · V
    
    → 每个 INP token 找到特征图中和自己最匹配的区域
    → 融合这些区域的共同特征
    → 6 个 INP token 覆盖了该产品的 6 种"正常模式"
```

> 💡 **为什么是 6 个 INP token？** 6 是论文实验下来的经验值。太少了覆盖不全（比如只有 2 个，纹理+边缘，缺了反光模式），太多了训练困难（INP 之间互相竞争，收敛慢）。

### 4.3 Gather Loss：INP token 的训练目标

INP-Former 的核心损失函数叫 **Gather Loss**。它的目标和直觉相反——不是"让异常样本远离正常"，而是"让 INP token 紧紧包裹住所有正常特征"。

```python
def gather_loss(query, keys):
    # query: 特征图的每个 patch token (1024个)
    # keys:  INP token (6个)
    
    # 计算每个 patch 到最近的 INP token 的距离
    distribution = 1 - cos_sim(query.unsqueeze(2), keys.unsqueeze(1))
    distance, cluster_index = torch.min(distribution, dim=2)
    
    # Loss = 所有 patch 到其最近 INP token 的平均距离
    return distance.mean()
```

> 💡 **类比：** 6 个 INP token 像是 6 个"集散中心"，1024 个 patch token 是 1024 个"包裹"。Gather Loss 的目标是最小化每个包裹到其最近集散中心的距离。训练后，6 个集散中心的位置能最好地覆盖所有正常包裹的分布范围。

**关键洞察：** Gather Loss 只在正常样本上训练。训练完后，INP token "习惯"了正常样本的特征分布。推理时，包含缺陷的测试样本——它的某个 patch 特征可能落在所有 6 个 INP token 的覆盖范围之外 → 距离大 → 异常分数高。

### 4.4 解码器：从 INP 匹配残差到像素级热力图

解码器由 8 个 Prototype_Block 组成。每个 Block 做一件事：用 INP 原型和特征图做交叉注意力，输出"重建"的特征。

```python
for i, blk in enumerate(self.decoder):  # 8次迭代
    x = blk(x, agg_prototype)  # x是特征图, agg_prototype是6个INP token
    de_list.append(x)
```

解码后的特征和原始编码器特征之间的**余弦距离** = 异常热力图。正常区域的特征能被 INP token 很好地"解释"（余弦距离小），异常区域的特征无法被任何 INP token 解释（余弦距离大）。

**和 Ch2 的核心区别：**

- Ch2：`异常 = 1 - cos(测试特征, 训练均值)` — 用固定的正常均值
- INP-Former：`异常 = 1 - cos(编码器特征, 解码器重建特征)` — 用 INP token 引导的"解释残差"

前者对所有区域一视同仁。后者的 INP token 可以学到"高纹理区域匹配不上 INP-3 是正常的，但匹配不上 INP-1 是异常的"——因为有 6 个不同的 INP token，每个代表不同的正常模式。

### 4.5 3类实测：INP-Former 热力图 vs Ch2

加载完整 INP-Former 权重（3,100 万参数），对 3 个类别跑推理（见文末 **图4.1**）：

| 类别 | INP-Former 分数 | Ch2 多尺度 | 差异 |
|------|----------------|-----------|------|
| `3_adapter` | 0.748 ± 0.018 | 0.604 ± 0.081 | +0.144 |
| `DVD_switch` | 0.778 ± 0.021 | 0.667 ± 0.132 | +0.111 |
| `D_sub_connector` | 0.812 ± 0.042 | 0.623 ± 0.111 | +0.189 |

> 🔑 **INP-Former 的分数更高，不是因为效果更差，而是因为解码器把微小的异常信号放大了。** Ch2 用固定均值做全局匹配——很多局部差异被"平均掉"了。INP-Former 的解码器逐层恢复空间细节，输出分辨率更高、对局部差异更敏感。

**标准差大幅缩小**（Ch2: 0.08~0.13 → INP-Former: 0.02~0.04）：INP token 的学习能力让模型对不同样本的判断更一致，不再因为"这一批的拍摄光线不同"就大幅翻分。

### 4.6 模型加载坑位：为什么我们花了两章才加载成功

你已经在 Ch2 中尝试过加载 INP-Former 失败。现在揭晓原因：

| | 代码默认值 | Checkpoint 实际 |
|---|---|---|
| 聚合块数量 | 2 (len(fuse_layer_encoder)) | **1** |
| 解码块数量 | 2 (len(fuse_layer_decoder)) | **8** (len(target_layers)) |

结论：**官方开源的代码中，fuse_layer_encoder/decoder 和训练用的不一致。** checkpoint 是按 `target_layers` 的长度（8 层）来构建解码块的，而不是按 `fuse_layer_decoder` 的组数。这是一个代码版本管理的坑——也是为什么你需要在实战中学会看 checkpoint keys 来反推正确架构。

## 🔬 本章检查清单

- INP 全称？→ Image-Neutral Prototype（与图片无关的"正常标准"）
- 为什么是 6 个 INP token？→ 经验值——6 个能覆盖产品的 6 种主要正常模式（纹理、反光、边缘、阴影等）
- Gather Loss 的目标？→ 让每个 patch 到最近 INP 的距离最小——INP "包裹"住所有正常特征
- INP-Former 和 Ch2 的核心区别？→ 固定余弦 vs 可学习 INP 原型 + 交叉注意力 + 解码器重建
- 为什么 INP-Former 分数更高？→ 解码器放大了局部异常信号，分辨率更高
- 为什么标准差大幅缩小？→ INP token 学习了对光照/纹理的容错
- Checkpoint 加载的坑？→ fuse_layer 不是块数，解码块数 = len(target_layers) = 8

## ❓ 新问题

你现在有一个能输出逐像素异常热力图的完整模型。但比赛评分不只看热力图——它要的是分类分数（30%）+ 分割分数（50%）+ 零样本分数（20%），还要控制阈值、平衡假阳性/假阴性。

光有热力图不够——你需要把它转化为比赛需要的分数格式。

**→ Ch5: 评分逻辑深度拆解 — 从热力图到提交分数**
