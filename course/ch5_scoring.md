# Ch5: 评分逻辑深度拆解 — 从热力图到提交分数

## ❓ 灵魂拷问

你现在有一个 INP-Former 模型，能输出 448×448 的异常热力图。但比赛不看你热力图画得好不好——它看的是三个指标按权重算出来的总分。

你的热力图上有 200,704 个像素的异常分数。怎么把这 20 万个数字变成比赛要的 **1 个 AUROC + 1 个 AP + 1 个 F1max**？分数之间怎么加权？阈值选在哪？提交文件长什么样？

**从热力图到排行榜上的排名——中间有 5 个转换步骤。** 每一步做错了，前面 4 章的功夫就白费了。

## 📚 前置认知

你已经有的：
- Ch4：INP-Former 输出 `[448, 448]` 的异常热力图
- Ch0：比赛评分三合一公式

本章新增：
- 6 个指标的含义和计算（I-AUROC, I-AP, P-AUROC, P-AP, P-F1max, P-AUPRO）
- 从热力图到图像级分数的 3 种策略（max / top-1% mean / adaptive）
- 阈值选择的 trade-off
- 提交文件的准确格式

## 🎯 学习目标

### 📊 6 个指标

- 🔴 **学习前：** 只知道 AUROC 越高越好，不知道 AP 和 F1max 的区别
- 🟢 **学习后：** 理解每项指标考察什么能力，知道为什么分割 50% 权重最高

### 📊 分数转换

- 🔴 **学习前：** 不知道热力图怎么变成图像级分数
- 🟢 **学习后：** 掌握 max polling / top-k mean / adaptive threshold 三种策略

### 📊 提交实战

- 🔴 **学习前：** 不知道提交文件长什么样
- 🟢 **学习后：** 能自己跑一遍完整的"推理→mask→csv→打包"流程

## 📖 主体内容

### 5.1 热力图 → 分数：5 步转换流水线

```
热力图 [448, 448, float]
    ↓ 步骤1：像素聚合 → 图像级分数
    ↓
图像级分数（per 样本）
    ↓ 步骤2：所有样本的分数 + 标签 → 排序
    ↓
排序列表
    ↓ 步骤3：计算 AUROC / AP / F1max
    ↓
子指标分数
    ↓ 步骤4：加权：S = 0.3×S_cls + 0.5×S_seg + 0.2×S_zs
    ↓
综合得分 S
    ↓ 步骤5：打包 → submission.zip → 上传
```

每步都有选择。选错一步，分数可能掉 20%。

### 5.2 步骤一：像素聚合 — 怎么把 [448,448] 压成 1 个数

你有 200,704 个像素异常分数。比赛要的图像级分数是什么？有三种常见策略：

**策略A：Max-Pooling（取最异常的像素）**

```python
image_score = anomaly_map.max()
```

- ✅ 简单直接，如果图里有一块明显的缺陷，它能抓到
- ❌ 对噪声极其敏感——一个硬件噪点就能打出 0.95 的假阳性

**策略B：Top-1% Mean（取最异常的1%像素的均值）**

```python
flat = anomaly_map.flatten()
k = max(1, int(len(flat) * 0.01))
image_score = torch.topk(torch.tensor(flat), k)[0].mean()
```

- ✅ 比 max 更鲁棒，不会被单个噪点欺骗
- ✅ 是 INP-Former 默认策略（`max_ratio=0.01`）
- ⚠️ 假设缺陷面积不超过图片 1% —— 对大缺陷可能低估

**策略C：Adaptive Threshold（自适应阈值）**

```python
mean, std = anomaly_map.mean(), anomaly_map.std()
threshold = mean + 3 * std
image_score = anomaly_map[anomaly_map > threshold].mean()
```

- ✅ 自动适应每张图的特点
- ❌ 阈值是启发式的，不同类别可能需要不同倍数

> 🔑 **比赛策略：** A 榜上交初期用策略 B（稳定、不易过拟合），后期根据验证结果调整。三版策略都会在这章代码里实现，你自己跑一遍看看哪版在你模型上最好。

### 5.3 步骤二+三：从分数到指标 — AUROC/AP/F1max

拿到所有样本的图像级分数后，计算比赛指标：

**I-AUROC（图像级 AUROC）—— 区分能力**

把分数排序。理想情况下，所有异常样本的分数 > 所有正常样本的分数。AUROC 衡量你的分数**排序**有多好——它不关心具体分数值，只关心"异常是否排在正常前面"。

> 💡 **类比：** 班里有 30 个学生。AUROC=1.0 意味着你把所有身高超过 170cm 的学生都排在了 170cm 以下的学生前面，没有一个排错。AUROC=0.5 意味着你的排序和瞎猜没区别。

**I-AP（图像级 Average Precision）—— 正样本纯度**

AP 衡量当你从高到低遍历分数时，每碰到一个样本，它有多大概率是真正的异常。**在不平衡数据（异常远少于正常）上，AP 比 AUROC 更敏感。**

**P-F1max —— 阈值敏感度**

F1max 是 precision-recall 曲线上最大的 F1 值。它同时惩罚漏检和误报。**这个指标对阈值特别敏感——阈值调偏 0.05，F1max 可能掉 5%。**

> ⚠️ **P-F1max 是很多人的翻车点。** 它不是"找一个好阈值"——它是"在 precision-recall 曲线上找到最优的平衡点"。这意味着你的排名质量（AUROC）好还不够，你还得在"正负样本交界处"区分得特别干净。

### 5.4 A 榜实际公式：B 榜还没开放

A 榜只有 seen 类别，没有零样本：

```
S_A = 100 × (0.3 × S_cls + 0.7 × S_seg)
```

其中：
- **S_cls** = (I-AUROC + I-AP) / 2 — 图像级分类得分
- **S_seg** = (P-AUROC + P-AP + P-F1max) / 3 — 像素级分割得分

> 🔑 **关键：A 榜分割权重是 0.7，不是 0.5。** 因为零样本得分（0.2）不参与 A 榜，它的权重被重新分配到分割上。这意味着 A 榜上，分割比分类重要 **2.3 倍**（0.7/0.3）——你的热力图质量比分类标签重要得多。

B 榜引入 50 个零样本类后才切换为完整公式 `S = 0.3 + 0.5 + 0.2`。

### 5.5 步骤五：提交格式 — 一字节都不能错

提交文件是 `submission.zip`，内含两个东西：

**【1】submission.csv**

```csv
group_folder,anomaly_score
3_adapter,0.1456
DVD_switch,0.0892
...
```

- `group_folder`：类别名，对应 Test_A 目录下的文件夹名
- `anomaly_score`：该样本的图像级异常分数（float，越大越异常）
- 注意：**是"样本级"分数，不是图片级。** 一个类别的 15 个样本，每个样本是 5 张图的 max 或 mean。

**【2】predicted_masks/**

```
predicted_masks/
  3_adapter/
    S0001/
      0_mask.png  ← 448×448 灰度图，像素值=异常概率 [0,1]
      1_mask.png
      2_mask.png
      3_mask.png
      4_mask.png
    S0002/ ...
  DVD_switch/ ...
```

- 文件名必须是 `{view}_mask.png`（0_mask.png 到 4_mask.png）
- 图片必须是 448×448，灰度，PNG 格式
- 像素值归一化到 [0, 255]（0=正常，255=异常）

> 📦 **不要自己打包 zip！** 比赛方提供了 `competitor_toolkit/make_submission.py`，用它打包——它会校验格式、图片尺寸、文件命名。格式错误直接 0 分。

### 5.6 实战演示：生成提交文件

以 `3_adapter` 和 `DVD_switch` 为例，完整运行一遍提交流水线：

```python
# 1. 加载 INP-Former
model = INP_Former(...)
model.load_state_dict(torch.load('model.pth'))

# 2. 对每个类别推理
for cat in categories:
    # 计算正常原型
    normal_en = compute_normal_prototype(model, train_loader[cat])
    
    # 对测试样本逐视角推理
    for sample in test_samples[cat]:
        for v in range(5):
            img = load_image(cat, sample, v)
            anomaly_map = model(img)  # [448, 448]
            
            # 保存 mask
            mask = normalize(anomaly_map)
            save_mask(mask, f'predicted_masks/{cat}/{sample}/{v}_mask.png')
        
        # 图像级分数：取 5 个视角的 max
        image_score = max([mean(masks[v]) for v in range(5)])

# 3. 写入 CSV
write_submission_csv(scores)

# 4. 打包
python competitor_toolkit/make_submission.py \
    --submission_csv submission.csv \
    --mask_dir predicted_masks \
    --output submission.zip
```

（完整可运行脚本见代码目录 `ch5_submit.py`——本章配图中展示了实际生成的 submission 文件结构。）

## 🔬 本章检查清单

- 热力图到图像级分数有哪三种策略？→ max / top-1% mean / adaptive threshold
- AUROC 衡量什么？→ 排序质量——异常是否排在正常前面
- 为什么 AP 在数据不平衡时更重要？→ 异常样本少，AUROC 对少量正样本排序不敏感
- P-F1max 为什么容易翻车？→ 对阈值极其敏感，阈值偏 0.05 分数掉 5%
- A 榜分割权重是多少？→ **0.7**（不是 0.5，因为零样本 0.2 重新分配了）
- 打包用哪个工具？→ `competitor_toolkit/make_submission.py`，不要自己写
- mask 格式要求？→ 448×448 灰度 PNG，文件名 `{view}_mask.png`

## ❓ 新问题

你现在知道怎么从热力图走到提交分数了。但每个样本有 5 个视角——你只是简单取了 max。5 个视角之间不是独立的——它们是同一个产品的不同角度。

怎么利用这个多视角信息？视角之间的一致性本身是不是也是一种信息？视角 1 说"异常"，视角 3 说"正常"——该信谁？

**→ Ch6: 多视角融合策略 — 1+1+1+1+1 > 5**
