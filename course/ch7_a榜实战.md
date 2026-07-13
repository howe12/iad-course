# Ch7: A 榜实战 — 从零到提交的完整流水线

## ❓ 灵魂拷问

前 6 章你分别掌握了：数据加载（Ch1）、热力图生成（Ch2）、DINOv2 特征提取（Ch3）、INP-Former 推理（Ch4）、评分转换（Ch5）、多视角融合（Ch6）。

但至今你还没提交过一次。每个模块都验证过能跑，但它们之间是**手动拼接**的——在容器上跑一段脚本，看结果，复制粘贴到下一步。

本章做一件事：**把 Ch1~Ch6 的所有模块串成一条自动化流水线，一键跑完 50 个类别，生成 submission.zip。**

## 📚 前置认知

### Ch6 留下了什么

- 多视角 Max 融合是最优策略（比 Mean 高 ~12%）
- 高斯平滑 σ=4.0、top-0.1% 分数提取——这两个参数是前 5 章验证过的
- 50 类的视角相关性在 0.14~0.28——视角近乎独立，Max 天然占优

本章不再重复验证这些结论，直接使用。

### 提交格式（关键）

比赛要求提交一个 `submission.zip`，包含：

```
submission.zip
  ├── submission.csv              ← 每个样本的异常分数
  │   格式: group_folder,anomaly_score
  │   示例: D_sub_connector/S0001,0.3254
  │
  └── predicted_masks/            ← 每个样本 5 个视角的异常掩码
       └── {category}/
            └── {sample_id}/
                 ├── 0_mask.png   ← 448×448 灰度图
                 ├── 1_mask.png
                 ├── 2_mask.png
                 ├── 3_mask.png
                 └── 4_mask.png
```

50 个类别（见文末 **表 7.1**），每类 15 个测试样本，每个样本 5 张图——总计 **3,750 次推理**。

### 本章执行计划

```
[容器上] 一键运行 ch7_pipeline.py
  → 加载 INP-Former（Ch4 的模型）
  → 遍历 50 类 × 15 样本 × 5 视角
  → 生成 anomaly maps → Max 融合分数
  → 保存 masks/ + scores.csv
  → 调用 make_submission.py 打包 zip
```

## 🎯 学习目标

- 🔴 **学习前：** 每个模块独立跑过，但不清楚怎么串起来
- 🟢 **学习后：** 掌握从原始图片到 submission.zip 的完整自动化流程

- 🔴 **学习前：** 不知道提交长什么样
- 🟢 **学习后：** 能生成符合比赛规范的 zip，并在本地验证格式

## 📖 主体内容

### 7.1 流水线设计：五步走

回顾 Ch1 的竞赛全景图，本章实现了最右侧的"提交"环节：

```
图片 → DINOv2 → INP-Former → 热力图 → Max融合 → 分数
                                                   ↓
                                          submission.csv
                                                   ↓
                                          masks/*.png
                                                   ↓
                                          submission.zip
```

核心脚本 `ch7_pipeline.py` 的五段结构：

```python
# 段 1：加载模型（Ch4 验证过的结构）
encoder = vit_encoder.load("dinov2reg_vit_base_14")
# 1 aggregation + 8 decoder blocks
model = INP_Former(encoder, Bottleneck, Agg, Dec, ...)
model.load_state_dict(torch.load(ckpt_path), strict=True)

# 段 2：推理函数（Ch2+Ch4+Ch5 的融合）
@torch.no_grad()
def infer(img_path):
    img = load_and_preprocess(img_path)
    en, de = model(img)[:2]                     # Ch4：INP-Former forward
    amap, _ = cal_anomaly_maps(en, de, 448)     # Ch2：余弦距离 → 热力图
    amap = gaussian_filter(amap, sigma=4.0)      # Ch2：高斯平滑
    score = top_k_mean(amap, k=0.1%)             # Ch5：分数提取
    return score, amap

# 段 3：遍历全量数据
for category in all_50_categories:
    for sample in category.samples:
        for view in [0,1,2,3,4]:
            score, amap = infer(view_img)
        final_score = max(view_scores)           # Ch6：Max 融合
        save_masks(view_maps)                    # 保存为 448×448 PNG
        save_score(category/sample, final_score)

# 段 4：打包提交
subprocess.run([
    "python3", "make_submission.py",
    "--scores-csv", "scores.csv",
    "--mask-root", "masks/",
    "--zip", "submission.zip"
])
```

### 7.2 模型加载：和 Ch6 实验完全一致

加载代码与 Ch6 的多视角实验一模一样——因为 Ch6 验证了这个构造方式能正确加载官方权重。

```
关键参数（与 Ch4 训练脚本对齐）：
  encoder:          dinov2reg_vit_base_14 (768维, 12头)
  target_layers:    [2,3,4,5,6,7,8,9]
  fuse_layer_enc:   [[0,1,2,3], [4,5,6,7]]
  fuse_layer_dec:   [[0,1,2,3], [4,5,6,7]]
  INP tokens:       6 个
  aggregation:      1 个 Aggregation_Block
  decoder:          8 个 Prototype_Block
```

> 🔑 **这个结构不是猜的。** 之前 Ch4 踩过坑——官方代码默认 `fuse_layer_num=2`（4+2+2 结构），但实际 checkpoint 是 1+8。通过 `torch.load` 检查 keys 才确认了正确结构。

### 7.3 核心推理循环：每一步都在干什么

每张图进入 `infer()` 后经过 5 步变换：

| 步骤 | 输入 | 输出 | 来源 |
|:--:|------|------|:--:|
| 1 | 448×448 RGB 图片 | (1,3,448,448) tensor | torchvision transforms |
| 2 | tensor | DINOv2 多尺度特征 | Ch3：冻结编码器 forward |
| 3 | encoder features + decoder features | 448×448 热力图 | Ch2：余弦距离 + 多尺度平均 |
| 4 | 热力图 | 高斯平滑热力图 | σ=4.0（Ch2/Ch5 验证） |
| 5 | 平滑热力图 | anomaly_score | top-0.1% 像素均值（Ch5） |

```python
# 步骤 2-5 逐行拆解
en, de = model(tensor)[:2]                    # 步骤2：encoder+decoder特征
amap, _ = cal_anomaly_maps(en, de, 448)       # 步骤3：多尺度余弦→(448,448)
amap = gaussian_filter(amap, sigma=4.0)        # 步骤4：高斯平滑
flat = amap.flatten()
K = max(1, int(len(flat) * 0.001))            # 步骤5：top-0.1%
score = float(flat[np.argpartition(flat, -K)[-K:]].mean())
```

> 💡 **为什么是 top-0.1% 而不是 max？** Ch2 解释过——单像素 max 对噪声太敏感。一张 448×448 图有 200,704 个像素，取最异常的 ~200 个像素求均值，既保留了异常信号又降低了随机噪声的影响。这个阈值是 INP-Former 论文的默认配置，Ch5 在本地验证过对 AUROC 的影响。

### 7.4 掩码保存：预测值，不是二值化结果

掩码保存的细节很多人搞错：

**❌ 错误做法：** 设一个阈值（如 0.5），把热力图二值化再保存
**✅ 正确做法：** 保存原始异常值（0~1 连续值），让比赛平台自己算 AUROC

```python
# 保存掩码——保留连续值，不二值化
m = np.clip(amap, 0, 1)           # 余弦距离范围 [0,2]，clip到[0,1]
mask_uint8 = (m * 255).astype(np.uint8)
Image.fromarray(mask_uint8).save(f"{vid}_mask.png")
```

> 🔑 **为什么不能二值化？** AUROC 需要在所有可能的阈值下评估。你自作主张设一个阈值，等于只提交了一个点（一个 TPR/FPR 对）而不是整条 ROC 曲线。比赛平台会拿你的连续值掩码和 ground truth 逐像素比较——保留连续值才能让它在所有阈值下算出真正的 AUROC。

### 7.5 全量运行结果

在 RTX 4090 上跑完 50 类：

| 指标 | 数值 |
|------|------|
| 类别数 | 50 |
| 总推理次数 | 3,750（50×15×5） |
| 总耗时 | ~10 分钟 |
| 平均分数 | 0.327 |
| 分数范围 | 0.24 ~ 0.56 |
| submission.zip | 100 MB |

**分数最高的 5 个类别：**

| 类别 | 均值分数 | 标准差 |
|------|:------:|:------:|
| battery | 0.559 | 0.080 |
| lithium_battery_plug | 0.411 | 0.013 |
| button_battery_holder | 0.380 | 0.014 |
| PLCC_socket | 0.376 | 0.012 |
| ingot_buckle | 0.372 | 0.032 |

**分数最低的 5 个类别：**

| 类别 | 均值分数 | 标准差 |
|------|:------:|:------:|
| smd_receiver_module | 0.278 | 0.025 |
| power_jack | 0.283 | 0.009 |
| lego_pin_connector_plate | 0.287 | 0.022 |
| motor_plug | 0.267 | 0.012 |
| purple_clay_pot | 0.240 | 0.039 |

（见文末 **图 7.1**：50 类分数排名柱状图）
（见文末 **图 7.2**：分数分布直方图）

### 7.6 分数解读：为什么 0.33 不是 0.5？

你可能觉得异常分数应该在 0.5 附近波动——但这是对异常检测分数的常见误解。AUROC 不在乎绝对值的大小，只在乎**排序一致性**：

> 如果一个类别的缺陷样本分数是 0.35，正常样本分数是 0.30，AUROC=1.0。
> 即使分数都在 0.3 附近，只要异常>正常，排序就是完美的。

0.33 的均值并不低。三个值得关注的信号：

1. **battery 明显偏高（0.56 ± 0.08）**：要么这个类别真的有明显缺陷，要么模型对它比较敏感。标准差 0.08 是所有类中最大的——暗示 battery 类内部的异常程度差异很大。

2. **38/50 类集中在 0.30~0.40**：紧凑的分布说明 INP-Former 对大多数类别的判断是稳定的。

3. **purple_clay_pot 垫底（0.24）**：可能是"最难"的类别——正常和异常的特征差异小。B 榜需要重点关注这类。

### 7.7 下一步：提交

你手上现在有一个 `submission.zip`（100 MB），可以去天池平台提交了。

但还有一个选择——**先在本地验证一下格式**，确保不会被平台拒掉：

```bash
# 把 submission.zip 和 standard_A.zip 放一起
python3 metrics_local.py \
    --standard-zip standard_A.zip \
    --submission-zip submission.zip \
    --out local_metrics.json
```

（`standard_A.zip` 包含 A 榜 ground truth——需要从比赛平台下载，这里不提供。）

这个脚本会算出你的本地 AUROC/AP/F1——跟平台评分一致。先本地看一眼分数，再决定是否提交。

> ⚠️ **A 榜每天只有 3 次提交机会**。不要浪费在格式错误上——先用 `metrics_local.py` 本地跑一遍。

## 🔬 本章检查清单

- 流水线跑几个类？→ **50 个类全部**，3,750 次推理
- 总分均值？→ 0.327（50 类平均）
- 分数最高的类？→ battery（0.559）
- 分数最低的类？→ purple_clay_pot（0.240）
- 融合策略用哪个？→ Max（Ch6 验证最优）
- 掩码用二值化还是连续值？→ **连续值**（比赛要整条 ROC 曲线）
- submission.zip 多大？→ 100 MB
- 提交前先做什么？→ 用 `metrics_local.py` 本地验证格式

## ❓ 新问题

你第一次跑通了 50 类的全自动流水线，拿到了 0.327 的均值分数。但你自己也看到了：battery 类 0.56，purple_clay_pot 类 0.24——不同类别的异常分数差异这么大，说明"一刀切"的推理策略对某些类不友好。

A 榜用到的 50 个类都是 INP-Former 训练时见过的。B 榜的 50 个新类呢？模型从没见过它们——没有可学习的原型，没有正常纹理的知识。

**→ Ch8: B 榜零样本 — 如何让模型判断从未见过的零件是否异常？**

---

## 📷 本章配图

**图 7.1：50 类 Anomaly Score 排名** — 红=高分(>0.4)，蓝=中等(0.3~0.4)，灰=低分(<0.3)。红色虚线=总体均值 0.327。

**图 7.2：分数分布直方图** — 38/50 类集中在 0.30~0.40 区间，2 类超过 0.4，10 类低于 0.3。

## 📋 50 类列表

3_adapter, DVD_switch, D_sub_connector, PLCC_socket, VR_joystick, accurate_detection_switch, battery, blade_switch, boost_converter_module, button_battery_holder, circuit_breaker, connector_housing_female, crimp_st_cable_mount_box, dc_jack, dc_power_connector, detection_switch, effect_transistor, electronic_watch_movement, ffc_connector_plug, ingot_buckle, laser_diode, lego_pin_connector_plate, limit_switch, lithium_battery_plug, littel_fuse, lock, miniature_lifting_motor, mobile_charging_connector, motor_bracket, motor_gear_reducer, motor_plug, pencil_sharpener, pinboard_connector, potentiometer, power_jack, power_strip_socket, purple_clay_pot, retaining_ring, rheostat, self_lock_switch, silicon_cell_sensor, single_switch, smd_receiver_module, suction_cup, toy_tire, travel_switch, vacuum_switch, vehicle_harness_conductor, vibration_motor, wireless_receiver_module
