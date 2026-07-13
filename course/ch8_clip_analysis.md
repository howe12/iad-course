# CLIP & WinCLIP 零样本异常检测实验分析

## 实验动机

排行榜最高分 99.82（东北大学）使用了 CLIP/VLM 路线，验证了视觉-语言模型在工业异常检测中的潜力。我们尝试复现这条路线。

## 实验 1：Naive CLIP Prompting

**方法**：用 CLIP ViT-L/14，对每张图提问 "a damaged {class_name}" vs "a perfect {class_name}"。

**结果**：所有 50 类分数集中在 0.5 附近，标准差接近 0。完全无区分度。

**根因**：CLIP 预训练数据（4 亿图文对）中几乎没有工业缺陷图片。CLIP 知道 "capacitor" 长什么样，但分不清 "鼓包的电容" 和 "完好的电容"。

| 指标 | INP-Former | CLIP Naive |
|------|:--:|:--:|
| 分数范围 | 0.24 ~ 0.56 | 0.496 ~ 0.513 |
| 区分度 | ✅ 有 | ❌ 无 |

## 实验 2：WinCLIP v1（全窗口版）

**方法**：基于 WinCLIP 论文（CVPR 2023），实现：
- 组合 prompt ensemble（8 状态词 × 5 模板 = 40 prompts/类）
- 三尺度滑窗（全图 + 3×3 grid + 5×5 grid = 35 窗口/图）
- 调和平均聚合

**结果**：3 类测试即超时。35 窗口/图 × 每窗口独立 CLIP forward = 过慢（~20 分钟/类）。

## 实验 3：WinCLIP v2（高效版）

**方法**：
- 全图 attention map（ViT 最后一层 patch features → CLIP 共享空间 → text prototype 内积）
- 4 象限窗口评分（裁剪 4 个 224×224 象限独立推理）
- Max-pooling over prompts（避免 fp16 平均溢出）

**结果**：

| 类别 | μ | σ | 范围 |
|------|:--:|:--:|------|
| D_sub_connector | 0.507 | 0.001 | 0.505-0.510 |
| 3_adapter | 0.509 | 0.000 | 0.509-0.509 |
| DVD_switch | 0.511 | 0.003 | 0.504-0.516 |

**结论：仍无区分度，所有分数 ≈ 0.5。**

### 根因分析

1. **语言-视觉不对齐**：CLIP 的 patch-level 特征未经 language supervision 训练，论文也确认了这点（Table 8）
2. **Attention map 无效**：ViT self-attention 聚合了全局上下文，局部细节被稀释
3. **Prompt 不匹配**：工业缺陷的视觉模式（微小划痕、色差、形变）不在 CLIP 的语义空间中

## 为什么排行榜有人能做到 99 分？

可能的技术路线（非单纯 zero-shot CLIP）：

1. **WinCLIP+**：用训练集正常样本做 visual reference，计算测试图和最近正常图的差异
2. **AnomalyCLIP / AdaCLIP**：在 CLIP 基础上加轻量 adapter，用少量数据训练
3. **DINOv2 + CLIP 融合**：DINOv2 提取局部特征 + CLIP 做语义引导
4. **Prompt learning**：可学习的 prompt token（不是固定文本）

## 下一步

1. **INP-Former v2**：提交 mask clip 修复版，验证分数（预期 50-55）
2. **WinCLIP+**：利用训练集的 20 张正常样本做 visual reference —— 这是 WinCLIP 论文的核心得分来源
3. **DINOv2 + 其他方法**：探索更强的 backbone 或 adapter 方案

## 相关文件

- `scripts/winclip_pipeline.py` — WinCLIP v2 管道（全图 attention + 4 象限窗口）
- `scripts/clip_pipeline.py` — Naive CLIP 管道
- `results/winclip_v2/` — v2 3 类测试结果（容器 `/root/gpufree-data/results/winclip_v2/`）

## 参考资料

- WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation (Jeong et al., CVPR 2023)
- Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)
