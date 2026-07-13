# 天池工业多视角异常检测 — 实战课程

基于 [天池「AI 大模型竞赛 — 工业多视角异常检测」](https://tianchi.aliyun.com/competition/entrance/532482) 赛道，从零到提交的完整实战课程。

## 课程结构（8 章）

| 章 | 内容 | 掌握能力 |
|:--:|------|---------|
| **Ch0** | 比赛规则、评分、生存指南 | 读懂比赛，知道分数怎么算 |
| **Ch1** | 数据探索 + DINOv2 全局基线 | 加载 Real-IAD 数据集，跑通第一个异常检测 |
| **Ch2** | 多尺度像素热力图 | 从"图有没有缺陷"到"缺陷在哪个像素" |
| **Ch3** | DINOv2 深度解析 | 理解自监督特征提取原理 |
| **Ch4** | INP-Former 完整拆解 | 可学习原型 + 交叉注意力 + 解码器 |
| **Ch5** | 评分逻辑与提交流程 | 6 指标详解 + submission.zip 打包 |
| **Ch6** | 多视角融合策略 | Spearman 相关性实测 + Max/Mean/Weighted 对比 |
| **Ch7** | A 榜实战（待写） | 完整流水线 → 第一次真实提交 |
| **Ch8** | B 榜零样本（待写） | 未见类别泛化策略 |

## 快速导航

- 📖 **课程正文**：[course/](./course/) — 每章 Markdown 源文件
- 🔬 **实验脚本**：[scripts/](./scripts/) — 可复现的实验代码
- 📊 **关键结果**：[results/](./results/) — 实验数据和可视化
- 🌐 **飞书阅读**：[Wiki 链接](https://my.feishu.cn/wiki/FDsUwBnkIiufTQkbEVAcFA62njb)

## 核心技术栈

```
图片 → DINOv2 编码器 → INP-Former 解码器 → 热力图 → 多视角融合 → 提交
       (冻结，提取特征)   (6个INP原型+8层解码器)   (Max策略，ρ=0.14~0.28)
```

- **DINOv2** (Meta, 2023): vit_base 版本，1.42 亿张图片预训练
- **INP-Former** (CVPR 2025): 工业异常检测 SoTA，6 个可学习原型
- **评分公式**: S = 100 × (0.3×S_cls + 0.5×S_seg + 0.2×S_zs)
- **硬件**: RTX 4090 (24GB)，推理单卡 ≤1s

## 环境

- Python 3.10 + PyTorch 2.0.0+cu118
- 数据集: Real-IAD Variety (PR 2026)，50 类，5 视角
- 容器: SSH `root@183.147.142.40:30426`

## 许可证

课程内容 CC-BY 4.0。实验代码 MIT。
