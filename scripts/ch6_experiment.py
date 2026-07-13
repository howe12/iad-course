#!/usr/bin/env python3
"""
Ch6 实验：多视角一致性分析 + 融合策略对比
基于官方 INP-Former Super Multi-Class 推理流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, sys, os
from pathlib import Path
from functools import partial
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ── 添加代码路径 ──
sys.path.insert(0, "/root/gpufree-data/code")
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import cal_anomaly_maps

# ── 配置 ──
RESULT_DIR = Path("/root/gpufree-data/results/ch6_multiview")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CLASSES = ["D_sub_connector", "3_adapter", "DVD_switch", "capacitor_elec", "resistor"]
DATA_DIR = Path("/root/gpufree-data/datasets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ── 构造模型（复用训练脚本的逻辑）──
embed_dim = 768
num_heads = 12
target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

# 编码器
encoder_name = "dinov2reg_vit_base_14"
encoder = vit_encoder.load(encoder_name)

# Bottleneck + INP + Aggregation + Decoder（1 agg + 8 decoder）
Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
INP = nn.ParameterList([nn.Parameter(torch.randn(6, embed_dim))])
INP_Extractor = nn.ModuleList([Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))])
INP_Guided_Decoder = nn.ModuleList([Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8)) for _ in range(8)])

model = INP_Former(
    encoder=encoder, bottleneck=Bottleneck,
    aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
    target_layers=target_layers, remove_class_token=True,
    fuse_layer_encoder=fuse_layer_encoder,
    fuse_layer_decoder=fuse_layer_decoder,
    prototype_token=INP
)

# 加载 checkpoint
ckpt_path = "/root/gpufree-data/code/saved_results/INP-Former-Super-Multi-Class_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt, strict=True)
model = model.to(device)
model.eval()
print(f"✅ 模型加载成功")

# ── 图像预处理 ──
img_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
gaussian_kernel = gaussian_filter  # 用 scipy 的高斯核

@torch.no_grad()
def infer_view(img_path):
    """对单张图片推理，返回 (anomaly_score, anomaly_map)"""
    img = Image.open(img_path).convert("RGB")
    img_tensor = img_transform(img).unsqueeze(0).to(device)
    
    output = model(img_tensor)
    en, de = output[0], output[1]
    
    # 计算异常图 → (1,1,H,W)
    anomaly_map, _ = cal_anomaly_maps(en, de, out_size=448)
    anomaly_map = anomaly_map.cpu().numpy()[0, 0]  # (448, 448)
    
    # 高斯平滑
    anomaly_map = gaussian_filter(anomaly_map, sigma=4.0)
    
    # 图像级分数：top-0.1% 均值（对应 max_ratio=0.001）
    flat = anomaly_map.flatten()
    K = max(1, int(len(flat) * 0.001))
    topk_idx = np.argpartition(flat, -K)[-K:]
    score = float(flat[topk_idx].mean())
    
    return score, anomaly_map


# ── 步骤 1：逐视角推理 ──
print("=" * 60)
print("步骤 1：逐视角 INP-Former 推理")
print("=" * 60)

all_results = {}

for cls_name in CLASSES:
    test_dir = DATA_DIR / "Test_A" / cls_name
    if not test_dir.exists():
        print(f"  ⚠️ 跳过 {cls_name}（不存在）")
        continue
    
    # 数据结构: Test_A/{cls}/{S0001}/0.png ... 4.png
    sample_dirs = sorted(test_dir.glob("S*"))
    
    all_results[cls_name] = {}
    print(f"\n📦 {cls_name}: {len(sample_dirs)} 样本")
    
    for sidx, sample_dir in enumerate(sample_dirs[:15]):
        sample_id = sample_dir.name  # e.g. "S0001"
        view_scores = {}
        for vid in range(5):
            img_path = sample_dir / f"{vid}.png"
            if not img_path.exists():
                continue
            try:
                score, _ = infer_view(str(img_path))
                view_scores[vid] = score
            except Exception as e:
                print(f"    ⚠️ {img_path}: {e}")
        
        if len(view_scores) == 5:
            all_results[cls_name][sample_id] = view_scores
            print(f"  [{sidx+1}/{min(15,len(sample_dirs))}] {sample_id}: " + 
                  " ".join(f"V{v}={view_scores[v]:.4f}" for v in range(5)))

# 保存
with open(RESULT_DIR / "raw_view_scores.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ 原始分数保存: {RESULT_DIR / 'raw_view_scores.json'}")

# ── 步骤 2：视角一致性分析 ──
print("\n" + "=" * 60)
print("步骤 2：视角一致性分析")
print("=" * 60)

for cls_name, samples in all_results.items():
    if len(samples) < 3:
        continue
    
    score_matrix = np.array([[s[v] for v in range(5)] for s in samples.values()])
    view_means = score_matrix.mean(axis=0)
    print(f"\n📊 {cls_name} ({len(samples)} 样本):")
    for v in range(5):
        print(f"  V{v} 均值: {view_means[v]:.4f}")
    
    # Spearman 相关性
    corr_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i == j:
                corr_matrix[i][j] = 1.0
            else:
                rho, _ = spearmanr(score_matrix[:, i], score_matrix[:, j])
                corr_matrix[i][j] = rho
    
    triu_idx = np.triu_indices(5, k=1)
    avg_corr = corr_matrix[triu_idx].mean()
    view_stds = score_matrix.std(axis=1)
    
    print(f"  平均相关系数: {avg_corr:.4f}")
    print(f"  视角分歧度(std): 均值={view_stds.mean():.4f} 最大={view_stds.max():.4f} 最小={view_stds.min():.4f}")
    
    np.savez(RESULT_DIR / f"{cls_name}_analysis.npz",
             score_matrix=score_matrix, view_means=view_means,
             corr_matrix=corr_matrix, avg_corr=avg_corr, view_stds=view_stds)

# ── 步骤 3：融合策略对比 ──
print("\n" + "=" * 60)
print("步骤 3：融合策略对比")
print("=" * 60)

FUSION_RESULTS = {}

for cls_name, samples in all_results.items():
    if len(samples) < 3:
        continue
    
    score_matrix = np.array([[s[v] for v in range(5)] for s in samples.values()])
    
    max_fusion = score_matrix.max(axis=1)
    mean_fusion = score_matrix.mean(axis=1)
    
    # Weighted: 各视角分数 × 自身 softmax 权重
    exp_scores = np.exp(score_matrix * 2)
    weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    weighted_fusion = (score_matrix * weights).sum(axis=1)
    
    # Conservative: Max × 一致性惩罚（分歧大的降权）
    view_ranges = score_matrix.max(axis=1) - score_matrix.min(axis=1)
    penalty = np.where(view_ranges > np.median(view_ranges), 0.85, 1.0)
    conservative_fusion = max_fusion * penalty
    
    strategies = {
        "Max": max_fusion,
        "Mean": mean_fusion,
        "Weighted": weighted_fusion,
        "Conservative": conservative_fusion
    }
    
    print(f"\n📊 {cls_name}:")
    print(f"  {'策略':<16} {'均值':>8} {'中位数':>8} {'标准差':>8} {'最小':>8} {'最大':>8}")
    print(f"  {'-'*56}")
    for name, scores in strategies.items():
        print(f"  {name:<16} {scores.mean():>8.4f} {np.median(scores):>8.4f} "
              f"{scores.std():>8.4f} {scores.min():>8.4f} {scores.max():>8.4f}")
    
    FUSION_RESULTS[cls_name] = {
        name: {"mean": float(s.mean()), "median": float(np.median(s)),
               "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}
        for name, s in strategies.items()
    }

with open(RESULT_DIR / "fusion_comparison.json", "w") as f:
    json.dump(FUSION_RESULTS, f, indent=2)

# ── 步骤 4：可视化 ──
print("\n" + "=" * 60)
print("步骤 4：可视化")
print("=" * 60)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

if all_results:
    # 4a. 相关性热力图
    first_cls = list(all_results.keys())[0]
    data = np.load(RESULT_DIR / f"{first_cls}_analysis.npz")
    corr = data['corr_matrix']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap='RdYlBu_r', vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f'View {i}' for i in range(5)], fontsize=11)
    ax.set_yticklabels([f'View {i}' for i in range(5)], fontsize=11)
    ax.set_title(f'View Correlation Matrix\n{first_cls} (Spearman ρ)', fontsize=13, fontweight='bold')
    for i in range(5):
        for j in range(5):
            color = 'white' if abs(corr[i,j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr[i,j]:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85).set_label('Spearman ρ', fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "view_correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {RESULT_DIR / 'view_correlation_heatmap.png'}")

    # 4b. 融合策略对比柱状图
    n_cls = len(FUSION_RESULTS)
    fig, axes = plt.subplots(1, n_cls, figsize=(5*n_cls, 5), squeeze=False)
    axes = axes[0]
    strategy_names = list(next(iter(FUSION_RESULTS.values())).keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for ax_idx, (cls_name, strats) in enumerate(FUSION_RESULTS.items()):
        ax = axes[ax_idx]
        means = [strats[n]['mean'] for n in strategy_names]
        stds = [strats[n]['std'] for n in strategy_names]
        bars = ax.bar(strategy_names, means, color=colors, edgecolor='white', linewidth=0.8)
        ax.errorbar(range(len(strategy_names)), means, yerr=stds, fmt='none', ecolor='gray', capsize=4)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_title(cls_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Anomaly Score', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='x', rotation=15, labelsize=8)
    
    plt.suptitle('Fusion Strategy Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "fusion_strategy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {RESULT_DIR / 'fusion_strategy_comparison.png'}")

    # 4c. 视角分歧直方图
    fig, axes = plt.subplots(1, n_cls, figsize=(5*n_cls, 4), squeeze=False)
    axes = axes[0]
    for ax_idx, (cls_name, samples) in enumerate(all_results.items()):
        if cls_name not in FUSION_RESULTS: continue
        ax = axes[ax_idx]
        score_matrix = np.array([[s[v] for v in range(5)] for s in samples.values()])
        view_ranges = score_matrix.max(axis=1) - score_matrix.min(axis=1)
        ax.hist(view_ranges, bins=12, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(view_ranges.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'μ={view_ranges.mean():.3f}')
        ax.set_xlabel('View Range (max−min)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(f'{cls_name}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
    plt.suptitle('View Disagreement Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "view_disagreement_hist.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {RESULT_DIR / 'view_disagreement_hist.png'}")

print("\n" + "=" * 60)
print("🎉 全部实验完成！")
print(f"结果目录: {RESULT_DIR}")
print("=" * 60)
