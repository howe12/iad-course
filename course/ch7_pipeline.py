#!/usr/bin/env python3
"""
Ch7 A榜统一推理流水线
50类 × 15样本 × 5视角 = 3750次推理 → submission.zip

输出结构:
  /root/gpufree-data/results/ch7_submit/
    scores.csv                              ← group_folder,anomaly_score
    masks/{category}/{sample_id}/0_mask.png ← 5视角异常掩码
    submission.zip                          ← 打包提交文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os, json, csv, time
from pathlib import Path
from functools import partial
from scipy.ndimage import gaussian_filter
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/root/gpufree-data/code")
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import cal_anomaly_maps

# ── 配置 ──
DATA_DIR = Path("/root/gpufree-data/datasets/Test_A")
RESULT_DIR = Path("/root/gpufree-data/results/ch7_submit")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
(RESULT_DIR / "masks").mkdir(exist_ok=True)

# 前6章验证过的最优配置
SIGMA = 4.0          # 高斯平滑
TOP_RATIO = 0.001    # top-0.1% 像素取均值
FUSION = "max"       # Max融合(Ch6验证最优)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ── 构造模型 ──
embed_dim, num_heads = 768, 12
target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
fuse_enc = [[0, 1, 2, 3], [4, 5, 6, 7]]
fuse_dec = [[0, 1, 2, 3], [4, 5, 6, 7]]

encoder = vit_encoder.load("dinov2reg_vit_base_14")
Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim)])
INP = nn.ParameterList([nn.Parameter(torch.randn(6, embed_dim))])
Agg = nn.ModuleList([Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))])
Dec = nn.ModuleList([Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8)) for _ in range(8)])

model = INP_Former(encoder, Bottleneck, Agg, Dec,
                   target_layers=target_layers, remove_class_token=True,
                   fuse_layer_encoder=fuse_enc, fuse_layer_decoder=fuse_dec,
                   prototype_token=INP)

ckpt = "/root/gpufree-data/code/saved_results/INP-Former-Super-Multi-Class_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"
model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
model = model.to(device)
model.eval()
print(f"✅ 模型加载完成 ({Path(ckpt).stat().st_size / 1e6:.0f} MB)")

img_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def infer(img_path):
    """推理单张图片 → (anomaly_score, anomaly_map_448x448)"""
    img = Image.open(img_path).convert("RGB")
    tensor = img_transform(img).unsqueeze(0).to(device)
    en, de = model(tensor)[:2]
    amap, _ = cal_anomaly_maps(en, de, out_size=448)
    amap = amap.cpu().numpy()[0, 0]
    amap = gaussian_filter(amap, sigma=SIGMA)
    # Top-K 分数（不做归一化，直接用原始异常值）
    flat = amap.flatten()
    K = max(1, int(len(flat) * TOP_RATIO))
    score = float(flat[np.argpartition(flat, -K)[-K:]].mean())
    return score, amap


# ── 主循环 ──
# 支持命令行参数: python3 ch7_pipeline.py --test (只跑3类)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="测试模式：只跑3个类别")
args_test = parser.parse_args()

if args_test.test:
    all_categories = ["D_sub_connector", "3_adapter", "DVD_switch"]
    print(f"🧪 测试模式: {all_categories}")
else:
    all_categories = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())

print(f"\n📦 共 {len(all_categories)} 个类别")
print(f"每类 15 样本 × 5 视角 = 75 次推理/类")
print(f"总计: {len(all_categories) * 75} 次推理\n")

scores_rows = []
cls_stats = {}
start_time = time.time()

for cat_idx, category in enumerate(all_categories):
    cat_dir = DATA_DIR / category
    samples = sorted(d.name for d in cat_dir.iterdir() if d.is_dir())
    
    cat_scores = []
    for sample_id in samples:
        view_scores = {}
        view_maps = {}
        
        for vid in range(5):
            img_path = cat_dir / sample_id / f"{vid}.png"
            if not img_path.exists():
                continue
            score, amap = infer(str(img_path))
            view_scores[vid] = score
            view_maps[vid] = amap
        
        if len(view_scores) < 5:
            continue
        
        # ── 融合 ──
        if FUSION == "max":
            final_score = max(view_scores.values())
        elif FUSION == "mean":
            final_score = np.mean(list(view_scores.values()))
        else:
            final_score = max(view_scores.values())
        
        cat_scores.append(final_score)
        
        # ── 保存 masks（保留完整余弦距离范围）──
        mask_dir = RESULT_DIR / "masks" / category / sample_id
        mask_dir.mkdir(parents=True, exist_ok=True)
        for vid in range(5):
            m = view_maps[vid]
            m = np.clip(m, 0, 2)  # 余弦距离理论范围 [0, 2]，不压缩到 [0, 1]
            mask_uint8 = (m * 127.5).astype(np.uint8)  # [0,2] → [0,255]
            Image.fromarray(mask_uint8).save(mask_dir / f"{vid}_mask.png")
        
        # ── 记录分数 ──
        group = f"{category}/{sample_id}"
        scores_rows.append({"group_folder": group, "anomaly_score": round(final_score, 6)})
    
    if cat_scores:
        cls_stats[category] = {
            "n": len(cat_scores),
            "mean": float(np.mean(cat_scores)),
            "std": float(np.std(cat_scores)),
            "min": float(np.min(cat_scores)),
            "max": float(np.max(cat_scores))
        }
    
    elapsed = time.time() - start_time
    eta = elapsed / (cat_idx + 1) * (len(all_categories) - cat_idx - 1)
    print(f"[{cat_idx+1:2d}/{len(all_categories)}] {category:<35s} "
          f"μ={np.mean(cat_scores):.4f} σ={np.std(cat_scores):.4f} "
          f"({elapsed/60:.0f}m elapsed, ~{eta/60:.0f}m remaining)")

# ── 保存 scores.csv ──
csv_path = RESULT_DIR / "scores.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["group_folder", "anomaly_score"])
    writer.writeheader()
    writer.writerows(scores_rows)
print(f"\n✅ scores.csv: {len(scores_rows)} 行")

# ── 类别统计 ──
with open(RESULT_DIR / "class_stats.json", "w") as f:
    json.dump(cls_stats, f, indent=2)

# Top/Bottom 5 类
sorted_cls = sorted(cls_stats.items(), key=lambda x: x[1]["mean"], reverse=True)
print("\n📊 Top-5 异常分数最高类:")
for name, s in sorted_cls[:5]:
    print(f"  {name:<40s} μ={s['mean']:.4f} σ={s['std']:.4f}")
print("📊 Bottom-5 异常分数最低类:")
for name, s in sorted_cls[-5:]:
    print(f"  {name:<40s} μ={s['mean']:.4f} σ={s['std']:.4f}")

# ── 打包 submission.zip ──
print("\n📦 打包 submission.zip ...")
import subprocess
subprocess.run([
    "python3", "/root/gpufree-data/code/competitor_toolkit/make_submission.py",
    "--scores-csv", str(csv_path),
    "--mask-root", str(RESULT_DIR / "masks"),
    "--out-dir", str(RESULT_DIR / "submission"),
    "--zip", str(RESULT_DIR / "submission.zip")
], check=True)

zip_size = (RESULT_DIR / "submission.zip").stat().st_size / 1e6
print(f"\n✅ submission.zip ({zip_size:.1f} MB)")

total_time = time.time() - start_time
print(f"\n⏱ 总耗时: {total_time/60:.1f} 分钟")
print(f"📁 结果目录: {RESULT_DIR}")
