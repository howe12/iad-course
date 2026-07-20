#!/usr/bin/env python3
"""
build_dinov2_bank.py — 用 DINOv2 ViT-B/14_reg 构建 per-class 多尺度特征 Bank

与 CLIP v7 的关键差异:
- 模型: DINOv2 ViT-B/14_reg (torch.hub) vs CLIP ViT-L/14
- 层数: 12 层 → hook [3, 6, 9, 11]
- 特征维度: 768 vs 1024
- Patch 分辨率: 32×32 (448px输入) vs 14×14 (224px)
- Register tokens: 去掉前 5 个 token (CLS + 4 register) → 留 1024 patches

用法:
    python build_dinov2_bank.py
    # 产出: clip_feature_bank_dinov2_multiscale/  (50个.pkl, 与v7同目录结构)
"""

import os, sys, pickle, glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
DATA_ROOT = "/root/gpufree-data/Real-IAD_Variety/train"
BANK_DIR = "/root/gpufree-data/clip_feature_bank_dinov2_multiscale"
LAYERS = [3, 6, 9, 11]  # DINOv2 共 12 层，取 4 个关键层
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(BANK_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Layers: {LAYERS}")

# ============================================================
# Load DINOv2
# ============================================================
print("Loading DINOv2 ViT-B/14_reg from torch.hub...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
model.eval().to(DEVICE)
transform = model.transform  # DINOv2 自带预处理

# ============================================================
# Register hooks
# ============================================================
hooked_features = {}

def make_hook(name):
    def hook_fn(module, input, output):
        hooked_features[name] = output.detach()
    return hook_fn

for layer in LAYERS:
    model.blocks[layer].register_forward_hook(make_hook(f"layer_{layer}"))

print(f"Hooks registered on layers {LAYERS}")

# ============================================================
# Build per-class bank
# ============================================================
classes = sorted(os.listdir(DATA_ROOT))
print(f"\nTotal classes: {len(classes)}")

for cls_idx, cls_name in enumerate(classes):
    cls_path = os.path.join(DATA_ROOT, cls_name)
    if not os.path.isdir(cls_path):
        continue

    bank = {layer: [] for layer in LAYERS}

    samples = sorted([d for d in os.listdir(cls_path)
                      if os.path.isdir(os.path.join(cls_path, d))])

    print(f"\n[{cls_idx+1}/{len(classes)}] {cls_name} ({len(samples)} samples)")

    for sample_dir in tqdm(samples, desc=cls_name):
        sample_path = os.path.join(cls_path, sample_dir)
        views = sorted(glob.glob(os.path.join(sample_path, "*.png")))

        for view_path in views:
            # Load & preprocess
            img = Image.open(view_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, 448, 448)

            # Forward
            hooked_features.clear()
            with torch.no_grad():
                _ = model.forward_features(img_tensor)

            # Extract patches for each layer
            for layer in LAYERS:
                feat = hooked_features[f"layer_{layer}"]  # (1, 1029, 768)
                # DINOv2_reg token order: [CLS, reg1, reg2, reg3, reg4, patch_1, ..., patch_1024]
                patches = feat[0, 5:, :]  # strip CLS + 4 register tokens → (1024, 768)
                patch_map = patches.reshape(32, 32, 768)  # → (32, 32, 768)
                # L2 normalize each patch (余弦距离 = 1 - L2归一化向量的内积)
                patch_map_norm = torch.nn.functional.normalize(
                    patch_map.reshape(-1, 768), dim=-1
                ).reshape(32, 32, 768)

                bank[layer].append(patch_map_norm.cpu().numpy())

    # Stack and save
    for layer in LAYERS:
        bank[layer] = np.stack(bank[layer], axis=0)  # (N_views, 32, 32, 768)

    pkl_path = os.path.join(BANK_DIR, f"{cls_name}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(bank, f)

    print(f"  Saved: {pkl_path}")
    for layer in LAYERS:
        print(f"    layer_{layer}: {bank[layer].shape}")

print(f"\n✅ Bank built: {BANK_DIR}/")
print(f"   Size: {sum(os.path.getsize(os.path.join(BANK_DIR, f)) for f in os.listdir(BANK_DIR)) / 1024 / 1024:.1f} MB")
