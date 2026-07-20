#!/usr/bin/env python3
"""
retrieval_inference_dinov2.py — DINOv2 多尺度 Patch 检索推理

用法:
    python retrieval_inference_dinov2.py
    
产出: results/perclass_retrieval_dinov2/submission.zip
"""

import os, sys, pickle, csv, zipfile, glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
BANK_DIR = "/root/gpufree-data/clip_feature_bank_dinov2_multiscale"
TEST_DIR = "/root/gpufree-data/Real-IAD_Variety/Test_A"
OUTPUT_DIR = "/root/gpufree-data/results/perclass_retrieval_dinov2"
LAYERS = [3, 6, 9, 11]
K_PATCH = 3              # 近邻数量
GF_RADIUS = 5            # Guided Filter 半径
TOP_N_PERCENT = 0.10     # top-N% 像素取均值作为分数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "anomaly_masks"), exist_ok=True)

# ============================================================
# Load DINOv2
# ============================================================
print("=" * 60)
print("[1/5] Loading DINOv2 ViT-B/14_reg...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
model.eval().to(DEVICE)
transform = model.transform

# Register hooks
hooked_features = {}
def make_hook(name):
    def hook_fn(module, input, output):
        hooked_features[name] = output.detach()
    return hook_fn

for layer in LAYERS:
    model.blocks[layer].register_forward_hook(make_hook(f"layer_{layer}"))

# ============================================================
# Load all banks
# ============================================================
print("[2/5] Loading feature banks...")
banks = {}
for pkl_path in sorted(glob.glob(os.path.join(BANK_DIR, "*.pkl"))):
    cls_name = os.path.splitext(os.path.basename(pkl_path))[0]
    with open(pkl_path, 'rb') as f:
        banks[cls_name] = pickle.load(f)
    shapes = {l: banks[cls_name][l].shape for l in LAYERS}
    print(f"  {cls_name}: {shapes}")

print(f"  Total classes in bank: {len(banks)}")

# ============================================================
# Scan test images
# ============================================================
print("[3/5] Scanning test images...")
test_images = []
classes = sorted(os.listdir(TEST_DIR))
for cls_name in classes:
    cls_path = os.path.join(TEST_DIR, cls_name)
    if not os.path.isdir(cls_path):
        continue
    for sample_dir in sorted(os.listdir(cls_path)):
        sample_path = os.path.join(cls_path, sample_dir)
        if not os.path.isdir(sample_path):
            continue
        views = sorted(glob.glob(os.path.join(sample_path, "*.png")))
        for view_path in views:
            test_images.append({
                'path': view_path,
                'class': cls_name,
                'sample': sample_dir,
                'view': os.path.splitext(os.path.basename(view_path))[0],
                'group': f"{cls_name}/{sample_dir}"
            })

print(f"  Found {len(test_images)} test images")

# ============================================================
# Guided Filter
# ============================================================
def guided_filter(guide, src, r=5, eps=1e-3):
    """用 guide(原图灰度)引导 src(热力图)的平滑"""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    mean_I = cv2.boxFilter(guide, -1, (r, r))
    mean_p = cv2.boxFilter(src, -1, (r, r))
    cov_Ip = cv2.boxFilter(guide * src, -1, (r, r)) - mean_I * mean_p
    var_I = cv2.boxFilter(guide * guide, -1, (r, r)) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    return mean_a * guide + mean_b

# ============================================================
# Patch retrieval
# ============================================================
def anomaly_map_cosine(patches_flat, bank_patches, k=3):
    """
    patches_flat: (N, 768)  测试图 patch 特征 (L2 已归一化)
    bank_patches: (M, 768)  Bank 中正常 patch 特征 (L2 已归一化)
    返回: (N,) 每个 patch 的异常分数
    """
    distances = 1 - (patches_flat @ bank_patches.T)  # (N, M)
    top_k = torch.topk(distances, k=k, dim=1, largest=False).values  # (N, k)
    return top_k.mean(dim=1)  # (N,)

# ============================================================
# Main inference loop
# ============================================================
print("[4/5] Running DINOv2 multi-scale inference...")
scores = {}  # group_folder → score
img_count = 0

for img_info in tqdm(test_images, desc="Inference"):
    cls_name = img_info['class']
    group = img_info['group']

    if cls_name not in banks:
        print(f"  WARNING: {cls_name} not in bank, skipping")
        continue

    # Load image
    img_pil = Image.open(img_info['path']).convert("RGB")
    img_np = np.array(img_pil)  # (448, 448, 3)
    H, W = img_np.shape[:2]
    guide_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    # Forward
    hooked_features.clear()
    with torch.no_grad():
        _ = model.forward_features(img_tensor)

    # Multi-scale heatmaps
    heatmaps = []
    for layer in LAYERS:
        feat = hooked_features[f"layer_{layer}"]  # (1, 1029, 768)
        patches = feat[0, 5:, :]                   # (1024, 768) — strip CLS + 4 register

        # Normalize
        Q = F.normalize(patches, dim=-1)            # (1024, 768)

        # Load bank for this class+layer (L2 already normalized during build)
        bank_tensor = torch.from_numpy(
            banks[cls_name][layer].reshape(-1, 768)  # (M, 768)
        ).to(DEVICE)

        # Retrieval
        score_per_patch = anomaly_map_cosine(Q, bank_tensor, k=K_PATCH)
        hmap = score_per_patch.reshape(32, 32).cpu().numpy()  # (32, 32)

        # Upsample to original size
        hmap_up = cv2.resize(hmap, (W, H), interpolation=cv2.INTER_LINEAR)
        heatmaps.append(hmap_up)

    # Max fusion + GF
    final_heatmap = np.maximum.reduce(heatmaps)  # (H, W)
    final_heatmap_gf = guided_filter(guide_gray, final_heatmap, r=GF_RADIUS)

    # Score: top-N% mean
    flat = np.sort(final_heatmap_gf.flatten())[::-1]
    score = flat[:int(TOP_N_PERCENT * len(flat))].mean()
    scores[group] = float(score)

    # Save mask
    mask_out = os.path.join(OUTPUT_DIR, "anomaly_masks",
                            cls_name, img_info['sample'],
                            f"{img_info['view']}.png")
    os.makedirs(os.path.dirname(mask_out), exist_ok=True)
    mask_uint8 = np.clip(final_heatmap_gf * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(mask_out, mask_uint8)

    img_count += 1

print(f"  Processed {img_count} images")

# ============================================================
# Build submission.zip
# ============================================================
print("[5/5] Building submission.zip...")

# Write CSV
csv_path = os.path.join(OUTPUT_DIR, "submission.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["group_folder", "anomaly_score"])
    for group in sorted(scores.keys()):
        writer.writerow([group, f"{scores[group]:.6f}"])

# Summary stats
score_vals = np.array(list(scores.values()))
print(f"  Scores: mean={score_vals.mean():.4f}, std={score_vals.std():.4f}, "
      f"min={score_vals.min():.4f}, max={score_vals.max():.4f}")

# Zip
zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(csv_path, "submission.csv")
    mask_root = os.path.join(OUTPUT_DIR, "anomaly_masks")
    for root, dirs, files in os.walk(mask_root):
        for fn in files:
            full = os.path.join(root, fn)
            arcname = "anomaly_masks/" + os.path.relpath(full, mask_root)
            zf.write(full, arcname)

zip_size = os.path.getsize(zip_path) / 1024 / 1024
print(f"\n✅ Done! {zip_path} ({zip_size:.1f} MB)")
print(f"   CSV rows: {len(scores)}")
print(f"   Mean: {score_vals.mean():.4f}  Std: {score_vals.std():.4f}")
print("\n📋 提交前检查:")
print(f"   1. head {csv_path}")
print(f"   2. unzip -l {zip_path} | head")
print(f"   3. 确认 mask 非空:")
first_mask_dir = os.path.join(OUTPUT_DIR, "anomaly_masks")
cls_list = os.listdir(first_mask_dir)
if cls_list:
    s_list = os.listdir(os.path.join(first_mask_dir, cls_list[0]))
    if s_list:
        v_list = os.listdir(os.path.join(first_mask_dir, cls_list[0], s_list[0]))
        if v_list:
            check_path = os.path.join(first_mask_dir, cls_list[0], s_list[0], v_list[0])
            print(f"   python -c \"from PIL import Image; import numpy as np; m=np.array(Image.open('{check_path}')); print('max=', m.max())\"")
