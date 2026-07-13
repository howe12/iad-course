#!/usr/bin/env python3
"""
WinCLIP v2 — 高效版
====================
基于已验证的 clip_pipeline.py，加入多尺度窗口评分。

核心改进：
  1. Full-image: CLIP encode + attention map → base anomaly map
  2. 4 Quads: 将 448×448 切成 4 个 224×224 象限，每象限独立评分
  3. Composite: 全图分数 + 各象限最高分 → 融合
  4. 总推理量: 5 forward/image（vs v1 的 35），速度快 7x
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys, os, csv, time, json
from pathlib import Path
from PIL import Image
import clip
from scipy.ndimage import gaussian_filter
import cv2

# ── Config ──
DATA_DIR = Path("/root/gpufree-data/datasets/Test_A")
RESULT_DIR = Path("/root/gpufree-data/results/winclip_v2")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
(RESULT_DIR / "masks").mkdir(exist_ok=True)

CLIP_MODEL = "ViT-L/14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# ── Prompts ──
NORMAL_STATES = [
    "perfect", "flawless", "good", "normal",
    "pristine", "intact", "undamaged", "unblemished",
]
DEFECT_STATES = [
    "damaged", "defective", "anomalous", "broken",
    "flawed", "imperfect", "faulty", "irregular",
]
TEMPLATES = [
    "a {} photo of a {}",
    "a {} {} in the image",
    "a photo of a {} for quality inspection",
    "a {} close-up of a {}",
    "a {} product photo of a {}",
]

GAUSSIAN_SIGMA = 4.0
TOP_K_FRAC = 0.001


def cat_to_name(cat: str) -> str:
    return cat.replace("_", " ")


@torch.no_grad()
def build_prototypes(cat_name: str, model, device):
    """Build prompt ensemble — keeps all individual prompt embeddings (max-pool at inference time)."""
    normal_texts, defect_texts = [], []
    for tmpl in TEMPLATES:
        for s in NORMAL_STATES:
            normal_texts.append(tmpl.format(s, cat_name))
        for s in DEFECT_STATES:
            defect_texts.append(tmpl.format(s, cat_name))

    nf = F.normalize(model.encode_text(clip.tokenize(normal_texts, truncate=True).to(device)), dim=-1)
    af = F.normalize(model.encode_text(clip.tokenize(defect_texts, truncate=True).to(device)), dim=-1)
    return nf, af  # (N_prompts, 768) — use max-pool at inference time


def get_quads(img_rgb: np.ndarray) -> list:
    """
    Split 448×448 image into 4 overlapping 224×224 quadrants.
    Each quad is resized to 224 for CLIP.
    Returns list of (crop_224x224, (x, y, w, h) in original coords).
    """
    h, w = img_rgb.shape[:2]
    half = 224
    quads = []

    # 4 corners + center hints
    positions = [
        (0, 0), (0, w - half),           # top
        (h - half, 0), (h - half, w - half),  # bottom
    ]
    for x, y in positions:
        crop = img_rgb[y:y + half, x:x + half]
        quads.append((crop, (x, y, half, half)))

    return quads


@torch.no_grad()
def get_attention_mask(img_tensor, model, anomaly_feat, normal_feat, device):
    """
    Extract CLIP ViT attention-based anomaly map.
    Uses final-layer patch features projected to CLIP space,
    compared against text prototypes.

    Returns: anomaly_map (16×16), image_score
    """
    x = model.visual.conv1(img_tensor.type(model.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x = torch.cat([
        model.visual.class_embedding.to(x.dtype) +
        torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype), x
    ], dim=1)
    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)

    for block in model.visual.transformer.resblocks:
        x = block(x)

    x = model.visual.ln_post(x)

    # CLS token → image-level score
    cls_feat = x[:, 0, :]
    cls_feat = cls_feat @ model.visual.proj
    cls_feat = F.normalize(cls_feat, dim=-1)

    sim_a = (cls_feat @ anomaly_feat.T).max(-1).values  # max over all anomaly prompts
    sim_n = (cls_feat @ normal_feat.T).max(-1).values
    image_score = float((sim_a / (sim_n + sim_a + 1e-8)).cpu())

    # Patch features → anomaly map
    patch_feats = x[:, 1:, :]  # (1, N_patches, dim)
    patch_feats = patch_feats @ model.visual.proj
    patch_feats = F.normalize(patch_feats, dim=-1)

    pa = (patch_feats @ anomaly_feat.T).max(-1).values  # (1, N)
    pn = (patch_feats @ normal_feat.T).max(-1).values
    amap = pa / (pn + pa + 1e-8)  # (1, N)

    side = int(np.sqrt(amap.shape[1]))
    amap = amap.reshape(side, side).cpu().numpy()
    amap = np.array(Image.fromarray((amap * 255).astype(np.uint8)).resize((448, 448), Image.BILINEAR)) / 255.0

    return amap, image_score


@torch.no_grad()
def score_crops(crops, normal_feat, anomaly_feat, model, preprocess, device):
    """Batch-score image crops via CLIP."""
    scores = []
    for i in range(0, len(crops), BATCH_SIZE):
        batch = crops[i:i + BATCH_SIZE]
        tensors = torch.stack([preprocess(Image.fromarray(c)) for c in batch]).to(device)
        feats = model.encode_image(tensors)
        feats = F.normalize(feats, dim=-1)
        sim_n = (feats @ normal_feat.T).max(-1).values
        sim_a = (feats @ anomaly_feat.T).max(-1).values
        s = sim_a / (sim_n + sim_a + 1e-8)
        scores.append(s.cpu().numpy())
    return np.concatenate(scores)


@torch.no_grad()
def infer_winclip(img_path, normal_feat, anomaly_feat, model, preprocess, device):
    """
    WinCLIP-lite: full-image attention map + quadrant window scoring.
    Returns (image_score, pixel_mask_448x448).
    """
    img = cv2.imread(img_path)
    if img is None:
        return 0.5, np.zeros((448, 448), dtype=np.float32)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── 1. Full-image attention map ──
    img_tensor = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    attn_map, cls_score = get_attention_mask(img_tensor, model, anomaly_feat, normal_feat, device)

    # ── 2. Quadrant window scoring ──
    quads = get_quads(img_rgb)
    q_scores = score_crops([q[0] for q in quads], normal_feat, anomaly_feat, model, preprocess, device)

    # Build quadrant map: each quadrant score assigned to its region
    quad_map = np.zeros((448, 448), dtype=np.float32)
    for (crop, (x, y, w, h)), qs in zip(quads, q_scores):
        quad_map[y:y + h, x:x + w] = np.maximum(quad_map[y:y + h, x:x + w], qs)

    # ── 3. Fuse: max of attention map and quadrant map (pessimistic — flag any anomaly) ──
    anomaly_map = np.maximum(attn_map, quad_map)

    # Gaussian smoothing
    anomaly_map = gaussian_filter(anomaly_map, sigma=GAUSSIAN_SIGMA)

    # Image-level: top-k% mean
    flat = anomaly_map.flatten()
    k = max(1, int(len(flat) * TOP_K_FRAC))
    image_score = float(np.mean(np.sort(flat)[-k:]))

    return image_score, anomaly_map


def save_mask(amap, path):
    if amap.shape[:2] != (448, 448):
        amap = cv2.resize(amap, (448, 448), interpolation=cv2.INTER_LINEAR)
    m = amap.copy()
    if m.max() > 0:
        m = m / m.max()
    m = np.clip(m * 255, 0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, m)


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true")
ap.add_argument("--classes", nargs="*", default=None)
a = ap.parse_args()

print(f"设备: {DEVICE}, 模型: {CLIP_MODEL}")
print(f"方法: Full-image attention + 4-Quad window scoring")
print(f"Prompt: {len(NORMAL_STATES)}×{len(TEMPLATES)} + {len(DEFECT_STATES)}×{len(TEMPLATES)} = {len(NORMAL_STATES)*len(TEMPLATES)+len(DEFECT_STATES)*len(TEMPLATES)} prompts\n")

model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)
model.eval()

if a.test and a.classes:
    all_cats = a.classes
elif a.test:
    all_cats = ["D_sub_connector", "3_adapter", "DVD_switch"]
    print(f"🧪 测试: {all_cats}")
else:
    all_cats = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())

print(f"📦 共 {len(all_cats)} 个类别\n")

scores_rows = []
cls_stats = {}
start_time = time.time()

for cat_idx, category in enumerate(all_cats):
    cat_dir = DATA_DIR / category
    samples = sorted(d.name for d in cat_dir.iterdir() if d.is_dir())
    cat_name = cat_to_name(category)

    nf, af = build_prototypes(cat_name, model, DEVICE)

    cat_scores = []
    t0 = time.time()
    for si, sample_id in enumerate(samples):
        vs, vms = [], []
        for vid in range(5):
            img_path = cat_dir / sample_id / f"{vid}.png"
            if not img_path.exists():
                continue
            score, mask = infer_winclip(str(img_path), nf, af, model, preprocess, DEVICE)
            vs.append(score)
            vms.append(mask)

        if len(vs) < 5:
            continue

        final_score = float(np.max(vs))
        cat_scores.append(final_score)

        mask_dir = RESULT_DIR / "masks" / category / sample_id
        for vid in range(5):
            save_mask(vms[vid], str(mask_dir / f"{vid}.png"))

        scores_rows.append({
            "group_folder": f"{category}/{sample_id}",
            "anomaly_score": round(final_score, 6),
        })

        # Progress every 3 samples
        if (si + 1) % 3 == 0:
            dt = time.time() - t0
            sps = dt / (si + 1)
            eta = sps * (len(samples) - si - 1)
            print(f"  [{category}] {si+1}/{len(samples)} samples, "
                  f"{sps:.1f}s/sample, eta {eta:.0f}s")

    if cat_scores:
        cls_stats[category] = {
            "n": len(cat_scores), "mean": float(np.mean(cat_scores)),
            "std": float(np.std(cat_scores)), "min": float(np.min(cat_scores)),
            "max": float(np.max(cat_scores)),
        }

    elapsed = time.time() - start_time
    eta = elapsed / (cat_idx + 1) * (len(all_cats) - cat_idx - 1) if cat_idx else 0
    mu_str = f"μ={np.mean(cat_scores):.4f}" if cat_scores else "N/A"
    print(f"[{cat_idx+1:2d}/{len(all_cats)}] {category:<35s} {mu_str:>12s}  "
          f"⏱{elapsed/60:.0f}m eta~{eta/60:.0f}m\n")

# ── Save ──
csv_path = RESULT_DIR / "scores.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["group_folder", "anomaly_score"])
    w.writeheader()
    w.writerows(scores_rows)

with open(RESULT_DIR / "class_stats.json", "w") as f:
    json.dump(cls_stats, f, indent=2)

# Summary
print("\n" + "=" * 60)
print("Score Distribution")
print("=" * 60)
sorted_cls = sorted(cls_stats.items(), key=lambda x: x[1]["mean"])
for name, s in sorted_cls:
    print(f"  {name:<35s} μ={s['mean']:.4f} σ={s['std']:.4f}  [{s['min']:.4f}, {s['max']:.4f}]")

all_scores = [r["anomaly_score"] for r in scores_rows]
print(f"\n  Overall: {len(all_scores):,} images, μ={np.mean(all_scores):.4f}, "
      f"σ={np.std(all_scores):.4f}")

# Make submission.zip
import subprocess
subprocess.run([
    "/root/gpufree-data/miniconda3/envs/comp/bin/python", "/root/gpufree-data/code/competitor_toolkit/make_submission.py",
    "--scores-csv", str(csv_path), "--mask-root", str(RESULT_DIR / "masks"),
    "--out-dir", str(RESULT_DIR / "submission"),
    "--zip", str(RESULT_DIR / "submission.zip"),
], check=True)

elapsed = time.time() - start_time
zip_size = (RESULT_DIR / "submission.zip").stat().st_size / 1e6
print(f"\n✅ submission.zip ({zip_size:.0f} MB), ⏱ {elapsed/60:.1f} min")
print(f"   {RESULT_DIR}/submission.zip")
