#!/usr/bin/env python3
"""Generate predicted_masks for PatchCore submission.

For each test image:
  1. Extract WideResNet50 layer3 features (14x14 grid)
  2. PCA transform + compute per-patch distance to nearest Coreset neighbor
  3. Reshape to 14x14 heatmap → resize to 448x448 → save as grayscale PNG
"""

import torch, numpy as np, faiss, pickle, os
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

device = "cuda"
bank_dir = "/root/gpufree-data/banks/patchcore_bank"
test_dir = "/root/gpufree-data/datasets/Test_A"
mask_dir = "/root/gpufree-data/results/patchcore/predicted_masks"
os.makedirs(mask_dir, exist_ok=True)

# Load PCA and banks
with open(os.path.join(bank_dir, "pca.pkl"), "rb") as f:
    pca = pickle.load(f)

banks = {}
for f in sorted(os.listdir(bank_dir)):
    if f.endswith("_bank.npy"):
        cls = f.replace("_bank.npy", "")
        banks[cls] = np.load(os.path.join(bank_dir, f))

# Load model
model = models.wide_resnet50_2(
    weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device).eval()
features = {}
model.get_submodule("layer3").register_forward_hook(
    lambda m, i, o: features.__setitem__("l3", o.detach()))

transform = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Collect test samples
test_samples = []
for cls in sorted(os.listdir(test_dir)):
    cls_path = os.path.join(test_dir, cls)
    if not os.path.isdir(cls_path): continue
    for item in sorted(os.listdir(cls_path)):
        item_path = os.path.join(cls_path, item)
        if os.path.isdir(item_path):
            for img in sorted(os.listdir(item_path)):
                if img.endswith(".png"):
                    test_samples.append((os.path.join(item_path, img), cls, item, img.replace(".png", "")))

print(f"Test samples: {len(test_samples)}")

for path, cls, sample, view_idx in tqdm(test_samples, desc="Masks"):
    if cls not in banks:
        continue

    # Output path: predicted_masks/class/sample/view_mask.png
    out_sample_dir = os.path.join(mask_dir, cls, sample)
    os.makedirs(out_sample_dir, exist_ok=True)
    out_path = os.path.join(out_sample_dir, f"{view_idx}_mask.png")

    if os.path.exists(out_path):
        continue

    # Extract features
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    features.clear()
    with torch.no_grad():
        model(t)

    f3 = features["l3"]  # (1, 1024, 14, 14)
    _, c, h, w = f3.shape
    f3_flat = f3.squeeze(0).reshape(c, h*w).T.cpu().numpy()  # (196, 1024)

    # PCA transform
    f3_pca = pca.transform(f3_flat).astype(np.float32)  # (196, 256)

    # Compute per-patch distance to nearest neighbor
    d = f3_pca.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(banks[cls].astype(np.float32))
    distances, _ = index.search(f3_pca, 1)  # (196, 1)

    # Reshape to 14x14 heatmap
    heatmap = distances.reshape(h, w)  # (14, 14)

    # Normalize to [0, 255]
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max() * 255
    heatmap = heatmap.astype(np.uint8)

    # Resize to 448x448
    mask_img = Image.fromarray(heatmap, mode="L")
    mask_img = mask_img.resize((448, 448), Image.BILINEAR)
    mask_img.save(out_path)

print(f"\nDone! Masks saved to {mask_dir}")
