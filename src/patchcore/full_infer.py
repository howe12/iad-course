#!/usr/bin/env python3
"""PatchCore full inference on Test_A."""
import torch, numpy as np, faiss, pickle, os, sys, pandas as pd
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

device = "cuda"
bank_dir = "/root/gpufree-data/banks/patchcore_bank"
test_dir = "/root/gpufree-data/datasets/Test_A"
out_dir = "/root/gpufree-data/results/patchcore"
os.makedirs(out_dir, exist_ok=True)

# Load PCA
with open(os.path.join(bank_dir, "pca.pkl"), "rb") as f:
    pca = pickle.load(f)

# Load banks
banks = {}
for f in sorted(os.listdir(bank_dir)):
    if f.endswith("_bank.npy"):
        cls = f.replace("_bank.npy", "")
        banks[cls] = np.load(os.path.join(bank_dir, f))
print(f"Loaded {len(banks)} banks", flush=True)

# Load model
model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device).eval()
features = {}
model.get_submodule("layer3").register_forward_hook(
    lambda m,i,o: features.__setitem__("l3", o.detach()))

transform = transforms.Compose([
    transforms.Resize((256,256)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

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
                    test_samples.append((os.path.join(item_path, img), cls, f"{cls}/{item}/{img}"))
        elif item.endswith(".png"):
            test_samples.append((item_path, cls, f"{cls}/{item}"))

print(f"Test samples: {len(test_samples)}", flush=True)

# Inference
top_k_pct = 0.01
results = []
for path, cls, rel in tqdm(test_samples, desc="Inference"):
    if cls not in banks:
        results.append({"filename": rel, "score": -1, "class": cls})
        continue
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    features.clear()
    with torch.no_grad(): model(t)
    f3 = features["l3"].squeeze(0)
    c,h,w = f3.shape
    f3_pca = pca.transform(f3.reshape(c, h*w).T.cpu().numpy()).astype(np.float32)
    
    d = f3_pca.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(banks[cls].astype(np.float32))
    distances, _ = index.search(f3_pca, 1)
    dists = distances.flatten()
    k = max(1, int(len(dists) * top_k_pct))
    score = float(np.sort(dists)[:k].mean())
    results.append({"filename": rel, "score": score, "class": cls})

df = pd.DataFrame(results)
df.to_csv(os.path.join(out_dir, "test_scores.csv"), index=False)

# Stats
print(f"\nOverall: mean={df['score'].mean():.4f}, std={df['score'].std():.4f}, "
      f"min={df['score'].min():.4f}, max={df['score'].max():.4f}", flush=True)

per_class = df.groupby("class")["score"].agg(["mean","std","count"])
per_class = per_class.sort_values("mean", ascending=False)
print("\nTop 10 highest-mean classes:")
print(per_class.head(10).to_string())

print("\nDone!", flush=True)

