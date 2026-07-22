
import torch, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os, pickle, numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.decomposition import PCA
import faiss

device = 'cuda'
data_root = '/root/gpufree-data/datasets/train'
bank_dir = '/root/gpufree-data/banks/patchcore_bank'
os.makedirs(bank_dir, exist_ok=True)

# Load model
model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device).eval()

# Hook for layer3 only
features = {}
model.get_submodule('layer3').register_forward_hook(
    lambda m,i,o: features.__setitem__('layer3', o.detach()))

transform = transforms.Compose([
    transforms.Resize((256,256)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# Collect samples
samples = defaultdict(list)
for cls in sorted(os.listdir(data_root)):
    cls_path = os.path.join(data_root, cls)
    if not os.path.isdir(cls_path): continue
    for item in os.listdir(cls_path):
        item_path = os.path.join(cls_path, item)
        if os.path.isdir(item_path):
            for img in os.listdir(item_path):
                if img.endswith('.png'):
                    samples[cls].append(os.path.join(item_path, img))
        elif item.endswith('.png'):
            samples[cls].append(item_path)

print(f"Classes: {len(samples)}, Total images: {sum(len(v) for v in samples.values())}")

# Phase 1: Extract layer3 features for all classes (save as float16 to save space)
print("\n=== Phase 1: Feature Extraction (layer3 only, float16) ===")
all_feats = {}
total_patches = 0
for cls_name, paths in tqdm(sorted(samples.items()), desc='Classes'):
    cls_feats = []
    for path in tqdm(paths, desc=cls_name, leave=False):
        img = Image.open(path).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        features.clear()
        with torch.no_grad():
            model(t)
        f3 = features['layer3'].squeeze(0)  # (C,H,W)
        c,h,w = f3.shape
        f3 = f3.reshape(c, h*w).T.cpu().half()  # (N,C) float16
        cls_feats.append(f3)
    all_feats[cls_name] = torch.cat(cls_feats, dim=0).float()  # back to float32 for PCA
    total_patches += all_feats[cls_name].shape[0]
    print(f"  {cls_name}: {all_feats[cls_name].shape[0]} patches")

print(f"\nTotal patches: {total_patches}")

# Phase 2: Fit PCA on sampled subset
print("\n=== Phase 2: PCA ===")
sample_size = min(100000, total_patches)
all_concat = np.concatenate([v.numpy() for v in all_feats.values()], axis=0)
indices = np.random.choice(len(all_concat), sample_size, replace=False)
pca = PCA(n_components=256, random_state=42)
pca.fit(all_concat[indices])
print(f"PCA explained: {pca.explained_variance_ratio_.sum():.3f}")

# Save PCA
with open(os.path.join(bank_dir, 'pca.pkl'), 'wb') as f:
    pickle.dump(pca, f)

# Phase 3: Per-class Coreset
print("\n=== Phase 3: Coreset ===")
coreset_size = 10000
for cls_name in tqdm(sorted(all_feats.keys()), desc='Coreset'):
    f3 = all_feats[cls_name].numpy()
    f3_pca = pca.transform(f3).astype(np.float32)
    n = f3_pca.shape[0]
    k = min(coreset_size, n)

    if n <= k:
        coreset = f3_pca
    else:
        kmeans = faiss.Kmeans(d=256, k=k, niter=25, verbose=False, seed=42)
        kmeans.train(f3_pca)
        _, idx = kmeans.index.search(f3_pca, 1)
        idx = np.unique(idx.flatten())
        if len(idx) < k:
            remaining = np.setdiff1d(np.arange(n), idx)
            extra = np.random.choice(remaining, k - len(idx), replace=False)
            idx = np.concatenate([idx, extra])
        coreset = f3_pca[idx]

    np.save(os.path.join(bank_dir, f'{cls_name}_bank.npy'), coreset)
    print(f"  {cls_name}: {n} -> {len(coreset)}")

print(f"\nDone! Bank saved to {bank_dir}")

