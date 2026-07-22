
import torch, numpy as np, faiss, pickle, os, pandas as pd
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score

device = 'cuda'
bank_dir = '/root/gpufree-data/banks/patchcore_bank'
test_dir = '/root/gpufree-data/datasets/Test_A'
train_dir = '/root/gpufree-data/datasets/train'
top_k_pct = 0.01  # top 1% patches

# Load PCA
with open(os.path.join(bank_dir, 'pca.pkl'), 'rb') as f:
    pca = pickle.load(f)

# Load banks
banks = {}
for f in sorted(os.listdir(bank_dir)):
    if f.endswith('_bank.npy'):
        cls = f.replace('_bank.npy', '')
        banks[cls] = np.load(os.path.join(bank_dir, f))
print(f"Loaded {len(banks)} banks")

# Load WRN50
model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device).eval()
features = {}
model.get_submodule('layer3').register_forward_hook(lambda m,i,o: features.__setitem__('l3', o.detach()))

transform = transforms.Compose([
    transforms.Resize((256,256)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def collect_samples(data_root):
    """Collect (path, class_name) pairs, maintaining original filenames for submission."""
    samples = []
    for cls in sorted(os.listdir(data_root)):
        cls_path = os.path.join(data_root, cls)
        if not os.path.isdir(cls_path): continue
        for item in sorted(os.listdir(cls_path)):
            item_path = os.path.join(cls_path, item)
            if os.path.isdir(item_path):
                for img in sorted(os.listdir(item_path)):
                    if img.endswith('.png'):
                        rel = f"{cls}/{item}/{img}"
                        samples.append((os.path.join(item_path, img), cls, rel))
            elif item.endswith('.png'):
                rel = f"{cls}/{item}"
                samples.append((item_path, cls, rel))
    return samples

def extract_test_feat(path):
    img = Image.open(path).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)
    features.clear()
    with torch.no_grad():
        model(t)
    f3 = features['l3'].squeeze(0)  # (C,H,W)
    c,h,w = f3.shape
    f3 = f3.reshape(c, h*w).T.cpu().numpy()  # (N,C)
    return pca.transform(f3).astype(np.float32)

def score_image(test_feats, bank):
    d = test_feats.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(bank.astype(np.float32))
    distances, _ = index.search(test_feats.astype(np.float32), 1)
    dists = distances.flatten()
    k = max(1, int(len(dists) * top_k_pct))
    return float(np.sort(dists)[:k].mean())

# --- Inference on Test_A ---
print("\n=== Inference on Test_A ===")
test_samples = collect_samples(test_dir)
results = []
for path, cls, rel in tqdm(test_samples, desc='Test_A'):
    if cls not in banks:
        results.append({'filename': rel, 'score': -1, 'class': cls})
        continue
    feats = extract_test_feat(path)
    score = score_image(feats, banks[cls])
    results.append({'filename': rel, 'score': score, 'class': cls})

df_test = pd.DataFrame(results)
os.makedirs('/root/gpufree-data/results/patchcore', exist_ok=True)
df_test.to_csv('/root/gpufree-data/results/patchcore/test_scores.csv', index=False)

# Per-class stats
class_stats = df_test.groupby('class')['score'].agg(['mean','std','min','max','count'])
class_stats = class_stats.sort_values('mean', ascending=False)
print(f"\nPer-class score stats (top 10 highest):")
print(class_stats.head(10).to_string())
print(f"\nOverall: mean={df_test['score'].mean():.4f}, std={df_test['score'].std():.4f}")

# --- Inference on Train (normal only) for baseline ---
print("\n=== Inference on Train ===")
train_samples = collect_samples(train_dir)
train_scores = []
for path, cls, rel in tqdm(train_samples, desc='Train'):
    if cls not in banks: continue
    feats = extract_test_feat(path)
    score = score_image(feats, banks[cls])
    train_scores.append(score)

train_arr = np.array(train_scores)
print(f"Train normal: mean={train_arr.mean():.4f}, std={train_arr.std():.4f}, "
      f"max={train_arr.max():.4f}")

# --- Cross-class validation (one-vs-all) ---
print("\n=== Cross-class AUROC ===")
auroc_scores = []
for cls in sorted(banks.keys()):
    cls_train = [s for p,c,r in train_samples if c == cls]
    # Use first 10 classes worth as "anomalies" for this class
    other_samples = [s for p,c,r in train_samples if c != cls]
    if not cls_train or not other_samples: continue
    
    # Run inference on a subset for speed
    cls_feats = []
    for path, _, _ in [x for x in train_samples if x[1] == cls][:20]:
        cls_feats.append(extract_test_feat(path))
    other_feats = []
    for path, _, _ in [x for x in train_samples if x[1] != cls][:50]:
        other_feats.append(extract_test_feat(path))
    
    cls_scores = [score_image(f, banks[cls]) for f in cls_feats]
    other_scores = [score_image(f, banks[cls]) for f in other_feats]
    
    y_true = [0]*len(cls_scores) + [1]*len(other_scores)
    y_score = cls_scores + other_scores
    try:
        auroc = roc_auc_score(y_true, y_score)
        auroc_scores.append(auroc)
    except: pass

if auroc_scores:
    print(f"Avg one-vs-all AUROC: {np.mean(auroc_scores):.4f} ± {np.std(auroc_scores):.4f}")
    print(f"Median: {np.median(auroc_scores):.4f}, Min: {min(auroc_scores):.4f}, Max: {max(auroc_scores):.4f}")

print("\nDone!")

