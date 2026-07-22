#!/usr/bin/env python3
"""PatchCore Inference: Score test images via KNN distance to Coreset Bank.

Scoring: top-1% patch distances → mean → per-image anomaly score.
"""

import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import os, pickle, argparse
from collections import defaultdict
import faiss
from tqdm import tqdm
import pandas as pd


def extract_test_features(model, img_path, transform, device='cuda'):
    """Extract layer3 features for a single test image."""
    features = {}
    def hook(name):
        def fn(m, i, o):
            features[name] = o.detach()
        return fn
    for name in ['layer3']:
        model.get_submodule(name).register_forward_hook(hook(name))

    img = Image.open(img_path).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(t)
    f3 = features['layer3'].squeeze(0)  # (C, H, W)
    c, h, w = f3.shape
    f3 = f3.reshape(c, h*w).T.cpu().numpy()  # (N, 1024)
    return f3


def score_image(test_feats, bank_feats, top_k_pct=0.01):
    """
    Score a single image against a class bank.

    Args:
        test_feats: (N, 256) PCA-transformed test features
        bank_feats: (M, 256) Coreset bank features
        top_k_pct: fraction of patches to use for scoring (e.g., 0.01 = top 1%)

    Returns:
        float: anomaly score (higher = more anomalous)
    """
    # Faiss L2 index
    d = test_feats.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(bank_feats.astype(np.float32))

    # Find nearest neighbor distance for each test patch
    distances, _ = index.search(test_feats.astype(np.float32), 1)  # (N, 1)
    dists = distances.flatten()  # (N,)

    # Top-k% scoring
    k = max(1, int(len(dists) * top_k_pct))
    top_dists = np.sort(dists)[:k]
    score = top_dists.mean()

    return float(score)


def run_inference(bank_dir, pca_path, test_dir, output_path, top_k_pct=0.01):
    """Full inference pipeline."""
    device = 'cuda'

    # Load PCA
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)

    # Load model
    model = models.wide_resnet50_2(
        weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
    ).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load all class banks
    banks = {}
    for f in sorted(os.listdir(bank_dir)):
        if f.endswith('_bank.npy'):
            cls_name = f.replace('_bank.npy', '')
            banks[cls_name] = np.load(os.path.join(bank_dir, f))
    print(f"Loaded {len(banks)} class banks")

    # Collect test samples
    test_samples = defaultdict(list)
    for cls in sorted(os.listdir(test_dir)):
        cls_path = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for item in os.listdir(cls_path):
            item_path = os.path.join(cls_path, item)
            if os.path.isdir(item_path):
                for img in os.listdir(item_path):
                    if img.endswith('.png'):
                        test_samples[cls].append((os.path.join(item_path, img), cls))
            elif item.endswith('.png'):
                test_samples[cls].append((item_path, cls))

    print(f"Test samples: {sum(len(v) for v in test_samples.values())} images across {len(test_samples)} classes")

    # Run inference
    results = []
    for cls_name, samples in tqdm(sorted(test_samples.items()), desc='Inference'):
        if cls_name not in banks:
            print(f"  WARNING: {cls_name} not in bank, skipping")
            continue
        bank = banks[cls_name]

        for img_path, _ in tqdm(samples, desc=f'  {cls_name}', leave=False):
            test_feats_raw = extract_test_features(model, img_path, transform, device)
            test_feats_pca = pca.transform(test_feats_raw).astype(np.float32)
            score = score_image(test_feats_pca, bank, top_k_pct=top_k_pct)
            results.append({
                'filename': os.path.relpath(img_path, test_dir),
                'score': score,
                'class': cls_name,
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"Score range: [{df['score'].min():.4f}, {df['score'].max():.4f}], mean={df['score'].mean():.4f}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank_dir', default='/root/gpufree-data/banks/patchcore_bank')
    parser.add_argument('--pca_path', default='/root/gpufree-data/banks/patchcore_bank/pca.pkl')
    parser.add_argument('--test_dir', default='/root/gpufree-data/datasets/Test_A')
    parser.add_argument('--output', default='/root/gpufree-data/results/patchcore/scores.csv')
    parser.add_argument('--top_k_pct', type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_inference(args.bank_dir, args.pca_path, args.test_dir, args.output, args.top_k_pct)
