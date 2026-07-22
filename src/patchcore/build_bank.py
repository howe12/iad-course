#!/usr/bin/env python3
"""PatchCore Bank Builder: PCA + Greedy Coreset Sampling

1. Load per-class features from extraction
2. PCA fit on all normal features → reduce to target_dim
3. Greedy Coreset sampling → select ~10K representative patches per class
4. Save bank
"""

import torch
import numpy as np
import os, pickle
from sklearn.decomposition import PCA
from tqdm import tqdm
import faiss
import argparse

def build_coreset_bank(feature_dir, bank_dir, target_dim=256, coreset_size=10000, seed=42):
    """
    Build Per-Class Coreset Bank.

    Args:
        feature_dir: Directory with per-class .pth files (output of extract_v2.py)
        bank_dir: Output directory for bank
        target_dim: PCA target dimension
        coreset_size: Number of features per class after Coreset
    """
    np.random.seed(seed)
    os.makedirs(bank_dir, exist_ok=True)

    # Step 1: Load all features and compute PCA
    print("=== Step 1: Loading features & fitting PCA ===")
    all_layer3 = []
    class_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.pth')])

    for cf in tqdm(class_files, desc='Loading'):
        data = torch.load(os.path.join(feature_dir, cf), map_location='cpu')
        # Use layer3 only for MVP (1024 dim)
        f3 = data['layer3'].numpy()
        all_layer3.append(f3)

    # Fit PCA on sampled subset (to save memory/time)
    sample_size = min(100000, sum(f.shape[0] for f in all_layer3))
    all_concat = np.concatenate(all_layer3, axis=0)
    indices = np.random.choice(len(all_concat), sample_size, replace=False)
    pca_sample = all_concat[indices]

    print(f"Total patches: {all_concat.shape[0]}, PCA sample: {sample_size}")
    pca = PCA(n_components=target_dim, random_state=seed)
    pca.fit(pca_sample)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Step 2: Transform per class + Greedy Coreset
    print("\n=== Step 2: Transform + Coreset per class ===")
    bank = {}  # {class_name: (features_np, indices)}

    for cf in tqdm(class_files, desc='Coreset'):
        cls_name = cf.replace('.pth', '')
        data = torch.load(os.path.join(feature_dir, cf), map_location='cpu')
        f3 = data['layer3'].numpy()  # (N, 1024)

        # PCA transform
        f3_pca = pca.transform(f3).astype(np.float32)  # (N, target_dim)

        # Greedy Coreset via Faiss
        n = f3_pca.shape[0]
        k = min(coreset_size, n)

        if n <= k:
            # Not enough patches, use all
            coreset_idx = np.arange(n)
        else:
            # Use faiss KMeans for approximate greedy coreset
            # Actual PatchCore uses: initialize with one patch, then greedily add farthest
            # Faiss k-means is a good approximation
            kmeans = faiss.Kmeans(d=f3_pca.shape[1], k=k, niter=25, verbose=False, seed=seed)
            kmeans.train(f3_pca)
            # Get cluster centers
            _, coreset_idx = kmeans.index.search(f3_pca, 1)
            coreset_idx = np.unique(coreset_idx.flatten())
            # Pad if needed
            if len(coreset_idx) < k:
                remaining = np.setdiff1d(np.arange(n), coreset_idx)
                extra = np.random.choice(remaining, k - len(coreset_idx), replace=False)
                coreset_idx = np.concatenate([coreset_idx, extra])

        coreset_feats = f3_pca[coreset_idx]

        bank[cls_name] = coreset_feats
        np.save(os.path.join(bank_dir, f'{cls_name}_bank.npy'), coreset_feats)
        print(f"  {cls_name}: {n} → {len(coreset_idx)} patches")

    # Save PCA model
    with open(os.path.join(bank_dir, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    print(f"\nDone! Bank saved to {bank_dir}/")
    print(f"Total bank size: {sum(v.shape[0] for v in bank.values())} patches")
    return bank, pca


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', default='/root/gpufree-data/banks/patchcore_features')
    parser.add_argument('--bank_dir', default='/root/gpufree-data/banks/patchcore_bank')
    parser.add_argument('--target_dim', type=int, default=256)
    parser.add_argument('--coreset_size', type=int, default=10000)
    args = parser.parse_args()

    build_coreset_bank(args.feature_dir, args.bank_dir,
                       args.target_dim, args.coreset_size)
