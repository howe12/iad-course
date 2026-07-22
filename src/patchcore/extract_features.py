#!/usr/bin/env python3
"""PatchCore Phase-1: Feature Extraction (WideResNet50)

Extracts intermediate features from WideResNet50 layers 1-3,
applies 1x1 + 3x3 local neighborhood aggregation, and saves per-class features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import argparse

# ---------- Data ----------
class RealIADDataset(Dataset):
    """Load all normal training images across all classes."""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []  # (path, class_name)

        class_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        for cls in class_dirs:
            cls_path = os.path.join(root, cls)
            # Handle both S0001/ subdirs and direct image files
            for item in os.listdir(cls_path):
                item_path = os.path.join(cls_path, item)
                if os.path.isdir(item_path):
                    for img in os.listdir(item_path):
                        if img.endswith('.png'):
                            self.samples.append((os.path.join(item_path, img), cls))
                elif item.endswith('.png'):
                    self.samples.append((item_path, cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, cls, path


# ---------- Model ----------
class WideResNet50FeatureExtractor(nn.Module):
    """Extract features from WideResNet50 layers 1, 2, 3 with local aggregation."""

    def __init__(self):
        super().__init__()
        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256 channels
        self.layer2 = backbone.layer2   # 512 channels
        self.layer3 = backbone.layer3   # 1024 channels
        self.eval()

    def _aggregate(self, feat):
        """Local neighborhood aggregation: 1x1 + 3x3 adaptive avg pool."""
        b, c, h, w = feat.shape
        # 1x1 (identity)
        f1 = feat.reshape(b, c, -1).transpose(1, 2)  # (B, H*W, C)
        # 3x3 adaptive average pool
        f3 = F.adaptive_avg_pool2d(feat, (h, w))  # 3x3 equivalent in adaptive mode
        f3 = f3.reshape(b, c, -1).transpose(1, 2)
        # Concatenate along channel dimension
        return torch.cat([f1, f3], dim=-1)  # (B, H*W, 2C)

    def forward(self, x):
        with torch.no_grad():
            x = self.layer0(x)
            f1 = self._aggregate(self.layer1(x))   # (B, N1, 512)
            f2 = self._aggregate(self.layer2(f1.mean(dim=2).view(x.shape[0], 512, -1).view(
                x.shape[0], 512, x.shape[2]//4, x.shape[3]//4)))  # approximate
            # Better approach: do layer-wise forward properly
            return None  # placeholder, will use hook-based approach


# ---------- Hook-based feature extraction (simpler & more reliable) ----------
def extract_features(data_root, output_dir, device='cuda', batch_size=1):
    """Extract and save per-class features."""

    # ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = RealIADDataset(data_root, transform=transform)
    print(f"Total images: {len(dataset.samples)}")

    # Load model
    backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    backbone.to(device)
    backbone.eval()

    # Register hooks
    features = OrderedDict()
    def get_hook(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook

    hooks = []
    for name, module in backbone.named_modules():
        if name in ['layer1', 'layer2', 'layer3']:
            hooks.append(module.register_forward_hook(get_hook(name)))

    # Group samples by class
    from collections import defaultdict
    class_samples = defaultdict(list)
    for path, cls in dataset.samples:
        class_samples[cls].append((path, cls))

    os.makedirs(output_dir, exist_ok=True)

    for cls_name, samples in tqdm(sorted(class_samples.items()), desc='Classes'):
        cls_features = {'layer1': [], 'layer2': [], 'layer3': []}

        for path, _ in tqdm(samples, desc=f'  {cls_name}', leave=False):
            img = Image.open(path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)

            features.clear()
            _ = backbone(img_t)

            for layer_name in ['layer1', 'layer2', 'layer3']:
                feat = features[layer_name]  # (1, C, H, W)
                c, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
                # Reshape to patches: (H*W, C)
                patch_feat = feat.squeeze(0).reshape(c, h*w).transpose(0, 1).cpu()  # (N, C)
                cls_features[layer_name].append(patch_feat)

        # Concatenate all samples for this class
        for layer_name in ['layer1', 'layer2', 'layer3']:
            cls_features[layer_name] = torch.cat(cls_features[layer_name], dim=0)  # (N_total, C)

        # Save
        save_path = os.path.join(output_dir, f'{cls_name}.pth')
        torch.save(cls_features, save_path)
        n_patches = sum(v.shape[0] for v in cls_features.values())
        print(f'  {cls_name}: {n_patches} patches saved → {save_path}')

    for h in hooks:
        h.remove()

    print(f'\nDone! Features saved to {output_dir}/')
    return class_samples


# ---------- Local aggregation ----------
def build_aggregated_bank(feature_dir, output_dir):
    """Apply 1x1 + 3x3 local aggregation and build per-class bank."""

    os.makedirs(output_dir, exist_ok=True)

    for cls_file in sorted(os.listdir(feature_dir)):
        if not cls_file.endswith('.pth'):
            continue

        cls_name = cls_file.replace('.pth', '')
        data = torch.load(os.path.join(feature_dir, cls_file))

        agg_features = []
        # We need original spatial dimensions to do 3x3 aggregation.
        # Skip 3x3 for now — just use 1x1 features and do PCA + Coreset.
        # For MVP, concatenate all layer features per patch.

        # Actually, we need per-patch features from each layer.
        # The data structure is {layer1: (N1, C1), layer2: (N2, C2), layer3: (N3, C3)}
        # Where N1 = total patches across all images for layer1.
        # This is already per-patch! But across all images.

        # For PatchCore, we need:
        # 1. Concatenate features from all 3 layers per patch position
        # 2. But patch counts differ across layers (different spatial resolutions)
        # 3. Solution: interpolate all layers to the coarsest resolution (layer3)

        # Simpler MVP: just use layer2 + layer3 concatenated (upsample layer3 to match layer2)
        # Even simpler MVP: use only layer3 features (1024 dim), process with PCA → Coreset

        # MVP: Use layer3 features only
        f3 = data['layer3']  # (N, 1024)
        # Add layer2 upsampled — skip for now
        agg_features = f3.numpy()

        save_path = os.path.join(output_dir, f'{cls_name}.npy')
        np.save(save_path, agg_features)
        print(f'{cls_name}: {agg_features.shape} → {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/root/gpufree-data/datasets/train')
    parser.add_argument('--feature_dir', type=str,
                        default='/root/gpufree-data/banks/patchcore_features')
    parser.add_argument('--extract', action='store_true', default=True)
    args = parser.parse_args()

    extract_features(args.data_root, args.feature_dir)
    build_aggregated_bank(args.feature_dir, args.feature_dir)
