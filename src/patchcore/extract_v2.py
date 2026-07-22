
import torch, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os, pickle, numpy as np
from tqdm import tqdm
from collections import defaultdict

device = 'cuda'
data_root = '/root/gpufree-data/datasets/train'
output_dir = '/root/gpufree-data/banks/patchcore_features'

# Load model
model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device).eval()

# Hooks
features = {}
for name in ['layer1','layer2','layer3']:
    model.get_submodule(name).register_forward_hook(
        lambda m,i,o,n=name: features.__setitem__(n, o.detach()))

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

os.makedirs(output_dir, exist_ok=True)

for cls_name, paths in tqdm(sorted(samples.items()), desc='Classes'):
    all_feats = {'layer1':[], 'layer2':[], 'layer3':[]}
    for path in tqdm(paths, desc=cls_name, leave=False):
        img = Image.open(path).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        features.clear()
        with torch.no_grad():
            model(t)
        for ln in ['layer1','layer2','layer3']:
            f = features[ln].squeeze(0)  # (C,H,W)
            c,h,w = f.shape
            f = f.reshape(c, h*w).T.cpu()  # (N,C)
            all_feats[ln].append(f)
    # Save
    out = {ln: torch.cat(all_feats[ln], dim=0) for ln in all_feats}
    torch.save(out, os.path.join(output_dir, f'{cls_name}.pth'))
    print(f"  {cls_name}: {sum(v.shape[0] for v in out.values())} patches")

print("Done!")

