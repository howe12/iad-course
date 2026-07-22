import torch, numpy as np, faiss, pickle, os
from torchvision import models, transforms
from PIL import Image
device="cuda"

# Load PCA and one bank
with open("/root/gpufree-data/banks/patchcore_bank/pca.pkl","rb") as f: pca=pickle.load(f)
bank=np.load("/root/gpufree-data/banks/patchcore_bank/3_adapter_bank.npy")
print(f"Bank: {bank.shape}, PCA: {pca.n_components_}")

# Load model
model=models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device).eval()
features={}
model.get_submodule("layer3").register_forward_hook(lambda m,i,o: features.__setitem__("l3",o.detach()))
transform=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# Test
img=Image.open("/root/gpufree-data/datasets/Test_A/3_adapter/S0001/0.png").convert("RGB")
t=transform(img).unsqueeze(0).to(device)
with torch.no_grad(): model(t)
f3=features["l3"].squeeze(0); c,h,w=f3.shape; f3=f3.reshape(c,h*w).T.cpu().numpy()
f3_pca=pca.transform(f3).astype(np.float32)

d=f3_pca.shape[1]; index=faiss.IndexFlatL2(d); index.add(bank.astype(np.float32))
distances,_=index.search(f3_pca,1); dists=distances.flatten()
k=max(1,int(len(dists)*0.01)); score=float(np.sort(dists)[:k].mean())
print(f"Score: {score:.4f}")
print("TEST OK")
