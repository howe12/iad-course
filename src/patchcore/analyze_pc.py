import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv("/root/gpufree-data/results/patchcore/test_scores.csv")
print(f"Total: {len(df)} images")
print(f"Overall: mean={df.score.mean():.4f}, std={df.score.std():.4f}, min={df.score.min():.4f}, max={df.score.max():.4f}")
print()

# Per-class
per_class = df.groupby("class")["score"].agg(["mean","std","count"]).sort_values("mean", ascending=False)
print("Top 10 highest-mean classes:")
print(per_class.head(10).to_string())
print()
print("Bottom 10 lowest-mean classes:")
print(per_class.tail(10).to_string())
print()

# Score distribution
print(f"Score percentiles: 10%={df.score.quantile(0.1):.4f}, 50%={df.score.quantile(0.5):.4f}, 90%={df.score.quantile(0.9):.4f}")

# Cross-class one-vs-all AUROC
auroc_list = []
for cls in df["class"].unique():
    cls_idx = (df["class"] == cls).astype(int)
    try:
        auroc = roc_auc_score(cls_idx, df["score"])
        auroc_list.append(auroc)
    except: pass
print(f"\nOne-vs-all AUROC: mean={np.mean(auroc_list):.4f}, std={np.std(auroc_list):.4f}, "
      f"min={np.min(auroc_list):.4f}, max={np.max(auroc_list):.4f}")

# Score spread vs v7
print(f"\nScore range ratio (max/min): {df.score.max()/df.score.min():.1f}x")
print(f"Coefficient of Variation: {df.score.std()/df.score.mean():.2f}")
