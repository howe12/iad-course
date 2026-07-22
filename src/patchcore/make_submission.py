import pandas as pd, numpy as np, os, zipfile

# Load PatchCore scores
df = pd.read_csv("/root/gpufree-data/results/patchcore/test_scores.csv")
print(f"Loaded {len(df)} per-image scores")

# Extract product ID (class/SXXXX from class/SXXXX/N.png)
df["product"] = df["filename"].apply(lambda x: "/".join(x.split("/")[:2]))

# Aggregate by product — max across 5 views
submission = df.groupby("product")["score"].max().reset_index()
submission.columns = ["group_folder", "anomaly_score"]
submission = submission.sort_values("group_folder")

print(f"Aggregated to {len(submission)} products")
print(f"Score range: [{submission.anomaly_score.min():.4f}, {submission.anomaly_score.max():.4f}]")

# Save
out_dir = "/root/gpufree-data/results/patchcore"
csv_path = os.path.join(out_dir, "submission.csv")
submission.to_csv(csv_path, index=False)

# Zip
zip_path = os.path.join(out_dir, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(csv_path, "submission.csv")

import os
size_mb = os.path.getsize(zip_path) / 1024
print(f"Submission: {zip_path} ({size_mb:.1f} KB)")
print(f"Products: {len(submission)}")
print(f"Head:\n{submission.head(5).to_string()}")
