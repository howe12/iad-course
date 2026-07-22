#!/usr/bin/env python3
"""Multi-View Voting: Aggregate v7 single-view scores per product.

Real-IAD provides 5 views (0.png-4.png) per product sample (S0001-S0020).
This script loads v7 single-view scores and aggregates by product.
"""

import pandas as pd
import numpy as np
import os, re, argparse
from collections import defaultdict


def load_v7_scores(v7_csv_path):
    """Load v7 per-image CLS scores. Expects columns: filename, score."""
    df = pd.read_csv(v7_csv_path)
    if 'filename' not in df.columns or 'score' not in df.columns:
        # Try to infer
        print(f"Columns: {df.columns.tolist()}")
        df.columns = df.columns.str.strip().str.lower()
        if 'filename' in df.columns and 'score' in df.columns:
            pass
        elif len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: 'filename', df.columns[1]: 'score'})
    return df


def parse_product_id(filename):
    """
    Extract product identifier from filename.

    Examples:
        '3_adapter/S0001/0.png' → '3_adapter/S0001'
        '3_adapter/0.png'        → '3_adapter'
    """
    parts = filename.replace('\\', '/').split('/')
    # Remove the image file
    if parts[-1].endswith('.png'):
        parts = parts[:-1]
    return '/'.join(parts)


def aggregate_by_product(df, method='mean'):
    """
    Aggregate per-view scores to per-product scores.

    Args:
        df: DataFrame with 'filename' and 'score' columns
        method: 'mean', 'max', 'min', or 'vote'

    Returns:
        DataFrame with 'filename' (product-level) and 'score'
    """
    df = df.copy()
    df['product_id'] = df['filename'].apply(parse_product_id)

    # print stats
    views_per_product = df.groupby('product_id').size()
    print(f"Products: {len(views_per_product)}")
    print(f"Views per product: min={views_per_product.min()}, max={views_per_product.max()}, mean={views_per_product.mean():.1f}")

    if method == 'mean':
        product_scores = df.groupby('product_id')['score'].mean()
    elif method == 'max':
        product_scores = df.groupby('product_id')['score'].max()
    elif method == 'min':
        product_scores = df.groupby('product_id')['score'].min()
    elif method == 'vote':
        # Majority vote: each view votes "normal" or "anomaly" at threshold,
        # then product score = fraction of "anomaly" votes
        threshold = df['score'].median()  # data-driven threshold
        df['anomaly_vote'] = (df['score'] > threshold).astype(float)
        product_scores = df.groupby('product_id')['anomaly_vote'].mean()
    else:
        raise ValueError(f"Unknown method: {method}")

    result = pd.DataFrame({
        'filename': product_scores.index,
        'score': product_scores.values,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v7_csv', required=True, help='Path to v7 single-view scores CSV')
    parser.add_argument('--output', default='/root/gpufree-data/results/multiview/scores.csv')
    parser.add_argument('--method', default='mean', choices=['mean', 'max', 'min', 'vote'])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load
    df = load_v7_scores(args.v7_csv)
    print(f"Loaded {len(df)} single-view scores")
    print(f"Score range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")

    # Aggregate
    for method in ['mean', 'max', 'min', 'vote']:
        result = aggregate_by_product(df, method=method)
        out = args.output.replace('.csv', f'_{method}.csv')
        result.to_csv(out, index=False)
        print(f"\n{method}: {len(result)} products, range=[{result['score'].min():.4f}, {result['score'].max():.4f}]")


if __name__ == '__main__':
    main()
