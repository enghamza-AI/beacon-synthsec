
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def segment_accounts(features: pd.DataFrame, config: dict) -> pd.DataFrame:
 
    seg_cfg = config["segmentation"]
    feature_cols = seg_cfg["clustering_features"]

    X = features[feature_cols].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=seg_cfg["n_clusters"],
        random_state=seg_cfg["random_state"],
        n_init=seg_cfg["n_init"],
    )
    cluster_ids = kmeans.fit_predict(X_scaled)

    sample_size = min(2000, len(X_scaled))
    sample_idx = np.random.default_rng(seg_cfg["random_state"]).choice(
        len(X_scaled), size=sample_size, replace=False
    )
    sil_score = silhouette_score(X_scaled[sample_idx], cluster_ids[sample_idx])

    out = features.copy()
    out["cluster_id"] = cluster_ids
    out["silhouette_avg"] = round(float(sil_score), 3)
    out["segment_name"] = _name_segments(out)

    return out


def _name_segments(df: pd.DataFrame) -> pd.Series:

    centroid_stats = df.groupby("cluster_id").agg(
        mean_trend=("login_trend_pct", "mean"),
        mean_logins=("avg_logins_last_4w", "mean"),
        mean_tenure=("tenure_weeks", "mean"),
        frac_new=("has_full_trend_window", lambda s: (~s).mean()),
    )

   
    remaining = list(centroid_stats.index)
    label_by_cluster = {}

    new_id = centroid_stats.loc[remaining, "frac_new"].idxmax()
    label_by_cluster[new_id] = "new"
    remaining.remove(new_id)

    at_risk_id = centroid_stats.loc[remaining, "mean_trend"].idxmin()
    label_by_cluster[at_risk_id] = "at_risk"
    remaining.remove(at_risk_id)

    dormant_id = centroid_stats.loc[remaining, "mean_logins"].idxmin()
    label_by_cluster[dormant_id] = "dormant"
    remaining.remove(dormant_id)

  
    for cid in remaining:
        label_by_cluster[cid] = "healthy"

    return df["cluster_id"].map(label_by_cluster)


if __name__ == "__main__":
  
    from config_loader import load_config
    from synthetic_data import generate_synthetic_usage
    from cleaning import clean_usage
    from feature_engineering import build_account_features

    cfg = load_config()
    raw = generate_synthetic_usage(cfg)
    clean = clean_usage(raw, cfg)
    features = build_account_features(clean, cfg)
    segmented = segment_accounts(features, cfg)
    print(segmented["segment_name"].value_counts())
    print(f"\nSilhouette score: {segmented['silhouette_avg'].iloc[0]}")
