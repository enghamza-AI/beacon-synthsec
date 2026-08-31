

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


MODEL_FEATURE_COLS = [
    "tenure_weeks",
    "avg_logins_last_4w",
    "avg_logins_prior_4w",
    "login_trend_pct",
    "avg_feature_events_last_4w",
    "support_tickets_last_4w",
    "total_support_tickets",
]


def train_churn_model(features: pd.DataFrame, config: dict) -> dict:

    scoring_cfg = config["scoring"]

    feature_cols = [c for c in MODEL_FEATURE_COLS if c in features.columns]
    X = features[feature_cols].to_numpy()
    y = features["has_churned"].to_numpy().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=scoring_cfg["test_size"],
        random_state=scoring_cfg["random_state"],
     
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=scoring_cfg["n_estimators"],
        max_depth=scoring_cfg["max_depth"],
        random_state=scoring_cfg["random_state"],
        class_weight="balanced", 
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "test_auc": round(float(test_auc), 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def predict_churn_risk(features: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:

    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    X = features[feature_cols].to_numpy()
 
    probs = model.predict_proba(X)[:, 1]

    out = features.copy()
    out["churn_risk_score"] = np.round(probs * 100, 1)
    out["primary_risk_reason"] = _determine_primary_reason(out)
    return out


def _determine_primary_reason(df: pd.DataFrame) -> pd.Series:

    reasons = []
    for _, row in df.iterrows():
        trend = row["login_trend_pct"]
        tickets = row["support_tickets_last_4w"]
        logins = row["avg_logins_last_4w"]

      
        if logins < 0.5:
            reasons.append("Almost no logins in the last 4 weeks")
        elif tickets >= 2:
            reasons.append(f"Support ticket spike ({tickets} tickets in the last 4 weeks)")
        elif trend <= -0.4:
            reasons.append(f"Logins dropped {abs(trend):.0%} vs. the prior month")
        elif trend <= -0.15:
            reasons.append(f"Usage trending down ({trend:.0%} vs. prior month)")
        else:
            reasons.append("No single dominant risk signal — usage is stable")

    return pd.Series(reasons, index=df.index)


if __name__ == "__main__":
   
    from config_loader import load_config
    from synthetic_data import generate_synthetic_usage
    from cleaning import clean_usage
    from feature_engineering import build_account_features
    from segmentation import segment_accounts

    cfg = load_config()
    raw = generate_synthetic_usage(cfg)
    clean = clean_usage(raw, cfg)
    features = build_account_features(clean, cfg)
    segmented = segment_accounts(features, cfg)

    bundle = train_churn_model(segmented, cfg)
    print(f"Trained on {bundle['n_train']} accounts, "
          f"tested on {bundle['n_test']}, AUC = {bundle['test_auc']}")

    scored = predict_churn_risk(segmented, bundle)
    print(scored[["account_id", "segment_name", "churn_risk_score",
                   "primary_risk_reason"]].sort_values(
        "churn_risk_score", ascending=False
    ).head())
