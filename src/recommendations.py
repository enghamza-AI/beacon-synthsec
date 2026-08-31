
from __future__ import annotations

import pandas as pd


def generate_recommendations(scored: pd.DataFrame, config: dict) -> pd.DataFrame:

    rec_cfg = config["recommendations"]
    high_threshold = rec_cfg["high_risk_threshold"]
    medium_threshold = rec_cfg["medium_risk_threshold"]

    out = scored.copy()

    actions = []
    urgencies = []

    for _, row in out.iterrows():
        score = row["churn_risk_score"]
        reason = row["primary_risk_reason"]
        segment = row["segment_name"]

        if segment == "new":
            action = (
                "Too new for a reliable risk read yet — focus on "
                "onboarding completion instead of a churn score."
            )
            urgency = "low"

        elif score >= high_threshold:
            action = (
                f"High churn risk ({score:.0f}/100) — {reason.lower()}. "
                f"Reach out personally this week, don't wait for a "
                f"cancellation."
            )
            urgency = "high"

        elif score >= medium_threshold:
            action = (
                f"Moderate churn risk ({score:.0f}/100) — {reason.lower()}. "
                f"Worth a check-in email or in-app nudge, not urgent "
                f"escalation yet."
            )
            urgency = "medium"

        else:
            action = (
                f"Low churn risk ({score:.0f}/100) — usage looks stable. "
                f"No action needed."
            )
            urgency = "low"

        actions.append(action)
        urgencies.append(urgency)

    out["recommended_action"] = actions
    out["urgency"] = urgencies
    return out


if __name__ == "__main__":
  
    from config_loader import load_config
    from synthetic_data import generate_synthetic_usage
    from cleaning import clean_usage
    from feature_engineering import build_account_features
    from segmentation import segment_accounts
    from scoring import train_churn_model, predict_churn_risk

    cfg = load_config()
    raw = generate_synthetic_usage(cfg)
    clean = clean_usage(raw, cfg)
    features = build_account_features(clean, cfg)
    segmented = segment_accounts(features, cfg)
    bundle = train_churn_model(segmented, cfg)
    scored = predict_churn_risk(segmented, bundle)
    recs = generate_recommendations(scored, cfg)
    print(recs["urgency"].value_counts())
    print(recs[["account_id", "segment_name", "urgency",
                 "recommended_action"]].head(3).to_string())
