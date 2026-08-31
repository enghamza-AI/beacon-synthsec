
from __future__ import annotations

import numpy as np
import pandas as pd


def build_account_features(usage: pd.DataFrame, config: dict) -> pd.DataFrame:

    fe_cfg = config["feature_engineering"]
    recent_w = fe_cfg["recent_window_weeks"]
    prior_w = fe_cfg["prior_window_weeks"]

    rows = []

    for account_id, group in usage.groupby("account_id"):
        group = group.sort_values("week_number")
        tenure_weeks = int(group["week_number"].max()) + 1
        has_churned = bool(group["churned_this_week"].any())

        recent = group.tail(recent_w)
       
        prior = group.iloc[max(0, len(group) - recent_w - prior_w): len(group) - recent_w]

        avg_logins_recent = float(recent["logins"].mean())
        avg_feature_events_recent = float(recent["feature_events"].mean())
        support_tickets_recent = int(recent["support_tickets"].sum())
        total_support_tickets = int(group["support_tickets"].sum())

        has_full_trend_window = len(prior) >= max(1, prior_w // 2)

        if has_full_trend_window and prior["logins"].mean() > 0:
            avg_logins_prior = float(prior["logins"].mean())
          
            login_trend_pct = (avg_logins_recent - avg_logins_prior) / avg_logins_prior
        else:
            avg_logins_prior = np.nan
            login_trend_pct = np.nan

        rows.append(
            {
                "account_id": account_id,
                "tenure_weeks": tenure_weeks,
                "avg_logins_last_4w": round(avg_logins_recent, 2),
                "avg_logins_prior_4w": (
                    round(avg_logins_prior, 2) if not np.isnan(avg_logins_prior) else np.nan
                ),
                "login_trend_pct": (
                    round(login_trend_pct, 4) if not np.isnan(login_trend_pct) else np.nan
                ),
                "avg_feature_events_last_4w": round(avg_feature_events_recent, 2),
                "support_tickets_last_4w": support_tickets_recent,
                "total_support_tickets": total_support_tickets,
                "has_full_trend_window": has_full_trend_window,
                "has_churned": has_churned,
            }
        )

    features = pd.DataFrame(rows)

   
    for col in ["avg_logins_prior_4w", "login_trend_pct"]:
        median_val = features[col].median()
        features[col] = features[col].fillna(median_val)

    return features


if __name__ == "__main__":
  
    from config_loader import load_config
    from synthetic_data import generate_synthetic_usage
    from cleaning import clean_usage

    cfg = load_config()
    raw = generate_synthetic_usage(cfg)
    clean = clean_usage(raw, cfg)
    features = build_account_features(clean, cfg)
    print(f"Built features for {len(features):,} accounts")
    print(features.describe(include="all"))
    print(f"\nChurned within window: {features['has_churned'].sum():,} "
          f"({features['has_churned'].mean():.1%})")
