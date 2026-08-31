

from __future__ import annotations

import pandas as pd


def clean_usage(usage: pd.DataFrame, config: dict) -> pd.DataFrame:
 
    rules = config["cleaning"]
    df = usage.copy()
    n_before = len(df)

    required_cols = [
        "account_id", "week_number", "week_date",
        "logins", "feature_events", "support_tickets", "churned_this_week",
    ]
    df = df.dropna(subset=required_cols)

    df["week_date"] = pd.to_datetime(df["week_date"], errors="coerce")
    df = df.dropna(subset=["week_date"])

    if rules.get("drop_negative_counts", True):
       
        for col in ["logins", "feature_events", "support_tickets"]:
            df = df[df[col] >= 0]

    df = df.drop_duplicates(subset=["account_id", "week_number"], keep="first")

   
    week_counts = df.groupby("account_id")["week_number"].transform("count")
    df = df[week_counts >= rules["min_weeks_per_account"]]

    df = df.sort_values(["account_id", "week_number"]).reset_index(drop=True)

    n_after = len(df)
    n_dropped = n_before - n_after
    if n_dropped > 0:
        print(f"[cleaning.clean_usage] dropped {n_dropped:,} of {n_before:,} rows "
              f"({n_dropped / n_before:.1%})")

    return df


if __name__ == "__main__":

    from config_loader import load_config
    from synthetic_data import generate_synthetic_usage

    cfg = load_config()
    raw = generate_synthetic_usage(cfg)
    clean = clean_usage(raw, cfg)
    print(f"Raw rows: {len(raw):,} -> Clean rows: {len(clean):,}")
    print(clean.dtypes)
