
from __future__ import annotations

import pandas as pd

from src.config_loader import load_config
from src.synthetic_data import generate_synthetic_usage
from src.cleaning import clean_usage
from src.feature_engineering import build_account_features
from src.segmentation import segment_accounts
from src.scoring import train_churn_model, predict_churn_risk
from src.recommendations import generate_recommendations


def run_pipeline(config: dict, data_mode: str = "demo") -> dict:

    if data_mode == "demo":
        csv_path = config["app"]["demo_csv_path"]
        raw_usage = pd.read_csv(csv_path, parse_dates=["week_date"])
    elif data_mode == "local":
        raw_usage = generate_synthetic_usage(config)
    else:
        raise ValueError(
            f"Unknown data_mode '{data_mode}'. Expected 'demo' or 'local'."
        )

    clean = clean_usage(raw_usage, config)
    features = build_account_features(clean, config)
    segmented = segment_accounts(features, config)
    model_bundle = train_churn_model(segmented, config)
    scored = predict_churn_risk(segmented, model_bundle)
    result = generate_recommendations(scored, config)

    return {
        "result": result,
        "model_auc": model_bundle["test_auc"],
        "silhouette_avg": float(result["silhouette_avg"].iloc[0]),
        "n_accounts": len(result),
    }


if __name__ == "__main__":
  
    cfg = load_config()
    output = run_pipeline(cfg, data_mode="local")
    print(f"Scored {output['n_accounts']:,} accounts")
    print(f"Model AUC: {output['model_auc']}")
    print(f"Silhouette score: {output['silhouette_avg']}")
    print(output["result"]["segment_name"].value_counts())
