

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_usage(config: dict) -> pd.DataFrame:

    sd = config["synthetic_data"]
    rng = np.random.default_rng(sd["random_seed"])

    n_accounts = sd["n_accounts"]
    sim_weeks = sd["simulation_weeks"]
    at_risk_frac = sd["at_risk_fraction"]

    sim_end_date = pd.Timestamp.today().normalize()

 
    is_at_risk = rng.random(n_accounts) < at_risk_frac

    rows = []

    for account_id in range(1, n_accounts + 1):
     
        baseline_logins = rng.lognormal(
            mean=sd["baseline_logins_lognormal_mean"],
            sigma=sd["baseline_logins_lognormal_sigma"],
        )
        baseline_feature_events = rng.lognormal(
            mean=sd["baseline_feature_events_lognormal_mean"],
            sigma=sd["baseline_feature_events_lognormal_sigma"],
        )

        signup_week_offset = rng.integers(0, sim_weeks)
        weeks_available = sim_weeks - signup_week_offset

        if is_at_risk[account_id - 1]:
            decay_rate = rng.lognormal(
                mean=sd["decay_rate_lognormal_mean"],
                sigma=sd["decay_rate_lognormal_sigma"],
            )
            ticket_lambda = (
                sd["support_ticket_poisson_lambda_healthy"]
                * sd["support_ticket_risk_multiplier"]
            )
        else:
            decay_rate = 0.0
            ticket_lambda = sd["support_ticket_poisson_lambda_healthy"]

        cancelled = False
        for w in range(weeks_available):
            if cancelled:
                break

            
            decay_multiplier = np.exp(-decay_rate * w)
            noise = rng.lognormal(mean=0.0, sigma=0.3)

            logins = max(0, int(round(baseline_logins * decay_multiplier * noise)))
            feature_events = max(
                0, int(round(baseline_feature_events * decay_multiplier * noise))
            )
            support_tickets = rng.poisson(ticket_lambda)

            week_date = sim_end_date - pd.Timedelta(
                weeks=int(weeks_available - w)
            )

           
            churns_now = (
                is_at_risk[account_id - 1]
                and decay_multiplier < sd["cancellation_engagement_threshold"]
            )

            rows.append(
                {
                    "account_id": account_id,
                    "week_number": w,
                    "week_date": week_date,
                    "logins": logins,
                    "feature_events": feature_events,
                    "support_tickets": int(support_tickets),
                    "churned_this_week": bool(churns_now),
                }
            )

            if churns_now:
                cancelled = True

    usage = pd.DataFrame(rows)
    usage = usage.sort_values(["account_id", "week_number"]).reset_index(drop=True)
    return usage


if __name__ == "__main__":
    
    from config_loader import load_config

    cfg = load_config()
    df = generate_synthetic_usage(cfg)
    n_churned = df.groupby("account_id")["churned_this_week"].any().sum()
    print(f"Generated {len(df):,} weekly rows for {df['account_id'].nunique():,} accounts")
    print(f"Accounts that churned within the window: {n_churned:,} "
          f"({n_churned / df['account_id'].nunique():.1%})")
    print(df.head())
