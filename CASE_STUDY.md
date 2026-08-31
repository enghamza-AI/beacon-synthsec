# Case study: Beacon — Subscription Health Monitor

## The problem

SaaS companies find out a customer was unhappy the moment they cancel — by then, there's nothing left to do. Usage almost always erodes gradually beforehand (fewer logins, narrower feature adoption, more support friction), but most teams have no systematic way to watch that erosion per account before it's too late.

## The approach

Beacon combines two techniques matched to the actual shape of the problem:

1. **Unsupervised health segmentation (K-Means)** groups accounts into healthy, at-risk, dormant, or too-new-to-score tiers, using trend features — how does an account's recent usage compare to its own prior usage — not just current activity level.
2. **Supervised churn-risk classification (RandomForestClassifier)** predicts a 0-100 churn probability per account, with `class_weight="balanced"` and a stratified train/test split to handle the fact that churners are a real minority (~7% in the demo data) — a naive model would otherwise just learn to always predict "safe."

A business-rule layer turns each score into one plain-English reason (e.g. "logins dropped 43% vs. the prior month") and a recommended action, with an explicit carve-out: brand-new accounts get told to focus on onboarding, not judged on an unreliable early risk score.

## Why synthetic data

No client data exists for this demo. The generator (`src/synthetic_data.py`) hides a secret decay trajectory per at-risk account and lets noisy weekly usage counts fall out of it — the model genuinely has to recover the churn signal from realistic noise, not read it off a labeled column.

## Results (on the bundled 2,000-account demo sample)

- 4 health segments recovered (silhouette 0.26 — honestly lower than Pulse's segmentation, reflecting that usage-trend behavior is noisier and less cleanly separable than purchase behavior)
- Churn-risk classifier ROC-AUC of 0.96 on a held-out, stratified test set
- Every high-risk score ships with a specific, plain-English reason, not just a number

## What this demonstrates for client work

The same pipeline (`src/pipeline.py`) is built to run unmodified against a real product-analytics or CRM usage export — swapping `DATA_MODE=local` and pointing the loader at a real file requires no changes to cleaning, feature engineering, segmentation, scoring, or recommendation logic. That's the second proof point (after Pulse) that this pipeline architecture generalizes across verticals, not just within one.
