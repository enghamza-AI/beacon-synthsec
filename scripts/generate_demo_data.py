
from __future__ import annotations

from src.config_loader import load_config
from src.synthetic_data import generate_synthetic_usage


def main() -> None:
    config = load_config()
    demo_max_rows = config["app"]["demo_max_rows"]
    demo_csv_path = config["app"]["demo_csv_path"]

    demo_config = {
        **config,
        "synthetic_data": {
            **config["synthetic_data"],
            "n_accounts": demo_max_rows,
        },
    }

    usage = generate_synthetic_usage(demo_config)
    usage.to_csv(demo_csv_path, index=False)

    print(f"Wrote {len(usage):,} rows "
          f"({usage['account_id'].nunique():,} accounts) to {demo_csv_path}")


if __name__ == "__main__":
    main()
