"""
Loads the German Credit ("credit-g") dataset from OpenML and prepares it
for fairness auditing: derives a `sex` sensitive attribute from the
`personal_status` field and encodes the target as 1 = good credit, 0 = bad.

Caches the raw pull to data/credit_g.csv so later runs don't need internet.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

DATA_DIR = Path(__file__).parent
RAW_CSV = DATA_DIR / "credit_g.csv"


def load_raw() -> pd.DataFrame:
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    bunch = fetch_openml("credit-g", version=1, as_frame=True)
    df = bunch.frame.copy()
    df.to_csv(RAW_CSV, index=False)
    return df


def load_processed() -> pd.DataFrame:
    """Returns a cleaned frame ready for modelling.

    Adds:
      - sex: 'male' / 'female', derived from personal_status
        (credit-g bundles sex and marital status into one field)
      - target: 1 = good credit, 0 = bad credit
    """
    df = load_raw()

    df["sex"] = df["personal_status"].apply(
        lambda v: "female" if "female" in v else "male"
    )
    df["target"] = (df["class"] == "good").astype(int)

    return df


if __name__ == "__main__":
    df = load_processed()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(df[["sex", "age", "foreign_worker", "target"]].head())
