"""Predict Boston house prices with the trained model.

Usage:
    python src/predict.py                      # demo on 3 example rows
    python src/predict.py --csv my_houses.csv  # CSV with a header row of feature names
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "boston_house_price_model.joblib"

# Three example neighbourhoods (feature order: see README)
EXAMPLES = pd.DataFrame(
    [
        [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.90, 4.98],
        [0.02731, 0.0, 7.07, 0, 0.469, 6.421, 78.9, 4.9671, 2, 242, 17.8, 396.90, 9.14],
        [0.22489, 12.5, 7.87, 0, 0.524, 6.377, 94.3, 6.3467, 5, 311, 15.2, 392.52, 20.45],
    ],
    columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
             "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
)


def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


def predict_price(rows: pd.DataFrame) -> pd.Series:
    """Return predicted prices (in $1000s) for a DataFrame of feature rows."""
    model, features = load_model()
    missing = [f for f in features if f not in rows.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return pd.Series(model.predict(rows[features]), index=rows.index, name="predicted_MEDV_k$")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, help="CSV file with a header of feature names")
    args = parser.parse_args()

    data = pd.read_csv(args.csv) if args.csv else EXAMPLES
    result = data.copy()
    result["predicted_price_$"] = (predict_price(data) * 1000).round(0)
    pd.set_option("display.width", 200)
    print(result[["RM", "LSTAT", "CRIM", "predicted_price_$"]] if not args.csv else result)
