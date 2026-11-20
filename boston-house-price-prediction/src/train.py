"""Train and evaluate models that predict Boston house prices.

Target: MEDV = median home value in $1000s.

Usage:
    python src/train.py                 # use all 13 features
    python src/train.py --drop-b        # exclude the "B" feature (see README)
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"
MODEL_PATH = ROOT / "models" / "boston_house_price_model.joblib"
FIG_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "metrics.json"

COLUMNS = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
]
TARGET = "MEDV"
SEED = 42


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset. Handles the headerless whitespace-separated version
    (the one in archive.zip) and the comma-separated Kaggle version with a header."""
    with open(path) as f:
        first_line = f.readline()
    if "CRIM" in first_line.upper():
        df = pd.read_csv(path)
        df.columns = [c.strip().upper() for c in df.columns]
    else:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_eda_plots(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[TARGET], bins=30, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("MEDV (median home value, $1000s)")
    ax.set_ylabel("Number of tracts")
    ax.set_title("Distribution of house prices")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_distribution.png", dpi=150)
    plt.close(fig)

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def main(drop_b: bool) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    save_eda_plots(df)

    features = [c for c in df.columns if c != TARGET]
    if drop_b:
        features.remove("B")
    X, y = df[features], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # ---- 1. Compare baseline models with 5-fold cross-validation ----------
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}
    print("\n5-fold cross-validation on the training set:")
    for name, model in candidates.items():
        scores = cross_validate(
            make_pipeline(model), X_train, y_train, cv=cv,
            scoring=("r2", "neg_root_mean_squared_error"),
        )
        cv_results[name] = {
            "cv_r2": float(scores["test_r2"].mean()),
            "cv_rmse": float(-scores["test_neg_root_mean_squared_error"].mean()),
        }
        print(f"  {name:<18} R2={cv_results[name]['cv_r2']:.3f}  RMSE={cv_results[name]['cv_rmse']:.3f}")

    # ---- 2. Tune the two tree-ensemble models ------------------------------
    grids = {
        "Random Forest": (
            RandomForestRegressor(random_state=SEED, n_jobs=-1),
            {"model__n_estimators": [200, 400], "model__max_depth": [None, 12],
             "model__min_samples_leaf": [1, 2]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=SEED),
            {"model__n_estimators": [200, 400], "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [2, 3, 4], "model__subsample": [0.8, 1.0]},
        ),
    }
    tuned = {}
    print("\nTuning with GridSearchCV:")
    for name, (model, grid) in grids.items():
        search = GridSearchCV(
            make_pipeline(model), grid, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned[name] = search
        print(f"  {name:<18} best CV RMSE={-search.best_score_:.3f}  params={search.best_params_}")

    best_name = max(tuned, key=lambda n: tuned[n].best_score_)  # higher neg-RMSE = better
    best_model = tuned[best_name].best_estimator_
    print(f"\nSelected model: {best_name}")

    # ---- 3. Final evaluation on the untouched test set ---------------------
    test_results = {}
    for name, model in candidates.items():
        pipe = make_pipeline(model).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        test_results[name] = {
            "r2": float(r2_score(y_test, pred)),
            "rmse": rmse(y_test, pred),
            "mae": float(mean_absolute_error(y_test, pred)),
        }
    for name, search in tuned.items():
        pred = search.best_estimator_.predict(X_test)
        test_results[f"{name} (tuned)"] = {
            "r2": float(r2_score(y_test, pred)),
            "rmse": rmse(y_test, pred),
            "mae": float(mean_absolute_error(y_test, pred)),
        }

    y_pred = best_model.predict(X_test)
    final = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": rmse(y_test, y_pred),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }
    print("\nTest-set results (20% hold-out):")
    for name, r in test_results.items():
        print(f"  {name:<26} R2={r['r2']:.3f}  RMSE={r['rmse']:.3f}  MAE={r['mae']:.3f}")
    print(f"\nFinal model ({best_name}, tuned): R2={final['r2']:.3f}  "
          f"RMSE={final['rmse']:.3f}  MAE={final['mae']:.3f}  (units: $1000s)")

    # ---- 4. Plots ----------------------------------------------------------
    names = list(test_results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, [test_results[n]["r2"] for n in names], color="#4C78A8")
    ax.set_xlabel("R² on test set")
    ax.set_title("Model comparison")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_test, y_pred, alpha=0.7, color="#4C78A8")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual price ($1000s)")
    ax.set_ylabel("Predicted price ($1000s)")
    ax.set_title(f"Actual vs predicted ({best_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.7, color="#4C78A8")
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted price ($1000s)")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residuals")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals.png", dpi=150)
    plt.close(fig)

    importances = best_model.named_steps["model"].feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(features)[order], importances[order], color="#4C78A8")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance ({best_name})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # ---- 5. Save model + metrics ------------------------------------------
    joblib.dump({"model": best_model, "features": features, "target": TARGET}, MODEL_PATH, compress=3)
    metrics = {
        "selected_model": best_name,
        "best_params": tuned[best_name].best_params_,
        "features_used": features,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "final_test_metrics": final,
        "cross_validation": cv_results,
        "test_results_all_models": test_results,
        "feature_importance": {f: float(i) for f, i in sorted(zip(features, importances), key=lambda t: -t[1])},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model   -> {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metrics -> {METRICS_PATH.relative_to(ROOT)}")
    print(f"Saved figures -> {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drop-b", action="store_true",
                        help="Exclude the 'B' feature (derived from racial demographics).")
    args = parser.parse_args()
    main(drop_b=args.drop_b)
