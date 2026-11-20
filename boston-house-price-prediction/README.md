# Boston House Price Prediction

A machine-learning project that predicts median house prices in the Boston area from 13 neighbourhood features (number of rooms, crime rate, pollution, distance to employment centres, and more).

The target is `MEDV`, the median home value in **$1000s**.

## Results

Models were compared with 5-fold cross-validation on the 80% training split. The two tree ensembles were then tuned with `GridSearchCV`, and the best one by CV RMSE (Gradient Boosting) was evaluated once on the 20% hold-out test set (102 rows).

| Model (test set) | R² | RMSE ($1000s) | MAE ($1000s) |
|---|---|---|---|
| Linear Regression | 0.669 | 4.93 | 3.19 |
| Ridge | 0.668 | 4.93 | 3.19 |
| Lasso | 0.650 | 5.07 | 3.24 |
| Random Forest (tuned) | 0.882 | 2.94 | 2.04 |
| **Gradient Boosting (tuned, selected)** | **0.882** | **2.95** | **2.01** |

The selected model is off by about **$2,000** on average. Tree ensembles clearly beat linear models, which points to non-linear relationships in the data. With only 102 test rows, small differences between the ensemble models are within noise, so exact numbers shift a little with a different random split.

Most important features: `RM` (rooms per dwelling) and `LSTAT` (% lower-status population) together account for roughly 78% of the importance. See `reports/figures/` for all plots and `reports/metrics.json` for every number.

## Project structure

```
.
├── data/housing.csv                          # dataset (506 rows, 14 columns)
├── src/
│   ├── train.py                              # compare, tune, evaluate, save
│   └── predict.py                            # use the saved model
├── models/boston_house_price_model.joblib    # trained model (pipeline + feature list)
├── reports/
│   ├── metrics.json
│   └── figures/                              # EDA and evaluation plots
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

python src/train.py          # retrains everything and overwrites models/ and reports/
python src/predict.py        # demo predictions on 3 example rows
python src/predict.py --csv my_houses.csv
```

If loading the saved model fails with a scikit-learn version error, just run `python src/train.py` to rebuild it for your installed version (the model was trained with scikit-learn 1.9.1).

For `--csv`, the file needs a header row with these columns:
`CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT`

Use from Python:

```python
import pandas as pd
from src.predict import predict_price

house = pd.DataFrame([[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98]],
                     columns=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"])
print(predict_price(house) * 1000)   # predicted price in dollars
```

## Features

| Column | Meaning |
|---|---|
| CRIM | Per-capita crime rate by town |
| ZN | % of residential land zoned for lots over 25,000 sq ft |
| INDUS | % of non-retail business acres per town |
| CHAS | 1 if the tract borders the Charles River, else 0 |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms per dwelling |
| AGE | % of owner-occupied units built before 1940 |
| DIS | Weighted distance to five Boston employment centres |
| RAD | Index of accessibility to radial highways |
| TAX | Property-tax rate per $10,000 |
| PTRATIO | Pupil–teacher ratio by town |
| B | 1000(Bk − 0.63)², where Bk is the proportion of Black residents by town |
| LSTAT | % lower-status population |
| **MEDV** | **Median home value in $1000s (target)** |

## Method

1. **Load**: handles both the headerless whitespace-separated file in this repo and the comma-separated Kaggle version with a header (missing values are median-imputed).
2. **Pipeline**: median imputation → standard scaling → model, so there is no data leakage from the test set.
3. **Compare**: Linear Regression, Ridge, Lasso, Random Forest and Gradient Boosting with 5-fold CV.
4. **Tune**: `GridSearchCV` for Random Forest and Gradient Boosting; the winner is chosen by CV RMSE.
5. **Evaluate**: one final pass on the hold-out test set, plus plots (actual vs predicted, residuals, feature importance).

## Notes and limitations

- **The `B` feature is ethically problematic.** It is derived from the racial composition of each town, and scikit-learn removed this dataset for that reason. It is kept by default to match the classic dataset. To train without it, run `python src/train.py --drop-b`. Do not use a model like this to make real housing or lending decisions.
- **The data is old (1970s) and small.** Prices are not comparable to today's market, so treat this as a learning/portfolio project.
- **Prices are capped at $50k** in the data (16 rows), so the model under-predicts the most expensive tracts (visible in the actual-vs-predicted plot).
