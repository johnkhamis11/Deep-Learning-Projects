# California Housing Price Prediction: ML vs ANN

A regression project that predicts the **median house value** of California districts, comparing Linear Regression, Random Forest and an Artificial Neural Network (Keras).

📓 Notebook: `California_Housing_ML_vs_ANN.ipynb`

---

## Dataset

- **Source:** `sklearn.datasets.fetch_california_housing` (1990 US Census), so no manual download is needed.
- **Size:** 20,640 districts × 8 features, with no missing values and no duplicates.
- **Target:** `MedHouseVal`, the median house value in units of **$100,000**.
- **Features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`.

## Workflow

1. **EDA:** shape, dtypes, duplicates, missing values, and a check of capped values and extreme outliers.
2. **Visualization:** target distribution, feature histograms, correlation heatmap, income vs value scatter, and a latitude/longitude map coloured by price.
3. **Preprocessing:**
   - Remove the **965 districts whose value is capped** at the maximum
   - Remove extreme outliers (`AveRooms ≥ 20`, `AveBedrms ≥ 5`, `AveOccup ≥ 10`), leaving **19,572 rows**
   - 70 / 15 / 15 train / validation / test split (13,700 / 2,936 / 2,936)
   - `StandardScaler` fitted on the training set only
4. **Models:**
   - Linear Regression (baseline)
   - Random Forest (100 trees), with a feature-importance plot
   - ANN: `Dense(64) → Dense(32) → Dense(16) → Dense(1)`, 3,201 parameters, Adam optimizer, MSE loss, early stopping
5. **Evaluation:** MAE, RMSE and R², plus an actual-vs-predicted plot and a residual histogram for the ANN.

## Results (test set)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest** | **0.299** | **0.451** | **0.783** |
| ANN | 0.336 | 0.478 | 0.756 |
| Linear Regression | 0.429 | 0.584 | 0.635 |

Errors are in units of $100,000, so the Random Forest's MAE of 0.299 is roughly **$29.9K**.

**Takeaway:** both non-linear models clearly beat Linear Regression. The Random Forest edges out the ANN on this small tabular dataset.

## Getting started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
jupyter notebook California_Housing_ML_vs_ANN.ipynb
```

Run all cells in order. The random seed is fixed (`SEED = 42`).

## Limitations and ideas

- Removing capped values and outliers improves training but means the models are **not evaluated on the most expensive districts**.
- The data is from 1990 and describes districts, not individual houses.
- Possible next steps: hyperparameter tuning (`GridSearchCV` / Keras Tuner), gradient boosting (XGBoost / LightGBM), and feature engineering such as distance to the coast or to major cities.
