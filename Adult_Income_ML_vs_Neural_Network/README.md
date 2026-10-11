# Adult Income Prediction: Classical ML vs Neural Network

Predicting whether a person earns **more than $50K per year** from census data, and comparing tuned classical models (Logistic Regression, Random Forest, XGBoost) against a Keras neural network.

📓 Notebook: `Adult_Income_ML_vs_Neural_Network.ipynb`

---

## Dataset

- **Source:** Adult Census Income (OpenML, `adult` v2), loaded with `fetch_openml`, so no manual download is needed.
- **Size:** 48,842 rows × 15 columns, with 45,222 rows left after dropping rows with missing values.
- **Target:** `class` (`<=50K` / `>50K`), imbalanced at about **76% / 24%**.
- **Features:** age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, native country, and others.
- **Missing values:** about 5.7% in `workclass` and `occupation`, and 1.8% in `native-country`.

## Workflow

1. **EDA:** shape, dtypes, duplicates, missing values, target distribution, high-income rate per occupation.
2. **Visualization:** histograms, boxplots by target, correlation heatmap, high-income rate by workclass / marital status / occupation / education.
3. **Preprocessing:**
   - Drop rows with missing values
   - One-hot encoding (`drop_first=True`), giving **97 features**
   - Stratified 80/20 train/test split
   - `StandardScaler` fitted on the training set only, to avoid data leakage
4. **Classical models:** `GridSearchCV` (3-fold, scored by ROC-AUC) for:
   - Logistic Regression
   - Random Forest
   - XGBoost
5. **Neural network (Keras):** `Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.3) → Dense(1, sigmoid)`, about 20.9K parameters, trained with Adam, binary cross-entropy and early stopping.
6. **Evaluation:** Accuracy, Precision, Recall, F1 and ROC-AUC. Accuracy alone is misleading on imbalanced data.
7. **Feature importance:** top 15 features from the best XGBoost model.

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0.866** | 0.771 | **0.651** | **0.706** | **0.927** |
| Random Forest | 0.858 | 0.786 | 0.587 | 0.672 | 0.914 |
| Neural Network | 0.849 | 0.745 | 0.594 | 0.661 | 0.907 |
| Logistic Regression | 0.846 | 0.731 | 0.598 | 0.658 | 0.905 |

**Takeaway:** on this tabular dataset, gradient boosting (XGBoost) beats the neural network on every metric except precision, where Random Forest leads. The neural network lands about even with the linear baseline.

## Saved artifacts

The last cell exports:

| File | Description |
|---|---|
| `xgboost_income.pkl` | Best tuned XGBoost model |
| `scaler.pkl` | Fitted `StandardScaler` |
| `columns.pkl` | Feature column order after one-hot encoding |
| `income_nn.keras` | Trained Keras network |

## Getting started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow joblib
jupyter notebook Adult_Income_ML_vs_Neural_Network.ipynb
```

Run all cells in order. The dataset is downloaded automatically on the first run, so an internet connection is required. The random seed is fixed (`SEED = 42`), but results may vary slightly across library versions.

## Limitations and ideas

- The dataset contains sensitive attributes (`sex`, `race`) and reflects historical census patterns. It is suited to learning and benchmarking, **not** to real decision-making about people.
- Rows with missing values were dropped rather than imputed.
- Recall for the `>50K` class is only about 0.6 at the default 0.5 threshold. Threshold tuning or class weighting could help.
- Possible next steps: SHAP explanations, a fairness analysis, and a wider hyperparameter search.
